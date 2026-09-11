"""NLLB translation model with a graph-based biaffine dependency head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.gelu(self.linear(value)))


class Biaffine(nn.Module):
    def __init__(self, left_dim: int, right_dim: int, outputs: int = 1) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(outputs, left_dim + 1, right_dim + 1))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left = torch.cat((left, torch.ones_like(left[..., :1])), dim=-1)
        right = torch.cat((right, torch.ones_like(right[..., :1])), dim=-1)
        return torch.einsum("id,odk,jk->oij", left, self.weight, right)


@dataclass
class ParseOutput:
    arc_scores: torch.Tensor
    relation_scores: torch.Tensor | None = None


class JointNLLBDependencyModel(nn.Module):
    """A shared NLLB decoder with its LM head plus a biaffine dependency head."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        relation_count: int,
        root_relation_id: int,
        parser_dim: int,
        dropout: float,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        torch_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        base = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        base.config.use_cache = False
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        lora = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        self.mt = get_peft_model(base, lora)
        self.root_relation_id = root_relation_id
        hidden = base.config.d_model
        self.root = nn.Parameter(torch.zeros(1, hidden))
        self.arc_dep = MLP(hidden, parser_dim, dropout)
        self.arc_head = MLP(hidden, parser_dim, dropout)
        self.rel_dep = MLP(hidden, parser_dim, dropout)
        self.rel_head = MLP(hidden, parser_dim, dropout)
        self.arc_biaffine = Biaffine(parser_dim, parser_dim, outputs=1)
        self.rel_biaffine = Biaffine(parser_dim, parser_dim, outputs=relation_count)

    def parse(self, word_states: torch.Tensor, gold_heads: torch.Tensor | None = None) -> ParseOutput:
        # Parser weights stay fp32 for stability even when the NLLB backbone uses fp16.
        words = word_states.float()
        heads = torch.cat((self.root, words), dim=0)
        arc_scores = self.arc_biaffine(self.arc_dep(words), self.arc_head(heads)).squeeze(0)
        relation_scores = None
        if gold_heads is not None:
            chosen = heads[gold_heads]
            # One head per dependent; use the diagonal of the pairwise relation tensor.
            pairwise = self.rel_biaffine(self.rel_dep(words), self.rel_head(chosen))
            relation_scores = pairwise.diagonal(dim1=1, dim2=2).transpose(0, 1)
        return ParseOutput(arc_scores=arc_scores, relation_scores=relation_scores)

    def parsing_losses(
        self,
        decoder_hidden: torch.Tensor,
        boundary_positions: Sequence[Sequence[int]],
        gold_heads: Sequence[torch.Tensor],
        gold_relations: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        arc_losses, relation_losses = [], []
        for batch_index, positions in enumerate(boundary_positions):
            states = decoder_hidden[batch_index, torch.as_tensor(positions, device=decoder_hidden.device)]
            heads = gold_heads[batch_index].to(decoder_hidden.device)
            relations = gold_relations[batch_index].to(decoder_hidden.device)
            parsed = self.parse(states, heads)
            arc_losses.append(F.cross_entropy(parsed.arc_scores, heads))
            relation_losses.append(F.cross_entropy(parsed.relation_scores, relations))
        return torch.stack(arc_losses).mean(), torch.stack(relation_losses).mean()

    def predict_tree(self, word_states: torch.Tensor) -> tuple[list[int], torch.Tensor]:
        parsed = self.parse(word_states)
        heads = one_root_mst(parsed.arc_scores.detach().float().cpu())
        head_tensor = torch.tensor(heads, dtype=torch.long, device=word_states.device)
        relation_scores = self.parse(word_states, head_tensor).relation_scores
        for dependent, head in enumerate(heads):
            if head == 0:
                relation_scores[dependent].fill_(-float("inf"))
                relation_scores[dependent, self.root_relation_id] = 0.0
            else:
                relation_scores[dependent, self.root_relation_id] = -float("inf")
        relations = relation_scores.argmax(-1)
        return heads, relations


def one_root_mst(arc_scores: torch.Tensor) -> list[int]:
    """Decode a non-projective, single-root maximum spanning arborescence."""
    if arc_scores.ndim != 2 or arc_scores.shape[1] != arc_scores.shape[0] + 1:
        raise ValueError(f"Expected [n,n+1] arc scores, got {tuple(arc_scores.shape)}")
    n = arc_scores.shape[0]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n + 1))
    magnitude = float(arc_scores.abs().max()) if arc_scores.numel() else 1.0
    root_penalty = (magnitude + 1.0) * (n + 1) * 10.0
    for dependent in range(1, n + 1):
        for head in range(0, n + 1):
            if head == dependent:
                continue
            score = float(arc_scores[dependent - 1, head])
            if head == 0:
                score -= root_penalty
            graph.add_edge(head, dependent, weight=score)
    tree = nx.maximum_spanning_arborescence(graph, attr="weight", default=-1e30)
    predicted = [-1] * n
    for head, dependent in tree.edges():
        if dependent:
            predicted[dependent - 1] = head
    if any(head < 0 for head in predicted) or sum(head == 0 for head in predicted) != 1:
        raise RuntimeError(f"Invalid decoded tree: {predicted}")
    return predicted
