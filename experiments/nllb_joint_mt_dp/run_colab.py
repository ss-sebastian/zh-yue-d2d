#!/usr/bin/env python3
"""Run the independent NLLB + biaffine two-head experiment in Google Colab."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import sacrebleu
import torch
import accelerate
import networkx
import peft
import safetensors
import transformers
from IPython.display import display
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from joint_model import JointNLLBDependencyModel


# ------------------------------- configuration -------------------------------
SEED = 42
MODEL_ID = "facebook/nllb-200-distilled-600M"
MODEL_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
SRC_LANG = "zho_Hant"
TGT_LANG = "yue_Hant"
UD_RELEASE = "r2.18"
TRAIN_RATIO, DEV_RATIO = 0.8, 0.1
MAX_SOURCE_TOKENS = 256
MAX_TARGET_TOKENS = 384
MAX_NEW_TOKENS = 256
TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
MAX_EPOCHS = 15
PATIENCE = 5
MIN_JOINT_IMPROVEMENT = 0.05
MT_LEARNING_RATE = 2e-4
PARSER_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.08
GRAD_CLIP = 1.0
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
PARSER_DIM = 256
PARSER_DROPOUT = 0.25
ARC_LOSS_WEIGHT = 0.5
RELATION_LOSS_WEIGHT = 0.5
NUM_BEAMS = 1
DEBUG_MODE = False

if DEBUG_MODE:
    MAX_EPOCHS, PATIENCE = 1, 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = DEVICE.type == "cuda"
LOAD_DTYPE = torch.float16 if USE_FP16 else torch.float32
BASE = Path("/content/nllb_joint_mt_dp") if Path("/content").exists() else Path("nllb_joint_mt_dp")
RAW = BASE / "raw"
RESULTS = BASE / "results"
CHECKPOINT = RESULTS / "checkpoint"
for directory in (RAW, RESULTS, CHECKPOINT, RESULTS / "splits", RESULTS / "plots", RESULTS / "code"):
    directory.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything()
print("Python", platform.python_version(), "PyTorch", torch.__version__, "device", DEVICE)
if DEVICE.type != "cuda":
    print("WARNING: select a GPU runtime for the full experiment.")


# ---------------------------------- dataset ----------------------------------
SOURCES = {
    "mandarin": {
        "repo": "UniversalDependencies/UD_Chinese-HK",
        "filename": "zh_hk-ud-test.conllu",
        "sha256": "a71ff348dd1e45c2cd525ed537a78d6c4b6c5edc15a19641bedc0a8465de820e",
    },
    "cantonese": {
        "repo": "UniversalDependencies/UD_Cantonese-HK",
        "filename": "yue_hk-ud-test.conllu",
        "sha256": "cbd843a195d0db4cdafbf6fcafb7b7b559afea750411006f4728311e70cc4e2a",
    },
}


@dataclass
class Sentence:
    sent_id: str
    parallel_id: str
    text: str
    forms: list[str]
    heads: list[int]
    relations: list[str]
    raw: str


@dataclass
class Pair:
    pair_id: str
    mandarin: Sentence
    cantonese: Sentence


def download_sources() -> dict:
    provenance = {"release": UD_RELEASE, "download_date": str(date.today()), "files": {}}
    for language, information in SOURCES.items():
        url = f"https://raw.githubusercontent.com/{information['repo']}/{UD_RELEASE}/{information['filename']}"
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        digest = hashlib.sha256(response.content).hexdigest()
        if digest != information["sha256"]:
            raise RuntimeError(f"Checksum mismatch for {language}: {digest}")
        path = RAW / information["filename"]
        path.write_bytes(response.content)
        shutil.copy2(path, RESULTS / information["filename"])
        provenance["files"][language] = {**information, "url": url}
    return provenance


def parse_conllu(path: Path) -> list[Sentence]:
    sentences = []
    text = path.read_text(encoding="utf-8").strip()
    for block in text.split("\n\n"):
        metadata, forms, heads, relations = {}, [], [], []
        for line in block.splitlines():
            if line.startswith("#"):
                if " = " in line:
                    key, value = line[2:].split(" = ", 1)
                    metadata[key] = value
                continue
            columns = line.split("\t")
            if len(columns) != 10 or not columns[0].isdigit():
                continue
            forms.append(columns[1])
            heads.append(int(columns[6]))
            relations.append(columns[7])
        if forms:
            sentences.append(
                Sentence(
                    sent_id=metadata["sent_id"],
                    parallel_id=metadata["parallel_id"],
                    text=metadata.get("text", "".join(forms)),
                    forms=forms,
                    heads=heads,
                    relations=relations,
                    raw=block + "\n\n",
                )
            )
    return sentences


def validate_tree(sentence: Sentence) -> None:
    n = len(sentence.forms)
    if len(sentence.heads) != n or len(sentence.relations) != n:
        raise ValueError(sentence.sent_id)
    if sum(head == 0 for head in sentence.heads) != 1:
        raise ValueError(f"Not single-root: {sentence.sent_id}")
    if any(head < 0 or head > n for head in sentence.heads):
        raise ValueError(f"Invalid head: {sentence.sent_id}")
    for dependent in range(1, n + 1):
        seen, node = set(), dependent
        while node:
            if node in seen:
                raise ValueError(f"Cycle: {sentence.sent_id}")
            seen.add(node)
            node = sentence.heads[node - 1]


def is_projective(heads: list[int]) -> bool:
    arcs = [(min(dependent, head), max(dependent, head)) for dependent, head in enumerate(heads, 1)]
    return not any(
        left_a < left_b < right_a < right_b or left_b < left_a < right_b < right_a
        for index, (left_a, right_a) in enumerate(arcs)
        for left_b, right_b in arcs[index + 1 :]
    )


provenance = download_sources()
mandarin_all = parse_conllu(RAW / SOURCES["mandarin"]["filename"])
cantonese_all = parse_conllu(RAW / SOURCES["cantonese"]["filename"])
for sentence in mandarin_all + cantonese_all:
    validate_tree(sentence)
mandarin_by_parallel = {sentence.parallel_id: sentence for sentence in mandarin_all}
cantonese_by_parallel = {sentence.parallel_id: sentence for sentence in cantonese_all}
if len(mandarin_by_parallel) != len(mandarin_all) or len(cantonese_by_parallel) != len(cantonese_all):
    raise RuntimeError("Duplicate parallel_id")
shared_ids = sorted(
    set(mandarin_by_parallel) & set(cantonese_by_parallel),
    key=lambda value: int(value.split("/")[-1]),
)
pairs = [Pair(key, mandarin_by_parallel[key], cantonese_by_parallel[key]) for key in shared_ids]
if len(pairs) != 1004:
    raise RuntimeError(f"Expected 1004 explicit pairs, found {len(pairs)}")

shuffled = list(pairs)
random.Random(SEED).shuffle(shuffled)
n_train = int(len(shuffled) * TRAIN_RATIO)
n_dev = int(len(shuffled) * DEV_RATIO)
train_pairs = shuffled[:n_train]
dev_pairs = shuffled[n_train : n_train + n_dev]
test_pairs = shuffled[n_train + n_dev :]
if DEBUG_MODE:
    train_pairs, dev_pairs, test_pairs = train_pairs[:12], dev_pairs[:3], test_pairs[:3]
if set(p.pair_id for p in train_pairs) & set(p.pair_id for p in dev_pairs + test_pairs):
    raise RuntimeError("Split overlap")
for name, items in (("train", train_pairs), ("dev", dev_pairs), ("test", test_pairs)):
    (RESULTS / "splits" / f"{name}_ids.txt").write_text(
        "\n".join(pair.pair_id for pair in items) + "\n", encoding="utf-8"
    )
print("pairs/splits", len(pairs), len(train_pairs), len(dev_pairs), len(test_pairs))

relation_vocabulary = ["<unk_rel>"] + sorted({relation for p in train_pairs for relation in p.cantonese.relations})
relation_to_id = {relation: index for index, relation in enumerate(relation_vocabulary)}
data_audit = {
    "total_pairs": len(pairs),
    "split_sizes": {"train": len(train_pairs), "dev": len(dev_pairs), "test": len(test_pairs)},
    "mandarin_dependency_tree_used_as_model_input": False,
    "target_nonprojective_trees": sum(not is_projective(pair.cantonese.heads) for pair in pairs),
    "target_nonprojective_trees_retained": sum(not is_projective(pair.cantonese.heads) for pair in pairs),
    "target_transition_oracle_used": False,
}
(RESULTS / "data_audit.json").write_text(json.dumps(data_audit, indent=2), encoding="utf-8")


# ------------------------- tokenizer and two-head model -----------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, src_lang=SRC_LANG, tgt_lang=TGT_LANG, use_fast=True
)
WORD_SEPARATOR = "|"
WORD_SEPARATOR_ID = tokenizer(WORD_SEPARATOR, add_special_tokens=False).input_ids[0]
if tokenizer(WORD_SEPARATOR, add_special_tokens=False).input_ids != [WORD_SEPARATOR_ID]:
    raise RuntimeError("The internal UD-word separator is not atomic")
if any(WORD_SEPARATOR in form for pair in pairs for form in pair.cantonese.forms):
    raise RuntimeError("Separator collides with a corpus token")


def serialize_forms(forms: list[str]) -> str:
    if not forms:
        raise ValueError("Cannot serialize an empty sentence")
    return f" {WORD_SEPARATOR} ".join(forms)


def boundary_positions(label_ids: list[int], expected_words: int) -> list[int]:
    separators = [i for i, token_id in enumerate(label_ids) if token_id == WORD_SEPARATOR_ID]
    eos = [i for i, token_id in enumerate(label_ids) if token_id == tokenizer.eos_token_id]
    if len(separators) != expected_words - 1 or not eos:
        raise RuntimeError(
            f"Target boundary failure: words={expected_words}, separators={len(separators)}, eos={eos}"
        )
    final_eos = next((position for position in eos if position > (separators[-1] if separators else -1)), None)
    if final_eos is None:
        raise RuntimeError("No EOS after the final target word")
    return separators + [final_eos]


def collate(batch: list[Pair]) -> dict:
    sources = ["".join(pair.mandarin.forms) for pair in batch]
    targets = [serialize_forms(pair.cantonese.forms) for pair in batch]
    encoded_source = tokenizer(
        sources, padding=True, truncation=True, max_length=MAX_SOURCE_TOKENS, return_tensors="pt"
    )
    encoded_target = tokenizer(
        text_target=targets, padding=True, truncation=True, max_length=MAX_TARGET_TOKENS, return_tensors="pt"
    )
    original_labels = encoded_target.input_ids
    boundaries = []
    for row, pair in zip(original_labels.tolist(), batch):
        valid = [token_id for token_id in row if token_id != tokenizer.pad_token_id]
        boundaries.append(boundary_positions(valid, len(pair.cantonese.forms)))
    labels = original_labels.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        "pairs": batch,
        "input_ids": encoded_source.input_ids,
        "attention_mask": encoded_source.attention_mask,
        "labels": labels,
        "boundaries": boundaries,
        "heads": [torch.tensor(pair.cantonese.heads, dtype=torch.long) for pair in batch],
        "relations": [
            torch.tensor([relation_to_id.get(rel, 0) for rel in pair.cantonese.relations], dtype=torch.long)
            for pair in batch
        ],
    }


model = JointNLLBDependencyModel(
    model_id=MODEL_ID,
    revision=MODEL_REVISION,
    relation_count=len(relation_vocabulary),
    root_relation_id=relation_to_id["root"],
    parser_dim=PARSER_DIM,
    dropout=PARSER_DROPOUT,
    lora_rank=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    torch_dtype=LOAD_DTYPE,
).to(DEVICE)
model.mt.print_trainable_parameters()


def move_batch(batch: dict) -> dict:
    for key in ("input_ids", "attention_mask", "labels"):
        batch[key] = batch[key].to(DEVICE)
    return batch


def joint_loss(batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    output = model.mt(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        output_hidden_states=True,
        return_dict=True,
    )
    arc_loss, relation_loss = model.parsing_losses(
        output.decoder_hidden_states[-1], batch["boundaries"], batch["heads"], batch["relations"]
    )
    total = output.loss + ARC_LOSS_WEIGHT * arc_loss + RELATION_LOSS_WEIGHT * relation_loss
    return total, output.loss, arc_loss, relation_loss


# ----------------------------- decoding and metrics ---------------------------
# Both SentencePiece forms decode as a vertical bar. Supervision always uses WORD_SEPARATOR_ID;
# accepting the alternate form makes inference robust without adding a new vocabulary row.
WORD_SEPARATOR_IDS = {
    tokenizer.get_vocab()[token]
    for token in ("▁|", "|")
    if token in tokenizer.get_vocab()
}
if WORD_SEPARATOR_ID not in WORD_SEPARATOR_IDS:
    raise RuntimeError("Primary separator ID missing from its accepted-ID set")
# Keep UNK inside a word segment: dropping it would silently remove a dependency node.
ALL_IGNORED_SPECIAL_IDS = set(tokenizer.all_special_ids) - {tokenizer.unk_token_id}
TARGET_LANGUAGE_ID = tokenizer.convert_tokens_to_ids(TGT_LANG)


def decode_generated_ids(ids: list[int]) -> tuple[list[str], dict]:
    retained = [token_id for token_id in ids if token_id not in ALL_IGNORED_SPECIAL_IDS]
    segments, current, empty_segments = [], [], 0
    separator_count = sum(token_id in WORD_SEPARATOR_IDS for token_id in retained)
    for token_id in retained:
        if token_id in WORD_SEPARATOR_IDS:
            if current:
                segments.append(current)
            else:
                empty_segments += 1
            current = []
        else:
            current.append(token_id)
    if current:
        segments.append(current)
    elif retained and retained[-1] in WORD_SEPARATOR_IDS:
        empty_segments += 1
    forms = []
    for segment in segments:
        form = tokenizer.decode(segment, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        form = form.strip().replace(" ", "")
        if form:
            forms.append(form)
        else:
            empty_segments += 1
    if not forms:
        forms = ["<unk>"]
    diagnostics = {
        "separator_count": separator_count,
        "empty_segment_count": empty_segments,
        "ended_with_eos": tokenizer.eos_token_id in ids,
        "well_formed": empty_segments == 0 and tokenizer.eos_token_id in ids,
    }
    return forms, diagnostics


def normalized_gold_forms(pair: Pair) -> list[str]:
    normalized = []
    for form in pair.cantonese.forms:
        ids = tokenizer(form, add_special_tokens=False).input_ids
        value = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        value = value.strip().replace(" ", "")
        normalized.append(value or tokenizer.unk_token)
    if len(normalized) != len(pair.cantonese.forms):
        raise RuntimeError("NLLB normalization changed the number of UD nodes")
    return normalized


@torch.no_grad()
def decoder_word_states(source_text: str, forms: list[str]) -> torch.Tensor:
    source = tokenizer(
        source_text, truncation=True, max_length=MAX_SOURCE_TOKENS, return_tensors="pt"
    ).to(DEVICE)
    target = tokenizer(
        text_target=serialize_forms(forms), truncation=True, max_length=MAX_TARGET_TOKENS, return_tensors="pt"
    )
    labels = target.input_ids.to(DEVICE)
    positions = boundary_positions(target.input_ids[0].tolist(), len(forms))
    output = model.mt(
        **source, labels=labels, output_hidden_states=True, return_dict=True
    )
    return output.decoder_hidden_states[-1][0, torch.tensor(positions, device=DEVICE)]


@torch.no_grad()
def translate_and_parse(source_text: str) -> dict:
    """Return Cantonese UD forms, HEADs, and DEPRELs from Mandarin text alone."""
    model.eval()
    source = tokenizer(
        source_text, truncation=True, max_length=MAX_SOURCE_TOKENS, return_tensors="pt"
    ).to(DEVICE)
    generated = model.mt.generate(
        **source,
        forced_bos_token_id=TARGET_LANGUAGE_ID,
        num_beams=NUM_BEAMS,
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
    )[0].tolist()
    forms, diagnostics = decode_generated_ids(generated)
    states = decoder_word_states(source_text, forms)
    heads, relation_ids = model.predict_tree(states)
    relations = [relation_vocabulary[index] for index in relation_ids.tolist()]
    return {
        "forms": forms,
        "heads": heads,
        "relations": relations,
        "generation_ids": generated,
        **diagnostics,
    }


def translate_and_parse_pair(pair: Pair) -> dict:
    return translate_and_parse("".join(pair.mandarin.forms))


@torch.no_grad()
def parse_gold_tokens(pair: Pair) -> dict:
    model.eval()
    states = decoder_word_states("".join(pair.mandarin.forms), pair.cantonese.forms)
    heads, relation_ids = model.predict_tree(states)
    return {
        "forms": pair.cantonese.forms,
        "heads": heads,
        "relations": [relation_vocabulary[index] for index in relation_ids.tolist()],
    }


def attachment_metrics(predictions: list[dict], items: list[Pair]) -> dict:
    total = uas = las = exact = 0
    for prediction, pair in zip(predictions, items):
        sentence_exact = True
        for predicted_head, predicted_relation, gold_head, gold_relation in zip(
            prediction["heads"], prediction["relations"], pair.cantonese.heads, pair.cantonese.relations
        ):
            total += 1
            head_ok = predicted_head == gold_head
            label_ok = predicted_relation == gold_relation
            uas += head_ok
            las += head_ok and label_ok
            sentence_exact &= head_ok and label_ok
        exact += sentence_exact
    return {"UAS": 100 * uas / total, "LAS": 100 * las / total, "exact_tree": 100 * exact / len(items)}


def translation_metrics(predictions: list[dict], items: list[Pair]) -> dict:
    surface_hypotheses = ["".join(prediction["forms"]) for prediction in predictions]
    raw_surface_references = [["".join(pair.cantonese.forms) for pair in items]]
    normalized_references_by_sentence = [normalized_gold_forms(pair) for pair in items]
    normalized_surface_references = [["".join(forms) for forms in normalized_references_by_sentence]]
    token_hypotheses = [" ".join(prediction["forms"]) for prediction in predictions]
    token_references = [[" ".join(forms) for forms in normalized_references_by_sentence]]
    exact = [prediction["forms"] == forms for prediction, forms in zip(predictions, normalized_references_by_sentence)]
    result = {
        "normalized_surface_BLEU_zh": sacrebleu.corpus_bleu(surface_hypotheses, normalized_surface_references, tokenize="zh").score,
        "normalized_surface_chrF": sacrebleu.corpus_chrf(surface_hypotheses, normalized_surface_references).score,
        "raw_reference_surface_BLEU_zh": sacrebleu.corpus_bleu(surface_hypotheses, raw_surface_references, tokenize="zh").score,
        "raw_reference_surface_chrF": sacrebleu.corpus_chrf(surface_hypotheses, raw_surface_references).score,
        "UD_token_BLEU": sacrebleu.corpus_bleu(token_hypotheses, token_references, tokenize="none").score,
        "exact_normalized_UD_token_match_rate": 100 * sum(exact) / len(exact),
        "mean_generated_UD_tokens": float(np.mean([len(prediction["forms"]) for prediction in predictions])),
        "mean_gold_UD_tokens": float(np.mean([len(pair.cantonese.forms) for pair in items])),
        "structurally_decodable_generation_rate": 100 * sum(p["well_formed"] for p in predictions) / len(predictions),
        "gold_boundary_count_match_rate": 100 * sum(
            prediction["separator_count"] == len(pair.cantonese.forms) - 1
            for prediction, pair in zip(predictions, items)
        ) / len(predictions),
    }
    exact_indices = [index for index, match in enumerate(exact) if match]
    if exact_indices:
        subset = attachment_metrics(
            [predictions[index] for index in exact_indices], [items[index] for index in exact_indices]
        )
        result["exact_token_subset_UAS"] = subset["UAS"]
        result["exact_token_subset_LAS"] = subset["LAS"]
        result["exact_token_subset_sentences"] = len(exact_indices)
    else:
        result.update(
            exact_token_subset_UAS=None, exact_token_subset_LAS=None, exact_token_subset_sentences=0
        )
    return result


@torch.no_grad()
def evaluate(items: list[Pair], description: str) -> tuple[dict, list[dict], list[dict]]:
    generated = [translate_and_parse_pair(pair) for pair in tqdm(items, desc=f"{description} generation")]
    gold_conditioned = [parse_gold_tokens(pair) for pair in tqdm(items, desc=f"{description} gold parsing")]
    metrics = {
        "translation": translation_metrics(generated, items),
        "gold_token_conditioned_parsing": attachment_metrics(gold_conditioned, items),
    }
    return metrics, generated, gold_conditioned


@torch.no_grad()
def mean_dev_losses(items: list[Pair]) -> dict:
    model.eval()
    totals = np.zeros(4)
    loader = DataLoader(items, batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=collate)
    for batch in loader:
        batch = move_batch(batch)
        values = joint_loss(batch)
        totals += np.array([float(value) for value in values]) * len(batch["pairs"])
    totals /= len(items)
    return dict(zip(("total_loss", "mt_loss", "arc_loss", "relation_loss"), totals.tolist()))


# ------------------------------- checkpoint I/O -------------------------------
def save_checkpoint(epoch: int, joint_score: float, dev_metrics: dict, dev_losses: dict) -> None:
    tensors = {}
    for key, value in get_peft_model_state_dict(model.mt).items():
        tensors[f"adapter::{key}"] = value.detach().cpu().contiguous()
    for key, value in model.state_dict().items():
        if not key.startswith("mt."):
            tensors[f"parser::{key}"] = value.detach().cpu().contiguous()
    save_file(tensors, CHECKPOINT / "joint_adapter_and_parser.safetensors")
    metadata = {
        "epoch": epoch,
        "joint_score": joint_score,
        "selection_rule": "0.5 * DEV NLLB-normalized surface_chrF + 0.5 * DEV gold-token-conditioned LAS",
        "dev_metrics": dev_metrics,
        "dev_losses": dev_losses,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "relations": relation_vocabulary,
    }
    (CHECKPOINT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint() -> dict:
    tensors = load_file(CHECKPOINT / "joint_adapter_and_parser.safetensors", device="cpu")
    adapter = {key.removeprefix("adapter::"): value for key, value in tensors.items() if key.startswith("adapter::")}
    parser = {key.removeprefix("parser::"): value for key, value in tensors.items() if key.startswith("parser::")}
    set_peft_model_state_dict(model.mt, adapter)
    incompatible = model.load_state_dict(parser, strict=False)
    if incompatible.unexpected_keys or any(not key.startswith("mt.") for key in incompatible.missing_keys):
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    return json.loads((CHECKPOINT / "metadata.json").read_text(encoding="utf-8"))


# ---------------------------------- training ---------------------------------
trainable_mt = [parameter for name, parameter in model.named_parameters() if name.startswith("mt.") and parameter.requires_grad]
trainable_parser = [parameter for name, parameter in model.named_parameters() if not name.startswith("mt.") and parameter.requires_grad]
optimizer = torch.optim.AdamW(
    [
        {"params": trainable_mt, "lr": MT_LEARNING_RATE},
        {"params": trainable_parser, "lr": PARSER_LEARNING_RATE},
    ],
    weight_decay=WEIGHT_DECAY,
)
train_loader = DataLoader(
    train_pairs, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate,
    generator=torch.Generator().manual_seed(SEED)
)
updates_per_epoch = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION)
total_updates = updates_per_epoch * MAX_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=max(1, int(total_updates * WARMUP_RATIO)), num_training_steps=total_updates
)
scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)

print("Running the frozen-pretrained DEV translation baseline before any update")
zero_shot_generated = [translate_and_parse_pair(pair) for pair in tqdm(dev_pairs, desc="zero-shot DEV")]
zero_shot_dev = translation_metrics(zero_shot_generated, dev_pairs)
(RESULTS / "zero_shot_dev_translation.json").write_text(
    json.dumps(zero_shot_dev, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(zero_shot_dev)

history, best_joint, stale = [], -float("inf"), 0
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    model.mt.config.use_cache = False
    optimizer.zero_grad(set_to_none=True)
    running = np.zeros(4)
    seen = 0
    for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}"), 1):
        batch = move_batch(batch)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_FP16):
            values = joint_loss(batch)
            scaled_loss = values[0] / GRADIENT_ACCUMULATION
        scaler.scale(scaled_loss).backward()
        size = len(batch["pairs"])
        running += np.array([float(value.detach()) for value in values]) * size
        seen += size
        if step % GRADIENT_ACCUMULATION == 0 or step == len(train_loader):
            scaler.unscale_(optimizer)
            clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    train_losses = dict(zip(("total_loss", "mt_loss", "arc_loss", "relation_loss"), (running / seen).tolist()))
    dev_losses = mean_dev_losses(dev_pairs)
    dev_metrics, _, _ = evaluate(dev_pairs, f"epoch {epoch} DEV")
    joint_score = 0.5 * (
        dev_metrics["translation"]["normalized_surface_chrF"]
        + dev_metrics["gold_token_conditioned_parsing"]["LAS"]
    )
    row = {
        "epoch": epoch,
        **{f"train_{key}": value for key, value in train_losses.items()},
        **{f"dev_{key}": value for key, value in dev_losses.items()},
        "dev_normalized_surface_BLEU_zh": dev_metrics["translation"]["normalized_surface_BLEU_zh"],
        "dev_normalized_surface_chrF": dev_metrics["translation"]["normalized_surface_chrF"],
        "dev_UAS": dev_metrics["gold_token_conditioned_parsing"]["UAS"],
        "dev_LAS": dev_metrics["gold_token_conditioned_parsing"]["LAS"],
        "dev_joint_score": joint_score,
    }
    history.append(row)
    print(row)
    if joint_score > best_joint + MIN_JOINT_IMPROVEMENT:
        best_joint, stale = joint_score, 0
        save_checkpoint(epoch, joint_score, dev_metrics, dev_losses)
    else:
        stale += 1
    if stale >= PATIENCE:
        print(f"Early stopping: DEV joint score did not improve for {PATIENCE} epochs")
        break

pd.DataFrame(history).to_csv(RESULTS / "training_history.csv", index=False)
best_metadata = load_checkpoint()
print("Loaded selected checkpoint", best_metadata)


# TEST is touched only here, after the one joint checkpoint has been selected.
dev_metrics, dev_generated, dev_gold_parse = evaluate(dev_pairs, "selected DEV")
test_metrics, test_generated, test_gold_parse = evaluate(test_pairs, "untouched TEST")
nonprojective_indices = [
    index for index, pair in enumerate(test_pairs) if not is_projective(pair.cantonese.heads)
]
projective_indices = [index for index in range(len(test_pairs)) if index not in set(nonprojective_indices)]
test_metrics["gold_token_conditioned_parsing_subgroups"] = {
    "nonprojective": attachment_metrics(
        [test_gold_parse[index] for index in nonprojective_indices],
        [test_pairs[index] for index in nonprojective_indices],
    ),
    "projective": attachment_metrics(
        [test_gold_parse[index] for index in projective_indices],
        [test_pairs[index] for index in projective_indices],
    ),
    "nonprojective_sentences": len(nonprojective_indices),
    "projective_sentences": len(projective_indices),
}
summary = {
    "selection": best_metadata,
    "zero_shot_dev_translation": zero_shot_dev,
    "selected_dev": dev_metrics,
    "test": test_metrics,
}
(RESULTS / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def write_predictions(path: Path, items: list[Pair], generated: list[dict], gold_parses: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for pair, generation, conditioned in zip(items, generated, gold_parses):
            record = {
                "pair_id": pair.pair_id,
                "mandarin": pair.mandarin.forms,
                "gold_cantonese": asdict(pair.cantonese),
                "generated_cantonese_and_tree": generation,
                "gold_token_conditioned_tree": conditioned,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


write_predictions(RESULTS / "dev_predictions.jsonl", dev_pairs, dev_generated, dev_gold_parse)
write_predictions(RESULTS / "test_predictions.jsonl", test_pairs, test_generated, test_gold_parse)


def generated_conllu(pair: Pair, prediction: dict) -> str:
    lines = [f"# pair_id = {pair.pair_id}", f"# source_text = {''.join(pair.mandarin.forms)}", f"# text = {''.join(prediction['forms'])}"]
    for index, (form, head, relation) in enumerate(
        zip(prediction["forms"], prediction["heads"], prediction["relations"]), 1
    ):
        lines.append("\t".join(map(str, (index, form, "_", "_", "_", "_", head, relation, "_", "_"))))
    return "\n".join(lines) + "\n\n"


(RESULTS / "test_generated.conllu").write_text(
    "".join(generated_conllu(pair, prediction) for pair, prediction in zip(test_pairs, test_generated)),
    encoding="utf-8",
)

history_frame = pd.DataFrame(history)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history_frame.epoch, history_frame.dev_normalized_surface_chrF, marker="o", label="DEV chrF")
axes[0].plot(history_frame.epoch, history_frame.dev_LAS, marker="o", label="DEV LAS")
axes[0].set(xlabel="Epoch", ylabel="Score (%)")
axes[0].legend()
axes[1].plot(history_frame.epoch, history_frame.dev_joint_score, marker="o")
axes[1].set(xlabel="Epoch", ylabel="DEV joint score")
fig.tight_layout()
fig.savefig(RESULTS / "plots" / "training_curves.png", dpi=180)
plt.close(fig)

configuration = {
    key: value
    for key, value in globals().items()
    if key
    in {
        "SEED", "MODEL_ID", "MODEL_REVISION", "SRC_LANG", "TGT_LANG", "UD_RELEASE",
        "MAX_SOURCE_TOKENS", "MAX_TARGET_TOKENS", "MAX_NEW_TOKENS", "TRAIN_BATCH_SIZE",
        "GRADIENT_ACCUMULATION", "MAX_EPOCHS", "PATIENCE", "MT_LEARNING_RATE",
        "PARSER_LEARNING_RATE", "LORA_RANK", "LORA_ALPHA", "LORA_DROPOUT", "PARSER_DIM",
        "ARC_LOSS_WEIGHT", "RELATION_LOSS_WEIGHT", "NUM_BEAMS", "DEBUG_MODE",
    }
}
configuration.update(
    device=str(DEVICE), word_separator=WORD_SEPARATOR, word_separator_id=WORD_SEPARATOR_ID,
    package_versions={
        "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
        "accelerate": accelerate.__version__, "safetensors": safetensors.__version__,
        "networkx": networkx.__version__, "sacrebleu": sacrebleu.__version__,
    },
    provenance=provenance,
)
(RESULTS / "config_and_provenance.json").write_text(
    json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8"
)
source_directory = Path(__file__).resolve().parent
for source_name in ("run_colab.py", "joint_model.py", "README.md"):
    shutil.copy2(source_directory / source_name, RESULTS / "code" / source_name)

report = f"""# NLLB two-head Mandarin→Cantonese results

- Architecture: pinned NLLB-200 distilled 600M with LoRA, existing LM head, and a biaffine graph dependency head.
- Mandarin dependency tree used: no.
- Target transition oracle/SWAP used: no; graph decoding permits non-projective trees.
- Split: {len(train_pairs)}/{len(dev_pairs)}/{len(test_pairs)}, seed {SEED}.
- Selected epoch: {best_metadata['epoch']} by `{best_metadata['selection_rule']}`.
- Zero-shot DEV NLLB-normalized surface BLEU/chrF: {zero_shot_dev['normalized_surface_BLEU_zh']:.2f}/{zero_shot_dev['normalized_surface_chrF']:.2f}.
- Adapted DEV NLLB-normalized surface BLEU/chrF: {dev_metrics['translation']['normalized_surface_BLEU_zh']:.2f}/{dev_metrics['translation']['normalized_surface_chrF']:.2f}.
- Adapted TEST NLLB-normalized surface BLEU/chrF: {test_metrics['translation']['normalized_surface_BLEU_zh']:.2f}/{test_metrics['translation']['normalized_surface_chrF']:.2f}.
- Adapted TEST scores against unnormalized original UD text: BLEU/chrF {test_metrics['translation']['raw_reference_surface_BLEU_zh']:.2f}/{test_metrics['translation']['raw_reference_surface_chrF']:.2f}.
- TEST gold-token-conditioned UAS/LAS: {test_metrics['gold_token_conditioned_parsing']['UAS']:.2f}/{test_metrics['gold_token_conditioned_parsing']['LAS']:.2f}.
- TEST non-projective subgroup UAS/LAS ({test_metrics['gold_token_conditioned_parsing_subgroups']['nonprojective_sentences']} sentences): {test_metrics['gold_token_conditioned_parsing_subgroups']['nonprojective']['UAS']:.2f}/{test_metrics['gold_token_conditioned_parsing_subgroups']['nonprojective']['LAS']:.2f}.
- TEST exact NLLB-normalized UD-token match: {test_metrics['translation']['exact_normalized_UD_token_match_rate']:.2f}%.
- TEST exact-token-subset UAS/LAS ({test_metrics['translation']['exact_token_subset_sentences']} sentences): {test_metrics['translation']['exact_token_subset_UAS']}/{test_metrics['translation']['exact_token_subset_LAS']}.
- TEST structurally decodable separator/EOS rate: {test_metrics['translation']['structurally_decodable_generation_rate']:.2f}%.
- TEST gold boundary-count match rate: {test_metrics['translation']['gold_boundary_count_match_rate']:.2f}%.

Gold-token-conditioned UAS/LAS isolates the dependency head. End-to-end attachment scores are reported only for exact generated/gold UD-token matches, because raw token indices are otherwise not comparable. The single checkpoint is selected on DEV only; TEST is evaluated once afterward. NLLB's CC BY-NC 4.0 license and research-use limitations apply.
"""
(RESULTS / "RESULTS.md").write_text(report, encoding="utf-8")
print(report)

# Demonstrate the simultaneous user-facing output from the selected two-head model.
demo_pair = test_pairs[0]
demo = translate_and_parse("".join(demo_pair.mandarin.forms))
display(pd.DataFrame({
    "ID": range(1, len(demo["forms"]) + 1),
    "FORM": demo["forms"],
    "HEAD": demo["heads"],
    "DEPREL": demo["relations"],
}))

zip_path = Path("nllb_joint_mt_dp_results.zip")
if zip_path.exists():
    zip_path.unlink()
shutil.make_archive(zip_path.with_suffix("").as_posix(), "zip", root_dir=RESULTS)
print("Result archive", zip_path.resolve(), f"{zip_path.stat().st_size / 1024 / 1024:.2f} MiB")
try:
    from google.colab import files

    files.download(str(zip_path))
except ImportError:
    print("Not in Colab; automatic download skipped")
