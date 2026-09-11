#!/usr/bin/env python3
"""Build the self-contained Colab notebook requested for the Wu-style experiment."""

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "wu2018_mandarin_cantonese_replication.ipynb"


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


cells = [
md(r"""
# Mandarin-to-Cantonese Dependency-to-Dependency Translation
## A Wu et al.-style replication/adaptation on a gold parallel dependency treebank

This notebook tests whether a dependency-aware recurrent model can learn

\[
(\text{Mandarin words + gold dependency tree}) \rightarrow
(\text{Cantonese words + gold dependency tree}).
\]

It is a **replication-style architectural adaptation**, not a reproduction of Wu et al.'s numerical results. It uses UD syntactic tokens throughout (no BPE), a parallel four-stream source encoder, additive attention, and interacting word/action recurrent decoders. The primary syntactic metrics are gold-target-conditioned standard UAS/LAS; raw attachment scores are never reported when freely generated tokens differ from the gold tokens.

### Sources

* Wong, Tak-sum, Kim Gerdes, Herman Leung, and John Lee. 2017. [Quantitative Comparative Syntax on the Cantonese-Mandarin Parallel Dependency Treebank](https://aclanthology.org/W17-6530/). Proceedings of Depling 2017, 266–275.
* Wu, Shuangzhi, Dongdong Zhang, Zhirui Zhang, Nan Yang, Mu Li, and Ming Zhou. 2018. [Dependency-to-Dependency Neural Machine Translation](https://doi.org/10.1109/TASLP.2018.2855968). *IEEE/ACM Transactions on Audio, Speech, and Language Processing* 26(11):2132–2141.
* Nivre, Joakim. 2009. [Non-Projective Dependency Parsing in Expected Linear Time](https://aclanthology.org/P09-1040/). ACL-IJCNLP 2009, 351–359.
* Machine-readable data: [UD Chinese-HK](https://github.com/UniversalDependencies/UD_Chinese-HK) and [UD Cantonese-HK](https://github.com/UniversalDependencies/UD_Cantonese-HK), pinned to UD release `r2.18`, CC BY-SA 4.0. Pairing uses only explicit `parallel_id` metadata.

### Faithful choices and documented deviations

Wu et al. define CES as dependency-tree preorder and HES as postorder; children are ordered here by original UD token index, which makes the traversal deterministic. Their paper uses recurrent encoders, additive attention, and two interacting recurrent target components; this implementation uses GRUs and a differentiable stack whose reduced head is composed from head, dependent, and relation embeddings. Gold target trees are available here, so genuine conditioned UAS/LAS are added. Plain Wu-style arc-standard cannot encode 118 non-projective Cantonese trees (11.75% of this scarce corpus), so the target system is extended with Nivre's established `SWAP` transition rather than discarding them. This **experiment-specific low-resource adaptation** preserves all gold pairs; `SWAP` itself is prior work, not claimed as a newly invented transition. The paper's large-scale training recipe is reduced to conservative Colab settings suitable for this 1,004-pair proof of concept.
"""),
code(r"""
# Fresh-Colab dependencies. PyTorch is supplied by the GPU runtime.
%pip -q install "conllu==6.0.0" "sacrebleu==2.5.1" "pandas>=2.0" "matplotlib>=3.7" "tqdm>=4.66" "tabulate==0.9.0"
"""),
md("""
## 1. Configuration and reproducibility

All important experimental settings are centralized here. `DEBUG_MODE=True` executes the whole pipeline on 25/4/4 pairs for one epoch. Full mode runs all four source ablations by default; switch `RUN_ABLATIONS=False` to train only the full CES+HES model.
"""),
code(r"""
from __future__ import annotations

import copy, csv, hashlib, json, math, os, platform, random, shutil, sys, time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import sacrebleu
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

# ---------- one configuration cell ----------
SEED = 42
TRAIN_RATIO, DEV_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
UD_RELEASE = "r2.18"
EMB_DIM = 128
HIDDEN_DIM = 192
ACTION_EMB_DIM = 64
REL_EMB_DIM = 48
DROPOUT = 0.20
BATCH_SIZE = 16                 # gradient-accumulation batch (sentences vary in size)
LEARNING_RATE = 8e-4
MAX_EPOCHS = 30
PATIENCE = 5
GRAD_CLIP = 5.0
LAMBDA_ACTION = 1.0
RUN_ABLATIONS = True
DEBUG_MODE = False
SAVE_TO_DRIVE = False
MAX_DECODE_LEN = 80
ALLOW_SWAP = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEBUG_MODE:
    MAX_EPOCHS, PATIENCE = 1, 1

CFG = {k: v for k, v in dict(
    SEED=SEED, TRAIN_RATIO=TRAIN_RATIO, DEV_RATIO=DEV_RATIO, TEST_RATIO=TEST_RATIO,
    UD_RELEASE=UD_RELEASE, EMB_DIM=EMB_DIM, HIDDEN_DIM=HIDDEN_DIM,
    ACTION_EMB_DIM=ACTION_EMB_DIM, REL_EMB_DIM=REL_EMB_DIM, DROPOUT=DROPOUT,
    BATCH_SIZE=BATCH_SIZE, LEARNING_RATE=LEARNING_RATE, MAX_EPOCHS=MAX_EPOCHS,
    PATIENCE=PATIENCE, GRAD_CLIP=GRAD_CLIP, LAMBDA_ACTION=LAMBDA_ACTION,
    RUN_ABLATIONS=RUN_ABLATIONS, DEBUG_MODE=DEBUG_MODE, SAVE_TO_DRIVE=SAVE_TO_DRIVE,
    MAX_DECODE_LEN=MAX_DECODE_LEN, ALLOW_SWAP=ALLOW_SWAP, DEVICE=str(DEVICE)
).items()}

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()
BASE = Path("/content/wu2018_zh_yue") if Path("/content").exists() else Path("wu2018_zh_yue")
RAW = BASE / "raw"; RESULTS = BASE / "results"
for p in [RAW, RESULTS, RESULTS/"splits", RESULTS/"checkpoints", RESULTS/"plots", RESULTS/"attention_examples", RESULTS/"data"]:
    p.mkdir(parents=True, exist_ok=True)
(RESULTS/"config.json").write_text(json.dumps(CFG, ensure_ascii=False, indent=2), encoding="utf-8")

print("Python:", sys.version.split()[0], "| PyTorch:", torch.__version__)
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "| CUDA:", torch.version.cuda)
else:
    print("WARNING: GPU not detected. In Colab select Runtime > Change runtime type > GPU.")
"""),
md("""
## 2. Download the public parallel treebank

Both files are pinned to the same UD release. SHA-256 checks prevent silent source changes. GitHub tag-reference hashes and the download date are recorded; no account, API key, Drive, or manual upload is needed.
"""),
code(r"""
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

provenance = {"release": UD_RELEASE, "download_date": str(date.today()), "files": {}}
for language, info in SOURCES.items():
    url = f"https://raw.githubusercontent.com/{info['repo']}/{UD_RELEASE}/{info['filename']}"
    response = requests.get(url, timeout=60); response.raise_for_status()
    digest = hashlib.sha256(response.content).hexdigest()
    assert digest == info["sha256"], f"Checksum mismatch for {language}: {digest}"
    path = RAW / info["filename"]; path.write_bytes(response.content)
    ref_url = f"https://api.github.com/repos/{info['repo']}/git/ref/tags/{UD_RELEASE}"
    ref = requests.get(ref_url, timeout=30); ref.raise_for_status()
    tag_object = ref.json()["object"]
    provenance["files"][language] = {**info, "url": url, "tag_ref_sha": tag_object["sha"], "tag_ref_type": tag_object["type"]}
    shutil.copy2(path, RESULTS/"data"/info["filename"])

(RESULTS/"data"/"provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
print(json.dumps(provenance, indent=2))
"""),
md("""
## 3. CoNLL-U parsing and explicit pair resolution

Only integer-ID basic UD tokens are modeled. Multiword-token range rows and empty-node decimal rows are preserved in raw CoNLL-U but excluded from model tensors; punctuation remains a normal syntactic token; ROOT is virtual index 0; enhanced DEPS is not modeled. Pair recovery is a strict key join on `parallel_id`, never row order.
"""),
code(r"""
@dataclass
class Sentence:
    metadata: Dict[str, str]
    ids: List[int]
    forms: List[str]
    upos: List[str]
    heads: List[int]
    deprels: List[str]
    raw: str
    n_mwt: int = 0
    n_empty: int = 0

    @property
    def text(self): return self.metadata.get("text", "".join(self.forms))
    @property
    def pair_id(self): return self.metadata.get("parallel_id")

@dataclass
class Pair:
    pair_id: str
    mandarin: Sentence
    cantonese: Sentence
    document: str = ""

def read_conllu(path: Path) -> List[Sentence]:
    text = path.read_text(encoding="utf-8").strip()
    sentences = []
    for block in text.split("\n\n"):
        meta, rows, n_mwt, n_empty = {}, [], 0, 0
        for line in block.splitlines():
            if line.startswith("#"):
                content = line[1:].strip()
                if " = " in content:
                    key, value = content.split(" = ", 1); meta[key] = value
                continue
            if not line.strip(): continue
            cols = line.split("\t")
            if "-" in cols[0]: n_mwt += 1; continue
            if "." in cols[0]: n_empty += 1; continue
            assert len(cols) == 10 and cols[0].isdigit()
            rows.append(cols)
        if rows:
            ids = [int(r[0]) for r in rows]
            assert ids == list(range(1, len(rows)+1)), "Non-contiguous basic token IDs"
            sentences.append(Sentence(meta, ids, [r[1] for r in rows], [r[3] for r in rows],
                                      [int(r[6]) for r in rows], [r[7] for r in rows], block+"\n\n", n_mwt, n_empty))
    return sentences

mandarin_all = read_conllu(RAW/SOURCES["mandarin"]["filename"])
cantonese_all = read_conllu(RAW/SOURCES["cantonese"]["filename"])

def keyed(sentences, side):
    counts = Counter(s.pair_id for s in sentences)
    missing = [s.metadata.get("sent_id", "?") for s in sentences if not s.pair_id]
    duplicates = sorted(k for k, v in counts.items() if k is not None and v > 1)
    assert not missing, f"{side} sentences without parallel_id: {missing[:10]}"
    return {s.pair_id: s for s in sentences}, duplicates

M, dup_m = keyed(mandarin_all, "Mandarin")
C, dup_c = keyed(cantonese_all, "Cantonese")
common_ids = sorted(M.keys() & C.keys(), key=lambda x: int(x.split("/")[-1]))
missing_m = sorted(C.keys() - M.keys()); missing_c = sorted(M.keys() - C.keys())

def document_for(pair_id):
    n = int(pair_id.split("/")[-1])
    return "Missing days" if n <= 410 else "Tempo in Temple" if n <= 547 else "What day is today" if n <= 650 else "LegCo 2016-10-12"

pairs = [Pair(pid, M[pid], C[pid], document_for(pid)) for pid in common_ids]
assert all(p.mandarin.pair_id == p.cantonese.pair_id == p.pair_id for p in pairs)

def distribution(sentences, attr): return dict(sorted(Counter(x for s in sentences for x in getattr(s, attr)).items()))
def length_summary(sentences):
    a = np.array([len(s.ids) for s in sentences])
    return {"min": int(a.min()), "median": float(np.median(a)), "mean": float(a.mean()), "p95": float(np.percentile(a,95)), "max": int(a.max()),
            "histogram": dict(sorted(Counter(map(int, a)).items()))}

audit = {
    "total_mandarin_trees": len(mandarin_all), "total_cantonese_trees": len(cantonese_all),
    "total_genuine_parallel_pairs": len(common_ids), "successfully_resolved_pairs": len(pairs),
    "duplicate_pair_identifiers": {"mandarin": dup_m, "cantonese": dup_c},
    "missing_mandarin_counterparts": missing_m, "missing_cantonese_counterparts": missing_c,
    "unmatched_examples": len(missing_m)+len(missing_c),
    "mandarin_token_count": sum(len(s.ids) for s in mandarin_all),
    "cantonese_token_count": sum(len(s.ids) for s in cantonese_all),
    "mandarin_upos_distribution": distribution(mandarin_all, "upos"),
    "cantonese_upos_distribution": distribution(cantonese_all, "upos"),
    "mandarin_deprel_distribution": distribution(mandarin_all, "deprels"),
    "cantonese_deprel_distribution": distribution(cantonese_all, "deprels"),
    "mandarin_sentence_lengths": length_summary(mandarin_all),
    "cantonese_sentence_lengths": length_summary(cantonese_all),
    "multiword_token_rows": {"mandarin": sum(s.n_mwt for s in mandarin_all), "cantonese": sum(s.n_mwt for s in cantonese_all)},
    "empty_node_rows": {"mandarin": sum(s.n_empty for s in mandarin_all), "cantonese": sum(s.n_empty for s in cantonese_all)},
    "document_counts": dict(Counter(p.document for p in pairs)),
}

pair_df = pd.DataFrame([dict(pair_id=p.pair_id, mandarin_text=p.mandarin.text, cantonese_text=p.cantonese.text,
                             mandarin_n_tokens=len(p.mandarin.ids), cantonese_n_tokens=len(p.cantonese.ids)) for p in pairs])
pair_df.to_csv(RESULTS/"parallel_pairs.csv", index=False)
pair_df.to_csv(RESULTS/"data"/"parallel_pairs.csv", index=False)
print(json.dumps(audit, ensure_ascii=False, indent=2))
display(pair_df.head())
"""),
md("""
## 4. Tree traversals, projectivity, and labeled arc-standard+SWAP actions

Edges are `head → dependent`. CES is root-first preorder; HES is children-first postorder. In both, sibling order is the original UD index. The action convention is arc-standard with a virtual ROOT preloaded: `LEFT_REDUCE:r` makes the top item head of the second-top item; `RIGHT_REDUCE:r` makes the second-top item head of the top item. `SWAP` moves the second-top stack item to the front of the buffer, following Nivre (2009), and makes the system complete for arbitrary dependency trees. The final `RIGHT_REDUCE:root` attaches the sentence root to virtual index 0. Projective trees retain the original exactly-*2n* derivation; non-projective trees add paired SWAP/re-SHIFT operations.
"""),
code(r"""
SHIFT = "SHIFT"
SWAP = "SWAP"
FINISH_WORDS = "FINISH_WORDS"
def children_and_root(heads):
    children = defaultdict(list); roots = []
    for dep, head in enumerate(heads, 1):
        (roots if head == 0 else children[head]).append(dep)
    assert len(roots) == 1, f"Expected one root, got {roots}"
    for h in children: children[h].sort()
    return children, roots[0]

def tree_traversals(heads):
    children, root = children_and_root(heads)
    pre, post = [], []
    def walk(node):
        pre.append(node)
        for child in children[node]: walk(child)
        post.append(node)
    walk(root)
    assert sorted(pre) == list(range(1, len(heads)+1)) == sorted(post)
    return pre, post

def is_projective(heads):
    # Include the virtual ROOT at position 0: root arcs can cross ordinary arcs.
    arcs = [(min(i,h), max(i,h)) for i,h in enumerate(heads,1)]
    return not any(a < c < b < d or c < a < d < b
                   for j,(a,b) in enumerate(arcs) for c,d in arcs[j+1:])

def projective_order(heads):
    # Nivre (2009): inorder traversal respecting surface order of each head and its children.
    children, root = children_and_root(heads); order=[]
    def visit(node):
        for item in sorted(children[node]+[node]):
            if item == node: order.append(node)
            else: visit(item)
    visit(root)
    assert sorted(order) == list(range(1,len(heads)+1))
    return order

def dependency_tree_to_actions(heads, deprels):
    # Static unrestricted arc-standard oracle (Nivre, 2009, Figure 4).
    n = len(heads); stack = [0]; buffer = list(range(1,n+1)); actions = []; attached = set()
    children = defaultdict(set)
    for dep, head in enumerate(heads, 1): children[head].add(dep)
    rank = {node:i for i,node in enumerate(projective_order(heads))}
    while buffer or len(stack) > 1:
        reduced = False
        if len(stack) >= 2:
            s1, s0 = stack[-2], stack[-1]
            if s1 != 0 and heads[s1-1] == s0 and children[s1].issubset(attached):
                actions.append(f"LEFT_REDUCE:{deprels[s1-1]}"); attached.add(s1); stack.pop(-2); reduced = True
            elif heads[s0-1] == s1 and children[s0].issubset(attached):
                actions.append(f"RIGHT_REDUCE:{deprels[s0-1]}"); attached.add(s0); stack.pop(); reduced = True
        if not reduced:
            if len(stack)>=2 and 0 < stack[-2] < stack[-1] and rank[stack[-1]] < rank[stack[-2]]:
                actions.append(SWAP); buffer.insert(0,stack.pop(-2))
            elif buffer:
                actions.append(SHIFT); stack.append(buffer.pop(0))
            else:
                raise ValueError(f"Oracle stuck: stack={stack}, attached={attached}")
    assert actions.count(SHIFT) == n + actions.count(SWAP)
    if is_projective(heads): assert actions.count(SWAP)==0 and len(actions)==2*n
    return actions

def actions_to_dependency_tree(actions, n_tokens):
    stack, buffer = [0], list(range(1,n_tokens+1))
    heads, rels = [None]*n_tokens, [None]*n_tokens
    for action in actions:
        if action == FINISH_WORDS: continue  # joint decoder control; no parser-state effect
        if action == SHIFT:
            if not buffer: raise ValueError("SHIFT with empty buffer")
            stack.append(buffer.pop(0)); continue
        if action == SWAP:
            if len(stack)<2 or not (0 < stack[-2] < stack[-1]): raise ValueError("Illegal SWAP")
            buffer.insert(0,stack.pop(-2)); continue
        if ":" not in action or len(stack) < 2: raise ValueError(f"Illegal action {action}")
        kind, rel = action.split(":", 1); s1, s0 = stack[-2], stack[-1]
        if kind == "LEFT_REDUCE":
            if s1 == 0: raise ValueError("ROOT cannot be dependent")
            heads[s1-1], rels[s1-1] = s0, rel; stack.pop(-2)
        elif kind == "RIGHT_REDUCE":
            heads[s0-1], rels[s0-1] = s1, rel; stack.pop()
        else: raise ValueError(f"Unknown action {action}")
    if buffer or stack != [0] or any(x is None for x in heads):
        raise ValueError("Incomplete action sequence")
    return heads, rels

def dependency_tree_to_joint_actions(heads, deprels):
    # Insert FINISH_WORDS immediately after the final token's first SHIFT.
    parser_actions=dependency_tree_to_actions(heads,deprels)
    stack=[0]; pending=[]; n_new=0; joint=[]
    for action in parser_actions:
        joint.append(action)
        if action==SHIFT:
            if pending: stack.append(pending.pop(0))
            else:
                n_new+=1; stack.append(n_new)
                if n_new==len(heads): joint.append(FINISH_WORDS)
        elif action==SWAP: pending.insert(0,stack.pop(-2))
        elif action.startswith("LEFT_REDUCE:"): stack.pop(-2)
        elif action.startswith("RIGHT_REDUCE:"): stack.pop()
    assert joint.count(FINISH_WORDS)==1 and n_new==len(heads)
    return joint

# Unit tests: identity-preserving traversals and exact action/tree round trips.
for sent in mandarin_all:
    ces, hes = tree_traversals(sent.heads)
    assert len(ces) == len(set(ces)) == len(sent.ids) and sorted(ces) == sent.ids
    assert len(hes) == len(set(hes)) == len(sent.ids) and sorted(hes) == sent.ids
    assert [sent.forms[i-1] for i in ces] == [sent.forms[original_id-1] for original_id in ces]
    assert [sent.forms[i-1] for i in hes] == [sent.forms[original_id-1] for original_id in hes]
swap_counts=[]
for sent in cantonese_all:
    actions = dependency_tree_to_actions(sent.heads, sent.deprels)
    h, r = actions_to_dependency_tree(actions, len(sent.ids))
    assert h == sent.heads and r == sent.deprels
    swap_counts.append(actions.count(SWAP))
    joint=dependency_tree_to_joint_actions(sent.heads,sent.deprels)
    h2,r2=actions_to_dependency_tree(joint,len(sent.ids))
    assert h2==sent.heads and r2==sent.deprels

nonproj_m = [p.pair_id for p in pairs if not is_projective(p.mandarin.heads)]
nonproj_c = [p.pair_id for p in pairs if not is_projective(p.cantonese.heads)]
audit["projectivity"] = {
    "mandarin": {"projective": len(pairs)-len(nonproj_m), "nonprojective": len(nonproj_m), "nonprojective_pct": 100*len(nonproj_m)/len(pairs)},
    "cantonese": {"projective": len(pairs)-len(nonproj_c), "nonprojective": len(nonproj_c), "nonprojective_pct": 100*len(nonproj_c)/len(pairs)},
}
audit["target_nonprojective_handling"] = "Retained: unrestricted arc-standard+SWAP exactly represents every gold tree. No pair is excluded."
audit["swap_transitions"] = {"sentences_using_swap":sum(x>0 for x in swap_counts),"total_swaps":sum(swap_counts),"max_swaps":max(swap_counts),"mean_swaps":float(np.mean(swap_counts))}
(RESULTS/"nonprojective_target_ids.txt").write_text("\n".join(nonproj_c)+"\n", encoding="utf-8")
(RESULTS/"excluded_nonprojective_ids.txt").write_text("", encoding="utf-8")
(RESULTS/"data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit["projectivity"], indent=2))
print("All CES/HES and all", len(pairs), "target arc-standard+SWAP round-trip tests passed; no exclusions.")
"""),
md("""
## 5. Inseparable train/dev/test split and train-only vocabularies

The four source documents contain 410, 137, 103, and 354 pairs. A document-disjoint 80/10/10 split is therefore not feasible without severe ratio distortion; the notebook uses the requested seeded pair-level split and explicitly reports potential within-document leakage. All projective and non-projective targets are retained through the SWAP extension. Vocabularies and target relation/action inventories are fitted on training data only.
"""),
code(r"""
eligible = list(pairs)
rng = random.Random(SEED); rng.shuffle(eligible)
n = len(eligible); n_train = int(n*TRAIN_RATIO); n_dev = int(n*DEV_RATIO)
train_pairs = eligible[:n_train]; dev_pairs = eligible[n_train:n_train+n_dev]; test_pairs = eligible[n_train+n_dev:]
assert not (set(p.pair_id for p in train_pairs) & set(p.pair_id for p in dev_pairs))
assert not (set(p.pair_id for p in train_pairs) & set(p.pair_id for p in test_pairs))

if DEBUG_MODE:
    train_pairs, dev_pairs, test_pairs = train_pairs[:25], dev_pairs[:4], test_pairs[:4]

split_map = [("train", train_pairs), ("dev", dev_pairs), ("test", test_pairs)]
manifest = []
for split, items in split_map:
    (RESULTS/"splits"/f"{split}_ids.txt").write_text("\n".join(p.pair_id for p in items)+"\n", encoding="utf-8")
    for p in items:
        manifest.append(dict(pair_id=p.pair_id, split=split, mandarin_text=p.mandarin.text, cantonese_text=p.cantonese.text,
                             mandarin_n_tokens=len(p.mandarin.ids), cantonese_n_tokens=len(p.cantonese.ids)))
pd.DataFrame(manifest).to_csv(RESULTS/"splits"/"split_manifest.csv", index=False)
print("Eligible/splits:", n, {s:len(x) for s,x in split_map})
print("Documents per split (leakage audit):")
display(pd.crosstab(pd.DataFrame([(s,p.document) for s,x in split_map for p in x], columns=["split","document"])["split"],
                    pd.DataFrame([(s,p.document) for s,x in split_map for p in x], columns=["split","document"])["document"]))

class Vocab:
    def __init__(self, items, specials):
        self.itos = list(specials) + sorted(set(items)-set(specials))
        self.stoi = {x:i for i,x in enumerate(self.itos)}
    def __len__(self): return len(self.itos)
    def get(self, item, fallback="<unk>"): return self.stoi.get(item, self.stoi[fallback])

src_vocab = Vocab((w for p in train_pairs for w in p.mandarin.forms), ["<pad>","<unk>"])
tgt_vocab = Vocab((w for p in train_pairs for w in p.cantonese.forms), ["<pad>","<unk>","<bos>","<eos>"])
train_rels = sorted(set(r for p in train_pairs for r in p.cantonese.deprels if r != "root"))
actions_inventory = ["<start_action>", SHIFT, SWAP, FINISH_WORDS, "LEFT_REDUCE:<unk_rel>", "RIGHT_REDUCE:<unk_rel>", "RIGHT_REDUCE:root"]
actions_inventory += [f"{d}:{r}" for d in ["LEFT_REDUCE","RIGHT_REDUCE"] for r in train_rels]
action_vocab = Vocab(actions_inventory, [])

def normalize_action(action):
    if action in action_vocab.stoi: return action
    if action.startswith("LEFT_REDUCE:"): return "LEFT_REDUCE:<unk_rel>"
    if action.startswith("RIGHT_REDUCE:"): return "RIGHT_REDUCE:<unk_rel>"
    raise KeyError(action)

print("Vocab sizes:", {"source":len(src_vocab), "target":len(tgt_vocab), "actions":len(action_vocab)})
"""),
md("""
## 6. Three genuine paired examples and reconstruction checks
"""),
code(r"""
def show_pair(p):
    ces, hes = tree_traversals(p.mandarin.heads)
    actions = dependency_tree_to_actions(p.cantonese.heads, p.cantonese.deprels)
    rh, rr = actions_to_dependency_tree(actions, len(p.cantonese.ids))
    assert rh == p.cantonese.heads and rr == p.cantonese.deprels
    print("="*90, "\nPAIR", p.pair_id)
    print("MANDARIN")
    for name, values in [("tokens",p.mandarin.forms),("UPOS",p.mandarin.upos),("HEAD",p.mandarin.heads),("DEPREL",p.mandarin.deprels)]: print(name, values)
    print("MANDARIN SOURCE REPRESENTATIONS")
    print("original", p.mandarin.forms)
    print("CES", [p.mandarin.forms[i-1] for i in ces], "original IDs", ces)
    print("HES", [p.mandarin.forms[i-1] for i in hes], "original IDs", hes)
    print("CANTONESE")
    for name, values in [("tokens",p.cantonese.forms),("UPOS",p.cantonese.upos),("HEAD",p.cantonese.heads),("DEPREL",p.cantonese.deprels)]: print(name, values)
    print("CANTONESE GOLD ACTION SEQUENCE", actions)
for p in eligible[:3]: show_pair(p)
"""),
md("""
## 7. Wu-style multi-stream encoder and interacting structured decoder

This is not a four-layer RNN. Forward sequence, backward sequence, CES, and HES are parallel GRU streams. CES/HES outputs are scattered back to original token identities before affine-sum/tanh fusion. The action GRU attends to fused Mandarin memory and reads the word state plus differentiable stack top states. On SHIFT, the word GRU uses the action state, attention, previous word, and parser context to predict the next Cantonese token; on a labeled reduction, a learned relation-aware composition updates the surviving head representation on the stack.
"""),
code(r"""
class BahdanauAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__(); self.q=nn.Linear(hidden,hidden,bias=False); self.k=nn.Linear(hidden,hidden,bias=False); self.v=nn.Linear(hidden,1,bias=False)
    def forward(self, query, memory):
        scores = self.v(torch.tanh(self.q(query).unsqueeze(0)+self.k(memory))).squeeze(-1)
        alpha = torch.softmax(scores, dim=0); return (alpha.unsqueeze(-1)*memory).sum(0), alpha

class WuDep2Dep(nn.Module):
    def __init__(self, variant="full"):
        super().__init__(); self.variant=variant; h=HIDDEN_DIM
        self.src_emb=nn.Embedding(len(src_vocab),EMB_DIM); self.tgt_emb=nn.Embedding(len(tgt_vocab),EMB_DIM)
        self.seq_f=nn.GRU(EMB_DIM,h,batch_first=True); self.seq_b=nn.GRU(EMB_DIM,h,batch_first=True)
        self.ces=nn.GRU(EMB_DIM,h,batch_first=True); self.hes=nn.GRU(EMB_DIM,h,batch_first=True)
        self.fuse_f=nn.Linear(h,h,bias=False); self.fuse_b=nn.Linear(h,h,bias=False)
        self.fuse_c=nn.Linear(h,h,bias=False); self.fuse_h=nn.Linear(h,h,bias=False); self.fuse_bias=nn.Parameter(torch.zeros(h))
        self.attn=BahdanauAttention(h)
        self.action_emb=nn.Embedding(len(action_vocab),ACTION_EMB_DIM)
        self.rel_emb=nn.Embedding(len(action_vocab),REL_EMB_DIM)
        self.action_gru=nn.GRUCell(ACTION_EMB_DIM+4*h,h)
        self.action_out=nn.Linear(h,len(action_vocab))
        self.word_gru=nn.GRUCell(EMB_DIM+4*h,h); self.word_out=nn.Linear(h,len(tgt_vocab))
        self.compose=nn.Linear(2*h+REL_EMB_DIM,h); self.root=nn.Parameter(torch.zeros(h)); self.drop=nn.Dropout(DROPOUT)

    def encode(self, sent):
        ids=torch.tensor([src_vocab.get(x) for x in sent.forms],device=DEVICE); emb=self.drop(self.src_emb(ids)).unsqueeze(0)
        f,_=self.seq_f(emb); rev=torch.flip(emb,[1]); b,_=self.seq_b(rev); b=torch.flip(b,[1])
        ces_ids,hes_ids=tree_traversals(sent.heads)
        def tree_stream(order,rnn):
            order0=torch.tensor([i-1 for i in order],device=DEVICE); out,_=rnn(emb[:,order0,:]); restored=torch.empty_like(out)
            restored[:,order0,:]=out
            return restored
        c=tree_stream(ces_ids,self.ces); hh=tree_stream(hes_ids,self.hes)
        z=self.fuse_f(f)+self.fuse_b(b)+self.fuse_bias
        if self.variant in ("ces","full"): z=z+self.fuse_c(c)
        if self.variant in ("hes","full"): z=z+self.fuse_h(hh)
        return torch.tanh(z.squeeze(0))

    @staticmethod
    def stack_context(stack):
        zero=torch.zeros(HIDDEN_DIM,device=DEVICE)
        top=stack[-1][1] if stack else zero; second=stack[-2][1] if len(stack)>1 else zero
        return torch.cat([top,second])

    def action_step(self, prev_action, word_state, action_state, stack, memory):
        ctx,alpha=self.attn(action_state,memory); sc=self.stack_context(stack)
        x=torch.cat([self.action_emb(torch.tensor(action_vocab.stoi[prev_action],device=DEVICE)),word_state,ctx,sc])
        state=self.action_gru(self.drop(x),action_state); return state,self.action_out(state),ctx,alpha,sc

    def word_step(self, prev_word_id, word_state, action_state, ctx, stack_context):
        x=torch.cat([self.tgt_emb(torch.tensor(prev_word_id,device=DEVICE)),action_state,ctx,stack_context])
        state=self.word_gru(self.drop(x),word_state); return state,self.word_out(state)

    def reduce_stack(self, stack, action, action_id):
        kind,rel=action.split(":",1); s1,s0=stack[-2],stack[-1]
        head,dep=(s0,s1) if kind=="LEFT_REDUCE" else (s1,s0)
        composed=torch.tanh(self.compose(torch.cat([head[1],dep[1],self.rel_emb(torch.tensor(action_id,device=DEVICE))])))
        if kind=="LEFT_REDUCE": stack[-1]=(head[0],composed); stack.pop(-2)
        else: stack[-2]=(head[0],composed); stack.pop()

    def teacher_forced_loss(self,pair):
        memory=self.encode(pair.mandarin); z=torch.zeros(HIDDEN_DIM,device=DEVICE)
        word_state=z.clone(); action_state=z.clone(); stack=[(0,self.root)]; prev_action="<start_action>"
        prev_word=tgt_vocab.stoi["<bos>"]; word_losses=[]; action_losses=[]; next_word=0; pending=[]; words_finished=False
        gold_actions=dependency_tree_to_joint_actions(pair.cantonese.heads,pair.cantonese.deprels)
        for gold_action in gold_actions:
            action_state,logits,ctx,alpha,sc=self.action_step(prev_action,word_state,action_state,stack,memory)
            normalized=normalize_action(gold_action); aid=action_vocab.stoi[normalized]
            action_losses.append(F.cross_entropy(logits.unsqueeze(0),torch.tensor([aid],device=DEVICE)))
            if gold_action==SHIFT:
                if pending:
                    stack.append(pending.pop(0))  # re-SHIFT an existing token: no duplicate word generation
                else:
                    word_state,wlogits=self.word_step(prev_word,word_state,action_state,ctx,sc)
                    wid=tgt_vocab.get(pair.cantonese.forms[next_word]); word_losses.append(F.cross_entropy(wlogits.unsqueeze(0),torch.tensor([wid],device=DEVICE)))
                    next_word+=1; prev_word=wid; stack.append((next_word,word_state))
            elif gold_action==SWAP:
                pending.insert(0,stack.pop(-2))
            elif gold_action==FINISH_WORDS:
                word_state,eos_logits=self.word_step(prev_word,word_state,action_state,ctx,sc)
                word_losses.append(F.cross_entropy(eos_logits.unsqueeze(0),torch.tensor([tgt_vocab.stoi["<eos>"]],device=DEVICE)))
                words_finished=True
            else: self.reduce_stack(stack,gold_action,aid)
            prev_action=normalized
        assert words_finished and next_word==len(pair.cantonese.forms) and not pending and len(stack)==1
        wl=torch.stack(word_losses).mean(); al=torch.stack(action_losses).mean(); return wl+LAMBDA_ACTION*al,wl,al
"""),
md("""
## 8. Constrained decoding and scientifically valid metrics

Conditioned decoding supplies the next gold Cantonese token only on a **first-time SHIFT**, while the action model constructs its own unrestricted tree. A token moved to the pending buffer by SWAP is re-SHIFTed with its existing Word-RNN representation and is not generated twice. The explicit `FINISH_WORDS` action closes the new-word stream and supervises the Word-RNN's EOS prediction; pending tokens and reductions then finish the tree. This gives training and inference the same stopping decision, even though a non-projective parse may still need SWAP/reduction actions after the last new token. Legal-action masking enforces Nivre's monotonic SWAP precondition and a complete single-root tree. Standard UAS/LAS is computed for free generation only on exact-token matches.
"""),
code(r"""
def legal_action_ids(stack, pending, n_new, n_tokens=None, ended=False, conditioned=False):
    legal=[]
    more_new=not ended and (n_tokens is None or n_new<n_tokens)
    if pending or more_new: legal.append(action_vocab.stoi[SHIFT])
    if not ended and n_new>0 and (not conditioned or n_new==n_tokens): legal.append(action_vocab.stoi[FINISH_WORDS])
    if len(stack)>=3:
        legal += [i for a,i in action_vocab.stoi.items() if a.startswith(("LEFT_REDUCE:","RIGHT_REDUCE:")) and not a.endswith(":root")]
    if ALLOW_SWAP and len(stack)>=2 and 0 < stack[-2][0] < stack[-1][0]: legal.append(action_vocab.stoi[SWAP])
    if len(stack)==2 and not pending and ended: legal.append(action_vocab.stoi["RIGHT_REDUCE:root"])
    return sorted(set(legal))

@torch.no_grad()
def decode_gold_conditioned(model,pair,save_attention=False):
    model.eval(); memory=model.encode(pair.mandarin); z=torch.zeros(HIDDEN_DIM,device=DEVICE)
    ws=z.clone(); ast=z.clone(); stack=[(0,model.root)]; prev_a="<start_action>"; prev_w=tgt_vocab.stoi["<bos>"]
    n_new=0; pending=[]; ended=False; arcs={}; attentions=[]; actions=[]; n=len(pair.cantonese.forms)
    for _ in range(2*n*n+4*n+10):
        if ended and len(stack)==1 and stack[0][0]==0 and n_new==n and not pending: break
        ast,logits,ctx,alpha,sc=model.action_step(prev_a,ws,ast,stack,memory)
        legal=legal_action_ids(stack,pending,n_new,n_tokens=n,ended=ended,conditioned=True); assert legal
        aid=max(legal,key=lambda i:float(logits[i])); action=action_vocab.itos[aid]; actions.append(action)
        if action==SHIFT:
            if pending:
                stack.append(pending.pop(0))
            else:
                wid=tgt_vocab.get(pair.cantonese.forms[n_new]); ws,_=model.word_step(prev_w,ws,ast,ctx,sc)
                n_new+=1; prev_w=wid; stack.append((n_new,ws)); attentions.append(alpha.cpu().numpy())
        elif action==SWAP:
            pending.insert(0,stack.pop(-2))
        elif action==FINISH_WORDS:
            ws,_=model.word_step(prev_w,ws,ast,ctx,sc); ended=True
        else:
            kind,rel=action.split(":",1); s1,s0=stack[-2][0],stack[-1][0]
            dep,head=(s1,s0) if kind=="LEFT_REDUCE" else (s0,s1); arcs[dep]=(head,rel)
            model.reduce_stack(stack,action,aid)
        prev_a=action
    heads=[arcs.get(i,(0,"dep"))[0] for i in range(1,n+1)]; rels=[arcs.get(i,(0,"dep"))[1] for i in range(1,n+1)]
    return {"tokens":pair.cantonese.forms,"heads":heads,"deprels":rels,"actions":actions,"attention":attentions}

@torch.no_grad()
def decode_free(model,pair):
    model.eval(); memory=model.encode(pair.mandarin); z=torch.zeros(HIDDEN_DIM,device=DEVICE)
    ws=z.clone(); ast=z.clone(); stack=[(0,model.root)]; prev_a="<start_action>"; prev_w=tgt_vocab.stoi["<bos>"]
    tokens=[]; pending=[]; arcs={}; ended=False; attentions=[]; actions=[]
    for _ in range(2*MAX_DECODE_LEN*MAX_DECODE_LEN+4*MAX_DECODE_LEN+10):
        if ended and len(stack)==1 and stack[0][0]==0 and not pending: break
        ast,logits,ctx,alpha,sc=model.action_step(prev_a,ws,ast,stack,memory)
        legal=legal_action_ids(stack,pending,len(tokens),n_tokens=MAX_DECODE_LEN,ended=ended)
        if not legal: ended=True; continue
        aid=max(legal,key=lambda i:float(logits[i])); action=action_vocab.itos[aid]; actions.append(action)
        if action==SHIFT:
            if pending:
                stack.append(pending.pop(0))
            else:
                ws,wlogits=model.word_step(prev_w,ws,ast,ctx,sc)
                for special in ["<pad>","<bos>","<eos>"]: wlogits[tgt_vocab.stoi[special]]=-float("inf")
                wid=int(wlogits.argmax()); token=tgt_vocab.itos[wid]; tokens.append(token); prev_w=wid
                stack.append((len(tokens),ws)); attentions.append(alpha.cpu().numpy())
        elif action==SWAP:
            pending.insert(0,stack.pop(-2))
        elif action==FINISH_WORDS:
            ws,_=model.word_step(prev_w,ws,ast,ctx,sc); ended=True
        else:
            kind,rel=action.split(":",1); s1,s0=stack[-2][0],stack[-1][0]
            dep,head=(s1,s0) if kind=="LEFT_REDUCE" else (s0,s1); arcs[dep]=(head,rel)
            model.reduce_stack(stack,action,aid)
        prev_a=action
    heads=[arcs.get(i,(0,"dep"))[0] for i in range(1,len(tokens)+1)]; rels=[arcs.get(i,(0,"dep"))[1] for i in range(1,len(tokens)+1)]
    complete=ended and len(stack)==1 and stack[0][0]==0 and not pending
    return {"tokens":tokens,"heads":heads,"deprels":rels,"actions":actions,"attention":attentions,
            "complete_parse":complete,"hit_max_decode_length":len(tokens)>=MAX_DECODE_LEN}

def tree_metrics(predictions,pairs):
    total=uas=las=roots=labels=exact=0; relstat=defaultdict(Counter)
    for pred,p in zip(predictions,pairs):
        gh,gr=p.cantonese.heads,p.cantonese.deprels; ph,pr=pred["heads"],pred["deprels"]
        sent_exact=True
        for h,r,h2,r2 in zip(gh,gr,ph,pr):
            total+=1; head_ok=h==h2; label_ok=r==r2
            uas+=head_ok; las+=head_ok and label_ok; labels+=label_ok; sent_exact &= head_ok and label_ok
            if h==0: roots+=head_ok
            relstat[r]["gold"]+=1; relstat[r2]["pred"]+=1
            if head_ok and label_ok: relstat[r]["tp"]+=1
        exact+=sent_exact
    root_n=sum(1 for p in pairs for h in p.cantonese.heads if h==0)
    metrics={"UAS":100*uas/total,"LAS":100*las/total,"root_accuracy":100*roots/root_n,
             "dependency_label_accuracy":100*labels/total,"exact_full_tree_match":100*exact/len(pairs)}
    rows=[]
    for r,c in sorted(relstat.items()):
        precision=c["tp"]/c["pred"] if c["pred"] else 0; recall=c["tp"]/c["gold"] if c["gold"] else 0
        rows.append({"deprel":r,**c,"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0})
    return metrics,pd.DataFrame(rows)

def text_metrics(free,pairs):
    hyps=[" ".join(x["tokens"]) for x in free]; refs=[[" ".join(p.cantonese.forms) for p in pairs]]
    bleu=sacrebleu.corpus_bleu(hyps,refs,tokenize="none").score; chrf=sacrebleu.corpus_chrf(hyps,refs).score
    matches=[x["tokens"]==p.cantonese.forms for x,p in zip(free,pairs)]
    result={"BLEU":bleu,"chrF":chrf,"exact_token_match_rate":100*sum(matches)/len(matches),
            "mean_hypothesis_length":float(np.mean([len(x["tokens"]) for x in free])),
            "mean_reference_length":float(np.mean([len(p.cantonese.forms) for p in pairs])),
            "max_decode_length_hit_count":sum(bool(x["hit_max_decode_length"]) for x in free),
            "complete_parse_rate":100*sum(bool(x["complete_parse"]) for x in free)/len(free)}
    idx=[i for i,m in enumerate(matches) if m]
    if idx:
        subset,_=tree_metrics([free[i] for i in idx],[pairs[i] for i in idx]); result["exact_match_subset_UAS"]=subset["UAS"]; result["exact_match_subset_LAS"]=subset["LAS"]
    else: result["exact_match_subset_UAS"]=result["exact_match_subset_LAS"]=None
    return result
"""),
md("""
## 9. Training, DEV-only early stopping, and checkpoints

The joint loss is `L_word + λ_action L_action`. Each reported sentence loss is normalized over its word/action decisions; gradients accumulate across `BATCH_SIZE` sentences, are clipped, and then updated. Training runs for at most 30 epochs and stops after 5 consecutive epochs without a meaningful DEV-total-loss improvement. DEV total loss selects the checkpoint used for free translation (BLEU/chrF). A separate checkpoint selected by DEV gold-target-conditioned LAS is used for gold-conditioned syntax (UAS/LAS). TEST remains untouched until both selections finish.
"""),
code(r"""
def mean_losses(model,items):
    model.eval(); sums=np.zeros(3)
    with torch.no_grad():
        for p in items: sums += np.array([float(x) for x in model.teacher_forced_loss(p)])
    return sums/len(items)

def conditioned_eval(model,items):
    preds=[decode_gold_conditioned(model,p) for p in items]; metrics,relations=tree_metrics(preds,items); return metrics,relations,preds

def train_variant(variant,label):
    seed_everything(); model=WuDep2Dep(variant).to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    best_loss=float("inf"); best_las=-1.; stale=0; history=[]
    loss_path=RESULTS/"checkpoints"/("best_by_dev_total_loss.pt" if variant=="full" else f"{variant}_best_by_dev_total_loss.pt")
    las_path=RESULTS/"checkpoints"/("best_by_dev_LAS.pt" if variant=="full" else f"{variant}_best_by_dev_LAS.pt")
    for epoch in range(1,MAX_EPOCHS+1):
        model.train(); order=list(train_pairs); random.Random(SEED+epoch).shuffle(order); opt.zero_grad(); sums=np.zeros(3)
        for j,p in enumerate(tqdm(order,desc=f"{label} epoch {epoch}",leave=False),1):
            total,wl,al=model.teacher_forced_loss(p); (total/BATCH_SIZE).backward(); sums += [x.detach().item() for x in (total,wl,al)]
            if j%BATCH_SIZE==0 or j==len(order):
                nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP); opt.step(); opt.zero_grad()
        train_mean=sums/len(order); dev_mean=mean_losses(model,dev_pairs); dev_tree,_,_=conditioned_eval(model,dev_pairs)
        row=dict(model=label,epoch=epoch,train_total_loss=train_mean[0],train_word_loss=train_mean[1],train_action_loss=train_mean[2],
                 dev_total_loss=dev_mean[0],dev_word_loss=dev_mean[1],dev_action_loss=dev_mean[2],dev_UAS=dev_tree["UAS"],dev_LAS=dev_tree["LAS"])
        history.append(row); print(row)
        payload={"model_state":model.state_dict(),"variant":variant,"epoch":epoch,"config":CFG,
                 "src_vocab":src_vocab.itos,"tgt_vocab":tgt_vocab.itos,"action_vocab":action_vocab.itos,"dev":row}
        if dev_mean[0] < best_loss-1e-5:
            best_loss=dev_mean[0]; stale=0; torch.save(payload,loss_path)
        else: stale+=1
        if dev_tree["LAS"] > best_las:
            best_las=dev_tree["LAS"]; torch.save(payload,las_path)
        if stale>=PATIENCE: print("Early stopping on DEV total loss."); break
    loss_checkpoint=torch.load(loss_path,map_location=DEVICE,weights_only=False)
    las_checkpoint=torch.load(las_path,map_location=DEVICE,weights_only=False)
    translation_model=WuDep2Dep(variant).to(DEVICE); translation_model.load_state_dict(loss_checkpoint["model_state"])
    syntax_model=WuDep2Dep(variant).to(DEVICE); syntax_model.load_state_dict(las_checkpoint["model_state"])
    selection={"translation_checkpoint":"best_by_dev_total_loss","syntax_checkpoint":"best_by_dev_LAS",
               "best_epoch_by_dev_total_loss":loss_checkpoint["epoch"],"best_epoch_by_dev_LAS":las_checkpoint["epoch"],
               "best_dev_total_loss":loss_checkpoint["dev"]["dev_total_loss"],"best_dev_LAS":las_checkpoint["dev"]["dev_LAS"]}
    return {"translation":translation_model,"syntax":syntax_model},history,selection,sum(x.numel() for x in model.parameters())

VARIANTS=[("sequence","Sequence-only"),("ces","Sequence+CES"),("hes","Sequence+HES"),("full","Full CES+HES")]
if not RUN_ABLATIONS: VARIANTS=[("full","Full CES+HES")]
models={}; all_history=[]; model_info={}
for variant,label in VARIANTS:
    selected,hist,selection,nparams=train_variant(variant,label); models[variant]=selected; all_history.extend(hist)
    model_info[variant]={"label":label,"parameter_count":nparams,**selection}
pd.DataFrame(all_history).to_csv(RESULTS/"training_history.csv",index=False)
print(model_info)
"""),
md("""
## 10. Frozen-checkpoint DEV and untouched TEST evaluation

All main-table UAS/LAS values below are **gold-target-conditioned standard attachment scores** on identical UD token sequences, evaluated from the DEV-LAS-selected checkpoint. End-to-end BLEU/chrF use the DEV-total-loss-selected checkpoint, space-separated UD tokens, and sacreBLEU `tokenize=none` for BLEU. Freely generated mismatched sequences receive no raw-index UAS/LAS.
"""),
code(r"""
summary={}; table_rows=[]; saved_full=None
for variant,label in VARIANTS:
    syntax_model=models[variant]["syntax"]; translation_model=models[variant]["translation"]
    dev_cond,dev_rel,dev_preds=conditioned_eval(syntax_model,dev_pairs); test_cond,test_rel,test_preds=conditioned_eval(syntax_model,test_pairs)
    dev_free=[decode_free(translation_model,p) for p in tqdm(dev_pairs,desc=f"{label} DEV free",leave=False)]
    test_free=[decode_free(translation_model,p) for p in tqdm(test_pairs,desc=f"{label} TEST free",leave=False)]
    dev_text=text_metrics(dev_free,dev_pairs); test_text=text_metrics(test_free,test_pairs)
    summary[variant]={"model":label,"model_info":model_info[variant],"dev_gold_conditioned":dev_cond,"test_gold_conditioned":test_cond,
                      "dev_end_to_end":dev_text,"test_end_to_end":test_text}
    table_rows.append({"Model":label,"Dev BLEU":dev_text["BLEU"],"Dev chrF":dev_text["chrF"],"Dev UAS":dev_cond["UAS"],"Dev LAS":dev_cond["LAS"],
                       "Test BLEU":test_text["BLEU"],"Test chrF":test_text["chrF"],"Test UAS":test_cond["UAS"],"Test LAS":test_cond["LAS"],
                       "End-to-end exact-token-match rate":test_text["exact_token_match_rate"]})
    if variant=="full" or (saved_full is None and len(VARIANTS)==1): saved_full=(syntax_model,test_preds,test_free,test_rel)

results_table=pd.DataFrame(table_rows)
(RESULTS/"metrics_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
results_table.to_csv(RESULTS/"metrics_summary.csv",index=False)
display(results_table.style.format(precision=2).set_caption("BLEU/chrF: best DEV total loss; UAS/LAS: best DEV LAS (gold-target-conditioned)"))

model,test_preds,test_free,test_rel=saved_full
test_rel.to_csv(RESULTS/"relation_metrics.csv",index=False)
records=[]
with (RESULTS/"test_predictions.jsonl").open("w",encoding="utf-8") as f:
    for p,cond,free in zip(test_pairs,test_preds,test_free):
        rec={"pair_id":p.pair_id,"gold_tokens":p.cantonese.forms,"gold_heads":p.cantonese.heads,"gold_deprels":p.cantonese.deprels,
             "gold_conditioned_prediction":{k:v for k,v in cond.items() if k!="attention"},"free_prediction":{k:v for k,v in free.items() if k!="attention"}}
        f.write(json.dumps(rec,ensure_ascii=False)+"\n"); records.append(rec)
pd.DataFrame([{"pair_id":p.pair_id,"reference":" ".join(p.cantonese.forms),"hypothesis":" ".join(x["tokens"]),
               "exact_token_match":x["tokens"]==p.cantonese.forms} for p,x in zip(test_pairs,test_free)]).to_csv(RESULTS/"translation_predictions.tsv",sep="\t",index=False)
"""),
md("""
## 11. CoNLL-U predictions, attention examples, plots, and human-readable report
"""),
code(r"""
def sentence_to_conllu(sent,heads=None,rels=None,prediction_note=None):
    heads=heads or sent.heads; rels=rels or sent.deprels; lines=[]
    for k,v in sent.metadata.items(): lines.append(f"# {k} = {v}")
    if prediction_note: lines.append(f"# prediction_mode = {prediction_note}")
    for i,(form,upos,head,rel) in enumerate(zip(sent.forms,sent.upos,heads,rels),1):
        lines.append("\t".join(map(str,[i,form,"_",upos,"_","_",head,rel,"_","_"])))
    return "\n".join(lines)+"\n\n"

(RESULTS/"test_gold.conllu").write_text("".join(p.cantonese.raw for p in test_pairs),encoding="utf-8")
(RESULTS/"test_pred_gold_conditioned.conllu").write_text("".join(sentence_to_conllu(p.cantonese,x["heads"],x["deprels"],"gold_target_conditioned") for p,x in zip(test_pairs,test_preds)),encoding="utf-8")

for p,pred in list(zip(test_pairs,test_preds))[:5]:
    if pred["attention"]:
        matrix=np.stack(pred["attention"]); np.save(RESULTS/"attention_examples"/f"{p.pair_id.replace('/','_')}.npy",matrix)
        fig,ax=plt.subplots(figsize=(max(6,len(p.mandarin.forms)*.45),max(3,len(p.cantonese.forms)*.35)))
        im=ax.imshow(matrix,aspect="auto",cmap="viridis"); ax.set_xticks(range(len(p.mandarin.forms)),p.mandarin.forms,rotation=60,ha="right")
        ax.set_yticks(range(len(p.cantonese.forms)),p.cantonese.forms); ax.set_xlabel("Mandarin source UD tokens"); ax.set_ylabel("Gold-conditioned Cantonese SHIFT tokens")
        fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(RESULTS/"attention_examples"/f"{p.pair_id.replace('/','_')}.png",dpi=160); plt.close(fig)

hist=pd.DataFrame(all_history)
fig,ax=plt.subplots(figsize=(8,5))
for name,g in hist.groupby("model"): ax.plot(g.epoch,g.dev_total_loss,marker="o",label=name)
ax.set(xlabel="Epoch",ylabel="DEV total loss"); ax.legend(); fig.tight_layout(); fig.savefig(RESULTS/"plots"/"loss_curve.png",dpi=180); plt.close(fig)
for metric,filename in [("dev_UAS","dev_uas_curve.png"),("dev_LAS","dev_las_curve.png")]:
    fig,ax=plt.subplots(figsize=(8,5))
    for name,g in hist.groupby("model"): ax.plot(g.epoch,g[metric],marker="o",label=name)
    ax.set(xlabel="Epoch",ylabel=metric.replace("dev_","DEV ")+" (%)"); ax.legend(); fig.tight_layout(); fig.savefig(RESULTS/"plots"/filename,dpi=180); plt.close(fig)

full_key="full" if "full" in summary else list(summary)[0]; sm=summary[full_key]
ablation_md=results_table.to_markdown(index=False,floatfmt=".2f")
report=f'''# RESULTS — Mandarin-to-Cantonese Dependency-to-Dependency Translation

- Dataset: UD Chinese-HK + UD Cantonese-HK `{UD_RELEASE}`, paired strictly by explicit `parallel_id`; {len(pairs)} genuine pairs.
- Basic UD tokens: {audit['mandarin_token_count']} Mandarin; {audit['cantonese_token_count']} Cantonese.
- Target non-projective trees retained: {len(nonproj_c)} / {len(pairs)} ({100*len(nonproj_c)/len(pairs):.2f}%). Arc-standard+SWAP round trips recover every gold HEAD/DEPREL; exclusions: 0.
- Modeling split over all pairs: train/dev/test = {len(train_pairs)}/{len(dev_pairs)}/{len(test_pairs)} (seed {SEED}). Pair-level random split; source documents span splits, so document leakage is possible.
- Architecture: parallel forward/backward/CES/HES GRUs, affine-sum/tanh fusion, Bahdanau attention, interacting word/action GRUs, relation-aware differentiable unrestricted arc-standard+SWAP stack.
- Full-model parameter count: {sm['model_info']['parameter_count']:,}; translation checkpoint epoch (best DEV total loss): {sm['model_info']['best_epoch_by_dev_total_loss']}; syntax checkpoint epoch (best DEV LAS): {sm['model_info']['best_epoch_by_dev_LAS']}.
- DEV gold-conditioned UAS/LAS: {sm['dev_gold_conditioned']['UAS']:.2f}/{sm['dev_gold_conditioned']['LAS']:.2f}.
- Untouched TEST BLEU/chrF: {sm['test_end_to_end']['BLEU']:.2f}/{sm['test_end_to_end']['chrF']:.2f}.
- TEST generated/reference mean length: {sm['test_end_to_end']['mean_hypothesis_length']:.2f}/{sm['test_end_to_end']['mean_reference_length']:.2f}; max-length hits: {sm['test_end_to_end']['max_decode_length_hit_count']}/{len(test_pairs)}; complete parses: {sm['test_end_to_end']['complete_parse_rate']:.2f}%.
- Untouched TEST gold-conditioned UAS/LAS: {sm['test_gold_conditioned']['UAS']:.2f}/{sm['test_gold_conditioned']['LAS']:.2f}.
- TEST end-to-end exact-token-match rate: {sm['test_end_to_end']['exact_token_match_rate']:.2f}%.

## Source-syntax ablations

BLEU/chrF use the best-DEV-total-loss checkpoint. UAS/LAS use the best-DEV-LAS checkpoint and are gold-target-conditioned standard scores.

{ablation_md}

## Deviations and limitations

Children are deterministically ordered by original UD index for CES/HES. GRUs and a relation-aware neural stack are a documented faithful interpretation of underspecified implementation details. The Nivre (2009) SWAP extension is an experiment-specific innovation relative to the Wu baseline: it avoids discarding 11.75% of an exceptionally scarce target treebank, but SWAP itself is established prior work. The small corpus, train-only word/action vocabulary, greedy constrained decoding, pair-level rather than document-level split, and potentially longer/error-prone SWAP derivations limit conclusions. This proof of concept does not reproduce Wu et al.'s original data or numerical results and does not implement Transformers, augmentation, or large-scale treebank repurposing.
'''
(RESULTS/"RESULTS.md").write_text(report,encoding="utf-8")
print(report)
"""),
md("""
## 12. Package every artifact and download

The ZIP contains the data audit and provenance, explicit pair mapping, exact splits, histories, both full-model checkpoints, metrics, relation scores, predictions, CoNLL-U outputs, attention examples, plots, and `RESULTS.md`.
"""),
code(r"""
zip_path=Path("wu2018_zh_yue_results.zip")
if zip_path.exists(): zip_path.unlink()
shutil.make_archive(zip_path.with_suffix("").as_posix(),"zip",root_dir=RESULTS)
print("ZIP path:",zip_path.resolve())
print("ZIP size:",f"{zip_path.stat().st_size/1024/1024:.2f} MiB")

try:
    from google.colab import files
    files.download(str(zip_path))
except ImportError:
    print("Not running in Colab; automatic browser download skipped.")
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": OUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {OUT} with {len(cells)} cells")
