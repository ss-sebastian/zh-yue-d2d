#!/usr/bin/env python3
"""Prepare a deterministic, leakage-aware split of UD Cantonese-HK.

The input sentence blocks are treated as immutable text.  Only whole blocks are
selected and concatenated into the three output files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


YUE_REPOSITORY = "https://github.com/UniversalDependencies/UD_Cantonese-HK"
YUE_TAG = "r2.18"
YUE_COMMIT = "fcc7dd5b5eb97004441c4aa20704055ddde7a748"
ZH_REPOSITORY = "https://github.com/UniversalDependencies/UD_Chinese-HK"
ZH_TAG = "r2.18"
ZH_COMMIT = "72dbb27668c13daa1fb456a14af823d445dc4c3a"
LICENSE = "CC BY-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
SPLIT_ORDER = ("train", "dev", "test")
TARGET_RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}


@dataclass(frozen=True)
class Sentence:
    position: int
    block: str
    block_sha256: str
    source_key: str
    sent_id: str | None
    parallel_id: str | None
    text: str | None
    rows: tuple[tuple[str, ...], ...]
    words: tuple[tuple[str, ...], ...]
    multiword_rows: tuple[tuple[str, ...], ...]
    empty_node_rows: tuple[tuple[str, ...], ...]
    forms: tuple[str, ...]
    exact_key: str
    joined_key: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def metadata_value(lines: Iterable[str], name: str) -> str | None:
    prefix = f"# {name} ="
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def parse_conllu(path: Path, dataset_key: str, version: str) -> tuple[list[Sentence], dict[str, Any]]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not valid UTF-8: {exc}") from exc

    blocks: list[str] = []
    current: list[str] = []
    blank_line_count = 0
    for line in text.splitlines(keepends=True):
        if line.strip():
            current.append(line)
        else:
            blank_line_count += 1
            if current:
                blocks.append("".join(current))
                current = []
    if current:
        blocks.append("".join(current))

    sentences: list[Sentence] = []
    for position, block in enumerate(blocks, start=1):
        lines = block.splitlines()
        token_rows: list[tuple[str, ...]] = []
        words: list[tuple[str, ...]] = []
        multiword: list[tuple[str, ...]] = []
        empty_nodes: list[tuple[str, ...]] = []
        for line_number, line in enumerate(lines, start=1):
            if line.startswith("#"):
                continue
            fields = tuple(line.split("\t"))
            if len(fields) != 10:
                raise ValueError(
                    f"{path}: sentence {position}, block line {line_number}: "
                    f"expected 10 columns, found {len(fields)}"
                )
            token_rows.append(fields)
            token_id = fields[0]
            if token_id.isdigit():
                words.append(fields)
            elif re.fullmatch(r"[0-9]+-[0-9]+", token_id):
                multiword.append(fields)
            elif re.fullmatch(r"[0-9]+\.[0-9]+", token_id):
                empty_nodes.append(fields)
            else:
                raise ValueError(f"{path}: sentence {position}: invalid token ID {token_id!r}")
        if not token_rows:
            raise ValueError(f"{path}: sentence {position} has no token rows")
        forms = tuple(unicodedata.normalize("NFC", row[1]) for row in words)
        exact_material = json.dumps(forms, ensure_ascii=False, separators=(",", ":"))
        joined_material = "".join(forms)
        block_hash = sha256_text(block)
        source_key = f"{dataset_key}:{version}:{position:04d}:{block_hash[:16]}"
        sentences.append(
            Sentence(
                position=position,
                block=block,
                block_sha256=block_hash,
                source_key=source_key,
                sent_id=metadata_value(lines, "sent_id"),
                parallel_id=metadata_value(lines, "parallel_id"),
                text=metadata_value(lines, "text"),
                rows=tuple(token_rows),
                words=tuple(words),
                multiword_rows=tuple(multiword),
                empty_node_rows=tuple(empty_nodes),
                forms=forms,
                exact_key=sha256_text(exact_material),
                joined_key=sha256_text(joined_material),
            )
        )
    diagnostics = {
        "path": str(path),
        "sha256": sha256_bytes(raw),
        "byte_count": len(raw),
        "sentence_count": len(sentences),
        "blank_line_count": blank_line_count,
        "line_endings": {
            "lf": raw.count(b"\n"),
            "crlf": raw.count(b"\r\n"),
        },
    }
    return sentences, diagnostics


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def make_groups(sentences: list[Sentence]) -> tuple[list[list[int]], dict[int, str]]:
    union_find = UnionFind(len(sentences))
    seen_exact: dict[str, int] = {}
    seen_joined: dict[str, int] = {}
    for index, sentence in enumerate(sentences):
        for key, seen in ((sentence.exact_key, seen_exact), (sentence.joined_key, seen_joined)):
            if key in seen:
                union_find.union(index, seen[key])
            else:
                seen[key] = index
    by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(sentences)):
        by_root[union_find.find(index)].append(index)
    groups = sorted(by_root.values(), key=lambda members: tuple(sentences[i].source_key for i in members))
    index_to_group: dict[int, str] = {}
    for members in groups:
        material = "\n".join(sentences[i].source_key for i in members)
        group_id = f"grp-{sha256_text(material)[:16]}"
        for index in members:
            index_to_group[index] = group_id
    return groups, index_to_group


def assign_groups(
    groups: list[list[int]], sentences: list[Sentence], seed: int
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    """Shuffle groups, then place them by cumulative sentence-count midpoint.

    Midpoint binning makes each boundary miss its sentence-count target by at
    most half of the group crossing it, while retaining randomized membership.
    """
    ordered = list(groups)
    ordered.sort(key=lambda members: tuple(sentences[i].source_key for i in members))
    random.Random(seed).shuffle(ordered)
    n_sentences = len(sentences)
    train_boundary = TARGET_RATIOS["train"] * n_sentences
    dev_boundary = (TARGET_RATIOS["train"] + TARGET_RATIOS["dev"]) * n_sentences
    assignments: dict[int, str] = {}
    trace: list[dict[str, Any]] = []
    cumulative = 0
    for random_order, members in enumerate(ordered):
        midpoint = cumulative + len(members) / 2
        if midpoint <= train_boundary:
            split = "train"
        elif midpoint <= dev_boundary:
            split = "dev"
        else:
            split = "test"
        for index in members:
            assignments[index] = split
        trace.append(
            {
                "random_order": random_order,
                "first_source_key": min(sentences[i].source_key for i in members),
                "size": len(members),
                "cumulative_before": cumulative,
                "midpoint": midpoint,
                "split": split,
            }
        )
        cumulative += len(members)
    return assignments, trace


def source_name(sent_id: str | None) -> str | None:
    if sent_id is None or not sent_id.isdigit():
        return None
    value = int(sent_id)
    if 1 <= value <= 410:
        return "Missing days / 小時光"
    if 411 <= value <= 547:
        return "Tempo in Temple / 廟眾樂樂"
    if 548 <= value <= 650:
        return "What day is today / 今日星期幾"
    if 651 <= value <= 1004:
        return "Election of President (LegCo, 2016-10-12)"
    return None


def validate_trees(sentences: list[Sentence]) -> dict[str, Any]:
    problems: list[dict[str, Any]] = []
    root_histogram: Counter[str] = Counter()
    for sentence in sentences:
        ids = {int(row[0]) for row in sentence.words}
        heads: dict[int, int] = {}
        roots: list[int] = []
        for row in sentence.words:
            node_id = int(row[0])
            try:
                head = int(row[6])
            except ValueError:
                problems.append(
                    {
                        "type": "non_integer_head",
                        "source_key": sentence.source_key,
                        "sent_id": sentence.sent_id,
                        "node_id": node_id,
                        "head": row[6],
                    }
                )
                continue
            heads[node_id] = head
            if head == 0:
                roots.append(node_id)
            elif head not in ids:
                problems.append(
                    {
                        "type": "head_out_of_range",
                        "source_key": sentence.source_key,
                        "sent_id": sentence.sent_id,
                        "node_id": node_id,
                        "head": head,
                    }
                )
        root_histogram[str(len(roots))] += 1
        if len(roots) != 1:
            problems.append(
                {
                    "type": "unexpected_root_count",
                    "source_key": sentence.source_key,
                    "sent_id": sentence.sent_id,
                    "root_count": len(roots),
                    "root_node_ids": roots,
                }
            )
        cycles: set[tuple[int, ...]] = set()
        for start in sorted(ids):
            path: list[int] = []
            offsets: dict[int, int] = {}
            current = start
            while current != 0 and current in heads:
                if current in offsets:
                    cycle = path[offsets[current] :]
                    rotations = [tuple(cycle[i:] + cycle[:i]) for i in range(len(cycle))]
                    cycles.add(min(rotations))
                    break
                offsets[current] = len(path)
                path.append(current)
                current = heads[current]
        for cycle in sorted(cycles):
            problems.append(
                {
                    "type": "dependency_cycle",
                    "source_key": sentence.source_key,
                    "sent_id": sentence.sent_id,
                    "node_ids": list(cycle),
                }
            )
    sent_ids: dict[str, list[str]] = defaultdict(list)
    for sentence in sentences:
        if sentence.sent_id is not None:
            sent_ids[sentence.sent_id].append(sentence.source_key)
    missing = [sentence.source_key for sentence in sentences if sentence.sent_id is None]
    duplicates = {key: values for key, values in sorted(sent_ids.items()) if len(values) > 1}
    return {
        "problem_count": len(problems),
        "problems": problems,
        "root_count_histogram": dict(sorted(root_histogram.items(), key=lambda item: int(item[0]))),
        "missing_sent_id_source_keys": missing,
        "duplicate_sent_ids": duplicates,
        "multiword_token_row_count": sum(len(sentence.multiword_rows) for sentence in sentences),
        "empty_node_row_count": sum(len(sentence.empty_node_rows) for sentence in sentences),
    }


def percentile(sorted_values: list[int], probability: float) -> float:
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def length_stats(lengths: list[int]) -> dict[str, Any]:
    ordered = sorted(lengths)
    histogram = Counter(str(value) for value in ordered)
    return {
        "min": min(ordered),
        "max": max(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p25": percentile(ordered, 0.25),
        "p75": percentile(ordered, 0.75),
        "p95": percentile(ordered, 0.95),
        "histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
    }


def duplicate_report(
    groups: list[list[int]], sentences: list[Sentence], index_to_group: dict[int, str]
) -> dict[str, Any]:
    column_names = ("ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "DEPS", "MISC")
    details: list[dict[str, Any]] = []
    for members in groups:
        if len(members) < 2:
            continue
        exact_keys = {sentences[i].exact_key for i in members}
        joined_keys = {sentences[i].joined_key for i in members}
        annotation_signatures = {
            sha256_text(json.dumps(sentences[i].rows, ensure_ascii=False, separators=(",", ":")))
            for i in members
        }
        reference = sentences[members[0]]
        annotation_differences: list[dict[str, Any]] = []
        for other_index in members[1:]:
            other = sentences[other_index]
            if other.rows == reference.rows:
                continue
            comparison: dict[str, Any] = {
                "reference_source_key": reference.source_key,
                "other_source_key": other.source_key,
                "row_count_reference": len(reference.rows),
                "row_count_other": len(other.rows),
                "cells": [],
            }
            if len(reference.rows) == len(other.rows):
                for reference_row, other_row in zip(reference.rows, other.rows):
                    for column_index, (reference_value, other_value) in enumerate(
                        zip(reference_row, other_row)
                    ):
                        if reference_value != other_value:
                            comparison["cells"].append(
                                {
                                    "token_id_reference": reference_row[0],
                                    "token_id_other": other_row[0],
                                    "form_reference": reference_row[1],
                                    "form_other": other_row[1],
                                    "column": column_names[column_index],
                                    "reference": reference_value,
                                    "other": other_value,
                                }
                            )
            else:
                comparison["note"] = "Token-row counts differ; inspect the preserved source blocks."
            annotation_differences.append(comparison)
        details.append(
            {
                "group_id": index_to_group[members[0]],
                "size": len(members),
                "source_keys": [sentences[i].source_key for i in members],
                "sent_ids": [sentences[i].sent_id for i in members],
                "texts": [sentences[i].text for i in members],
                "match_modes": {
                    "exact_form_sequence_collision": len(exact_keys) < len(members),
                    "joined_form_string_collision": len(joined_keys) < len(members),
                },
                "tokenization_differs": len({sentences[i].forms for i in members}) > 1,
                "annotation_inconsistent": len(annotation_signatures) > 1,
                "annotation_row_sha256": {
                    sentences[i].source_key: sha256_text(
                        json.dumps(sentences[i].rows, ensure_ascii=False, separators=(",", ":"))
                    )
                    for i in members
                },
                "annotation_differences_against_first": annotation_differences,
            }
        )
    return {
        "duplicate_group_count": len(details),
        "sentences_in_duplicate_groups": sum(item["size"] for item in details),
        "annotation_inconsistent_group_count": sum(item["annotation_inconsistent"] for item in details),
        "groups": details,
    }


def split_statistics(sentences: list[Sentence], indices: list[int], total: int) -> dict[str, Any]:
    subset = [sentences[i] for i in indices]
    lengths = [len(sentence.words) for sentence in subset]
    upos = Counter(row[3] for sentence in subset for row in sentence.words)
    deprels = Counter(row[7] for sentence in subset for row in sentence.words)
    sources = Counter(source_name(sentence.sent_id) or "unresolved" for sentence in subset)
    return {
        "sentence_count": len(subset),
        "sentence_ratio": len(subset) / total,
        "word_node_count": sum(lengths),
        "sentence_length_word_nodes": length_stats(lengths),
        "upos_distribution": dict(sorted(upos.items())),
        "deprel_distribution": dict(sorted(deprels.items())),
        "source_distribution": dict(sorted(sources.items())),
    }


def build_parallel_exclusions(
    yue_sentences: list[Sentence],
    assignments: dict[int, str],
    zh_sentences: list[Sentence],
    zh_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    zh_by_parallel: dict[str, list[Sentence]] = defaultdict(list)
    for sentence in zh_sentences:
        if sentence.parallel_id is not None:
            zh_by_parallel[sentence.parallel_id].append(sentence)
    yue_parallel_counts = Counter(
        sentence.parallel_id for sentence in yue_sentences if sentence.parallel_id is not None
    )
    exclusions: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for index, sentence in enumerate(yue_sentences):
        split = assignments[index]
        if split not in {"dev", "test"}:
            continue
        candidates = zh_by_parallel.get(sentence.parallel_id or "", [])
        if sentence.parallel_id is None:
            reason = "Cantonese sentence lacks parallel_id"
        elif yue_parallel_counts[sentence.parallel_id] != 1:
            reason = "Cantonese parallel_id is not unique"
        elif len(candidates) != 1:
            reason = f"Chinese parallel_id match count is {len(candidates)}, expected 1"
        else:
            match = candidates[0]
            exclusions.append(
                {
                    "split": split,
                    "parallel_id": sentence.parallel_id,
                    "yue_source_key": sentence.source_key,
                    "yue_sent_id": sentence.sent_id,
                    "yue_text": sentence.text,
                    "zh_source_key": match.source_key,
                    "zh_sent_id": match.sent_id,
                    "zh_original_position": match.position,
                    "zh_text": match.text,
                }
            )
            continue
        unresolved.append(
            {
                "split": split,
                "yue_source_key": sentence.source_key,
                "yue_sent_id": sentence.sent_id,
                "parallel_id": sentence.parallel_id,
                "reason": reason,
            }
        )
    return {
        "purpose": "Evidence-backed Mandarin counterparts to exclude from any future Mandarin-side training data used with Cantonese dev/test.",
        "status": "resolved" if not unresolved else "partially_unresolved",
        "mapping_evidence": "Exact equality of explicit # parallel_id metadata in both official r2.18 treebanks; IDs are checked for uniqueness on both sides.",
        "warning": "This list does not establish what any future pretrained Mandarin parser saw. Check checkpoint provenance separately.",
        "chinese_source": {
            "repository": ZH_REPOSITORY,
            "tag": ZH_TAG,
            "commit": ZH_COMMIT,
            "raw_url": f"{ZH_REPOSITORY}/raw/{ZH_TAG}/zh_hk-ud-test.conllu",
            "file_sha256": zh_diagnostics["sha256"],
            "license": LICENSE,
            "license_url": LICENSE_URL,
        },
        "excluded_sentence_count": len(exclusions),
        "unresolved_count": len(unresolved),
        "exclusions": exclusions,
        "unresolved": unresolved,
    }


def output_bytes_for_split(sentences: list[Sentence], indices: list[int]) -> bytes:
    chunks: list[str] = []
    for index in sorted(indices, key=lambda i: sentences[i].position):
        block = sentences[index].block
        chunks.append(block if block.endswith(("\n", "\r")) else block + "\n")
        chunks.append("\n")
    return "".join(chunks).encode("utf-8")


def write_if_safe(path: Path, content: bytes) -> str:
    if path.exists():
        existing = path.read_bytes()
        if existing != content:
            raise FileExistsError(
                f"Refusing to overwrite different existing output: {path}. "
                "Choose a new --output-dir for a different split/version."
            )
        return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return "written"


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    yue_sentences, yue_diagnostics = parse_conllu(args.input, "yue_hk", YUE_TAG)
    zh_sentences, zh_diagnostics = parse_conllu(args.parallel_input, "zh_hk", ZH_TAG)
    checks = validate_trees(yue_sentences)
    groups, index_to_group = make_groups(yue_sentences)
    assignments, assignment_trace = assign_groups(groups, yue_sentences, args.seed)
    split_indices = {
        split: [i for i in range(len(yue_sentences)) if assignments[i] == split]
        for split in SPLIT_ORDER
    }
    duplicates = duplicate_report(groups, yue_sentences, index_to_group)

    rendered = {
        f"{split}.conllu": output_bytes_for_split(yue_sentences, split_indices[split])
        for split in SPLIT_ORDER
    }
    output_block_hashes: dict[str, list[str]] = {}
    for split in SPLIT_ORDER:
        parsed, _ = parse_conllu_bytes(rendered[f"{split}.conllu"], f"generated-{split}")
        output_block_hashes[split] = [sentence.block_sha256 for sentence in parsed]

    all_output_keys = {
        split: {yue_sentences[i].source_key for i in indices}
        for split, indices in split_indices.items()
    }
    keys_disjoint = all(
        all_output_keys[left].isdisjoint(all_output_keys[right])
        for pos, left in enumerate(SPLIT_ORDER)
        for right in SPLIT_ORDER[pos + 1 :]
    )
    group_leakage = [
        index_to_group[members[0]]
        for members in groups
        if len({assignments[i] for i in members}) != 1
    ]
    expected_block_hashes = {
        split: [yue_sentences[i].block_sha256 for i in sorted(indices)]
        for split, indices in split_indices.items()
    }
    validation = {
        "each_original_sentence_exactly_once": (
            sorted(i for indices in split_indices.values() for i in indices)
            == list(range(len(yue_sentences)))
        ),
        "split_source_keys_pairwise_disjoint": keys_disjoint,
        "duplicate_groups_crossing_splits": group_leakage,
        "output_sentence_blocks_identical_to_source": output_block_hashes == expected_block_hashes,
        "sentence_count_across_splits": sum(len(indices) for indices in split_indices.values()),
    }
    if not all(
        (
            validation["each_original_sentence_exactly_once"],
            validation["split_source_keys_pairwise_disjoint"],
            not validation["duplicate_groups_crossing_splits"],
            validation["output_sentence_blocks_identical_to_source"],
        )
    ):
        raise AssertionError(f"Internal acceptance validation failed: {validation}")

    split_stats = {
        split: split_statistics(yue_sentences, split_indices[split], len(yue_sentences))
        for split in SPLIT_ORDER
    }
    train_deprels = set(split_stats["train"]["deprel_distribution"])
    unseen_deprels = {
        split: sorted(set(split_stats[split]["deprel_distribution"]) - train_deprels)
        for split in ("dev", "test")
    }
    stats = {
        "dataset": {
            "sentence_count": len(yue_sentences),
            "word_node_count": sum(len(sentence.words) for sentence in yue_sentences),
            "official_readme_reference_counts": {"sentence_count": 1004, "word_count": 13918},
            "difference_from_official_readme": {
                "sentence_count": len(yue_sentences) - 1004,
                "word_node_count": sum(len(sentence.words) for sentence in yue_sentences) - 13918,
            },
        },
        "splits": split_stats,
        "duplicate_groups": duplicates,
        "deprels_present_outside_train_only": unseen_deprels,
        "data_checks": checks,
    }
    parallel_exclusions = build_parallel_exclusions(
        yue_sentences, assignments, zh_sentences, zh_diagnostics
    )
    stats_bytes = json_bytes(stats)
    exclusions_bytes = json_bytes(parallel_exclusions)
    rendered["stats.json"] = stats_bytes
    rendered["mandarin_parallel_exclusions.json"] = exclusions_bytes

    output_hashes = {name: sha256_bytes(content) for name, content in sorted(rendered.items())}
    manifest_sentences = []
    for index, sentence in enumerate(yue_sentences):
        manifest_sentences.append(
            {
                "source_key": sentence.source_key,
                "original_sent_id": sentence.sent_id,
                "parallel_id": sentence.parallel_id,
                "original_position": sentence.position,
                "source_block_sha256": sentence.block_sha256,
                "exact_form_sequence_key_sha256": sentence.exact_key,
                "joined_form_string_key_sha256": sentence.joined_key,
                "group_id": index_to_group[index],
                "split": assignments[index],
                "source": source_name(sentence.sent_id),
            }
        )
    manifest = {
        "schema_version": 1,
        "data": {
            "repository": YUE_REPOSITORY,
            "tag": YUE_TAG,
            "commit": YUE_COMMIT,
            "raw_url": f"{YUE_REPOSITORY}/raw/{YUE_TAG}/yue_hk-ud-test.conllu",
            "input_path": str(args.input),
            "input_sha256": yue_diagnostics["sha256"],
            "license": LICENSE,
            "license_url": LICENSE_URL,
            "official_file_role": "test",
            "experiment_split_role": "custom train/dev/test; not the official UD split",
        },
        "parallel_data_evidence": {
            "repository": ZH_REPOSITORY,
            "tag": ZH_TAG,
            "commit": ZH_COMMIT,
            "input_path": str(args.parallel_input),
            "input_sha256": zh_diagnostics["sha256"],
            "mapping_field": "parallel_id",
        },
        "split": {
            "seed": args.seed,
            "target_ratios": TARGET_RATIOS,
            "unit": "sentence blocks",
            "algorithm": (
                "Union sentences sharing either NFC-normalized integer-ID FORM sequence or the "
                "concatenation of those FORMs; sort groups by source key, shuffle with Python "
                "random.Random(seed), then assign by cumulative sentence-count midpoint at the "
                "80% and 90% boundaries. Sort each output by original position."
            ),
            "duplicate_rule": (
                "Any shared exact FORM-sequence key or joined-FORM-string key creates an edge; "
                "connected components are indivisible groups. No deduplication is performed."
            ),
            "assignment_trace": assignment_trace,
        },
        "source_metadata": {
            "status": "resolved from numeric sent_id ranges documented in the official r2.18 Cantonese README",
            "evidence_path": "data/raw/ud_cantonese_hk-r2.18/README.md",
            "ranges": [
                {"sent_id": "1-410", "source": "Missing days / 小時光"},
                {"sent_id": "411-547", "source": "Tempo in Temple / 廟眾樂樂"},
                {"sent_id": "548-650", "source": "What day is today / 今日星期幾"},
                {"sent_id": "651-1004", "source": "Election of President (LegCo, 2016-10-12)"},
            ],
        },
        "sentences": manifest_sentences,
        "output_sha256": output_hashes,
        "validation": validation,
    }
    rendered["split_manifest.json"] = json_bytes(manifest)

    actions = {}
    for name, content in sorted(rendered.items()):
        actions[name] = write_if_safe(args.output_dir / name, content)
    return {
        "output_dir": str(args.output_dir),
        "actions": actions,
        "output_sha256": {name: sha256_bytes(content) for name, content in sorted(rendered.items())},
        "split_sentence_counts": {
            split: len(split_indices[split]) for split in SPLIT_ORDER
        },
        "split_word_node_counts": {
            split: split_stats[split]["word_node_count"] for split in SPLIT_ORDER
        },
        "duplicate_group_count": duplicates["duplicate_group_count"],
        "tree_problem_count": checks["problem_count"],
        "parallel_unresolved_count": parallel_exclusions["unresolved_count"],
        "validation": validation,
    }


def parse_conllu_bytes(raw: bytes, label: str) -> tuple[list[Sentence], dict[str, Any]]:
    """Parse generated bytes through a temporary-like in-memory adaptation."""
    text = raw.decode("utf-8")
    blocks = [block for block in re.split(r"\n\s*\n", text.strip()) if block.strip()]
    sentences: list[Sentence] = []
    for position, block in enumerate(blocks, start=1):
        normalized_block = block + "\n"
        lines = normalized_block.splitlines()
        rows = tuple(tuple(line.split("\t")) for line in lines if line and not line.startswith("#"))
        words = tuple(row for row in rows if row[0].isdigit())
        multiword = tuple(row for row in rows if re.fullmatch(r"[0-9]+-[0-9]+", row[0]))
        empty_nodes = tuple(row for row in rows if re.fullmatch(r"[0-9]+\.[0-9]+", row[0]))
        forms = tuple(unicodedata.normalize("NFC", row[1]) for row in words)
        block_hash = sha256_text(normalized_block)
        exact_material = json.dumps(forms, ensure_ascii=False, separators=(",", ":"))
        sentences.append(
            Sentence(
                position=position,
                block=normalized_block,
                block_sha256=block_hash,
                source_key=f"{label}:{position}:{block_hash[:16]}",
                sent_id=metadata_value(lines, "sent_id"),
                parallel_id=metadata_value(lines, "parallel_id"),
                text=metadata_value(lines, "text"),
                rows=rows,
                words=words,
                multiword_rows=multiword,
                empty_node_rows=empty_nodes,
                forms=forms,
                exact_key=sha256_text(exact_material),
                joined_key=sha256_text("".join(forms)),
            )
        )
    return sentences, {"sha256": sha256_bytes(raw), "sentence_count": len(sentences)}


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/ud_cantonese_hk-r2.18/yue_hk-ud-test.conllu"),
    )
    parser.add_argument(
        "--parallel-input",
        type=Path,
        default=Path("data/raw/ud_chinese_hk-r2.18/zh_hk-ud-test.conllu"),
        help="Evidence-only Mandarin parallel file; never added to a training split.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/yue_hk"))
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = argument_parser().parse_args()
    try:
        result = prepare(args)
    except (ValueError, FileExistsError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
