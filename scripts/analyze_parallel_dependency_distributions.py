#!/usr/bin/env python3
"""Compare gold dependency distributions in parallel UD Cantonese-HK/Chinese-HK.

The primary analysis uses only Cantonese train sentences and their official
parallel_id-matched Chinese translations.  A full-corpus appendix is emitted
for descriptive use only and must not be used to select model settings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YUE = ROOT / "data/raw/ud_cantonese_hk-r2.18/yue_hk-ud-test.conllu"
DEFAULT_ZH = ROOT / "data/raw/ud_chinese_hk-r2.18/zh_hk-ud-test.conllu"
DEFAULT_MANIFEST = ROOT / "data/processed/yue_hk/split_manifest.json"
DEFAULT_OUT = ROOT / "analysis/dependency_distribution_r2.18"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_conllu(path: Path) -> list[dict]:
    sentences = []
    for position, block in enumerate(path.read_text(encoding="utf-8").strip().split("\n\n"), 1):
        meta = {}
        rows = []
        for line in block.splitlines():
            if line.startswith("# ") and " = " in line:
                key, value = line[2:].split(" = ", 1)
                meta[key] = value
            elif line and not line.startswith("#"):
                cols = line.split("\t")
                if cols[0].isdigit():
                    rows.append(cols)
        parallel_id = meta.get("parallel_id")
        if not parallel_id:
            raise ValueError(f"Missing parallel_id at {path}:{position}")
        sentences.append({"position": position, "meta": meta, "rows": rows})
    return sentences


def distance_bucket(distance: int) -> str:
    if distance == 0:
        return "ROOT"
    if distance <= 2:
        return "1-2"
    if distance <= 5:
        return "3-5"
    return "6+"


def base_relation(label: str) -> str:
    return label.split(":", 1)[0]


def arcs(sentences: list[dict]) -> list[dict]:
    output = []
    for sent in sentences:
        rows = sent["rows"]
        by_id = {int(row[0]): row for row in rows}
        for row in rows:
            token_id = int(row[0])
            head = int(row[6])
            distance = 0 if head == 0 else abs(token_id - head)
            direction = "ROOT" if head == 0 else ("head_left" if head < token_id else "head_right")
            output.append({
                "parallel_id": sent["meta"]["parallel_id"],
                "sent_id": sent["meta"].get("sent_id"),
                "text": sent["meta"].get("text"),
                "token_id": token_id,
                "form": row[1],
                "upos": row[3],
                "head": head,
                "head_upos": "ROOT" if head == 0 else by_id[head][3],
                "deprel": row[7],
                "base_deprel": base_relation(row[7]),
                "direction": direction,
                "distance": distance,
                "distance_bucket": distance_bucket(distance),
            })
    return output


def probability(counter: Counter, key, total: int) -> float:
    return counter[key] / total if total else 0.0


def js_divergence_bits(left: Counter, right: Counter) -> float:
    keys = sorted(set(left) | set(right), key=repr)
    nl, nr = sum(left.values()), sum(right.values())
    result = 0.0
    for key in keys:
        p = probability(left, key, nl)
        q = probability(right, key, nr)
        m = (p + q) / 2
        if p:
            result += 0.5 * p * math.log2(p / m)
        if q:
            result += 0.5 * q * math.log2(q / m)
    return result


def total_variation(left: Counter, right: Counter) -> float:
    keys = sorted(set(left) | set(right), key=repr)
    nl, nr = sum(left.values()), sum(right.values())
    return 0.5 * sum(abs(probability(left, k, nl) - probability(right, k, nr)) for k in keys)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    return numerator / math.sqrt(dx * dy) if dx and dy else None


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_pair(name: str, yue_sentences: list[dict], zh_sentences: list[dict], out: Path) -> dict:
    ya, za = arcs(yue_sentences), arcs(zh_sentences)
    y_count, z_count = len(ya), len(za)

    dimensions = {
        "deprel_strict": (Counter(x["deprel"] for x in ya), Counter(x["deprel"] for x in za)),
        "deprel_conll18_base": (Counter(x["base_deprel"] for x in ya), Counter(x["base_deprel"] for x in za)),
        "direction": (Counter(x["direction"] for x in ya), Counter(x["direction"] for x in za)),
        "distance_bucket": (Counter(x["distance_bucket"] for x in ya), Counter(x["distance_bucket"] for x in za)),
        "dependent_upos": (Counter(x["upos"] for x in ya), Counter(x["upos"] for x in za)),
        "head_upos": (Counter(x["head_upos"] for x in ya), Counter(x["head_upos"] for x in za)),
        "attachment_signature": (
            Counter((x["upos"], x["head_upos"], x["direction"], x["distance_bucket"], x["base_deprel"]) for x in ya),
            Counter((x["upos"], x["head_upos"], x["direction"], x["distance_bucket"], x["base_deprel"]) for x in za),
        ),
    }
    divergences = {
        key: {"jensen_shannon_bits": js_divergence_bits(a, b), "total_variation": total_variation(a, b)}
        for key, (a, b) in dimensions.items()
    }

    y_rel = Counter(x["deprel"] for x in ya)
    z_rel = Counter(x["deprel"] for x in za)
    relation_rows = []
    for relation in sorted(set(y_rel) | set(z_rel)):
        yr = [x for x in ya if x["deprel"] == relation]
        zr = [x for x in za if x["deprel"] == relation]
        yc, zc = len(yr), len(zr)
        relation_rows.append({
            "deprel": relation,
            "yue_count": yc,
            "zh_count": zc,
            "yue_per_1000_tokens": 1000 * yc / y_count,
            "zh_per_1000_tokens": 1000 * zc / z_count,
            "yue_minus_zh_per_1000": 1000 * yc / y_count - 1000 * zc / z_count,
            "yue_per_sentence": yc / len(yue_sentences),
            "zh_per_sentence": zc / len(zh_sentences),
            "yue_minus_zh_per_sentence": yc / len(yue_sentences) - zc / len(zh_sentences),
            "yue_head_left_percent": 100 * sum(x["direction"] == "head_left" for x in yr) / yc if yc else "",
            "zh_head_left_percent": 100 * sum(x["direction"] == "head_left" for x in zr) / zc if zc else "",
            "yue_long_6plus_percent": 100 * sum(x["distance"] >= 6 for x in yr) / yc if yc else "",
            "zh_long_6plus_percent": 100 * sum(x["distance"] >= 6 for x in zr) / zc if zc else "",
            "yue_mean_distance_nonroot": sum(x["distance"] for x in yr if x["head"] != 0) / sum(x["head"] != 0 for x in yr) if any(x["head"] != 0 for x in yr) else "",
            "zh_mean_distance_nonroot": sum(x["distance"] for x in zr if x["head"] != 0) / sum(x["head"] != 0 for x in zr) if any(x["head"] != 0 for x in zr) else "",
        })
    relation_rows.sort(key=lambda r: (-abs(r["yue_minus_zh_per_sentence"]), r["deprel"]))
    write_csv(out / f"{name}_relation_distribution.csv", relation_rows, list(relation_rows[0]))

    context_y = Counter((x["upos"], x["head_upos"], x["direction"], x["distance_bucket"], x["base_deprel"]) for x in ya)
    context_z = Counter((x["upos"], x["head_upos"], x["direction"], x["distance_bucket"], x["base_deprel"]) for x in za)
    context_rows = []
    for signature in sorted(set(context_y) | set(context_z)):
        yc, zc = context_y[signature], context_z[signature]
        context_rows.append({
            "dependent_upos": signature[0], "head_upos": signature[1], "direction": signature[2],
            "distance_bucket": signature[3], "base_deprel": signature[4],
            "yue_count": yc, "zh_count": zc,
            "yue_per_1000_tokens": 1000 * yc / y_count,
            "zh_per_1000_tokens": 1000 * zc / z_count,
            "absolute_rate_difference": abs(1000 * yc / y_count - 1000 * zc / z_count),
        })
    context_rows.sort(key=lambda r: (-r["absolute_rate_difference"], r["dependent_upos"], r["head_upos"]))
    write_csv(out / f"{name}_attachment_signature_shifts.csv", context_rows, list(context_rows[0]))

    y_by_id = {s["meta"]["parallel_id"]: s for s in yue_sentences}
    z_by_id = {s["meta"]["parallel_id"]: s for s in zh_sentences}
    pair_rows = []
    for parallel_id in sorted(y_by_id, key=lambda value: int(value.split("/")[-1])):
        ys, zs = y_by_id[parallel_id], z_by_id[parallel_id]
        yarcs, zarcs = arcs([ys]), arcs([zs])
        yn, zn = len(yarcs), len(zarcs)
        y_nonroot = [x["distance"] for x in yarcs if x["head"]]
        z_nonroot = [x["distance"] for x in zarcs if x["head"]]
        yrels, zrels = Counter(x["base_deprel"] for x in yarcs), Counter(x["base_deprel"] for x in zarcs)
        pair_rows.append({
            "parallel_id": parallel_id,
            "yue_sent_id": ys["meta"].get("sent_id"), "zh_sent_id": zs["meta"].get("sent_id"),
            "yue_text": ys["meta"].get("text"), "zh_text": zs["meta"].get("text"),
            "yue_tokens": yn, "zh_tokens": zn,
            "absolute_token_difference": abs(yn - zn),
            "yue_mean_arc_distance": sum(y_nonroot) / len(y_nonroot) if y_nonroot else 0,
            "zh_mean_arc_distance": sum(z_nonroot) / len(z_nonroot) if z_nonroot else 0,
            "yue_long_arc_percent": 100 * sum(x >= 6 for x in y_nonroot) / len(y_nonroot) if y_nonroot else 0,
            "zh_long_arc_percent": 100 * sum(x >= 6 for x in z_nonroot) / len(z_nonroot) if z_nonroot else 0,
            "base_relation_JS_bits": js_divergence_bits(yrels, zrels),
        })
    pair_rows.sort(key=lambda r: (-r["base_relation_JS_bits"], -r["absolute_token_difference"], r["parallel_id"]))
    write_csv(out / f"{name}_parallel_sentence_structural_shifts.csv", pair_rows, list(pair_rows[0]))

    original_order = sorted(pair_rows, key=lambda r: int(r["parallel_id"].split("/")[-1]))
    summary = {
        "sentence_pairs": len(yue_sentences),
        "tokens": {"yue": y_count, "zh": z_count},
        "mean_tokens_per_sentence": {"yue": y_count / len(yue_sentences), "zh": z_count / len(zh_sentences)},
        "mean_nonroot_arc_distance": {
            "yue": sum(x["distance"] for x in ya if x["head"]) / sum(x["head"] != 0 for x in ya),
            "zh": sum(x["distance"] for x in za if x["head"]) / sum(x["head"] != 0 for x in za),
        },
        "long_arc_6plus_percent": {
            "yue": 100 * sum(x["distance"] >= 6 for x in ya) / y_count,
            "zh": 100 * sum(x["distance"] >= 6 for x in za) / z_count,
        },
        "head_left_percent_nonroot": {
            "yue": 100 * sum(x["direction"] == "head_left" for x in ya) / sum(x["head"] != 0 for x in ya),
            "zh": 100 * sum(x["direction"] == "head_left" for x in za) / sum(x["head"] != 0 for x in za),
        },
        "strict_labels_only_in_yue": sorted(set(y_rel) - set(z_rel)),
        "strict_labels_only_in_zh": sorted(set(z_rel) - set(y_rel)),
        "divergences": divergences,
        "parallel_sentence_correlations": {
            "token_count": pearson([r["yue_tokens"] for r in original_order], [r["zh_tokens"] for r in original_order]),
            "mean_arc_distance": pearson([r["yue_mean_arc_distance"] for r in original_order], [r["zh_mean_arc_distance"] for r in original_order]),
            "long_arc_percent": pearson([r["yue_long_arc_percent"] for r in original_order], [r["zh_long_arc_percent"] for r in original_order]),
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yue", type=Path, default=DEFAULT_YUE)
    parser.add_argument("--zh", type=Path, default=DEFAULT_ZH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    yue = read_conllu(args.yue)
    zh = read_conllu(args.zh)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    split_by_position = {int(s["original_position"]): s["split"] for s in manifest["sentences"]}
    yue_by_pid = {s["meta"]["parallel_id"]: s for s in yue}
    zh_by_pid = {s["meta"]["parallel_id"]: s for s in zh}
    assert len(yue_by_pid) == len(yue) == 1004
    assert len(zh_by_pid) == len(zh) == 1004
    assert set(yue_by_pid) == set(zh_by_pid)

    train_ids = {s["meta"]["parallel_id"] for s in yue if split_by_position[s["position"]] == "train"}
    yue_train = [s for s in yue if s["meta"]["parallel_id"] in train_ids]
    zh_train = [zh_by_pid[s["meta"]["parallel_id"]] for s in yue_train]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "scope": {
            "primary": "Cantonese custom train (803 sentences) and official parallel_id-matched Chinese-HK translations",
            "appendix": "all 1004 parallel pairs; descriptive only, not for model/configuration selection",
            "no_model_loaded_or_modified": True,
            "no_dev_test_used_in_primary_analysis": True,
        },
        "sources": {
            "yue": {"path": str(args.yue), "sha256": sha256(args.yue), "tag": "r2.18"},
            "zh": {"path": str(args.zh), "sha256": sha256(args.zh), "tag": "r2.18"},
            "split_manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
            "pairing_evidence": "exact equality of unique official # parallel_id fields; no row-number assumption",
        },
        "primary_train_only": summarize_pair("train_only", yue_train, zh_train, args.output_dir),
        "descriptive_full_corpus_appendix": summarize_pair("full_corpus_descriptive", yue, zh, args.output_dir),
        "limitations": [
            "Chinese-HK is parallel reference data, not evidence of the ELECTRA parser checkpoint's actual training distribution.",
            "Translations have different tokenization and lexical realization, so arcs are not aligned token-to-token.",
            "Marginal label similarity cannot establish equality of conditional HEAD decisions or internal representations.",
            "The full-corpus appendix includes custom dev/test and must not guide hyperparameter or architecture selection.",
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    primary = result["primary_train_only"]
    (args.output_dir / "README.md").write_text(f"""# Parallel dependency-distribution analysis

Primary scope: the 803 custom Cantonese train sentences and their Chinese-HK
translations matched by the official unique `parallel_id`.  No model was loaded
or modified.  The full-corpus files are descriptive appendices only and must not
be used for architecture or hyperparameter selection.

Key train-only measurements:

- Tokens: Cantonese {primary['tokens']['yue']:,}; Chinese-HK {primary['tokens']['zh']:,}.
- Mean non-root arc distance: Cantonese {primary['mean_nonroot_arc_distance']['yue']:.3f}; Chinese-HK {primary['mean_nonroot_arc_distance']['zh']:.3f}.
- Arcs of distance >=6 (denominator: all integer-ID tokens): Cantonese {primary['long_arc_6plus_percent']['yue']:.2f}%; Chinese-HK {primary['long_arc_6plus_percent']['zh']:.2f}%.
- Strict DEPREL JS divergence: {primary['divergences']['deprel_strict']['jensen_shannon_bits']:.4f} bits.
- CoNLL18 base-DEPREL JS divergence: {primary['divergences']['deprel_conll18_base']['jensen_shannon_bits']:.4f} bits.
- Conditional attachment-signature JS divergence: {primary['divergences']['attachment_signature']['jensen_shannon_bits']:.4f} bits.

The attachment signature is `(dependent UPOS, head UPOS, whether the head is to
the left/right, distance bucket, base DEPREL)`.  Its larger divergence shows why
marginal label frequencies alone do not determine `P(HEAD, relation | sentence)`.

Limitations: Chinese-HK is not the documented training corpus of the Mandarin
ELECTRA parser; translations are not word-aligned and use different tokenization;
these results therefore describe the two annotated parallel treebanks rather than
the checkpoint's learned representation.
""", encoding="utf-8")

    checksums = {}
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.json":
            checksums[path.name] = sha256(path)
    (args.output_dir / "SHA256SUMS.json").write_text(json.dumps(checksums, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["primary_train_only"], ensure_ascii=False, indent=2))
    print("Wrote", args.output_dir)


if __name__ == "__main__":
    main()
