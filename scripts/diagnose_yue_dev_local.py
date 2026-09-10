#!/usr/bin/env python3
"""Run dev-only error diagnostics for the trained Yue LoRA parser locally."""

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import unicodedata
import zipfile
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import torch


SEED = 42
HF_REPO = "hfl/chinese-electra-180g-large-discriminator"
HF_REVISION = "d017e219578df8e4885484edbc8969dbdea9cbe0"
EXPECTED_CHECKPOINT_SHA = "3a56254dca03e20ba77d8fc2310123cce9a129963ff5954c472f0f9c02840f97"
EXPECTED_DEV_CACHE_SHA = "e88aa03ff7453469deda68ed30c29dd16769e9839d2010b36abf91faace3bd97"
EXPECTED_RESOURCES_SHA = "4e41c1df152146fa26ed0c006a08feea7a60bb3414bb6d57dbda24ad2e3cb99c"
EVALUATOR_URL = "https://universaldependencies.org/conll18/conll18_ud_eval.py"
EXPECTED_EVALUATOR_SHA = "1072e02af00b1a56205b5e8216d51dee9b8944a104d80744afaccc78859fcb16"
EXPECTED_ARTIFACTS = {
    "tokenize": ("gsdsimp", "962f2578e2a3dabeb4671053372eb1bd092357921904233556c8d77a46440882"),
    "pos": ("gsdsimp_electra-large", "f7a8cd0ae5c92c07655b7f3e8078d33541f66450dd75541881b6c2740a1b89b4"),
    "lemma": ("gsdsimp_charlm", "b940e3e3195403228cac8e873e4276ceac4691472310fbd413c344143a59c4ba"),
    "depparse": ("gsdsimp_electra-large", "6531bcc2dbfbe1e3b19d4deb855533fc2379f7305162423f7b53216b1e03434a"),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blocks(path):
    return Path(path).read_text(encoding="utf-8").strip().split("\n\n")


def integer_rows(block):
    return [
        line.split("\t")
        for line in block.splitlines()
        if line and not line.startswith("#") and line.split("\t", 1)[0].isdigit()
    ]


def sentence_meta(block):
    result = {}
    for line in block.splitlines():
        if line.startswith("# ") and " = " in line:
            key, value = line[2:].split(" = ", 1)
            result[key] = value
    return result


def safe_extract(zip_path, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
        archive.extractall(destination)


def make_pretagged(pipeline, source, destination, chunk_size=32):
    source_blocks = blocks(source)
    output = []
    for start in range(0, len(source_blocks), chunk_size):
        subset = source_blocks[start : start + chunk_size]
        forms = [[row[1] for row in integer_rows(block)] for block in subset]
        document = pipeline(forms)
        assert len(document.sentences) == len(subset)
        for block, sentence, gold_forms in zip(subset, document.sentences, forms):
            assert [word.text for word in sentence.words] == gold_forms
            predictions = iter(sentence.words)
            lines = []
            for line in block.splitlines():
                if line and not line.startswith("#") and line.split("\t", 1)[0].isdigit():
                    columns = line.split("\t")
                    word = next(predictions)
                    columns[2] = word.lemma or "_"
                    columns[3] = word.upos or "_"
                    columns[4] = word.xpos or "_"
                    columns[5] = word.feats or "_"
                    line = "\t".join(columns)
                lines.append(line)
            try:
                next(predictions)
                raise AssertionError("Extra predicted word")
            except StopIteration:
                pass
            output.append("\n".join(lines))
    destination.write_text("\n\n".join(output) + "\n", encoding="utf-8")


def make_goldpos_predlemma(predicted_cache, gold_path, destination):
    predicted_blocks = blocks(predicted_cache)
    gold_blocks = blocks(gold_path)
    assert len(predicted_blocks) == len(gold_blocks)
    output = []
    for predicted_block, gold_block in zip(predicted_blocks, gold_blocks):
        gold_words = iter(integer_rows(gold_block))
        lines = []
        for line in predicted_block.splitlines():
            if line and not line.startswith("#") and line.split("\t", 1)[0].isdigit():
                columns = line.split("\t")
                gold = next(gold_words)
                assert columns[:2] == gold[:2]
                columns[3:6] = gold[3:6]
                line = "\t".join(columns)
            lines.append(line)
        output.append("\n".join(lines))
    destination.write_text("\n\n".join(output) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-zip", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = (args.output_dir or root / "experiments/runs/yue_lora_electra_r8/dev_diagnostics").resolve()
    cache = root / ".cache/yue_dev_diagnostics"
    model_dir = cache / "stanza_resources_1.14.0"
    hf_home = cache / "hf_home"
    extracted = cache / "result_bundle"
    output.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    # The Xet transport can stall indefinitely on large files behind some
    # local proxy setups.  Standard HTTP is slower but resumable and reliable.
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cpu")

    if extracted.exists():
        shutil.rmtree(extracted)
    safe_extract(args.result_zip.resolve(), extracted)
    checkpoint = extracted / "best_dev_electra_yue_lora_r8.pt"
    assert sha256(checkpoint) == EXPECTED_CHECKPOINT_SHA
    run_result = json.loads((extracted / "result.json").read_text(encoding="utf-8"))
    assert run_result["hf_revision"] == HF_REVISION

    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            HF_REPO,
            revision=HF_REVISION,
            cache_dir=hf_home / "hub",
            allow_patterns=[
                "config.json",
                "pytorch_model.bin",
                "vocab.txt",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
            ],
        )
    )
    assert snapshot.name == HF_REVISION

    import stanza
    from stanza.pipeline.core import DownloadMethod
    from stanza.resources.common import download_resources_json, load_resources_json

    assert stanza.__version__ == "1.14.0"
    download_resources_json(model_dir=str(model_dir))
    assert sha256(model_dir / "resources.json") == EXPECTED_RESOURCES_SHA
    packages = {name: value[0] for name, value in EXPECTED_ARTIFACTS.items()}
    stanza.download("zh-hans", model_dir=str(model_dir), package=None, processors=packages, verbose=True)
    resources = load_resources_json(model_dir=str(model_dir))
    artifact_hashes = {}
    for processor, (package, expected_hash) in EXPECTED_ARTIFACTS.items():
        path = model_dir / "zh-hans" / processor / f"{package}.pt"
        actual = sha256(path)
        assert actual == expected_hash, (processor, actual, expected_hash)
        artifact_hashes[processor] = actual

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tagger = stanza.Pipeline(
        lang="zh-hans",
        dir=str(model_dir),
        processors={key: value for key, value in packages.items() if key != "depparse"},
        tokenize_pretokenized=True,
        use_gpu=False,
        download_method=DownloadMethod.REUSE_RESOURCES,
        verbose=False,
    )
    for processor in ("pos", "lemma"):
        model = tagger.processors[processor]._trainer.model
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

    dev_gold = root / "data/processed/yue_hk/dev.conllu"
    dev_cache = output / "dev.predposlemma.conllu"
    make_pretagged(tagger, dev_gold, dev_cache)
    del tagger
    gc.collect()
    actual_cache_hash = sha256(dev_cache)
    assert actual_cache_hash == EXPECTED_DEV_CACHE_SHA, (actual_cache_hash, EXPECTED_DEV_CACHE_SHA)

    evaluator_path = cache / "conll18_ud_eval.py"
    if not evaluator_path.exists() or sha256(evaluator_path) != EXPECTED_EVALUATOR_SHA:
        import urllib.request

        urllib.request.urlretrieve(EVALUATOR_URL, evaluator_path)
    assert sha256(evaluator_path) == EXPECTED_EVALUATOR_SHA
    spec = importlib.util.spec_from_file_location("official_conll18", evaluator_path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)

    def conll18(gold, system):
        return official.evaluate(official.load_conllu_file(str(gold)), official.load_conllu_file(str(system)))

    from stanza.models.common.doc import DEPREL, HEAD
    from stanza.models.common.pretrain import Pretrain
    from stanza.models.depparse.data import DataLoader
    from stanza.models.depparse.trainer import GraphTrainer
    from stanza.models.depparse.utils import predict_dataset
    from stanza.utils.conll import CoNLL

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    load_args = {}
    dependencies = resources["zh-hans"]["depparse"]["gsdsimp_electra-large"].get("dependencies", [])

    def dependency_path(model_type):
        names = [item["package"] for item in dependencies if item.get("model") == model_type]
        paths = [model_dir / "zh-hans" / model_type / f"{name}.pt" for name in names]
        paths = [path for path in paths if path.exists()]
        if len(paths) != 1:
            paths = list((model_dir / "zh-hans" / model_type).glob("*.pt"))
        assert len(paths) == 1, (model_type, paths)
        return paths[0].resolve()

    if saved["config"].get("charlm"):
        load_args["charlm_forward_file"] = str(dependency_path("forward_charlm"))
        load_args["charlm_backward_file"] = str(dependency_path("backward_charlm"))
    pretrain_obj = None
    if saved["config"].get("pretrain"):
        pretrain_obj = Pretrain(filename=str(dependency_path("pretrain")))
    trainer = GraphTrainer.load(str(checkpoint), pretrain=pretrain_obj, args=load_args, device=device)
    trainer.model.eval()

    goldpos_input = output / "dev.goldpos_predlemma.conllu"
    make_goldpos_predlemma(dev_cache, dev_gold, goldpos_input)

    def predict(input_path, output_path):
        document = CoNLL.conll2doc(input_file=str(input_path))
        loader = DataLoader(
            document,
            900,
            trainer.args,
            pretrain_obj,
            vocab=trainer.vocab,
            evaluation=True,
            sort_during_eval=True,
            bert_tokenizer=trainer.model.bert_tokenizer,
        )
        predictions = predict_dataset(trainer, loader)
        loader.doc.set([HEAD, DEPREL], [item for sentence in predictions for item in sentence])
        output_path.write_text(f"{loader.doc:C}\n\n", encoding="utf-8")
        return conll18(dev_gold, output_path)

    predicted_output = output / "dev.predposlemma.best.pred.conllu"
    goldpos_output = output / "dev.goldpos_predlemma.best.pred.conllu"
    predicted_scores = predict(dev_cache, predicted_output)
    goldpos_scores = predict(goldpos_input, goldpos_output)
    expected_dev = run_result["best_dev_conll18_las_percent"] / 100
    assert abs(predicted_scores["LAS"].f1 - expected_dev) < 1e-12

    gold_blocks = blocks(dev_gold)
    input_blocks = blocks(dev_cache)
    predicted_blocks = blocks(predicted_output)
    goldpos_blocks = blocks(goldpos_output)
    records = []
    sentence_records = []
    for sentence_index, (gold_block, input_block, predicted_block, goldpos_block) in enumerate(
        zip(gold_blocks, input_blocks, predicted_blocks, goldpos_blocks), 1
    ):
        gold_rows, input_rows, predicted_rows, goldpos_rows = map(
            integer_rows, (gold_block, input_block, predicted_block, goldpos_block)
        )
        meta = sentence_meta(gold_block)
        correct = 0
        for gold, source, predicted, goldpos in zip(gold_rows, input_rows, predicted_rows, goldpos_rows):
            assert gold[:2] == source[:2] == predicted[:2] == goldpos[:2]
            gold_head, predicted_head, goldpos_head = int(gold[6]), int(predicted[6]), int(goldpos[6])
            base = lambda relation: relation.split(":", 1)[0]
            distance = 0 if gold_head == 0 else abs(int(gold[0]) - gold_head)
            bucket = "ROOT" if gold_head == 0 else "1" if distance == 1 else "2" if distance == 2 else "3-5" if distance <= 5 else "6-10" if distance <= 10 else "11+"
            las = gold_head == predicted_head and base(gold[7]) == base(predicted[7])
            correct += las
            records.append(
                {
                    "sentence_index": sentence_index,
                    "sent_id": meta.get("sent_id"),
                    "text": meta.get("text"),
                    "token_id": int(gold[0]),
                    "form": gold[1],
                    "gold_upos": gold[3],
                    "predicted_upos": source[3],
                    "upos_correct": gold[3] == source[3],
                    "full_morph_correct": gold[3:6] == source[3:6],
                    "gold_head": gold_head,
                    "predicted_head": predicted_head,
                    "goldpos_pred_head": goldpos_head,
                    "gold_deprel": gold[7],
                    "pred_deprel": predicted[7],
                    "goldpos_pred_deprel": goldpos[7],
                    "head_distance": distance,
                    "distance_bucket": bucket,
                    "uas": gold_head == predicted_head,
                    "conll18_las": las,
                    "strict_las": gold_head == predicted_head and gold[7] == predicted[7],
                    "goldpos_uas": gold_head == goldpos_head,
                    "goldpos_conll18_las": gold_head == goldpos_head and base(gold[7]) == base(goldpos[7]),
                    "goldpos_strict_las": gold_head == goldpos_head and gold[7] == goldpos[7],
                }
            )
        sentence_records.append(
            {
                "sentence_index": sentence_index,
                "sent_id": meta.get("sent_id"),
                "text": meta.get("text"),
                "tokens": len(gold_rows),
                "conll18_las": correct / len(gold_rows),
            }
        )

    frame = pd.DataFrame(records)
    sentences = pd.DataFrame(sentence_records)
    frame.to_csv(output / "token_diagnostics.csv", index=False)
    sentences.sort_values(["conll18_las", "tokens"]).to_csv(output / "worst_sentences.csv", index=False)

    def grouped_stats(column):
        return (
            frame.groupby(column, dropna=False)
            .agg(
                tokens=("token_id", "size"),
                upos_accuracy=("upos_correct", "mean"),
                UAS=("uas", "mean"),
                CoNLL18_LAS=("conll18_las", "mean"),
                strict_LAS=("strict_las", "mean"),
                goldPOS_CoNLL18_LAS=("goldpos_conll18_las", "mean"),
            )
            .reset_index()
        )

    grouped_stats("distance_bucket").to_csv(output / "distance_stats.csv", index=False)
    grouped_stats("gold_deprel").sort_values(["CoNLL18_LAS", "tokens"], ascending=[True, False]).to_csv(
        output / "relation_stats.csv", index=False
    )
    grouped_stats("upos_correct").to_csv(output / "pos_error_association.csv", index=False)
    confusion = (
        frame[frame["uas"] & ~frame["strict_las"]]
        .groupby(["gold_deprel", "pred_deprel"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    confusion.to_csv(output / "relation_confusions_when_head_correct.csv", index=False)

    groups = {}
    for index, block in enumerate(gold_blocks):
        rows = integer_rows(block)
        forms = tuple(unicodedata.normalize("NFC", row[1]) for row in rows)
        joined = unicodedata.normalize("NFC", "".join(row[1] for row in rows))
        for key in (("tokens", forms), ("joined", joined)):
            groups.setdefault(key, set()).add(index)
    duplicate_audit = []
    for key, indices in groups.items():
        if len(indices) < 2:
            continue
        analyses = {
            tuple((row[6], row[7]) for row in integer_rows(gold_blocks[index])) for index in indices
        }
        duplicate_audit.append(
            {
                "key_type": key[0],
                "indices_1_based": [index + 1 for index in sorted(indices)],
                "sent_ids": [sentence_meta(gold_blocks[index]).get("sent_id") for index in sorted(indices)],
                "annotation_disagreement": len(analyses) > 1,
            }
        )
    (output / "exact_duplicate_annotation_audit.json").write_text(
        json.dumps(duplicate_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    texts = ["".join(row[1] for row in integer_rows(block)) for block in gold_blocks]
    near = []
    for left in range(len(texts)):
        for right in range(left + 1, len(texts)):
            similarity = SequenceMatcher(None, texts[left], texts[right], autojunk=False).ratio()
            if similarity >= 0.85 and texts[left] != texts[right]:
                near.append(
                    {
                        "similarity": similarity,
                        "index_a": left + 1,
                        "index_b": right + 1,
                        "sent_id_a": sentence_meta(gold_blocks[left]).get("sent_id"),
                        "sent_id_b": sentence_meta(gold_blocks[right]).get("sent_id"),
                        "text_a": sentence_meta(gold_blocks[left]).get("text"),
                        "text_b": sentence_meta(gold_blocks[right]).get("text"),
                    }
                )
    pd.DataFrame(near).to_csv(output / "near_similar_manual_review_candidates.csv", index=False)

    pos_right = frame[frame["upos_correct"]]
    pos_wrong = frame[~frame["upos_correct"]]
    percent = lambda value: 100 * float(value)
    summary = {
        "split": "dev",
        "sentences": len(gold_blocks),
        "tokens": len(frame),
        "checkpoint_sha256": sha256(checkpoint),
        "hf_revision": HF_REVISION,
        "frozen_dev_cache_sha256": actual_cache_hash,
        "processor_artifact_sha256": artifact_hashes,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "stanza": stanza.__version__,
            "device": str(device),
        },
        "predicted_condition": {
            "UAS_percent": percent(predicted_scores["UAS"].f1),
            "CoNLL18_LAS_percent": percent(predicted_scores["LAS"].f1),
            "strict_LAS_percent": percent(frame["strict_las"].mean()),
        },
        "gold_pos_morph_predicted_lemma_condition": {
            "UAS_percent": percent(goldpos_scores["UAS"].f1),
            "CoNLL18_LAS_percent": percent(goldpos_scores["LAS"].f1),
            "strict_LAS_percent": percent(frame["goldpos_strict_las"].mean()),
        },
        "gold_pos_delta_points": percent(goldpos_scores["LAS"].f1 - predicted_scores["LAS"].f1),
        "predicted_UPOS_accuracy_percent": percent(frame["upos_correct"].mean()),
        "dependency_LAS_given_UPOS_correct_percent": percent(pos_right["conll18_las"].mean()),
        "dependency_LAS_given_UPOS_wrong_percent": percent(pos_wrong["conll18_las"].mean()),
        "tokens_UPOS_correct": len(pos_right),
        "tokens_UPOS_wrong": len(pos_wrong),
        "exact_duplicate_matching_records": len(duplicate_audit),
        "exact_duplicate_records_with_annotation_disagreement": sum(
            item["annotation_disagreement"] for item in duplicate_audit
        ),
        "near_similar_pairs_for_manual_review": len(near),
        "interpretation_warning": "Gold POS/morph is a diagnostic distribution shift because training used predicted tags. Near-similar pairs are not automatically annotation errors.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = shutil.make_archive(str(output), "zip", root_dir=output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Diagnostics: {output}")
    print(f"Archive: {archive}")


if __name__ == "__main__":
    main()
