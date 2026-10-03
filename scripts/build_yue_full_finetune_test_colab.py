#!/usr/bin/env python3
"""Build a self-contained Run-all full-fine-tuning replication Colab."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "yue_full_finetune_vs_lora.ipynb"
source = json.loads(SOURCE.read_text(encoding="utf-8"))


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


def cell_source(index: int) -> str:
    return "".join(source["cells"][index]["source"])


install = cell_source(1)
setup = cell_source(2).replace(
    "/content/yue_full_finetune_comparison", "/content/yue_full_finetune_replicate"
)

download_data = cell_source(3)
old_split_loop = "for split in ('train','dev'):\n    positions = SPLIT_POSITIONS[split]"
assert old_split_loop in download_data
download_data = download_data.replace(
    old_split_loop, "for split, positions in SPLIT_POSITIONS.items():"
)

resolve_hf = cell_source(4)
download_stanza = cell_source(5)

tag_data = cell_source(6)
tag_data = tag_data.replace(
    "# Run frozen POS/lemma exactly once and build pretagged train/dev caches.",
    "# Run frozen POS/lemma exactly once and build pretagged train/dev/test caches.",
)
assert "for split in ('train','dev'):" in tag_data
tag_data = tag_data.replace(
    "for split in ('train','dev'):", "for split in ('train','dev','test'):"
)
old_cache = """EXPECTED_CACHE_SHA256={'train':'806b3d42c4f6838d6ba9c6565f29062a0d7b6e136e12d158e4449c3e7e970e89',
                       'dev':'e88aa03ff7453469deda68ed30c29dd16769e9839d2010b36abf91faace3bd97'}"""
new_cache = """EXPECTED_CACHE_SHA256={'train':'806b3d42c4f6838d6ba9c6565f29062a0d7b6e136e12d158e4449c3e7e970e89',
                       'dev':'e88aa03ff7453469deda68ed30c29dd16769e9839d2010b36abf91faace3bd97',
                       'test':'8c5ee2a043ceb7870d636d798c3c2863c9da82adebcac9dffd53670cadb26a7f'}"""
assert old_cache in tag_data
tag_data = tag_data.replace(old_cache, new_cache)

full_init = cell_source(7)
full_train = cell_source(8)

test_eval = r'''# Evaluate test exactly once after DEV has selected the checkpoint.
# The custom test was previously used to select the Mandarin model family, but
# it is not consulted anywhere in this full-FT training/checkpoint selection loop.
del trainer
gc.collect()
torch.cuda.empty_cache()

best_trainer=GraphTrainer.load(str(best_path),pretrain=pretrain_obj,args=load_args,device=DEVICE)
for parameter in best_trainer.model.parameters(): parameter.requires_grad=False
best_trainer.optimizer={}; best_trainer.scheduler={}

test_doc=CoNLL.conll2doc(input_file=str(DATA/'test.predposlemma.conllu'))
test_loader=DataLoader(test_doc,CFG['batch_size'],best_trainer.args,pretrain_obj,
    vocab=best_trainer.vocab,evaluation=True,sort_during_eval=True,
    bert_tokenizer=best_trainer.model.bert_tokenizer)
test_pred=OUT/'test.full.best_dev.pred.conllu'
predict_to_file(best_trainer,test_loader,test_pred)
test_scores=conll18(DATA/'test.conllu',test_pred)

def strict_las(gold_path,pred_path,exclude_punct=False):
    gold_blocks=split_blocks(gold_path.read_text(encoding='utf-8'))
    pred_blocks=split_blocks(pred_path.read_text(encoding='utf-8'))
    assert len(gold_blocks)==len(pred_blocks)==100
    correct=total=0
    for gold_block,pred_block in zip(gold_blocks,pred_blocks):
        gold_rows=integer_rows(gold_block); pred_rows=integer_rows(pred_block)
        assert len(gold_rows)==len(pred_rows)
        for gold,pred in zip(gold_rows,pred_rows):
            assert gold[:2]==pred[:2]
            if exclude_punct and gold[3]=='PUNCT': continue
            total+=1
            correct+=int(gold[6]==pred[6] and gold[7]==pred[7])
    return correct/total

result={
 'split':'test',
 'seed':SEED,
 'checkpoint_selection':'DEV CoNLL-2018 LAS only; earliest checkpoint wins ties',
 'best_dev_step':best_step,
 'best_dev_conll18_las_percent':100*best_score,
 'test_conll18_las_percent':100*test_scores['LAS'].f1,
 'test_conll18_uas_percent':100*test_scores['UAS'].f1,
 'test_strict_full_deprel_percent':100*strict_las(DATA/'test.conllu',test_pred),
 'test_strict_full_deprel_no_punct_percent':100*strict_las(DATA/'test.conllu',test_pred,True),
 'reference_prior_full_ft':{'best_dev_step':900,'dev_las_percent':75.65831727681439,
                            'test_las_percent':76.76470588235294},
 'reference_lora':{'best_dev_step':1200,'dev_las_percent':74.82337829158638,
                   'test_las_percent':75.66176470588235},
 'initialization':{'stanza_depparse_sha256':sha256(base_path),'hf_repo':HF_REPO,
                   'hf_revision':HF_COMMIT,'old_parser_weights_preserved':True,
                   'new_deprel_labels':new_labels},
 'controls':{'gold_sentence_boundaries':True,'gold_tokenization':True,
             'frozen_predicted_pos_lemma':True,'cache_sha256':cache_hashes,
             'split_sha256':SPLIT_SHA256},
 'config':CFG,
 'checkpoint_sha256':sha256(best_path),
 'test_prediction_sha256':sha256(test_pred),
 'test_selection_caveat':('This custom test had previously been used to choose the Mandarin '
                           'model family; the full-FT checkpoint itself was selected only on dev.'),
 'replication_note':('GPU training may not be bitwise deterministic across Colab hardware/CUDA '
                     'versions; compare the selected step and LAS, not only file hashes.')}
(OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
'''

export = r'''# Package the reproducibility record, predictions, and dev history.
# The small results ZIP downloads automatically.  The 1+ GB checkpoint is kept
# separately and also packaged for users who want to retain the trained model.
from google.colab import files
run_record={'config':CFG,'seed':SEED,
 'software':{'stanza':'1.14.0','transformers':'4.56.2','huggingface_hub':'0.34.4'},
 'input_hashes':{'train':SPLIT_SHA256['train'],'dev':SPLIT_SHA256['dev'],'test':SPLIT_SHA256['test'],
                 'train_predposlemma':cache_hashes['train'],
                 'dev_predposlemma':cache_hashes['dev'],
                 'test_predposlemma':cache_hashes['test']},
 'hf_revision':HF_COMMIT,'test_evaluated_once_after_dev_selection':True}
(OUT/'run_record.json').write_text(json.dumps(run_record,ensure_ascii=False,indent=2),encoding='utf-8')

small=WORK/'results_bundle'; small.mkdir(exist_ok=True)
for path in (OUT/'result.json',OUT/'run_record.json',OUT/'full_dev_history.json',test_pred):
    shutil.copy2(path,small/path.name)
small_zip=shutil.make_archive('/content/yue_full_finetune_test_results','zip',root_dir=small)

complete=WORK/'complete_bundle'; complete.mkdir(exist_ok=True)
for path in small.iterdir(): shutil.copy2(path,complete/path.name)
shutil.copy2(best_path,complete/best_path.name)
complete_zip=shutil.make_archive('/content/yue_full_finetune_test_complete','zip',root_dir=complete)

print('results archive:',small_zip,sha256(Path(small_zip)))
print('complete archive:',complete_zip,sha256(Path(complete_zip)))
print('checkpoint:',best_path,sha256(best_path))
print('The complete archive is over 1 GB; keep it if you need the replicated model.')
files.download(small_zip)
files.download(complete_zip)
'''

cells = [
    md(
        """# Cantonese ELECTRA-large full fine-tuning: train, DEV selection, and fixed TEST

This notebook is self-contained and safe to **Run all** after selecting a GPU runtime. It reproduces the full-Transformer fine-tuning condition from the original Mandarin Stanza 1.14 ELECTRA-large dependency checkpoint. It requires no uploaded files and does not train LoRA.

Protocol: fixed UD Cantonese-HK r2.18 source; custom grouped 803/101/100 train/dev/test split; gold sentence boundaries and tokenization; frozen Mandarin predicted POS/lemma; gold HEAD/DEPREL supervision; complete Transformer plus parsing-layer updates; seed 42; checkpoint selection only by DEV CoNLL-2018 LAS; one TEST evaluation after selection.

Use an **A100 GPU** if available. Full ELECTRA-large fine-tuning, gradients, optimizer state, and checkpoint storage require substantially more GPU memory and disk than LoRA. The notebook verifies all source, split, processor, predicted-feature-cache, checkpoint, and prediction hashes available at each stage. Exact checkpoint bytes can still vary across Colab GPU/CUDA environments even with fixed seeds.

The fixed test was previously used in this project to select the Mandarin model family. Therefore, the resulting test score is comparable with the earlier runs but must not be described as coming from a completely untouched model-selection set.
"""
    ),
    code(install),
    code(setup),
    code(download_data),
    code(resolve_hf),
    code(download_stanza),
    code(tag_data),
    code(full_init),
    code(full_train),
    code(test_eval),
    code(export),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "yue_full_finetune_test_replicate.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

destination = ROOT / "notebooks" / "yue_full_finetune_test_replicate.ipynb"
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

pack = ROOT / "colab" / "yue_full_finetune_test_replicate"
if pack.exists():
    shutil.rmtree(pack)
pack.mkdir(parents=True)
shutil.copy2(destination, pack / destination.name)
(pack / "README.txt").write_text(
    "Upload yue_full_finetune_test_replicate.ipynb to Google Colab, choose an A100 GPU "
    "runtime if available, and click Runtime > Run all. No file upload is required. The "
    "notebook downloads a small result archive and a large archive containing the checkpoint.\n",
    encoding="utf-8",
)
print(destination)
print(pack)
