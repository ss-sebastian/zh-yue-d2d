#!/usr/bin/env python3
"""Build a standalone, Run-all-safe, dev-diagnostics-only Colab."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "data/processed/yue_hk/split_manifest.json").read_text())
DEV_POSITIONS = [x["original_position"] for x in MANIFEST["sentences"] if x["split"] == "dev"]
DEV_SHA = MANIFEST["output_sha256"]["dev.conllu"]
TRAINING_NB = json.loads((ROOT / "notebooks/finetune_yue_stanza_electra_lora.ipynb").read_text())
DIAGNOSTIC_SOURCE = next(
    "".join(cell["source"])
    for cell in TRAINING_NB["cells"]
    if cell["cell_type"] == "code" and "# DEV ONLY: error attribution" in "".join(cell["source"])
)
DIAGNOSTIC_SOURCE = DIAGNOSTIC_SOURCE.replace(
    "diag_trainer=GraphTrainer.load(str(best_path),pretrain=pretrain_obj,device=DEVICE)\n", ""
)


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


cells = [
    md("""# Cantonese LoRA parser — dev diagnostics only

This notebook is safe to **Run all**. It never reads the train split, calls no training update, changes no weight, and does not evaluate test. Immediately after Stanza loads the checkpoint, every parameter is frozen and the loader-created optimizer/scheduler references are discarded. It uploads the already-trained result ZIP, verifies the dev-best checkpoint, reconstructs only the fixed dev split, reproduces the frozen predicted POS/lemma cache, and runs two dev inference conditions:

1. frozen predicted POS/lemma (the main protocol condition);
2. gold UPOS/XPOS/FEATS with the same frozen predicted lemma (diagnostic intervention).

Near-similar sentence pairs are manual-review candidates only. No annotation is automatically changed."""),
    code(r"""# Install the same software versions as the training run.
%pip install -q stanza==1.14.0 transformers==4.56.2 peft==0.17.1 huggingface-hub==0.34.4 pandas==2.2.3
"""),
    code(fr"""from pathlib import Path
import os, json, hashlib, random, shutil, gc, importlib.util, zipfile
import numpy as np
import pandas as pd
import torch
from google.colab import files

SEED=42
WORK=Path('/content/yue_dev_diagnostics_only')
DATA=WORK/'data'; MODELS=WORK/'stanza_resources_1.14.0'; OUT=WORK/'outputs'; HF_HOME=WORK/'hf_home'
for p in (DATA,MODELS,OUT,HF_HOME): p.mkdir(parents=True,exist_ok=True)
os.environ['HF_HOME']=str(HF_HOME)
os.environ['HF_HUB_DISABLE_XET']='1'

RAW_URL='https://raw.githubusercontent.com/UniversalDependencies/UD_Cantonese-HK/r2.18/yue_hk-ud-test.conllu'
RAW_SHA256='cbd843a195d0db4cdafbf6fcafb7b7b559afea750411006f4728311e70cc4e2a'
DEV_POSITIONS={DEV_POSITIONS!r}
DEV_SHA256={DEV_SHA!r}
EXPECTED_DEV_CACHE_SHA='e88aa03ff7453469deda68ed30c29dd16769e9839d2010b36abf91faace3bd97'
EXPECTED_CHECKPOINT_SHA='3a56254dca03e20ba77d8fc2310123cce9a129963ff5954c472f0f9c02840f97'
EXPECTED_BEST_DEV=0.7482337829158638
HF_REPO='hfl/chinese-electra-180g-large-discriminator'
HF_REVISION='d017e219578df8e4885484edbc8969dbdea9cbe0'
EVAL_URL='https://universaldependencies.org/conll18/conll18_ud_eval.py'
EVAL_SHA256='1072e02af00b1a56205b5e8216d51dee9b8944a104d80744afaccc78859fcb16'
CFG={{'batch_size':900}}

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert DEVICE.type=='cuda','Select a Colab GPU runtime; this notebook does inference only but uses ELECTRA-large.'

print('Upload yue_lora_electra_r8_results.zip')
uploaded=files.upload()
zip_names=[name for name in uploaded if name.endswith('.zip')]
assert len(zip_names)==1,zip_names
result_zip=Path('/content')/zip_names[0]
with zipfile.ZipFile(result_zip) as z:
    member='best_dev_electra_yue_lora_r8.pt'
    assert member in z.namelist()
    z.extract(member,WORK)
best_path=WORK/member
assert sha256(best_path)==EXPECTED_CHECKPOINT_SHA
best_step=1200; best_score=EXPECTED_BEST_DEV
print('checkpoint verified:',sha256(best_path))
"""),
    code(r"""# Reconstruct ONLY dev and verify the project hash.
import urllib.request
raw_path=DATA/'yue_hk-ud-test.r2.18.conllu'
urllib.request.urlretrieve(RAW_URL,raw_path)
assert sha256(raw_path)==RAW_SHA256
raw=raw_path.read_text(encoding='utf-8')
all_blocks=raw.strip().split('\n\n')
assert len(all_blocks)==1004 and len(DEV_POSITIONS)==101
dev_gold=DATA/'dev.conllu'
dev_gold.write_text('\n\n'.join(all_blocks[i-1] for i in DEV_POSITIONS)+'\n\n',encoding='utf-8')
assert sha256(dev_gold)==DEV_SHA256

eval_path=WORK/'conll18_ud_eval.py'
urllib.request.urlretrieve(EVAL_URL,eval_path)
assert sha256(eval_path)==EVAL_SHA256
spec=importlib.util.spec_from_file_location('official_conll18',eval_path)
official=importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
def conll18(gold,system):
    return official.evaluate(official.load_conllu_file(str(gold)),official.load_conllu_file(str(system)))

def split_blocks(text): return text.strip().split('\n\n')
def integer_rows(block):
    return [line.split('\t') for line in block.splitlines()
            if line and not line.startswith('#') and line.split('\t',1)[0].isdigit()]
"""),
    code(r"""# Download only inference dependencies at the exact recorded versions.
from huggingface_hub import snapshot_download
snapshot=Path(snapshot_download(HF_REPO,revision=HF_REVISION,cache_dir=HF_HOME/'hub',
    allow_patterns=['config.json','pytorch_model.bin','vocab.txt','tokenizer.json',
                    'tokenizer_config.json','special_tokens_map.json','added_tokens.json']))
assert snapshot.name==HF_REVISION

import stanza
from stanza.resources.common import download_resources_json,load_resources_json
from stanza.pipeline.core import DownloadMethod
assert stanza.__version__=='1.14.0'
download_resources_json(model_dir=str(MODELS))
assert sha256(MODELS/'resources.json')=='4e41c1df152146fa26ed0c006a08feea7a60bb3414bb6d57dbda24ad2e3cb99c'
packages={'tokenize':'gsdsimp','pos':'gsdsimp_electra-large','lemma':'gsdsimp_charlm'}
stanza.download('zh-hans',model_dir=str(MODELS),package=None,processors=packages,verbose=True)

expected={
 'tokenize':'962f2578e2a3dabeb4671053372eb1bd092357921904233556c8d77a46440882',
 'pos':'f7a8cd0ae5c92c07655b7f3e8078d33541f66450dd75541881b6c2740a1b89b4',
 'lemma':'b940e3e3195403228cac8e873e4276ceac4691472310fbd413c344143a59c4ba'}
for proc,pkg in packages.items(): assert sha256(MODELS/'zh-hans'/proc/f'{pkg}.pt')==expected[proc]

os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
tagger=stanza.Pipeline(lang='zh-hans',dir=str(MODELS),processors=packages,
    tokenize_pretokenized=True,use_gpu=True,download_method=DownloadMethod.REUSE_RESOURCES,verbose=False)
for proc in ('pos','lemma'):
    tagger.processors[proc]._trainer.model.eval()
    for p in tagger.processors[proc]._trainer.model.parameters(): p.requires_grad=False

def make_pretagged(src,dst,chunk=32):
    bs=split_blocks(src.read_text(encoding='utf-8')); out=[]
    for start in range(0,len(bs),chunk):
        sub=bs[start:start+chunk]; forms=[[r[1] for r in integer_rows(b)] for b in sub]
        doc=tagger(forms); assert len(doc.sentences)==len(sub)
        for block,sent,gold_forms in zip(sub,doc.sentences,forms):
            assert [w.text for w in sent.words]==gold_forms
            pred=iter(sent.words); lines=[]
            for line in block.splitlines():
                if line and not line.startswith('#') and line.split('\t',1)[0].isdigit():
                    c=line.split('\t'); w=next(pred)
                    c[2]=w.lemma or '_'; c[3]=w.upos or '_'; c[4]=w.xpos or '_'; c[5]=w.feats or '_'
                    line='\t'.join(c)
                lines.append(line)
            out.append('\n'.join(lines))
    dst.write_text('\n\n'.join(out)+'\n',encoding='utf-8')

dev_predcache=DATA/'dev.predposlemma.conllu'
make_pretagged(dev_gold,dev_predcache)
assert sha256(dev_predcache)==EXPECTED_DEV_CACHE_SHA,(sha256(dev_predcache),EXPECTED_DEV_CACHE_SHA)
del tagger; gc.collect(); torch.cuda.empty_cache()
print('frozen dev cache reproduced:',sha256(dev_predcache))
"""),
    code(r"""# Load the existing checkpoint, then force inference-only state.
from stanza.models.common.pretrain import Pretrain
from stanza.models.depparse.trainer import GraphTrainer
from stanza.models.depparse.data import DataLoader
from stanza.models.depparse.utils import predict_dataset
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import HEAD,DEPREL

resources=load_resources_json(model_dir=str(MODELS))
saved=torch.load(best_path,map_location='cpu',weights_only=True)
dependencies=resources['zh-hans']['pos']['gsdsimp_electra-large'].get('dependencies',[])

def one_local(model_type):
    paths=list((MODELS/'zh-hans'/model_type).glob('*.pt'))
    assert len(paths)==1,(model_type,paths)
    return paths[0].resolve()

load_args={}
if saved['config'].get('charlm'):
    load_args['charlm_forward_file']=str(one_local('forward_charlm'))
    load_args['charlm_backward_file']=str(one_local('backward_charlm'))
pretrain_obj=Pretrain(filename=str(one_local('pretrain'))) if saved['config'].get('pretrain') else None
diag_trainer=GraphTrainer.load(str(best_path),pretrain=pretrain_obj,args=load_args,device=DEVICE)
diag_trainer.model.eval()
for parameter in diag_trainer.model.parameters(): parameter.requires_grad=False
diag_trainer.optimizer=None
diag_trainer.scheduler=None
assert all(not p.requires_grad for p in diag_trainer.model.parameters()), 'Inference model must be fully frozen'

def predict_to_file(tr,loader,path):
    with torch.inference_mode(): preds=predict_dataset(tr,loader)
    loader.doc.set([HEAD,DEPREL],[y for x in preds for y in x])
    path.write_text(f'{loader.doc:C}\n\n',encoding='utf-8')
    return path
"""),
    md("""## Execute dev diagnostics

The next cell performs only two frozen dev inference passes and exports diagnostic tables. It contains no training update."""),
    code(DIAGNOSTIC_SOURCE),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "yue_dev_diagnostics_only.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

destination = ROOT / "notebooks/yue_dev_diagnostics_only.ipynb"
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(destination)
