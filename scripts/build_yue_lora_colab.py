#!/usr/bin/env python3
"""Build the self-contained Cantonese ELECTRA LoRA Colab notebook."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "data/processed/yue_hk/split_manifest.json").read_text())
POSITIONS = {
    split: [x["original_position"] for x in MANIFEST["sentences"] if x["split"] == split]
    for split in ("train", "dev", "test")
}
HASHES = MANIFEST["output_sha256"]


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


cells = [
md("""# Cantonese-HK dependency parser: ELECTRA-large LoRA fine-tuning

This notebook performs one pre-registered run only: Stanza 1.14.0 Mandarin ELECTRA parser initialization, frozen Mandarin POS/lemma predictions, LoRA rank 8 on the Transformer, and full parsing-layer updates. Model/checkpoint selection uses only custom **dev CoNLL-2018 LAS**. The custom test is evaluated in the final, separately marked cell exactly once.

Important disclosure: this custom test was already used to compare Mandarin model families; the historical ELECTRA baseline is 41.32%. It is therefore not an untouched test set. The notebook does not add a gold-POS condition, augmentation, SRep, or Mandarin training data.

Use a GPU runtime. Runtime can be substantial because ELECTRA-large is used both to freeze POS predictions and in parser training."""),
code(r"""# Pinned Python packages.  Restart the runtime after this cell if Colab asks.
%pip install -q stanza==1.14.0 transformers==4.56.2 peft==0.17.1 huggingface-hub==0.34.4
"""),
code(fr"""from pathlib import Path
import os, json, hashlib, random, shutil, gc, copy, importlib.util, logging
import numpy as np
import torch

SEED = 42
WORK = Path('/content/yue_lora_electra_r8')
DATA = WORK / 'data'
MODELS = WORK / 'stanza_resources_1.14.0'
OUT = WORK / 'outputs'
HF_HOME = WORK / 'hf_home'
for p in (DATA, MODELS, OUT, HF_HOME): p.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(HF_HOME)

RAW_URL = 'https://raw.githubusercontent.com/UniversalDependencies/UD_Cantonese-HK/r2.18/yue_hk-ud-test.conllu'
RAW_SHA256 = 'cbd843a195d0db4cdafbf6fcafb7b7b559afea750411006f4728311e70cc4e2a'
SPLIT_POSITIONS = {POSITIONS!r}
SPLIT_SHA256 = {{'train': {HASHES['train.conllu']!r}, 'dev': {HASHES['dev.conllu']!r}, 'test': {HASHES['test.conllu']!r}}}
EVAL_URL = 'https://universaldependencies.org/conll18/conll18_ud_eval.py'
EVAL_SHA256 = '1072e02af00b1a56205b5e8216d51dee9b8944a104d80744afaccc78859fcb16'
HF_REPO = 'hfl/chinese-electra-180g-large-discriminator'
# If the earlier 41.32% run recorded a HF commit, paste it here.  None resolves HEAD once,
# records the exact commit, and then forces offline reuse for the rest of this run.
HF_ELECTRA_REVISION = None

CFG = dict(rank=8, alpha=16, dropout=0.10,
           targets=['query', 'value', 'output.dense', 'intermediate.dense'],
           parser_lr=1e-3, lora_lr=2e-5, batch_size=900,
           max_steps=4000, eval_interval=100, patience_steps=600,
           max_grad_norm=1.0)

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert DEVICE.type == 'cuda', 'ELECTRA-large training requires a Colab GPU runtime.'
print('device:', DEVICE, 'config:', CFG)
"""),
code(r"""# Download the pinned UD source and reconstruct the project's exact custom splits.
import urllib.request
raw_path = DATA / 'yue_hk-ud-test.r2.18.conllu'
urllib.request.urlretrieve(RAW_URL, raw_path)
assert sha256(raw_path) == RAW_SHA256

raw = raw_path.read_text(encoding='utf-8')
blocks = raw.strip().split('\n\n')
assert len(blocks) == 1004
for split, positions in SPLIT_POSITIONS.items():
    # CoNLL-U requires a blank line between sentence blocks.  The project files
    # also end in one blank line, hence the final two newline characters.
    text = '\n\n'.join(blocks[i-1] for i in positions) + '\n\n'
    path = DATA / f'{split}.conllu'
    path.write_text(text, encoding='utf-8')
    assert sha256(path) == SPLIT_SHA256[split], (split, sha256(path))
    print(split, len(positions), sha256(path))

eval_path = WORK / 'conll18_ud_eval.py'
urllib.request.urlretrieve(EVAL_URL, eval_path)
assert sha256(eval_path) == EVAL_SHA256
spec=importlib.util.spec_from_file_location('official_conll18', eval_path)
official=importlib.util.module_from_spec(spec); spec.loader.exec_module(official)

def conll18(path_gold, path_system):
    return official.evaluate(official.load_conllu_file(str(path_gold)),
                             official.load_conllu_file(str(path_system)))
"""),
code(r"""# Resolve one exact HF commit and download it into an isolated cache.
from huggingface_hub import snapshot_download
snapshot = Path(snapshot_download(HF_REPO, revision=HF_ELECTRA_REVISION, cache_dir=HF_HOME/'hub'))
HF_COMMIT = snapshot.name
assert len(HF_COMMIT) == 40
print('frozen HF revision:', HF_COMMIT)
"""),
code(r"""# Download the exact Stanza 1.14 Mandarin processors and verify registry/artifact hashes.
import stanza
from stanza.resources.common import download_resources_json, load_resources_json
from stanza.pipeline.core import DownloadMethod
assert stanza.__version__ == '1.14.0', f'Expected Stanza 1.14.0, found {stanza.__version__}. Restart runtime and rerun.'
download_resources_json(model_dir=str(MODELS))
resources_path=MODELS/'resources.json'
assert sha256(resources_path) == '4e41c1df152146fa26ed0c006a08feea7a60bb3414bb6d57dbda24ad2e3cb99c'
packages={'tokenize':'gsdsimp','pos':'gsdsimp_electra-large','lemma':'gsdsimp_charlm','depparse':'gsdsimp_electra-large'}
stanza.download('zh-hans', model_dir=str(MODELS), package=None, processors=packages, verbose=True)
resources=load_resources_json(model_dir=str(MODELS))
expected_md5={'tokenize':'48f993223d568afedc2893f7cd76719c','pos':'73859e5ec15bedc545d6deafd6ddba94','lemma':'b49edd41abb063a87b125ec53aa5b96c','depparse':'c7ea98d93459b22720337a9dcba1a848'}
artifacts={}
for proc,pkg in packages.items():
    p=MODELS/'zh-hans'/proc/f'{pkg}.pt'
    registry=resources['zh-hans'][proc][pkg]['md5']
    assert registry == expected_md5[proc] and p.exists()
    artifacts[proc]={'path':str(p),'registry_md5':registry,'sha256':sha256(p)}
print(json.dumps(artifacts, indent=2))
# Processor artifacts and the exact Transformer snapshot are now present.  Forbid
# later Hub lookups so every POS/parser load reuses this one cached revision.
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
"""),
code(r"""# Run frozen POS/lemma exactly once and build pretagged train/dev/test caches.
# Gold sentence boundaries and integer-ID FORM tokenization are fed to the pipeline.
# Only LEMMA/UPOS/XPOS/FEATS are replaced; gold HEAD/DEPREL remain training targets.
tagger = stanza.Pipeline(lang='zh-hans', dir=str(MODELS),
    processors={k:v for k,v in packages.items() if k != 'depparse'},
    tokenize_pretokenized=True, use_gpu=True,
    download_method=DownloadMethod.REUSE_RESOURCES, verbose=False)
for proc in ('pos','lemma'):
    model=tagger.processors[proc]._trainer.model
    model.eval()
    for p in model.parameters(): p.requires_grad=False
    assert not any(p.requires_grad for p in model.parameters()), f'{proc} unexpectedly trainable'

def split_blocks(text): return text.strip().split('\n\n')
def integer_rows(block):
    return [line.split('\t') for line in block.splitlines()
            if line and not line.startswith('#') and line.split('\t',1)[0].isdigit()]

def make_pretagged(src, dst, chunk=32):
    bs=split_blocks(src.read_text(encoding='utf-8')); out=[]
    for start in range(0,len(bs),chunk):
        sub=bs[start:start+chunk]
        forms=[[r[1] for r in integer_rows(b)] for b in sub]
        doc=tagger(forms)
        assert len(doc.sentences)==len(sub)
        for block,sent,gold_forms in zip(sub,doc.sentences,forms):
            assert [w.text for w in sent.words] == gold_forms
            pred=iter(sent.words); lines=[]
            for line in block.splitlines():
                if line and not line.startswith('#') and line.split('\t',1)[0].isdigit():
                    c=line.split('\t'); w=next(pred)
                    c[2]=w.lemma or '_'; c[3]=w.upos or '_'; c[4]=w.xpos or '_'; c[5]=w.feats or '_'
                    line='\t'.join(c)
                lines.append(line)
            try: next(pred); raise AssertionError('extra predicted word')
            except StopIteration: pass
            out.append('\n'.join(lines))
    dst.write_text('\n\n'.join(out)+'\n',encoding='utf-8')
    assert len(split_blocks(dst.read_text()))==len(bs)

cache_hashes={}
for split in ('train','dev','test'):
    dst=DATA/f'{split}.predposlemma.conllu'
    make_pretagged(DATA/f'{split}.conllu',dst)
    cache_hashes[split]=sha256(dst)
    print(split,'frozen cache',cache_hashes[split])
del tagger; gc.collect(); torch.cuda.empty_cache()
"""),
code(r"""# Build a compatible r=8 checkpoint: extend only DEPREL output units, preserve every old unit.
# Also seed an initial LoRA adapter so Stanza 1.14 can load PEFT continuation safely.
from stanza.models.pos.vocab import MultiVocab
from stanza.models.common.vocab import VOCAB_PREFIX_SIZE
from stanza.models.common.bert_embedding import load_bert
from stanza.models.common.peft_config import build_peft_wrapper
from peft import get_peft_model_state_dict

base_path=Path(artifacts['depparse']['path'])
base_ckpt=torch.load(base_path,map_location='cpu',weights_only=True)
assert base_ckpt.get('model_type','graph') == 'graph'
old_vocab=MultiVocab.load_state_dict(base_ckpt['vocab'])
old_units=list(old_vocab['deprel']._id2unit)
train_labels=sorted({r[7] for b in split_blocks((DATA/'train.conllu').read_text()) for r in integer_rows(b)})
new_labels=[x for x in train_labels if x not in old_vocab['deprel']]

expanded=copy.deepcopy(base_ckpt)
dep_state=expanded['vocab']['deprel']
dep_state['_id2unit']=old_units+new_labels
dep_state['_unit2id']={u:i for i,u in enumerate(dep_state['_id2unit'])}
old_n=len(old_units)-VOCAB_PREFIX_SIZE; new_n=old_n+len(new_labels)
expanded_names=[]
if new_labels:
    for name,t in list(expanded['model'].items()):
        if name == 'deprel.scorer.W_bilin.weight':
            z=t.new_zeros(t.shape[0],t.shape[1],new_n); z[:,:,:old_n]=t; expanded['model'][name]=z; expanded_names.append(name)
        elif name == 'deprel.scorer.W_bilin.bias':
            z=t.new_zeros(new_n); z[:old_n]=t; expanded['model'][name]=z; expanded_names.append(name)
        elif name.startswith('deprel_linear.') and t.ndim in (1,2) and t.shape[0]==old_n:
            z=t.new_zeros((new_n,)+tuple(t.shape[1:])); z[:old_n]=t; expanded['model'][name]=z; expanded_names.append(name)
    assert expanded_names, 'New labels exist but relation output tensor was not found'

args=copy.deepcopy(expanded['config'])
args.update(use_peft=True, bert_finetune=True, lora_rank=CFG['rank'], lora_alpha=CFG['alpha'],
            lora_dropout=CFG['dropout'], lora_target_modules=CFG['targets'], lora_modules_to_save=[],
            optim='adamw', second_optim=None, lr=CFG['parser_lr'],
            bert_learning_rate=CFG['lora_lr']/CFG['parser_lr'], bert_start_finetuning=0,
            bert_warmup_steps=0, weight_decay=0.0, bert_weight_decay=0.0,
            max_grad_norm=CFG['max_grad_norm'], batch_size=CFG['batch_size'], seed=SEED,
            enable_gradient_checkpointing=True, augment_nopunct=0.0)
assert args['bert_model'] == HF_REPO

# Distributed checkpoints retain absolute CharLM paths from Stanford's build
# machine.  Resolve their declared resource dependencies into this Colab's
# pinned MODEL_DIR before saving/loading the continuation checkpoint.
def one_dependency_path(model_type):
    deps=resources['zh-hans']['depparse']['gsdsimp_electra-large'].get('dependencies',[])
    names=[d['package'] for d in deps if d.get('model')==model_type]
    paths=[MODELS/'zh-hans'/model_type/f'{name}.pt' for name in names]
    paths=[p for p in paths if p.exists()]
    if len(paths)!=1:
        paths=list((MODELS/'zh-hans'/model_type).glob('*.pt'))
    assert len(paths)==1, f'Cannot unambiguously resolve {model_type}: {paths}'
    return paths[0]

if args.get('charlm'):
    args['charlm_forward_file']=str(one_dependency_path('forward_charlm'))
    args['charlm_backward_file']=str(one_dependency_path('backward_charlm'))
    print('redirected CharLM:',args['charlm_forward_file'],args['charlm_backward_file'])
bert,_=load_bert(HF_REPO, enable_gradient_checkpointing=True)
pefted=build_peft_wrapper(bert,args,logging.getLogger('stanza'),adapter_name='depparse')
expanded['bert_lora']=get_peft_model_state_dict(pefted,adapter_name='depparse')
expanded['config']=args; expanded['global_step']=0; expanded['last_best_step']=0; expanded['dev_score_history']=[]
compat_path=OUT/'mandarin_electra_expanded_deprel_lora_r8_init.pt'
torch.save(expanded,compat_path,_use_new_zipfile_serialization=False)
del bert,pefted,expanded; gc.collect(); torch.cuda.empty_cache()
print('old relation labels:',old_n,'new labels:',new_labels,'expanded tensors:',expanded_names)
"""),
code(r"""# Load the compatible checkpoint, assert preservation/freeze policy, then train.
from stanza.models.depparse.trainer import GraphTrainer
from stanza.models.depparse.data import DataLoader, InfiniteBatch
from stanza.models.depparse.utils import predict_dataset
from stanza.models.common.pretrain import Pretrain
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import HEAD, DEPREL

cfg0=base_ckpt['config']
pretrain_obj=None
if cfg0.get('pretrain'):
    dependencies=resources['zh-hans']['depparse']['gsdsimp_electra-large'].get('dependencies',[])
    pretrain_names=[d['package'] for d in dependencies if d.get('model')=='pretrain']
    candidates=[MODELS/'zh-hans'/'pretrain'/f'{name}.pt' for name in pretrain_names]
    candidates=[p for p in candidates if p.exists()]
    if len(candidates)!=1:  # defensive fallback, still refuses an ambiguous choice
        candidates=list((MODELS/'zh-hans'/'pretrain').glob('*.pt'))
    assert len(candidates)==1, f'Cannot unambiguously select parser pretrain: {candidates}'
    pretrain_obj=Pretrain(filename=str(candidates[0]))

# Defensive redirect here as well: this makes the cell safe even if compat_path
# was created by an older notebook which still contains Stanford build paths.
load_args=dict(args)
# Trainer.load() restores this serialized string to a SubtreeCombination enum.
# Do not override that restored value with the raw string copied from the base
# checkpoint, otherwise Trainer.save() later fails on `.name`.
load_args.pop('transition_subtree_combination',None)
forward_candidates=list((MODELS/'zh-hans'/'forward_charlm').glob('*.pt'))
backward_candidates=list((MODELS/'zh-hans'/'backward_charlm').glob('*.pt'))
assert len(forward_candidates)==1, f'Expected one forward CharLM: {forward_candidates}'
assert len(backward_candidates)==1, f'Expected one backward CharLM: {backward_candidates}'
load_args['charlm_forward_file']=str(forward_candidates[0].resolve())
load_args['charlm_backward_file']=str(backward_candidates[0].resolve())
assert Path(load_args['charlm_forward_file']).is_file()
assert Path(load_args['charlm_backward_file']).is_file()
print('loading local CharLM:',load_args['charlm_forward_file'],load_args['charlm_backward_file'])
trainer=GraphTrainer.load(str(compat_path),pretrain=pretrain_obj,args=load_args,device=DEVICE,reset_history=True)
from stanza.models.depparse.transition.model import SubtreeCombination
assert isinstance(trainer.args['transition_subtree_combination'],SubtreeCombination)

# Exact initialization audit for all parser tensors; old DEPREL slices must equal base checkpoint.
loaded=trainer.model.get_params(skip_modules=True)
for name,old in base_ckpt['model'].items():
    got=loaded[name].detach().cpu()
    if name == 'deprel.scorer.W_bilin.weight': assert torch.equal(got[:,:,:old_n],old)
    elif name == 'deprel.scorer.W_bilin.bias' or (name.startswith('deprel_linear.') and old.ndim in (1,2) and old.shape[0]==old_n): assert torch.equal(got[:old_n],old)
    else: assert got.shape==old.shape and torch.equal(got,old), name

base_trainable=[]; lora_trainable=[]; frozen_transformer=[]
for n,p in trainer.model.named_parameters():
    if n.startswith('bert_model.'):
        (lora_trainable if p.requires_grad else frozen_transformer).append(n)
    elif p.requires_grad: base_trainable.append(n)
assert lora_trainable and frozen_transformer and base_trainable
assert all(('lora_' in n or 'modules_to_save' in n) for n in lora_trainable), lora_trainable[:10]
print('trainable parser tensors:',len(base_trainable),'trainable LoRA tensors:',len(lora_trainable),'frozen Transformer tensors:',len(frozen_transformer))

train_doc=CoNLL.conll2doc(input_file=str(DATA/'train.predposlemma.conllu'))
dev_doc=CoNLL.conll2doc(input_file=str(DATA/'dev.predposlemma.conllu'))
train_loader=DataLoader(train_doc,CFG['batch_size'],trainer.args,pretrain_obj,vocab=trainer.vocab,evaluation=False,bert_tokenizer=trainer.model.bert_tokenizer)
dev_loader=DataLoader(dev_doc,CFG['batch_size'],trainer.args,pretrain_obj,vocab=trainer.vocab,evaluation=True,sort_during_eval=True,bert_tokenizer=trainer.model.bert_tokenizer)
infinite=InfiniteBatch(train_loader)

def predict_to_file(tr,loader,path):
    preds=predict_dataset(tr,loader)
    loader.doc.set([HEAD,DEPREL],[y for x in preds for y in x])
    path.write_text(f'{loader.doc:C}\n\n',encoding='utf-8')
    return path

best_path=OUT/'best_dev_electra_yue_lora_r8.pt'
history=[]
dev_pred=OUT/'dev.step0000.conllu'
predict_to_file(trainer,dev_loader,dev_pred)
best_score=conll18(DATA/'dev.conllu',dev_pred)['LAS'].f1
best_step=0; trainer.save(str(best_path)); print('step 0 dev LAS',best_score*100)

for step in range(1,CFG['max_steps']+1):
    loss,_=trainer.update(infinite.next_batch(),eval=False)
    trainer.global_step=step
    if step % 20 == 0: print(f'step {step} loss {loss:.5f}')
    if step % CFG['eval_interval'] == 0:
        pred=OUT/f'dev.step{step:04d}.conllu'; predict_to_file(trainer,dev_loader,pred)
        score=conll18(DATA/'dev.conllu',pred)['LAS'].f1
        history.append({'step':step,'dev_conll18_las':score,'loss':loss,'prediction_sha256':sha256(pred)})
        print(f'== step {step}: dev CoNLL18 LAS {score*100:.2f}% ==')
        if score > best_score:  # deterministic tie policy: earliest checkpoint wins
            best_score=score; best_step=step; trainer.save(str(best_path))
        if step-best_step >= CFG['patience_steps']:
            print('early stop'); break

(OUT/'dev_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
print('BEST DEV:',best_step,best_score*100,sha256(best_path))
"""),
md("""## Dev-only error diagnostics (optional, does not alter the checkpoint)

Run this after training. It compares the frozen predicted-POS/lemma condition with a diagnostic gold POS/morph condition on **dev only**. In the gold condition, UPOS/XPOS/FEATS come from gold while lemma remains the same frozen prediction, so lemma is not another changed variable. Because the parser was trained with predicted tags, this is a diagnostic intervention, not an alternative main result.

Exact duplicate annotation disagreements are reported directly. Near-similar pairs are only manual-review candidates; different analyses are not automatically called annotation errors and no annotation is edited."""),
code(r"""# DEV ONLY: error attribution and gold-POS/morph diagnostic.
import pandas as pd
import unicodedata
from difflib import SequenceMatcher

DIAG=WORK/'dev_diagnostics'; DIAG.mkdir(exist_ok=True)
diag_trainer=GraphTrainer.load(str(best_path),pretrain=pretrain_obj,device=DEVICE)

def make_goldpos_predlemma(pred_cache,gold_path,dst):
    pb=split_blocks(Path(pred_cache).read_text(encoding='utf-8'))
    gb=split_blocks(Path(gold_path).read_text(encoding='utf-8'))
    assert len(pb)==len(gb)
    out=[]
    for pblock,gblock in zip(pb,gb):
        gold_iter=iter(integer_rows(gblock)); lines=[]
        for line in pblock.splitlines():
            if line and not line.startswith('#') and line.split('\t',1)[0].isdigit():
                c=line.split('\t'); g=next(gold_iter)
                assert c[0:2]==g[0:2]
                c[3:6]=g[3:6]       # gold UPOS/XPOS/FEATS
                # c[2] remains the frozen predicted lemma
                line='\t'.join(c)
            lines.append(line)
        try: next(gold_iter); raise AssertionError('missing gold word')
        except StopIteration: pass
        out.append('\n'.join(lines))
    dst.write_text('\n\n'.join(out)+'\n',encoding='utf-8')

dev_gold=DATA/'dev.conllu'
dev_predcache=DATA/'dev.predposlemma.conllu'
dev_goldpos=DIAG/'dev.goldpos_predlemma.conllu'
make_goldpos_predlemma(dev_predcache,dev_gold,dev_goldpos)

def run_dev_condition(input_path,output_path):
    doc=CoNLL.conll2doc(input_file=str(input_path))
    loader=DataLoader(doc,CFG['batch_size'],diag_trainer.args,pretrain_obj,
                      vocab=diag_trainer.vocab,evaluation=True,sort_during_eval=True,
                      bert_tokenizer=diag_trainer.model.bert_tokenizer)
    predict_to_file(diag_trainer,loader,output_path)
    return conll18(dev_gold,output_path)

pred_out=DIAG/'dev.predposlemma.best.pred.conllu'
goldpos_out=DIAG/'dev.goldpos_predlemma.best.pred.conllu'
pred_scores=run_dev_condition(dev_predcache,pred_out)
goldpos_scores=run_dev_condition(dev_goldpos,goldpos_out)
assert abs(pred_scores['LAS'].f1-best_score)<1e-12, (pred_scores['LAS'].f1,best_score)

def sentence_meta(block):
    meta={}
    for line in block.splitlines():
        if line.startswith('# ') and ' = ' in line:
            k,v=line[2:].split(' = ',1); meta[k]=v
    return meta

gold_blocks=split_blocks(dev_gold.read_text(encoding='utf-8'))
input_blocks=split_blocks(dev_predcache.read_text(encoding='utf-8'))
pred_blocks=split_blocks(pred_out.read_text(encoding='utf-8'))
oracle_blocks=split_blocks(goldpos_out.read_text(encoding='utf-8'))
records=[]; sent_records=[]
for si,(gb,ib,pb,ob) in enumerate(zip(gold_blocks,input_blocks,pred_blocks,oracle_blocks),1):
    gr,ir,pr,orr=map(integer_rows,(gb,ib,pb,ob)); meta=sentence_meta(gb)
    assert len(gr)==len(ir)==len(pr)==len(orr)
    correct=0
    for g,inp,p,o in zip(gr,ir,pr,orr):
        assert g[0:2]==inp[0:2]==p[0:2]==o[0:2]
        gh,ph,oh=int(g[6]),int(p[6]),int(o[6]); gd,pd,od=g[7],p[7],o[7]
        base=lambda x:x.split(':',1)[0]
        dep_len=0 if gh==0 else abs(int(g[0])-gh)
        if gh==0: bucket='ROOT'
        elif dep_len==1: bucket='1'
        elif dep_len==2: bucket='2'
        elif dep_len<=5: bucket='3-5'
        elif dep_len<=10: bucket='6-10'
        else: bucket='11+'
        las=(gh==ph and base(gd)==base(pd)); correct+=las
        records.append(dict(sentence_index=si,sent_id=meta.get('sent_id'),text=meta.get('text'),
            token_id=int(g[0]),form=g[1],gold_upos=g[3],predicted_upos=inp[3],
            upos_correct=g[3]==inp[3],full_morph_correct=g[3:6]==inp[3:6],
            gold_head=gh,pred_head=ph,goldpos_pred_head=oh,gold_deprel=gd,
            pred_deprel=pd,goldpos_pred_deprel=od,head_distance=dep_len,distance_bucket=bucket,
            uas=gh==ph,conll18_las=las,strict_las=gh==ph and gd==pd,
            goldpos_uas=gh==oh,goldpos_conll18_las=gh==oh and base(gd)==base(od),
            goldpos_strict_las=gh==oh and gd==od))
    sent_records.append(dict(sentence_index=si,sent_id=meta.get('sent_id'),text=meta.get('text'),
                             tokens=len(gr),conll18_las=correct/len(gr)))

# The loop above uses `pd` as a short-lived predicted-DEPREL variable.
# Restore the pandas module alias before constructing tables.
import pandas as pd
df=pd.DataFrame(records); sdf=pd.DataFrame(sent_records)
df.to_csv(DIAG/'token_diagnostics.csv',index=False)
sdf.sort_values(['conll18_las','tokens']).to_csv(DIAG/'worst_sentences.csv',index=False)

def grouped_stats(column):
    return (df.groupby(column,dropna=False)
      .agg(tokens=('token_id','size'),upos_accuracy=('upos_correct','mean'),
           UAS=('uas','mean'),CoNLL18_LAS=('conll18_las','mean'),strict_LAS=('strict_las','mean'),
           goldPOS_CoNLL18_LAS=('goldpos_conll18_las','mean')).reset_index())

distance_stats=grouped_stats('distance_bucket')
distance_stats['distance_bucket']=pd.Categorical(distance_stats['distance_bucket'],['ROOT','1','2','3-5','6-10','11+'],ordered=True)
distance_stats.sort_values('distance_bucket').to_csv(DIAG/'distance_stats.csv',index=False)
relation_stats=grouped_stats('gold_deprel').sort_values(['CoNLL18_LAS','tokens'],ascending=[True,False])
relation_stats.to_csv(DIAG/'relation_stats.csv',index=False)
upos_stats=grouped_stats('upos_correct'); upos_stats.to_csv(DIAG/'pos_error_association.csv',index=False)

conf=(df[df['uas'] & ~df['strict_las']]
      .groupby(['gold_deprel','pred_deprel']).size().reset_index(name='count')
      .sort_values('count',ascending=False))
conf.to_csv(DIAG/'relation_confusions_when_head_correct.csv',index=False)

# Exact duplicates and annotation disagreements inside dev.
groups={}
for i,b in enumerate(gold_blocks):
    r=integer_rows(b); forms=tuple(unicodedata.normalize('NFC',x[1]) for x in r)
    joined=unicodedata.normalize('NFC',''.join(x[1] for x in r))
    for key in (('tokens',forms),('joined',joined)): groups.setdefault(key,set()).add(i)
dup=[]
for key,idxs in groups.items():
    if len(idxs)<2: continue
    analyses={tuple((x[6],x[7]) for x in integer_rows(gold_blocks[i])) for i in idxs}
    dup.append({'key_type':key[0],'indices_1_based':[i+1 for i in sorted(idxs)],
                'sent_ids':[sentence_meta(gold_blocks[i]).get('sent_id') for i in sorted(idxs)],
                'annotation_disagreement':len(analyses)>1})
(DIAG/'exact_duplicate_annotation_audit.json').write_text(json.dumps(dup,ensure_ascii=False,indent=2),encoding='utf-8')

# Near-similar forms are candidates for manual review only.
near=[]
texts=[''.join(x[1] for x in integer_rows(b)) for b in gold_blocks]
for i in range(len(texts)):
    for j in range(i+1,len(texts)):
        ratio=SequenceMatcher(None,texts[i],texts[j],autojunk=False).ratio()
        if ratio>=0.85 and texts[i]!=texts[j]:
            near.append({'similarity':ratio,'index_a':i+1,'index_b':j+1,
                         'sent_id_a':sentence_meta(gold_blocks[i]).get('sent_id'),
                         'sent_id_b':sentence_meta(gold_blocks[j]).get('sent_id'),
                         'text_a':sentence_meta(gold_blocks[i]).get('text'),
                         'text_b':sentence_meta(gold_blocks[j]).get('text')})
pd.DataFrame(near).sort_values('similarity',ascending=False).to_csv(DIAG/'near_similar_manual_review_candidates.csv',index=False) if near else (DIAG/'near_similar_manual_review_candidates.csv').write_text('similarity,index_a,index_b,sent_id_a,sent_id_b,text_a,text_b\n')

def pct(x): return 100*float(x)
pos_wrong=df[~df.upos_correct]; pos_right=df[df.upos_correct]
summary={
 'split':'dev','checkpoint_selected_without_test':True,'best_step':best_step,
 'predicted_condition':{'UAS_percent':pct(pred_scores['UAS'].f1),'CoNLL18_LAS_percent':pct(pred_scores['LAS'].f1),
                        'strict_LAS_percent':pct(df.strict_las.mean())},
 'gold_pos_morph_predicted_lemma_condition':{'UAS_percent':pct(goldpos_scores['UAS'].f1),
                        'CoNLL18_LAS_percent':pct(goldpos_scores['LAS'].f1),
                        'strict_LAS_percent':pct(df.goldpos_strict_las.mean())},
 'gold_pos_delta_points':pct(goldpos_scores['LAS'].f1-pred_scores['LAS'].f1),
 'predicted_UPOS_accuracy_percent':pct(df.upos_correct.mean()),
 'dependency_LAS_given_UPOS_correct_percent':pct(pos_right.conll18_las.mean()),
 'dependency_LAS_given_UPOS_wrong_percent':pct(pos_wrong.conll18_las.mean()),
 'tokens_UPOS_correct':len(pos_right),'tokens_UPOS_wrong':len(pos_wrong),
 'exact_duplicate_groups':len(dup),
 'exact_duplicate_groups_with_annotation_disagreement':sum(x['annotation_disagreement'] for x in dup),
 'near_similar_pairs_for_manual_review':len(near),
 'interpretation_warning':'Gold POS/morph is a diagnostic distribution shift because training used predicted tags. Near-similar pairs are not automatically annotation errors.'}
(DIAG/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
display(upos_stats, distance_stats, relation_stats.head(20), conf.head(20), sdf.sort_values('conll18_las').head(15))
diag_zip=shutil.make_archive('/content/yue_dev_diagnostics','zip',root_dir=DIAG)
from google.colab import files
files.download(diag_zip)
"""),
md("""## Final test — run once only

Do not use the result below to change LoRA settings or select another checkpoint. If any training configuration is changed, a clean confirmatory test requires a new untouched evaluation set; rerunning this same test does not restore independence."""),
code(r"""# FINAL TEST: load the already selected dev-best checkpoint and evaluate this test once.
sentinel=OUT/'.final_test_completed'
assert not sentinel.exists(), 'Final test already completed in this run directory; refusing to evaluate it again.'
final_trainer=GraphTrainer.load(str(best_path),pretrain=pretrain_obj,device=DEVICE)
test_doc=CoNLL.conll2doc(input_file=str(DATA/'test.predposlemma.conllu'))
test_loader=DataLoader(test_doc,CFG['batch_size'],final_trainer.args,pretrain_obj,vocab=final_trainer.vocab,evaluation=True,sort_during_eval=True,bert_tokenizer=final_trainer.model.bert_tokenizer)
test_pred=OUT/'test.best_dev.pred.conllu'; predict_to_file(final_trainer,test_loader,test_pred)
scores=conll18(DATA/'test.conllu',test_pred)

def strict(gold_path,pred_path,no_punct=False):
    g=[r for b in split_blocks(Path(gold_path).read_text()) for r in integer_rows(b)]
    p=[r for b in split_blocks(Path(pred_path).read_text()) for r in integer_rows(b)]
    assert len(g)==len(p) and all(a[0:2]==b[0:2] for a,b in zip(g,p))
    keep=[(a,b) for a,b in zip(g,p) if not(no_punct and a[3]=='PUNCT')]
    return sum(a[6]==b[6] and a[7]==b[7] for a,b in keep)/len(keep)

las=scores['LAS'].f1*100
result={'historical_test_baseline_conll18_las_percent':41.32,
        'adapted_test_conll18_las_percent':las,
        'delta_vs_historical_41_32_points':las-41.32,
        'strict_full_deprel_percent':strict(DATA/'test.conllu',test_pred)*100,
        'strict_full_deprel_no_punct_percent':strict(DATA/'test.conllu',test_pred,True)*100,
        'best_dev_step':best_step,'best_dev_conll18_las_percent':best_score*100,
        'test_was_previously_used_for_model_family_selection':True,
        'hf_revision':HF_COMMIT,'config':CFG,'processor_artifacts':artifacts,
        'frozen_cache_sha256':cache_hashes,'checkpoint_sha256':sha256(best_path),
        'test_prediction_sha256':sha256(test_pred),
        'historical_baseline_hf_revision_was_supplied':HF_ELECTRA_REVISION is not None,
        'comparison_caveat':('Historical revision was explicitly supplied; verify it matches hf_revision.' if HF_ELECTRA_REVISION is not None else '41.32 run HF commit was not supplied here; exact same-version equivalence requires matching its recorded commit to hf_revision.')}
(OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
sentinel.write_text(json.dumps({'completed':True,'prediction_sha256':sha256(test_pred)}),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
bundle=WORK/'download_bundle'; bundle.mkdir(exist_ok=True)
for p in (best_path, OUT/'result.json', OUT/'dev_history.json', test_pred):
    shutil.copy2(p,bundle/p.name)
archive=shutil.make_archive('/content/yue_lora_electra_r8_results','zip',root_dir=bundle)
from google.colab import files
files.download(archive)
""")]

notebook = {
    "cells": cells,
    "metadata": {"accelerator": "GPU", "colab": {"name": "finetune_yue_stanza_electra_lora.ipynb", "provenance": []},
                 "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3"}},
    "nbformat": 4, "nbformat_minor": 5,
}
dest = ROOT / "notebooks/finetune_yue_stanza_electra_lora.ipynb"
dest.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(dest)

# Convenience script for a still-running Colab session which has already
# completed training.  It intentionally relies on the notebook globals and
# does not retrain or touch test.
diag_cell = next(c for c in cells if c["cell_type"] == "code" and
                 "# DEV ONLY: error attribution" in "".join(c["source"]))
diag_dest = ROOT / "notebooks/run_yue_dev_diagnostics.py"
diag_dest.write_text("".join(diag_cell["source"]), encoding="utf-8")
print(diag_dest)
