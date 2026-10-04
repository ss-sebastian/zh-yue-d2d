#!/usr/bin/env python3
"""Build a self-contained, DEV-only LoRA rank sweep Colab."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "finetune_yue_stanza_electra_lora.ipynb"
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
setup = cell_source(2)
setup = setup.replace("/content/yue_lora_electra_r8", "/content/yue_lora_rank_sweep")
setup = setup.replace(
    "HF_ELECTRA_REVISION = None",
    "HF_ELECTRA_REVISION = 'd017e219578df8e4885484edbc8969dbdea9cbe0'",
)
old_cfg = """CFG = dict(rank=8, alpha=16, dropout=0.10,
           targets=['query', 'value', 'output.dense', 'intermediate.dense'],
           parser_lr=1e-3, lora_lr=2e-5, batch_size=900,
           max_steps=4000, eval_interval=100, patience_steps=600,
           max_grad_norm=1.0)"""
new_cfg = """RANKS = [4, 8, 16, 32]
ALPHA_TO_RANK = 2                 # alpha/r stays fixed at 2
CFG = dict(dropout=0.10,
           targets=['query', 'value', 'output.dense', 'intermediate.dense'],
           parser_lr=1e-3, lora_lr=2e-5, batch_size=900,
           max_steps=4000, eval_interval=100, patience_steps=600,
           max_grad_norm=1.0)
assert len(RANKS)==len(set(RANKS)) and all(isinstance(r,int) and r>0 for r in RANKS)"""
assert old_cfg in setup
setup = setup.replace(old_cfg, new_cfg)

download_data = cell_source(3)
download_data = download_data.replace(
    "for split, positions in SPLIT_POSITIONS.items():",
    "for split in ('train','dev'):\n    positions = SPLIT_POSITIONS[split]",
)
resolve_hf = cell_source(4)
download_stanza = cell_source(5)
tag_data = cell_source(6)
tag_data = tag_data.replace(
    "# Run frozen POS/lemma exactly once and build pretagged train/dev/test caches.",
    "# Run frozen POS/lemma exactly once and build pretagged train/dev caches.",
)
tag_data = tag_data.replace(
    "for split in ('train','dev','test'):", "for split in ('train','dev'):"
)
tag_data += r'''
EXPECTED_CACHE_SHA256={
 'train':'806b3d42c4f6838d6ba9c6565f29062a0d7b6e136e12d158e4449c3e7e970e89',
 'dev':'e88aa03ff7453469deda68ed30c29dd16769e9839d2010b36abf91faace3bd97'}
assert cache_hashes==EXPECTED_CACHE_SHA256,(cache_hashes,EXPECTED_CACHE_SHA256)
print('Frozen POS/lemma caches match the original r=8 run: OK')
'''

prepare_base = r'''# Prepare the common Mandarin initialization once.
# Only new Cantonese DEPREL output units are initialized; all original parser
# tensors and old relation slices are kept exactly unchanged.
from stanza.models.pos.vocab import MultiVocab
from stanza.models.common.vocab import VOCAB_PREFIX_SIZE

base_path=Path(artifacts['depparse']['path'])
base_ckpt=torch.load(base_path,map_location='cpu',weights_only=True)
assert base_ckpt.get('model_type','graph')=='graph'
old_vocab=MultiVocab.load_state_dict(base_ckpt['vocab'])
old_units=list(old_vocab['deprel']._id2unit)
train_labels=sorted({row[7] for block in split_blocks((DATA/'train.conllu').read_text())
                     for row in integer_rows(block)})
new_labels=[label for label in train_labels if label not in old_vocab['deprel']]

expanded=copy.deepcopy(base_ckpt)
dep_state=expanded['vocab']['deprel']
dep_state['_id2unit']=old_units+new_labels
dep_state['_unit2id']={unit:i for i,unit in enumerate(dep_state['_id2unit'])}
old_n=len(old_units)-VOCAB_PREFIX_SIZE
new_n=old_n+len(new_labels)
expanded_names=[]
if new_labels:
    for name,tensor in list(expanded['model'].items()):
        if name=='deprel.scorer.W_bilin.weight':
            value=tensor.new_zeros(tensor.shape[0],tensor.shape[1],new_n)
            value[:,:,:old_n]=tensor; expanded['model'][name]=value; expanded_names.append(name)
        elif name=='deprel.scorer.W_bilin.bias':
            value=tensor.new_zeros(new_n)
            value[:old_n]=tensor; expanded['model'][name]=value; expanded_names.append(name)
        elif name.startswith('deprel_linear.') and tensor.ndim in (1,2) and tensor.shape[0]==old_n:
            value=tensor.new_zeros((new_n,)+tuple(tensor.shape[1:]))
            value[:old_n]=tensor; expanded['model'][name]=value; expanded_names.append(name)
    assert expanded_names

def one_dependency_path(model_type):
    dependencies=resources['zh-hans']['depparse']['gsdsimp_electra-large'].get('dependencies',[])
    names=[item['package'] for item in dependencies if item.get('model')==model_type]
    paths=[MODELS/'zh-hans'/model_type/f'{name}.pt' for name in names]
    paths=[path for path in paths if path.exists()]
    if len(paths)!=1: paths=list((MODELS/'zh-hans'/model_type).glob('*.pt'))
    assert len(paths)==1,f'Cannot resolve {model_type}: {paths}'
    return paths[0]

common_args=copy.deepcopy(expanded['config'])
if common_args.get('charlm'):
    common_args['charlm_forward_file']=str(one_dependency_path('forward_charlm').resolve())
    common_args['charlm_backward_file']=str(one_dependency_path('backward_charlm').resolve())
expanded['config']=common_args
expanded['global_step']=0; expanded['last_best_step']=0; expanded['dev_score_history']=[]
expanded_base_path=OUT/'mandarin_electra_expanded_deprel_base.pt'
torch.save(expanded,expanded_base_path,_use_new_zipfile_serialization=False)
del expanded
gc.collect(); torch.cuda.empty_cache()
print('common base prepared:',sha256(expanded_base_path))
print('new DEPREL labels:',new_labels)
'''

sweep = r'''# Train each rank independently from the same Mandarin initialization.
from peft import get_peft_model_state_dict
from stanza.models.common.bert_embedding import load_bert
from stanza.models.common.peft_config import build_peft_wrapper
from stanza.models.common.pretrain import Pretrain
from stanza.models.common.doc import HEAD,DEPREL
from stanza.models.depparse.data import DataLoader,InfiniteBatch
from stanza.models.depparse.trainer import GraphTrainer
from stanza.models.depparse.transition.model import SubtreeCombination
from stanza.models.depparse.utils import predict_dataset
from stanza.utils.conll import CoNLL
import pandas as pd
import matplotlib.pyplot as plt

pretrain_obj=Pretrain(filename=str(one_dependency_path('pretrain'))) if base_ckpt['config'].get('pretrain') else None

def reset_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def predict_to_file(trainer,loader,path):
    trainer.model.eval()
    with torch.inference_mode(): predictions=predict_dataset(trainer,loader)
    loader.doc.set([HEAD,DEPREL],[item for sentence in predictions for item in sentence])
    path.write_text(f'{loader.doc:C}\n\n',encoding='utf-8')
    return path

rank_results=[]
all_histories={}
best_paths={}
for rank in RANKS:
    print(f'\n========== START RANK {rank} (alpha={ALPHA_TO_RANK*rank}) ==========')
    reset_seed()
    rank_dir=OUT/f'rank_{rank}'; rank_dir.mkdir(exist_ok=True)
    initial=torch.load(expanded_base_path,map_location='cpu',weights_only=True)
    args=copy.deepcopy(initial['config'])
    args.update(use_peft=True,bert_finetune=True,lora_rank=rank,lora_alpha=ALPHA_TO_RANK*rank,
        lora_dropout=CFG['dropout'],lora_target_modules=CFG['targets'],lora_modules_to_save=[],
        optim='adamw',second_optim=None,lr=CFG['parser_lr'],
        bert_learning_rate=CFG['lora_lr']/CFG['parser_lr'],bert_start_finetuning=0,
        bert_warmup_steps=0,weight_decay=0.0,bert_weight_decay=0.0,
        max_grad_norm=CFG['max_grad_norm'],batch_size=CFG['batch_size'],seed=SEED,
        enable_gradient_checkpointing=True,augment_nopunct=0.0)

    # Seed a rank-specific adapter for Stanza 1.14 PEFT continuation loading.
    bert,_=load_bert(HF_REPO,enable_gradient_checkpointing=True)
    pefted=build_peft_wrapper(bert,args,logging.getLogger('stanza'),adapter_name='depparse')
    initial['bert_lora']=get_peft_model_state_dict(pefted,adapter_name='depparse')
    initial['config']=args
    compat_path=rank_dir/f'mandarin_electra_lora_r{rank}_init.pt'
    torch.save(initial,compat_path,_use_new_zipfile_serialization=False)
    del bert,pefted,initial
    gc.collect(); torch.cuda.empty_cache()

    load_args=dict(args)
    load_args.pop('transition_subtree_combination',None)
    load_args['charlm_forward_file']=str(one_dependency_path('forward_charlm').resolve())
    load_args['charlm_backward_file']=str(one_dependency_path('backward_charlm').resolve())
    trainer=GraphTrainer.load(str(compat_path),pretrain=pretrain_obj,args=load_args,
                              device=DEVICE,reset_history=True)
    assert isinstance(trainer.args['transition_subtree_combination'],SubtreeCombination)

    loaded=trainer.model.get_params(skip_modules=True)
    for name,old in base_ckpt['model'].items():
        got=loaded[name].detach().cpu()
        if name=='deprel.scorer.W_bilin.weight': assert torch.equal(got[:,:,:old_n],old)
        elif name=='deprel.scorer.W_bilin.bias' or (name.startswith('deprel_linear.') and old.ndim in (1,2) and old.shape[0]==old_n):
            assert torch.equal(got[:old_n],old)
        else: assert got.shape==old.shape and torch.equal(got,old),name

    lora_parameters=[(name,p) for name,p in trainer.model.named_parameters()
                     if name.startswith('bert_model.') and p.requires_grad]
    frozen_parameters=[(name,p) for name,p in trainer.model.named_parameters()
                       if name.startswith('bert_model.') and not p.requires_grad]
    parser_parameters=[(name,p) for name,p in trainer.model.named_parameters()
                       if not name.startswith('bert_model.') and p.requires_grad]
    assert lora_parameters and frozen_parameters and parser_parameters
    assert all(('lora_' in name or 'modules_to_save' in name) for name,_ in lora_parameters)
    lora_parameter_count=sum(p.numel() for _,p in lora_parameters)
    parser_parameter_count=sum(p.numel() for _,p in parser_parameters)

    # Reset again immediately before constructing loaders so data order is the
    # same across ranks and is not affected by rank-specific adapter creation.
    reset_seed()
    train_doc=CoNLL.conll2doc(input_file=str(DATA/'train.predposlemma.conllu'))
    dev_doc=CoNLL.conll2doc(input_file=str(DATA/'dev.predposlemma.conllu'))
    train_loader=DataLoader(train_doc,CFG['batch_size'],trainer.args,pretrain_obj,
        vocab=trainer.vocab,evaluation=False,bert_tokenizer=trainer.model.bert_tokenizer)
    dev_loader=DataLoader(dev_doc,CFG['batch_size'],trainer.args,pretrain_obj,
        vocab=trainer.vocab,evaluation=True,sort_during_eval=True,
        bert_tokenizer=trainer.model.bert_tokenizer)
    infinite=InfiniteBatch(train_loader)

    best_path=rank_dir/f'best_dev_electra_yue_lora_r{rank}.pt'
    history=[]
    prediction=rank_dir/'dev.step0000.conllu'
    predict_to_file(trainer,dev_loader,prediction)
    best_score=conll18(DATA/'dev.conllu',prediction)['LAS'].f1
    best_step=0
    trainer.save(str(best_path))
    last_loss=None
    for step in range(1,CFG['max_steps']+1):
        loss,_=trainer.update(infinite.next_batch(),eval=False)
        last_loss=float(loss); trainer.global_step=step
        if step%20==0: print(f'rank {rank} step {step} loss {last_loss:.5f}')
        if step%CFG['eval_interval']==0:
            prediction=rank_dir/f'dev.step{step:04d}.conllu'
            predict_to_file(trainer,dev_loader,prediction)
            score=conll18(DATA/'dev.conllu',prediction)['LAS'].f1
            row={'rank':rank,'alpha':ALPHA_TO_RANK*rank,'step':step,
                 'dev_conll18_las':score,'loss':last_loss,
                 'prediction_sha256':sha256(prediction)}
            history.append(row)
            print(f'== rank {rank} step {step}: dev LAS {100*score:.2f}% ==')
            if score>best_score:
                best_score=score; best_step=step; trainer.save(str(best_path))
            if step-best_step>=CFG['patience_steps']:
                print('early stop'); break

    history_path=rank_dir/'dev_history.json'
    history_path.write_text(json.dumps(history,indent=2),encoding='utf-8')
    result={'rank':rank,'alpha':ALPHA_TO_RANK*rank,'alpha_over_rank':ALPHA_TO_RANK,
            'best_step':best_step,'best_dev_conll18_las_percent':100*best_score,
            'lora_trainable_parameters':lora_parameter_count,
            'parser_trainable_parameters':parser_parameter_count,
            'checkpoint_sha256':sha256(best_path),'history_sha256':sha256(history_path)}
    rank_results.append(result); all_histories[str(rank)]=history; best_paths[rank]=best_path
    print(json.dumps(result,indent=2))

    del trainer,train_loader,dev_loader,infinite,train_doc,dev_doc
    gc.collect(); torch.cuda.empty_cache()

# Predeclared selection rule: highest DEV LAS; ties go to smaller rank.
selected=min(rank_results,key=lambda row:(-row['best_dev_conll18_las_percent'],row['rank']))
selected_rank=selected['rank']; selected_path=best_paths[selected_rank]
for row in rank_results:
    row['delta_vs_r8_dev_las_points']=row['best_dev_conll18_las_percent']-next(
        item['best_dev_conll18_las_percent'] for item in rank_results if item['rank']==8)
    row['selected_by_dev']=row['rank']==selected_rank

results_df=pd.DataFrame(rank_results).sort_values('rank')
results_df.to_csv(OUT/'rank_sweep_dev_results.csv',index=False)
(OUT/'rank_sweep_dev_results.json').write_text(json.dumps({
    'selection_split':'dev','test_evaluated':False,'seed':SEED,'ranks':RANKS,
    'alpha_policy':'alpha=2*rank, keeping alpha/r=2',
    'selected_rank':selected_rank,'results':rank_results,
    'reference_original_r8_dev_las_percent':74.82337829158638,
    'interpretation_warning':('One seed measures rank sensitivity for this run only. Small '
                              'differences may be smaller than seed-to-seed variation.')},
    ensure_ascii=False,indent=2),encoding='utf-8')

plt.figure(figsize=(7,4.5))
plt.plot(results_df['rank'],results_df['best_dev_conll18_las_percent'],marker='o')
for _,row in results_df.iterrows():
    plt.annotate(f"{row['best_dev_conll18_las_percent']:.2f}",(row['rank'],row['best_dev_conll18_las_percent']),
                 textcoords='offset points',xytext=(0,7),ha='center')
plt.xscale('log',base=2); plt.xticks(RANKS,RANKS)
plt.xlabel('LoRA rank'); plt.ylabel('Best DEV CoNLL-2018 LAS (%)')
plt.title('Cantonese dependency parser: DEV-only LoRA rank sweep')
plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(OUT/'rank_sweep_dev_las.png',dpi=180)
display(results_df)
print('DEV-selected rank:',selected_rank,'checkpoint:',selected_path)
print('TEST HAS NOT BEEN EVALUATED.')
'''

export = r'''# Export the DEV-only comparison and the selected rank checkpoint.
from google.colab import files
report_dir=WORK/'rank_sweep_report'; report_dir.mkdir(exist_ok=True)
for path in (OUT/'rank_sweep_dev_results.csv',OUT/'rank_sweep_dev_results.json',
             OUT/'rank_sweep_dev_las.png'):
    shutil.copy2(path,report_dir/path.name)
for rank in RANKS:
    history_path=OUT/f'rank_{rank}'/'dev_history.json'
    shutil.copy2(history_path,report_dir/f'rank_{rank}_dev_history.json')
run_record={'seed':SEED,'ranks':RANKS,'alpha_to_rank':ALPHA_TO_RANK,'config':CFG,
 'data':{'train_sha256':SPLIT_SHA256['train'],'dev_sha256':SPLIT_SHA256['dev'],
         'train_predposlemma_sha256':cache_hashes['train'],
         'dev_predposlemma_sha256':cache_hashes['dev']},
 'initialization':{'stanza_depparse_sha256':sha256(base_path),'hf_revision':HF_COMMIT},
 'selection':{'split':'dev','metric':'CoNLL-2018 LAS','selected_rank':selected_rank},
 'test_evaluated':False}
(report_dir/'run_record.json').write_text(json.dumps(run_record,ensure_ascii=False,indent=2),encoding='utf-8')
report_zip=shutil.make_archive('/content/yue_lora_rank_sweep_dev_results','zip',root_dir=report_dir)

selected_dir=WORK/'selected_rank_bundle'; selected_dir.mkdir(exist_ok=True)
shutil.copy2(selected_path,selected_dir/selected_path.name)
for path in report_dir.iterdir(): shutil.copy2(path,selected_dir/path.name)
selected_zip=shutil.make_archive('/content/yue_lora_rank_sweep_selected_checkpoint','zip',root_dir=selected_dir)
print('DEV report:',report_zip,sha256(Path(report_zip)))
print('Selected-rank checkpoint archive:',selected_zip,sha256(Path(selected_zip)))
files.download(report_zip)
files.download(selected_zip)
'''

cells = [
    md(
        """# Cantonese dependency parser — LoRA rank sweep (DEV only)

This self-contained Colab compares LoRA ranks **4, 8, 16, and 32**. It is designed for **Runtime → Run all** and requires no uploaded files. Every rank starts independently from the same Stanza 1.14 Mandarin ELECTRA-large dependency checkpoint. The Transformer base weights remain frozen; LoRA parameters and all parsing layers are trained.

To isolate rank capacity, `alpha/r` is held constant at 2 (`r=4/8/16/32`, `alpha=8/16/32/64`). All other inputs and hyperparameters match the original rank-8 experiment: UD Cantonese-HK r2.18, the fixed custom 803/101 train/dev split, seed 42, frozen Mandarin predicted POS/lemma, learning rates, evaluation interval, and early stopping.

**This notebook never reads or evaluates test.** Rank is selected using DEV CoNLL-2018 LAS only; ties go to the smaller rank. Because this is a one-seed diagnostic, small differences should not be interpreted as stable rank effects without a later multi-seed confirmation.

Use an A100 GPU if possible. Four sequential ELECTRA-large runs can take several hours, although only one model is kept in GPU memory at a time.
"""
    ),
    code(install),
    code(setup),
    code(download_data),
    code(resolve_hf),
    code(download_stanza),
    code(tag_data),
    code(prepare_base),
    code(sweep),
    code(export),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "yue_lora_rank_sweep_dev_only.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

destination = ROOT / "notebooks" / "yue_lora_rank_sweep_dev_only.ipynb"
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(destination)
