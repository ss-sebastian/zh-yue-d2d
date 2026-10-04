#!/usr/bin/env python3
"""Build a resumable LoRA-vs-full-FT data-scaling Colab."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "finetune_yue_stanza_electra_lora.ipynb"
source = json.loads(SOURCE.read_text(encoding="utf-8"))
manifest = json.loads((ROOT / "data/processed/yue_hk/split_manifest.json").read_text(encoding="utf-8"))
train_records = [row for row in manifest["sentences"] if row["split"] == "train"]


def stable_priority(group_id: str) -> str:
    return hashlib.sha256(f"42:{group_id}".encode()).hexdigest()


groups: dict[str, list[dict]] = defaultdict(list)
for record in train_records:
    groups[record["group_id"]].append(record)
sources = sorted({record["source"] for record in train_records})
total_source = Counter(record["source"] for record in train_records)
source_proportion = {source: total_source[source] / len(train_records) for source in sources}

# Produce one deterministic, source-balanced order of complete duplicate groups.
# Every scaling subset is a prefix of this order and is therefore nested.
remaining = []
for group_id, records in groups.items():
    remaining.append(
        {
            "group_id": group_id,
            "records": records,
            "counts": Counter(record["source"] for record in records),
            "size": len(records),
            "priority": stable_priority(group_id),
        }
    )
selected_groups = []
selected_source = Counter()
selected_count = 0
while remaining:
    def score(group):
        new_total = selected_count + group["size"]
        imbalance = sum(
            ((selected_source[source] + group["counts"][source]) / new_total - source_proportion[source]) ** 2
            for source in sources
        )
        return imbalance, group["priority"]

    chosen = min(remaining, key=score)
    remaining.remove(chosen)
    selected_groups.append(chosen)
    selected_source.update(chosen["counts"])
    selected_count += chosen["size"]

raw_blocks = (ROOT / "data/raw/ud_cantonese_hk-r2.18/yue_hk-ud-test.conllu").read_text(encoding="utf-8").strip().split("\n\n")
proportions = [0.10, 0.25, 0.50, 1.00]
subsets = {}
cumulative = 0
cumulative_counts = []
for group in selected_groups:
    cumulative += group["size"]
    cumulative_counts.append(cumulative)

previous_end = 0
for proportion in proportions:
    target = round(len(train_records) * proportion)
    end = min(
        range(previous_end + 1, len(cumulative_counts) + 1),
        key=lambda idx: (abs(cumulative_counts[idx - 1] - target), idx),
    )
    previous_end = end
    records = [record for group in selected_groups[:end] for record in group["records"]]
    positions = sorted(record["original_position"] for record in records)
    text = "\n\n".join(raw_blocks[position - 1] for position in positions) + "\n\n"
    key = f"p{int(100 * proportion):03d}"
    subsets[key] = {
        "proportion": proportion,
        "target_sentence_count": target,
        "sentence_count": len(positions),
        "group_count": end,
        "positions": positions,
        "gold_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_distribution": dict(sorted(Counter(record["source"] for record in records).items())),
    }

assert set(subsets["p010"]["positions"]) <= set(subsets["p025"]["positions"])
assert set(subsets["p025"]["positions"]) <= set(subsets["p050"]["positions"])
assert set(subsets["p050"]["positions"]) <= set(subsets["p100"]["positions"])
assert subsets["p100"]["sentence_count"] == 803


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


def cell_source(index: int) -> str:
    return "".join(source["cells"][index]["source"])


install = cell_source(1)
setup = cell_source(2).replace("/content/yue_lora_electra_r8", "/content/yue_data_scaling")
setup = setup.replace(
    "HF_ELECTRA_REVISION = None",
    "HF_ELECTRA_REVISION = 'd017e219578df8e4885484edbc8969dbdea9cbe0'",
)
old_cfg = """CFG = dict(rank=8, alpha=16, dropout=0.10,
           targets=['query', 'value', 'output.dense', 'intermediate.dense'],
           parser_lr=1e-3, lora_lr=2e-5, batch_size=900,
           max_steps=4000, eval_interval=100, patience_steps=600,
           max_grad_norm=1.0)"""
new_cfg = f"""SUBSETS = {subsets!r}
METHODS = ['lora_r8', 'full_ft']
CFG = dict(rank=8, alpha=16, dropout=0.10,
           targets=['query', 'value', 'output.dense', 'intermediate.dense'],
           parser_lr=1e-3, transformer_lr=2e-5, batch_size=900,
           max_epochs=120, eval_every_epochs=5, patience_epochs=40,
           max_grad_norm=1.0)
assert list(SUBSETS)==['p010','p025','p050','p100']"""
assert old_cfg in setup
setup = setup.replace(old_cfg, new_cfg)

download_data = cell_source(3)
resolve_hf = cell_source(4)
download_stanza = cell_source(5)
tag_data = cell_source(6)
tag_data += r'''
EXPECTED_CACHE_SHA256={
 'train':'806b3d42c4f6838d6ba9c6565f29062a0d7b6e136e12d158e4449c3e7e970e89',
 'dev':'e88aa03ff7453469deda68ed30c29dd16769e9839d2010b36abf91faace3bd97',
 'test':'8c5ee2a043ceb7870d636d798c3c2863c9da82adebcac9dffd53670cadb26a7f'}
assert cache_hashes==EXPECTED_CACHE_SHA256,(cache_hashes,EXPECTED_CACHE_SHA256)
print('Frozen POS/lemma caches match the original experiments: OK')
'''

drive_and_subsets = r'''# Persist every completed condition to Drive so Colab interruption is recoverable.
from google.colab import drive
drive.mount('/content/drive')
PERSIST=Path('/content/drive/MyDrive/yue_lora_fullft_data_scaling')
PERSIST.mkdir(parents=True,exist_ok=True)

train_positions=SPLIT_POSITIONS['train']
train_index={position:index for index,position in enumerate(train_positions)}
full_gold_blocks=split_blocks((DATA/'train.conllu').read_text(encoding='utf-8'))
full_pred_blocks=split_blocks((DATA/'train.predposlemma.conllu').read_text(encoding='utf-8'))
assert len(full_gold_blocks)==len(full_pred_blocks)==803

for subset_id,meta in SUBSETS.items():
    subset_dir=DATA/subset_id; subset_dir.mkdir(exist_ok=True)
    indices=sorted(train_index[position] for position in meta['positions'])
    gold_text='\n\n'.join(full_gold_blocks[index] for index in indices)+'\n\n'
    pred_text='\n\n'.join(full_pred_blocks[index] for index in indices)+'\n'
    gold_path=subset_dir/'train.conllu'; pred_path=subset_dir/'train.predposlemma.conllu'
    gold_path.write_text(gold_text,encoding='utf-8')
    pred_path.write_text(pred_text,encoding='utf-8')
    assert sha256(gold_path)==meta['gold_sha256'],(subset_id,sha256(gold_path),meta['gold_sha256'])
    meta['predposlemma_sha256']=sha256(pred_path)
    assert len(split_blocks(gold_text))==meta['sentence_count']
    print(subset_id,meta['sentence_count'],meta['source_distribution'],meta['gold_sha256'])

(PERSIST/'subset_manifest.json').write_text(json.dumps(SUBSETS,ensure_ascii=False,indent=2),encoding='utf-8')
'''

training = r'''# Run the predeclared 4 x 2 factorial experiment.
# Completed conditions are skipped.  An interrupted condition resumes from its latest DEV evaluation.
import time
import shutil
from peft import get_peft_model_state_dict
from stanza.models.common.bert_embedding import load_bert
from stanza.models.common.peft_config import build_peft_wrapper
from stanza.models.common.pretrain import Pretrain
from stanza.models.common.doc import HEAD,DEPREL
from stanza.models.common.vocab import VOCAB_PREFIX_SIZE
from stanza.models.depparse.data import DataLoader,InfiniteBatch
from stanza.models.depparse.trainer import GraphTrainer
from stanza.models.depparse.transition.model import SubtreeCombination
from stanza.models.depparse.utils import predict_dataset
from stanza.models.pos.vocab import MultiVocab
from stanza.utils.conll import CoNLL

base_path=Path(artifacts['depparse']['path'])
base_ckpt=torch.load(base_path,map_location='cpu',weights_only=True)
assert base_ckpt.get('model_type','graph')=='graph'
old_vocab=MultiVocab.load_state_dict(base_ckpt['vocab'])
old_units=list(old_vocab['deprel']._id2unit)
old_n=len(old_units)-VOCAB_PREFIX_SIZE

def one_dependency_path(model_type):
    dependencies=resources['zh-hans']['depparse']['gsdsimp_electra-large'].get('dependencies',[])
    names=[item['package'] for item in dependencies if item.get('model')==model_type]
    paths=[MODELS/'zh-hans'/model_type/f'{name}.pt' for name in names]
    paths=[path for path in paths if path.exists()]
    if len(paths)!=1: paths=list((MODELS/'zh-hans'/model_type).glob('*.pt'))
    assert len(paths)==1,f'Cannot resolve {model_type}: {paths}'
    return paths[0]

pretrain_obj=Pretrain(filename=str(one_dependency_path('pretrain'))) if base_ckpt['config'].get('pretrain') else None

def reset_seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def predict_to_file(trainer,loader,path):
    trainer.model.eval()
    with torch.inference_mode(): predictions=predict_dataset(trainer,loader)
    loader.doc.set([HEAD,DEPREL],[item for sentence in predictions for item in sentence])
    path.write_text(f'{loader.doc:C}\n\n',encoding='utf-8')
    return path

def strict_las(gold_path,pred_path,exclude_punct=False):
    correct=total=0
    for gold_block,pred_block in zip(split_blocks(gold_path.read_text()),split_blocks(pred_path.read_text())):
        gold_rows=integer_rows(gold_block); pred_rows=integer_rows(pred_block)
        assert len(gold_rows)==len(pred_rows)
        for gold,pred in zip(gold_rows,pred_rows):
            assert gold[:2]==pred[:2]
            if exclude_punct and gold[3]=='PUNCT': continue
            total+=1; correct+=int(gold[6]==pred[6] and gold[7]==pred[7])
    return correct/total

def make_expanded_base(subset_id,condition_dir):
    train_path=DATA/subset_id/'train.conllu'
    train_labels=sorted({row[7] for block in split_blocks(train_path.read_text()) for row in integer_rows(block)})
    new_labels=[label for label in train_labels if label not in old_vocab['deprel']]
    expanded=copy.deepcopy(base_ckpt)
    dep_state=expanded['vocab']['deprel']
    dep_state['_id2unit']=old_units+new_labels
    dep_state['_unit2id']={unit:index for index,unit in enumerate(dep_state['_id2unit'])}
    new_n=old_n+len(new_labels)
    if new_labels:
        changed=[]
        for name,tensor in list(expanded['model'].items()):
            if name=='deprel.scorer.W_bilin.weight':
                value=tensor.new_zeros(tensor.shape[0],tensor.shape[1],new_n); value[:,:,:old_n]=tensor
                expanded['model'][name]=value; changed.append(name)
            elif name=='deprel.scorer.W_bilin.bias':
                value=tensor.new_zeros(new_n); value[:old_n]=tensor
                expanded['model'][name]=value; changed.append(name)
            elif name.startswith('deprel_linear.') and tensor.ndim in (1,2) and tensor.shape[0]==old_n:
                value=tensor.new_zeros((new_n,)+tuple(tensor.shape[1:])); value[:old_n]=tensor
                expanded['model'][name]=value; changed.append(name)
        assert changed
    args=copy.deepcopy(expanded['config'])
    args['charlm_forward_file']=str(one_dependency_path('forward_charlm').resolve())
    args['charlm_backward_file']=str(one_dependency_path('backward_charlm').resolve())
    expanded['config']=args; expanded['global_step']=0; expanded['last_best_step']=0; expanded['dev_score_history']=[]
    path=condition_dir/'expanded_base.pt'
    torch.save(expanded,path,_use_new_zipfile_serialization=False)
    del expanded
    return path,args,new_labels

def run_condition(subset_id,method):
    persistent=PERSIST/subset_id/method
    persistent.mkdir(parents=True,exist_ok=True)
    meta=SUBSETS[subset_id]
    signature_payload={'protocol_version':2,'subset_id':subset_id,'method':method,'gold_sha256':meta['gold_sha256'],
                       'pred_sha256':meta['predposlemma_sha256'],'seed':SEED,'cfg':CFG,
                       'hf_revision':HF_COMMIT,'base_sha256':sha256(base_path)}
    signature=hashlib.sha256(json.dumps(signature_payload,sort_keys=True).encode()).hexdigest()
    completed=persistent/'result.json'
    if completed.exists():
        result=json.loads(completed.read_text())
        assert result['experiment_signature']==signature,(completed,result.get('experiment_signature'),signature)
        print('SKIP verified completed condition:',subset_id,method)
        return result

    resume_state_path=persistent/'resume_state.json'

    def atomic_json(path,payload):
        tmp=path.with_suffix(path.suffix+'.tmp')
        tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
        tmp.replace(path)

    def atomic_torch(path,payload):
        tmp=path.with_suffix(path.suffix+'.tmp')
        torch.save(payload,tmp,_use_new_zipfile_serialization=False)
        tmp.replace(path)

    reset_seed(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    condition_dir=OUT/f'{subset_id}_{method}'; condition_dir.mkdir(exist_ok=True)
    expanded_path,args,new_labels=make_expanded_base(subset_id,condition_dir)
    initial=torch.load(expanded_path,map_location='cpu',weights_only=True)
    if method=='lora_r8':
        args.update(use_peft=True,bert_finetune=True,lora_rank=CFG['rank'],lora_alpha=CFG['alpha'],
            lora_dropout=CFG['dropout'],lora_target_modules=CFG['targets'],lora_modules_to_save=[],
            optim='adamw',second_optim=None,lr=CFG['parser_lr'],
            bert_learning_rate=CFG['transformer_lr']/CFG['parser_lr'],bert_start_finetuning=0,
            bert_warmup_steps=0,weight_decay=0.0,bert_weight_decay=0.0,
            max_grad_norm=CFG['max_grad_norm'],batch_size=CFG['batch_size'],seed=SEED,
            enable_gradient_checkpointing=True,augment_nopunct=0.0)
        bert,_=load_bert(HF_REPO,enable_gradient_checkpointing=True)
        pefted=build_peft_wrapper(bert,args,logging.getLogger('stanza'),adapter_name='depparse')
        initial['bert_lora']=get_peft_model_state_dict(pefted,adapter_name='depparse')
        del bert,pefted
    elif method=='full_ft':
        for key in list(args):
            if key.startswith('lora_') or key in ('peft_name','bert_lora'): args.pop(key,None)
        args.update(use_peft=False,bert_finetune=True,optim='adamw',second_optim=None,
            lr=CFG['parser_lr'],bert_learning_rate=CFG['transformer_lr']/CFG['parser_lr'],
            bert_start_finetuning=0,bert_warmup_steps=0,bert_finetune_layers=None,
            weight_decay=0.0,bert_weight_decay=0.0,max_grad_norm=CFG['max_grad_norm'],
            batch_size=CFG['batch_size'],seed=SEED,enable_gradient_checkpointing=True,augment_nopunct=0.0)
        initial.pop('bert_lora',None)
    else: raise ValueError(method)
    initial['config']=args
    compat_path=condition_dir/'initial.pt'
    torch.save(initial,compat_path,_use_new_zipfile_serialization=False)
    del initial
    gc.collect(); torch.cuda.empty_cache()

    load_args=dict(args); load_args.pop('transition_subtree_combination',None)
    load_args['charlm_forward_file']=str(one_dependency_path('forward_charlm').resolve())
    load_args['charlm_backward_file']=str(one_dependency_path('backward_charlm').resolve())
    resume_state=json.loads(resume_state_path.read_text()) if resume_state_path.exists() else None
    if resume_state is not None:
        assert resume_state['experiment_signature']==signature,(resume_state_path,resume_state.get('experiment_signature'),signature)
        resume_model=persistent/resume_state['current_model']
        resume_optim=persistent/resume_state['optimizer_state']
        resume_rng=persistent/resume_state['rng_state']
        for required in (resume_model,resume_optim,resume_rng,persistent/resume_state['best_model']):
            assert required.exists() and required.stat().st_size>0,required
        trainer=GraphTrainer.load(str(resume_model),pretrain=pretrain_obj,args=load_args,device=DEVICE,reset_history=False)
    else:
        trainer=GraphTrainer.load(str(compat_path),pretrain=pretrain_obj,args=load_args,device=DEVICE,reset_history=True)
    assert isinstance(trainer.args['transition_subtree_combination'],SubtreeCombination)
    if method=='full_ft':
        assert 'bert_model' in trainer.model.unsaved_modules
        trainer.model.unsaved_modules.remove('bert_model')
        for parameter in trainer.model.bert_model.parameters(): parameter.requires_grad=True
        trainer._Trainer__init_optim()

    if resume_state is not None:
        saved_optim=torch.load(resume_optim,map_location='cpu',weights_only=True)
        for key,state in saved_optim['optimizer'].items(): trainer.optimizer[key].load_state_dict(state)
        for key,state in saved_optim['scheduler'].items(): trainer.scheduler[key].load_state_dict(state)

    # Confirm old parser tensors are exactly preserved only at fresh initialization.
    if resume_state is None:
        loaded=trainer.model.get_params(skip_modules=True)
        for name,old in base_ckpt['model'].items():
            got=loaded[name].detach().cpu()
            if name=='deprel.scorer.W_bilin.weight': assert torch.equal(got[:,:,:old_n],old)
            elif name=='deprel.scorer.W_bilin.bias' or (name.startswith('deprel_linear.') and old.ndim in (1,2) and old.shape[0]==old_n):
                assert torch.equal(got[:old_n],old)
            else: assert got.shape==old.shape and torch.equal(got,old),name
        del loaded,got,old

    trainable=sum(parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad)
    transformer_trainable=sum(parameter.numel() for name,parameter in trainer.model.named_parameters()
                              if name.startswith('bert_model.') and parameter.requires_grad)
    reset_seed()
    train_doc=CoNLL.conll2doc(input_file=str(DATA/subset_id/'train.predposlemma.conllu'))
    dev_doc=CoNLL.conll2doc(input_file=str(DATA/'dev.predposlemma.conllu'))
    train_loader=DataLoader(train_doc,CFG['batch_size'],trainer.args,pretrain_obj,vocab=trainer.vocab,
                            evaluation=False,bert_tokenizer=trainer.model.bert_tokenizer)
    dev_loader=DataLoader(dev_doc,CFG['batch_size'],trainer.args,pretrain_obj,vocab=trainer.vocab,
                          evaluation=True,sort_during_eval=True,bert_tokenizer=trainer.model.bert_tokenizer)
    steps_per_epoch=len(train_loader)
    assert steps_per_epoch>0
    max_steps=CFG['max_epochs']*steps_per_epoch
    eval_interval=CFG['eval_every_epochs']*steps_per_epoch
    patience_steps=CFG['patience_epochs']*steps_per_epoch
    if resume_state is None:
        infinite=InfiniteBatch(train_loader)
        history=[]
        dev_prediction=condition_dir/'dev.step0000.conllu'
        predict_to_file(trainer,dev_loader,dev_prediction)
        best_score=conll18(DATA/'dev.conllu',dev_prediction)['LAS'].f1
        best_step=0; start_step=0; prior_training_seconds=0.0
    else:
        # Checkpoints are written only at complete epoch boundaries.  Restore RNG, then
        # reshuffle once: this is exactly what the uninterrupted InfiniteBatch would do
        # on its next call after exhausting the previous epoch.
        rng=torch.load(resume_rng,map_location='cpu',weights_only=False)
        random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch'])
        if torch.cuda.is_available(): torch.cuda.set_rng_state_all(rng['cuda'])
        train_loader.reshuffle(); infinite=InfiniteBatch(train_loader)
        history=resume_state['history']; best_score=resume_state['best_score']; best_step=resume_state['best_step']
        start_step=resume_state['step']; prior_training_seconds=resume_state['training_seconds']
        assert trainer.global_step==start_step,(trainer.global_step,start_step)
        print('RESUME',subset_id,method,'from epoch',start_step/steps_per_epoch)

    def save_resume(step,training_seconds):
        model_name=f'resume_model.step{step:06d}.pt'
        optim_name=f'resume_optim.step{step:06d}.pt'
        rng_name=f'resume_rng.step{step:06d}.pt'
        model_path=persistent/model_name
        model_tmp=model_path.with_suffix('.pt.tmp')
        trainer.save(str(model_tmp),save_optimizer=False)
        assert model_tmp.exists() and model_tmp.stat().st_size>0
        model_tmp.replace(model_path)
        atomic_torch(persistent/optim_name,{
            'optimizer':{key:value.state_dict() for key,value in trainer.optimizer.items()},
            'scheduler':{key:value.state_dict() for key,value in trainer.scheduler.items()}})
        atomic_torch(persistent/rng_name,{
            'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []})
        previous_best=resume_state.get('best_model') if resume_state else None
        best_model=model_name if best_step==step else previous_best
        assert best_model is not None
        state={'experiment_signature':signature,'step':step,'best_step':best_step,'best_score':best_score,
               'best_model':best_model,'current_model':model_name,'optimizer_state':optim_name,
               'rng_state':rng_name,'training_seconds':training_seconds,'history':history}
        atomic_json(resume_state_path,state)
        # Keep only the current model and the DEV-best model; remove superseded recovery files.
        keep={model_name,best_model,optim_name,rng_name}
        for pattern in ('resume_model.step*.pt','resume_optim.step*.pt','resume_rng.step*.pt'):
            for old_path in persistent.glob(pattern):
                if old_path.name not in keep: old_path.unlink()
        return state

    if resume_state is None:
        resume_state=save_resume(0,0.0)
    start=time.monotonic(); last_loss=None
    loop_end=start_step if start_step-best_step>=patience_steps else max_steps
    for step in range(start_step+1,loop_end+1):
        loss,_=trainer.update(infinite.next_batch(),eval=False); last_loss=float(loss); trainer.global_step=step
        if step%eval_interval==0:
            dev_prediction=condition_dir/f'dev.step{step:06d}.conllu'
            predict_to_file(trainer,dev_loader,dev_prediction)
            score=conll18(DATA/'dev.conllu',dev_prediction)['LAS'].f1
            history.append({'step':step,'epoch':step/steps_per_epoch,'dev_conll18_las':score,
                            'loss':last_loss,'prediction_sha256':sha256(dev_prediction)})
            print(subset_id,method,'epoch',step/steps_per_epoch,'dev LAS',100*score)
            if score>best_score:
                best_score=score; best_step=step
            training_seconds=prior_training_seconds+time.monotonic()-start
            resume_state=save_resume(step,training_seconds)
            if step-best_step>=patience_steps: break
    training_seconds=prior_training_seconds+time.monotonic()-start
    peak_allocated=torch.cuda.max_memory_allocated()/2**30
    peak_reserved=torch.cuda.max_memory_reserved()/2**30
    best_path=persistent/resume_state['best_model']
    checkpoint_sha=sha256(best_path); checkpoint_bytes=best_path.stat().st_size

    # Reload the DEV-selected checkpoint before the single predeclared TEST evaluation.
    del trainer,train_loader,dev_loader,infinite,train_doc,dev_doc
    gc.collect(); torch.cuda.empty_cache()
    best_trainer=GraphTrainer.load(str(best_path),pretrain=pretrain_obj,args=load_args,device=DEVICE)
    for parameter in best_trainer.model.parameters(): parameter.requires_grad=False
    best_trainer.optimizer={}; best_trainer.scheduler={}
    dev_doc=CoNLL.conll2doc(input_file=str(DATA/'dev.predposlemma.conllu'))
    test_doc=CoNLL.conll2doc(input_file=str(DATA/'test.predposlemma.conllu'))
    dev_loader=DataLoader(dev_doc,CFG['batch_size'],best_trainer.args,pretrain_obj,vocab=best_trainer.vocab,
        evaluation=True,sort_during_eval=True,bert_tokenizer=best_trainer.model.bert_tokenizer)
    test_loader=DataLoader(test_doc,CFG['batch_size'],best_trainer.args,pretrain_obj,vocab=best_trainer.vocab,
        evaluation=True,sort_during_eval=True,bert_tokenizer=best_trainer.model.bert_tokenizer)
    dev_best=persistent/'dev.best.pred.conllu'; test_best=persistent/'test.best.pred.conllu'
    predict_to_file(best_trainer,dev_loader,dev_best); predict_to_file(best_trainer,test_loader,test_best)
    dev_score=conll18(DATA/'dev.conllu',dev_best)['LAS'].f1
    assert abs(dev_score-best_score)<1e-12,(dev_score,best_score)
    test_scores=conll18(DATA/'test.conllu',test_best)
    subset_labels=sorted({row[7] for block in split_blocks((DATA/subset_id/'train.conllu').read_text()) for row in integer_rows(block)})
    dev_labels=sorted({row[7] for block in split_blocks((DATA/'dev.conllu').read_text()) for row in integer_rows(block)})
    result={'experiment_signature':signature,'subset_id':subset_id,'proportion':meta['proportion'],
      'train_sentences':meta['sentence_count'],'method':method,'seed':SEED,
      'steps_per_epoch':steps_per_epoch,'best_step':best_step,'best_epoch':best_step/steps_per_epoch,
      'dev_conll18_las_percent':100*dev_score,'test_conll18_las_percent':100*test_scores['LAS'].f1,
      'test_uas_percent':100*test_scores['UAS'].f1,
      'test_strict_las_percent':100*strict_las(DATA/'test.conllu',test_best),
      'test_strict_las_no_punct_percent':100*strict_las(DATA/'test.conllu',test_best,True),
      'trainable_parameters':trainable,'transformer_trainable_parameters':transformer_trainable,
      'training_seconds':training_seconds,'peak_gpu_allocated_gib':peak_allocated,
      'peak_gpu_reserved_gib':peak_reserved,'checkpoint_bytes':checkpoint_bytes,
      'checkpoint_sha256':checkpoint_sha,'dev_prediction_sha256':sha256(dev_best),
      'test_prediction_sha256':sha256(test_best),'new_deprel_labels':new_labels,
      'dev_labels_unseen_in_subset':sorted(set(dev_labels)-set(subset_labels)),
      'test_was_previously_used_for_mandarin_model_family_selection':True}
    atomic_json(persistent/'history.json',history)
    atomic_json(completed,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

    del best_trainer,dev_loader,test_loader,dev_doc,test_doc
    gc.collect(); torch.cuda.empty_cache()
    for path in condition_dir.glob('*'):
        if path.is_file(): path.unlink()
    condition_dir.rmdir()
    # Result and predictions are the durable artifacts; large recovery checkpoints are temporary.
    for pattern in ('resume_model.step*.pt','resume_optim.step*.pt','resume_rng.step*.pt'):
        for path in persistent.glob(pattern): path.unlink()
    resume_state_path.unlink(missing_ok=True)
    return result

all_results=[]
for subset_id in SUBSETS:
    for method in METHODS:
        print('\n==========',subset_id,method,'==========')
        all_results.append(run_condition(subset_id,method))
        gc.collect(); torch.cuda.empty_cache()
'''

report = r'''# Aggregate fixed-TEST learning curves and export a compact report.
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files

all_results=[]
for subset_id in SUBSETS:
    for method in METHODS:
        result_path=PERSIST/subset_id/method/'result.json'
        assert result_path.exists(),result_path
        all_results.append(json.loads(result_path.read_text()))
df=pd.DataFrame(all_results).sort_values(['train_sentences','method'])
df.to_csv(PERSIST/'data_scaling_results.csv',index=False)

pivot=df.pivot(index='train_sentences',columns='method',values='test_conll18_las_percent')
pivot['full_minus_lora_las_points']=pivot['full_ft']-pivot['lora_r8']
pivot.to_csv(PERSIST/'test_las_comparison.csv')

fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for method,label,color in [('lora_r8','LoRA r=8','#2563eb'),('full_ft','Full FT','#dc2626')]:
    part=df[df.method==method]
    axes[0].plot(part.train_sentences,part.dev_conll18_las_percent,marker='o',label=label,color=color)
    axes[1].plot(part.train_sentences,part.test_conll18_las_percent,marker='o',label=label,color=color)
axes[0].set_title('DEV learning curve'); axes[1].set_title('Fixed TEST learning curve')
for ax in axes:
    ax.set_xlabel('Cantonese training sentences'); ax.set_ylabel('CoNLL-2018 LAS (%)')
    ax.grid(alpha=.25); ax.legend(frameon=False)
fig.suptitle('Cantonese dependency parsing: LoRA vs full fine-tuning')
fig.tight_layout(); fig.savefig(PERSIST/'data_scaling_curves.png',dpi=180)

summary={'design':{'proportions':[meta['proportion'] for meta in SUBSETS.values()],
 'methods':METHODS,'seed':SEED,'max_epochs':CFG['max_epochs'],
 'eval_every_epochs':CFG['eval_every_epochs'],'patience_epochs':CFG['patience_epochs'],
 'nested_grouped_source_balanced_subsets':True,'fixed_dev':True,'fixed_test':True},
 'results':all_results,
 'test_caveat':'The fixed custom test had already been used for Mandarin model-family selection.',
 'single_seed_caveat':'This Run-all default uses one seed; small differences require multi-seed confirmation.'}
(PERSIST/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
display(df,pivot)

report_dir=WORK/'final_report'; report_dir.mkdir(exist_ok=True)
for name in ('subset_manifest.json','data_scaling_results.csv','test_las_comparison.csv','data_scaling_curves.png','summary.json'):
    shutil.copy2(PERSIST/name,report_dir/name)
for subset_id in SUBSETS:
    for method in METHODS:
        source_dir=PERSIST/subset_id/method
        for name in ('result.json','history.json','dev.best.pred.conllu','test.best.pred.conllu'):
            shutil.copy2(source_dir/name,report_dir/f'{subset_id}_{method}_{name}')
archive=shutil.make_archive('/content/yue_lora_vs_fullft_data_scaling_results','zip',root_dir=report_dir)
print('report archive:',archive,sha256(Path(archive)))
files.download(archive)
'''

cells = [
    md(
        """# Cantonese data scaling: LoRA r=8 vs full fine-tuning

This self-contained Colab compares **LoRA r=8 + parsing-layer fine-tuning** with **full Transformer + parsing-layer fine-tuning** at four nested Cantonese training sizes: approximately 10%, 25%, 50%, and 100% of the fixed 803-sentence train split. Select an A100 GPU and choose **Runtime → Run all**.

The custom 101-sentence dev and 100-sentence test sets remain exactly fixed. Duplicate-text groups are never divided, and training subsets are deterministic, nested, and greedily balanced across the four official source domains. Each condition starts from the same Stanza 1.14 Mandarin ELECTRA-large parser, uses the same frozen predicted POS/lemma inputs, and expands the output vocabulary only with Cantonese relations observed in that condition's training subset.

Training is controlled by epochs rather than a fixed number of steps: 120 maximum epochs, evaluation every 5 epochs, and early stopping after 40 epochs without DEV LAS improvement. This prevents smaller subsets from receiving many more passes over their data. DEV selects each checkpoint; the predeclared fixed TEST is evaluated once afterward for every condition.

The notebook mounts Google Drive and writes a recovery checkpoint after every DEV evaluation (every 5 epochs), including model, optimizer, scheduler, RNG states, and progress. If Colab disconnects, reconnect and Run all again: verified completed conditions are skipped, and the interrupted condition resumes from its latest completed evaluation. Full-FT recovery files are large and make training somewhat slower, but they prevent a long condition from being lost. Recovery checkpoints are deleted after that condition's DEV and TEST predictions and result record are safely written.

Important limitation: this custom test was previously used to select the Mandarin model family, so it is fixed and comparable but not untouched. The default experiment uses one seed; small method differences require later multi-seed confirmation.
"""
    ),
    code(install),
    code(setup),
    code(download_data),
    code(resolve_hf),
    code(download_stanza),
    code(tag_data),
    code(drive_and_subsets),
    code(training),
    code(report),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "yue_lora_vs_fullft_data_scaling.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

destination = ROOT / "notebooks" / "yue_lora_vs_fullft_data_scaling.ipynb"
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
(ROOT / "data/processed/yue_hk/scaling_subsets_seed42.json").write_text(
    json.dumps(subsets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(destination)
