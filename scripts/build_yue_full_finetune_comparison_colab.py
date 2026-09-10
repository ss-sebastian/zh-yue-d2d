#!/usr/bin/env python3
"""Build a Run-all Colab for full ELECTRA fine-tuning vs the existing LoRA run."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NB = ROOT / "notebooks/finetune_yue_stanza_electra_lora.ipynb"
source = json.loads(SOURCE_NB.read_text(encoding="utf-8"))


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


def cell_source(index):
    return "".join(source["cells"][index]["source"])


install = cell_source(1).replace("peft==0.17.1 ", "")
setup = cell_source(2)
setup = setup.replace("/content/yue_lora_electra_r8", "/content/yue_full_finetune_comparison")
setup = setup.replace("HF_ELECTRA_REVISION = None", "HF_ELECTRA_REVISION = 'd017e219578df8e4885484edbc8969dbdea9cbe0'")
old_cfg = """CFG = dict(rank=8, alpha=16, dropout=0.10,
           targets=['query', 'value', 'output.dense', 'intermediate.dense'],
           parser_lr=1e-3, lora_lr=2e-5, batch_size=900,
           max_steps=4000, eval_interval=100, patience_steps=600,
           max_grad_norm=1.0)"""
new_cfg = """CFG = dict(parser_lr=1e-3, transformer_lr=2e-5, batch_size=900,
           max_steps=4000, eval_interval=100, patience_steps=600,
           max_grad_norm=1.0, selected_sentence_count=36)"""
assert old_cfg in setup
setup = setup.replace(old_cfg, new_cfg)

download_data = cell_source(3)
download_data = download_data.replace("for split, positions in SPLIT_POSITIONS.items():", "for split in ('train','dev'):\n    positions = SPLIT_POSITIONS[split]")
resolve_hf = cell_source(4)
download_stanza = cell_source(5)
tag_data = cell_source(6).replace("# Run frozen POS/lemma exactly once and build pretagged train/dev/test caches.", "# Run frozen POS/lemma exactly once and build pretagged train/dev caches.")
tag_data = tag_data.replace("for split in ('train','dev','test'):", "for split in ('train','dev'):")
tag_data += """
EXPECTED_CACHE_SHA256={'train':'806b3d42c4f6838d6ba9c6565f29062a0d7b6e136e12d158e4449c3e7e970e89',
                       'dev':'e88aa03ff7453469deda68ed30c29dd16769e9839d2010b36abf91faace3bd97'}
assert cache_hashes == EXPECTED_CACHE_SHA256,(cache_hashes,EXPECTED_CACHE_SHA256)
print('Frozen POS/lemma caches exactly match the LoRA run: OK')
"""

full_init = r'''# Construct the full-fine-tuning initialization from the original Mandarin checkpoint.
# Only new Cantonese DEPREL output units are initialized; every old parser tensor is preserved.
from stanza.models.pos.vocab import MultiVocab
from stanza.models.common.vocab import VOCAB_PREFIX_SIZE

base_path=Path(artifacts['depparse']['path'])
EXPECTED_DEP_REGISTRY_MD5='c7ea98d93459b22720337a9dcba1a848'
PRIOR_RECORDED_DEP_SHA256='6531bcc2dbfbe1e3b19d4deb855533fc2379f7305162423f7b53216b1e03434'
def file_md5(path):
    h=hashlib.md5()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()
actual_dep_md5=file_md5(base_path)
actual_dep_sha256=sha256(base_path)
assert artifacts['depparse']['registry_md5']==EXPECTED_DEP_REGISTRY_MD5,artifacts['depparse']
assert actual_dep_md5==EXPECTED_DEP_REGISTRY_MD5,(
    'Downloaded depparse file does not match the pinned Stanza registry MD5',
    str(base_path),actual_dep_md5,EXPECTED_DEP_REGISTRY_MD5)
if actual_dep_sha256 != PRIOR_RECORDED_DEP_SHA256:
    print('WARNING: registry-verified depparse artifact SHA-256 differs from the prior run record.')
    print('prior recorded SHA-256:',PRIOR_RECORDED_DEP_SHA256)
    print('current verified SHA-256:',actual_dep_sha256)
    print('The official Stanza registry MD5 matches; this discrepancy is retained in the run record.')
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
for key in list(args):
    if key.startswith('lora_') or key in ('peft_name','bert_lora'):
        args.pop(key,None)
args.update(use_peft=False, bert_finetune=True, optim='adamw', second_optim=None,
            lr=CFG['parser_lr'],
            # Stanza expresses the Transformer LR as a ratio to the parser LR.
            bert_learning_rate=CFG['transformer_lr']/CFG['parser_lr'],
            bert_start_finetuning=0, bert_warmup_steps=0,
            bert_finetune_layers=None,
            weight_decay=0.0, bert_weight_decay=0.0,
            max_grad_norm=CFG['max_grad_norm'], batch_size=CFG['batch_size'], seed=SEED,
            enable_gradient_checkpointing=True, augment_nopunct=0.0)
assert args['bert_model'] == HF_REPO

def one_dependency_path(model_type):
    deps=resources['zh-hans']['depparse']['gsdsimp_electra-large'].get('dependencies',[])
    names=[d['package'] for d in deps if d.get('model')==model_type]
    paths=[MODELS/'zh-hans'/model_type/f'{name}.pt' for name in names]
    paths=[p for p in paths if p.exists()]
    if len(paths)!=1: paths=list((MODELS/'zh-hans'/model_type).glob('*.pt'))
    assert len(paths)==1, f'Cannot unambiguously resolve {model_type}: {paths}'
    return paths[0]

if args.get('charlm'):
    args['charlm_forward_file']=str(one_dependency_path('forward_charlm').resolve())
    args['charlm_backward_file']=str(one_dependency_path('backward_charlm').resolve())
expanded.pop('bert_lora',None)
expanded['config']=args; expanded['global_step']=0; expanded['last_best_step']=0; expanded['dev_score_history']=[]
compat_path=OUT/'mandarin_electra_expanded_deprel_full_init.pt'
torch.save(expanded,compat_path,_use_new_zipfile_serialization=False)
del expanded; gc.collect(); torch.cuda.empty_cache()
print('original parser SHA-256:',sha256(base_path))
print('HF revision:',HF_COMMIT)
print('old relation labels:',old_n,'new labels:',new_labels,'expanded tensors:',expanded_names)
'''

full_train = r'''# Load the initialization, audit it, and perform full Transformer + parser fine-tuning.
from stanza.models.depparse.trainer import GraphTrainer
from stanza.models.depparse.data import DataLoader, InfiniteBatch
from stanza.models.depparse.utils import predict_dataset
from stanza.models.common.pretrain import Pretrain
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import HEAD, DEPREL
from stanza.models.depparse.transition.model import SubtreeCombination

pretrain_obj=Pretrain(filename=str(one_dependency_path('pretrain'))) if base_ckpt['config'].get('pretrain') else None
load_args=dict(args)
load_args.pop('transition_subtree_combination',None)
load_args['charlm_forward_file']=str(one_dependency_path('forward_charlm').resolve())
load_args['charlm_backward_file']=str(one_dependency_path('backward_charlm').resolve())
trainer=GraphTrainer.load(str(compat_path),pretrain=pretrain_obj,args=load_args,device=DEVICE,reset_history=True)
assert isinstance(trainer.args['transition_subtree_combination'],SubtreeCombination)

# Stanza 1.14 freezes an externally reloaded Transformer and registers it as
# unsaved when the source parser checkpoint itself did not contain BERT tensors.
# For genuine full fine-tuning we must unfreeze it, make future checkpoints save
# it, and rebuild BOTH optimizers after changing requires_grad.
assert trainer.args['bert_finetune'] and not trainer.args['use_peft']
assert 'bert_model' in trainer.model.unsaved_modules,trainer.model.unsaved_modules
trainer.model.unsaved_modules.remove('bert_model')
for parameter in trainer.model.bert_model.parameters():
    parameter.requires_grad=True
trainer._Trainer__init_optim()

# Exact audit of every original parsing tensor and every old DEPREL slice.
loaded=trainer.model.get_params(skip_modules=True)
for name,old in base_ckpt['model'].items():
    got=loaded[name].detach().cpu()
    if name == 'deprel.scorer.W_bilin.weight': assert torch.equal(got[:,:,:old_n],old)
    elif name == 'deprel.scorer.W_bilin.bias' or (name.startswith('deprel_linear.') and old.ndim in (1,2) and old.shape[0]==old_n): assert torch.equal(got[:old_n],old)
    else: assert got.shape==old.shape and torch.equal(got,old),name

transformer=[(n,p) for n,p in trainer.model.named_parameters() if n.startswith('bert_model.')]
parser=[(n,p) for n,p in trainer.model.named_parameters() if not n.startswith('bert_model.')]
assert transformer and parser
assert all(p.requires_grad for _,p in transformer),[n for n,p in transformer if not p.requires_grad][:10]
assert any(p.requires_grad for _,p in parser)
assert not any('lora_' in n for n,_ in transformer)
assert 'bert_optimizer' in trainer.optimizer,trainer.optimizer.keys()
optimizer_parameter_ids={id(p) for opt in trainer.optimizer.values() for group in opt.param_groups for p in group['params']}
assert {id(p) for _,p in transformer}.issubset(optimizer_parameter_ids)
bert_lrs={group['lr'] for group in trainer.optimizer['bert_optimizer'].param_groups}
assert bert_lrs=={CFG['transformer_lr']},(bert_lrs,CFG['transformer_lr'])
assert 'bert_model' not in trainer.model.unsaved_modules
trainable=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
total=sum(p.numel() for p in trainer.model.parameters())
print('Original Mandarin parser tensors preserved: OK')
print('trainable Transformer tensors:',len(transformer),'trainable total parameters:',trainable,'/',total)

train_doc=CoNLL.conll2doc(input_file=str(DATA/'train.predposlemma.conllu'))
dev_doc=CoNLL.conll2doc(input_file=str(DATA/'dev.predposlemma.conllu'))
train_loader=DataLoader(train_doc,CFG['batch_size'],trainer.args,pretrain_obj,vocab=trainer.vocab,evaluation=False,bert_tokenizer=trainer.model.bert_tokenizer)
dev_loader=DataLoader(dev_doc,CFG['batch_size'],trainer.args,pretrain_obj,vocab=trainer.vocab,evaluation=True,sort_during_eval=True,bert_tokenizer=trainer.model.bert_tokenizer)
infinite=InfiniteBatch(train_loader)

def predict_to_file(tr,loader,path):
    tr.model.eval()
    with torch.inference_mode(): preds=predict_dataset(tr,loader)
    loader.doc.set([HEAD,DEPREL],[y for x in preds for y in x])
    path.write_text(f'{loader.doc:C}\n\n',encoding='utf-8')
    return path

best_path=OUT/'best_dev_electra_yue_full_finetune.pt'
history=[]
dev_pred=OUT/'dev.full.step0000.conllu'
predict_to_file(trainer,dev_loader,dev_pred)
best_score=conll18(DATA/'dev.conllu',dev_pred)['LAS'].f1
best_step=0; trainer.save(str(best_path))
checkpoint_audit=torch.load(best_path,map_location='cpu',weights_only=True)
assert any(name.startswith('bert_model.') for name in checkpoint_audit['model']),(
    'Full-fine-tuned checkpoint did not save Transformer weights')
del checkpoint_audit
print('step 0 dev LAS',best_score*100,'Transformer checkpoint persistence: OK')

for step in range(1,CFG['max_steps']+1):
    loss,_=trainer.update(infinite.next_batch(),eval=False)
    trainer.global_step=step
    if step % 20 == 0: print(f'step {step} loss {loss:.5f}')
    if step % CFG['eval_interval'] == 0:
        pred=OUT/f'dev.full.step{step:04d}.conllu'; predict_to_file(trainer,dev_loader,pred)
        score=conll18(DATA/'dev.conllu',pred)['LAS'].f1
        history.append({'step':step,'dev_conll18_las':score,'loss':float(loss),'prediction_sha256':sha256(pred)})
        print(f'== step {step}: dev CoNLL18 LAS {score*100:.2f}% ==')
        if score > best_score:
            best_score=score; best_step=step; trainer.save(str(best_path))
        if step-best_step >= CFG['patience_steps']:
            print('early stop'); break

(OUT/'full_dev_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
print('BEST FULL DEV:',best_step,best_score*100,sha256(best_path))
'''

compare = r'''# Upload the prior LoRA DEV diagnostics, run the selected full model, and compare them.
# No test data is evaluated in this notebook.
import zipfile, unicodedata
import pandas as pd
from google.colab import files

print('Upload the existing yue_dev_diagnostics.zip')
uploaded=files.upload()
zip_names=[name for name in uploaded if name.endswith('.zip')]
assert len(zip_names)==1,zip_names
lora_zip=Path('/content')/zip_names[0]
EXPECTED_LORA_ZIP_SHA='6165dd6dbcb22d7871fc9f9a725eebf8c2a65cda95a20b9ae7cf449c9d882cc0'
EXPECTED_LORA_DEV_PRED_SHA='02e222057654b97dd3ced80d8c456014b84192185900c14ded78af7ceda420fd'
assert sha256(lora_zip)==EXPECTED_LORA_ZIP_SHA,(sha256(lora_zip),EXPECTED_LORA_ZIP_SHA)
with zipfile.ZipFile(lora_zip) as z:
    assert z.testzip() is None
    member='dev.predposlemma.best.pred.conllu'
    assert member in z.namelist()
    z.extract(member,WORK/'prior_lora')
lora_pred=WORK/'prior_lora'/member
assert sha256(lora_pred)==EXPECTED_LORA_DEV_PRED_SHA

del trainer; gc.collect(); torch.cuda.empty_cache()
best_trainer=GraphTrainer.load(str(best_path),pretrain=pretrain_obj,args=load_args,device=DEVICE)
full_doc=CoNLL.conll2doc(input_file=str(DATA/'dev.predposlemma.conllu'))
full_loader=DataLoader(full_doc,CFG['batch_size'],best_trainer.args,pretrain_obj,vocab=best_trainer.vocab,
                       evaluation=True,sort_during_eval=True,bert_tokenizer=best_trainer.model.bert_tokenizer)
full_pred=OUT/'dev.full.best.pred.conllu'
predict_to_file(best_trainer,full_loader,full_pred)
full_score=conll18(DATA/'dev.conllu',full_pred)['LAS'].f1
assert abs(full_score-best_score)<1e-12,(full_score,best_score)

def sentence_meta(block):
    out={}
    for line in block.splitlines():
        if line.startswith('# ') and ' = ' in line:
            k,v=line[2:].split(' = ',1); out[k]=v
    return out

gold_blocks=split_blocks((DATA/'dev.conllu').read_text(encoding='utf-8'))
input_blocks=split_blocks((DATA/'dev.predposlemma.conllu').read_text(encoding='utf-8'))
lora_blocks=split_blocks(lora_pred.read_text(encoding='utf-8'))
full_blocks=split_blocks(full_pred.read_text(encoding='utf-8'))
assert len(gold_blocks)==len(input_blocks)==len(lora_blocks)==len(full_blocks)==101
rows=[]
base=lambda x:x.split(':',1)[0]
for si,(gb,ib,lb,fb) in enumerate(zip(gold_blocks,input_blocks,lora_blocks,full_blocks),1):
    gr,ir,lr,fr=map(integer_rows,(gb,ib,lb,fb)); meta=sentence_meta(gb)
    assert len(gr)==len(ir)==len(lr)==len(fr)
    forms={int(x[0]):x[1] for x in gr}; forms[0]='ROOT'
    for g,inp,l,f in zip(gr,ir,lr,fr):
        assert g[:2]==inp[:2]==l[:2]==f[:2]
        tid=int(g[0]); gh,lh,fh=map(int,(g[6],l[6],f[6])); gd,ld,fd=g[7],l[7],f[7]
        dist=0 if gh==0 else abs(tid-gh)
        bucket='ROOT' if gh==0 else ('1-2' if dist<=2 else ('3-5' if dist<=5 else '6+'))
        l_uas=gh==lh; f_uas=gh==fh
        l_las=l_uas and base(gd)==base(ld); f_las=f_uas and base(gd)==base(fd)
        change='fixed' if (not l_las and f_las) else ('regressed' if (l_las and not f_las) else ('stable_correct' if l_las else 'persistent_error'))
        rows.append(dict(sentence_index=si,sent_id=meta.get('sent_id'),text=meta.get('text'),token_id=tid,
            form=g[1],predicted_upos=inp[3],gold_head=gh,gold_head_form=forms[gh],gold_deprel=gd,
            lora_head=lh,lora_head_form=forms.get(lh,'?'),lora_deprel=ld,lora_uas=l_uas,lora_las=l_las,
            full_head=fh,full_head_form=forms.get(fh,'?'),full_deprel=fd,full_uas=f_uas,full_las=f_las,
            distance=dist,distance_bucket=bucket,change=change,
            manual_error_class='',manual_notes=''))
df=pd.DataFrame(rows)

def aggregate(column):
    out=(df.groupby(column,dropna=False).agg(tokens=('token_id','size'),
         lora_UAS=('lora_uas','mean'),full_UAS=('full_uas','mean'),
         lora_LAS=('lora_las','mean'),full_LAS=('full_las','mean')).reset_index())
    out['UAS_delta_points']=100*(out.full_UAS-out.lora_UAS)
    out['LAS_delta_points']=100*(out.full_LAS-out.lora_LAS)
    return out

comparison=WORK/'comparison'; comparison.mkdir(exist_ok=True)
distance=aggregate('distance_bucket')
distance['distance_bucket']=pd.Categorical(distance.distance_bucket,['ROOT','1-2','3-5','6+'],ordered=True)
distance=distance.sort_values('distance_bucket')
relation=aggregate('gold_deprel').sort_values(['tokens','gold_deprel'],ascending=[False,True])
distance.to_csv(comparison/'distance_comparison.csv',index=False)
relation.to_csv(comparison/'relation_comparison.csv',index=False)
df.to_csv(comparison/'token_level_comparison.csv',index=False)

# Deterministically select 36 sentences.  Greedy coverage spans short/medium/long/root,
# fixed/regressed/persistent errors, and the predefined low-performing relations.
priority_rel={'advcl','reparandum','obl','conj','compound','parataxis','mark','vocative','amod','xcomp','mark:rel','obl:tmod'}
error_df=df[~(df.lora_las & df.full_las)].copy()
features={}
for si,g in error_df.groupby('sentence_index'):
    fs={'distance:'+x for x in g.distance_bucket.unique()}
    fs|={'change:'+x for x in g.change.unique() if x!='stable_correct'}
    fs|={'relation:'+x for x in g.gold_deprel.unique() if x in priority_rel}
    features[int(si)]=fs
universe=set().union(*features.values()) if features else set()
selected=[]; covered=set()
while len(selected)<CFG['selected_sentence_count'] and len(selected)<len(features):
    candidates=[si for si in features if si not in selected]
    best=max(candidates,key=lambda si:(len(features[si]-covered),len(error_df[error_df.sentence_index==si]),-si))
    selected.append(best); covered|=features[best]

review=df[df.sentence_index.isin(selected)].copy()
review['selection_order']=review.sentence_index.map({si:i+1 for i,si in enumerate(selected)})
review=review.sort_values(['selection_order','token_id'])
review.to_csv(comparison/'manual_review_36_sentences_token_rows.csv',index=False)
sentence_review=(review.groupby(['selection_order','sentence_index','sent_id','text'],dropna=False)
  .agg(tokens=('token_id','size'),lora_errors=('lora_las',lambda x:(~x).sum()),
       full_errors=('full_las',lambda x:(~x).sum()),fixed=('change',lambda x:(x=='fixed').sum()),
       regressed=('change',lambda x:(x=='regressed').sum()),max_gold_distance=('distance','max')).reset_index())
sentence_review['manual_primary_diagnosis']=''
sentence_review['manual_notes']=''
sentence_review.to_csv(comparison/'manual_review_36_sentences_index.csv',index=False)
(comparison/'manual_review_36_sentences.conllu').write_text(
    '\n\n'.join(gold_blocks[si-1] for si in selected)+'\n\n',encoding='utf-8')

lora_las=float(df.lora_las.mean()); full_las=float(df.full_las.mean())
fixed=int((df.change=='fixed').sum()); regressed=int((df.change=='regressed').sum())
eligible_rel=relation[relation.tokens>=10]
summary={'split':'dev','test_evaluated':False,'seed':SEED,
 'initialization':{'stanza_depparse_sha256':sha256(base_path),'hf_repo':HF_REPO,'hf_revision':HF_COMMIT,
                   'old_parser_weights_preserved':True,'new_deprel_labels':new_labels},
 'controls':{'same_frozen_predicted_pos_lemma_cache':True,'cache_sha256':cache_hashes,
             'same_train_dev_split':True,'same_parser_lr':CFG['parser_lr'],
             'same_transformer_lr':CFG['transformer_lr'],'same_dev_selection_rule':True},
 'lora':{'best_step':1200,'dev_conll18_las_percent':100*lora_las,'prediction_sha256':sha256(lora_pred)},
 'full_finetune':{'best_step':best_step,'dev_conll18_las_percent':100*full_las,
                  'prediction_sha256':sha256(full_pred),'checkpoint_sha256':sha256(best_path)},
 'full_minus_lora_las_points':100*(full_las-lora_las),'tokens_fixed':fixed,'tokens_regressed':regressed,
 'head_errors_fixed':int(((~df.lora_uas)&df.full_uas).sum()),
 'head_errors_regressed':int((df.lora_uas&(~df.full_uas)).sum()),
 'distance_buckets_las_improved':int((distance.LAS_delta_points>0).sum()),
 'distance_buckets_total':len(distance),
 'relations_support_ge_10_las_improved':int((eligible_rel.LAS_delta_points>0).sum()),
 'relations_support_ge_10_total':len(eligible_rel),
 'manual_review_sentences':len(selected),
 'interpretation_warning':'One matched schedule and one seed do not prove a general LoRA capacity limitation; full fine-tuning may have a different optimal learning rate.'}
(comparison/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
display(distance,relation.head(25),sentence_review)
'''

export = r'''# Package all reports and the full model.  The complete archive may be over 1 GB.
run_record={'config':CFG,'seed':SEED,'software':{'stanza':'1.14.0','transformers':'4.56.2','huggingface_hub':'0.34.4'},
            'input_hashes':{'train':SPLIT_SHA256['train'],'dev':SPLIT_SHA256['dev'],
                            'train_predposlemma':cache_hashes['train'],'dev_predposlemma':cache_hashes['dev']},
            'hf_revision':HF_COMMIT,'test_evaluated':False}
(OUT/'run_record.json').write_text(json.dumps(run_record,ensure_ascii=False,indent=2),encoding='utf-8')

small=WORK/'analysis_bundle'; small.mkdir(exist_ok=True)
for p in comparison.iterdir(): shutil.copy2(p,small/p.name)
for p in (OUT/'full_dev_history.json',OUT/'run_record.json',full_pred): shutil.copy2(p,small/p.name)
small_zip=shutil.make_archive('/content/yue_full_vs_lora_dev_analysis','zip',root_dir=small)

complete=WORK/'complete_bundle'; complete.mkdir(exist_ok=True)
for p in small.iterdir(): shutil.copy2(p,complete/p.name)
shutil.copy2(best_path,complete/best_path.name)
complete_zip=shutil.make_archive('/content/yue_full_finetune_complete','zip',root_dir=complete)
print('analysis archive:',small_zip,sha256(small_zip))
print('complete archive:',complete_zip,sha256(complete_zip))
print('The complete archive includes the dev-selected full-fine-tuned model. Keep its printed SHA-256.')
files.download(small_zip)
files.download(complete_zip)
'''

cells = [
    md("""# Cantonese ELECTRA-large: full fine-tuning control vs LoRA

This Run-all notebook performs one controlled **full fine-tuning** run from the original Stanza 1.14 Mandarin ELECTRA dependency checkpoint. It updates the complete Transformer and parsing layers. POS and lemma processors remain frozen, and the exact custom train/dev split, predicted POS/lemma inputs, learning rates, maximum steps, evaluation interval, patience, seed, and dev CoNLL-2018 LAS selection rule match the prior LoRA run.

The notebook does **not** evaluate test. It uploads the existing `yue_dev_diagnostics.zip` only to obtain the verified LoRA dev prediction, compares LoRA and full fine-tuning token by token, and exports a deterministic 36-sentence manual review sample covering distance ranges, principal low-scoring relations, fixed errors, regressions, and persistent errors.

Use an A100 GPU if available. Full ELECTRA-large fine-tuning and its optimizer require much more memory and storage than LoRA. A single matched run answers whether full fine-tuning wins under this schedule; it cannot establish a universal LoRA capacity limitation."""),
    code(install), code(setup), code(download_data), code(resolve_hf), code(download_stanza), code(tag_data),
    code(full_init), code(full_train),
    md("""## Dev-only comparison and manual review

When prompted, upload the exact `yue_dev_diagnostics.zip` produced previously. The comparison uses its verified LoRA dev prediction; it does not retrain LoRA and does not inspect test.

In the exported manual-review CSV, fill `manual_primary_diagnosis` / `manual_error_class` with categories such as `local attachment`, `clause scope`, `repair/disfluency`, `coordination`, `relation only`, or `annotation question`. The notebook deliberately does not infer these linguistic diagnoses from dependency distance alone."""),
    code(compare), code(export),
]

notebook = {
    "cells": cells,
    "metadata": {"accelerator": "GPU", "colab": {"name": "yue_full_finetune_vs_lora.ipynb", "provenance": []},
                 "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3"}},
    "nbformat": 4, "nbformat_minor": 5,
}

dest = ROOT / "notebooks/yue_full_finetune_vs_lora.ipynb"
dest.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")

pack_dir = ROOT / "colab/yue_full_finetune_vs_lora"
pack_dir.mkdir(parents=True,exist_ok=True)
(pack_dir / dest.name).write_bytes(dest.read_bytes())
(pack_dir / "README.txt").write_text(
    "Upload yue_full_finetune_vs_lora.ipynb to Google Colab, choose an A100 GPU if available, "
    "and Run all. When prompted, upload the exact yue_dev_diagnostics.zip from the prior run. "
    "The notebook trains only on train, selects only on dev, and never evaluates test.\n",
    encoding="utf-8")
print(dest)
print(pack_dir)
