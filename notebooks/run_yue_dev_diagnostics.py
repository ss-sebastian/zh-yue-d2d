# DEV ONLY: error attribution and gold-POS/morph diagnostic.
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
