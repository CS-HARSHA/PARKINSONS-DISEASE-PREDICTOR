"""WP8: Leakage-safe cross-cohort evaluation shared by every model."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv,json
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix
from . import config
from .features import macd
@dataclass
class EvalResult:
    name:str; n_subjects:int; auc_pooled:float; auc_by_direction:dict; ci_low:float; ci_high:float; subject_table:list[dict]; metadata:dict

@dataclass
class SubjectExample:
    subject_id:str; cohort:str; label:int; feature:float

def aggregate_subjects(sessions, feature_fn=macd):
    grouped={}
    for s in sessions: grouped.setdefault((s.cohort,s.subject_id),[]).append(s)
    out=[]
    for (cohort,sid), rows in sorted(grouped.items()):
        vals=[feature_fn(np.asarray(s.hold_times_s)) for s in rows]; vals=[v for v in vals if np.isfinite(v)]
        if vals: out.append(SubjectExample(sid,cohort,rows[0].label,float(np.mean(vals))))
    return out

def cross_cohort_evaluation(examples):
    rows=list(examples); by={c:[r for r in rows if r.cohort==c] for c in config.COHORTS.values()}; predictions=[]; folds=[]
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score
    for train_cohort,test_cohort in [(config.COHORTS['early_pd'],config.COHORTS['de_novo_pd']),(config.COHORTS['de_novo_pd'],config.COHORTS['early_pd'])]:
        train,test=by[train_cohort],by[test_cohort]
        if {r.subject_id for r in train}&{r.subject_id for r in test}: raise AssertionError('Subject leakage detected between train and test')
        model=LogisticRegression(max_iter=2000,random_state=config.SEED).fit(np.asarray([[r.feature] for r in train]),[r.label for r in train])
        for r in test:
            score=float(model.predict_proba([[r.feature]])[0,1]); predictions.append({'subject_id':r.subject_id,'cohort':r.cohort,'label':r.label,'score':r.feature,'model_score':score,'fold':train_cohort})
        folds.append({'train_cohort':train_cohort,'test_cohort':test_cohort,'n_train':len(train),'n_test':len(test)})
    y=np.asarray([r['label'] for r in predictions]); s=np.asarray([r['score'] for r in predictions]); cut=youden_cutoff(y,s); return {'n_subjects':len(predictions),'auc':float(roc_auc_score(y,s)),'auc_ci_95_subject_bootstrap':bootstrap_auc_ci(y,s),'youden_cutoff_pooled_descriptive':cut,'accuracy_at_pooled_descriptive_cutoff':float(accuracy_score(y,s>=cut)),'confusion_matrix_at_pooled_descriptive_cutoff':confusion_matrix(y,s>=cut).tolist(),'folds':folds,'predictions':predictions}

def _subject_scores(sessions,scores):
    grouped={}
    for s,v in zip(sessions,scores):
        if np.isfinite(v): grouped.setdefault((s.subject_id,s.cohort),[]).append((s.label,float(v)))
    rows=[]
    for (sid,cohort),vals in sorted(grouped.items()): rows.append({'subject_id':sid,'cohort':cohort,'label':vals[0][0],'score':float(np.mean([v for _,v in vals]))})
    return rows

def bootstrap_auc_ci(y,s,reps=2000,seed=42):
    rng=np.random.default_rng(seed); vals=[]; y=np.asarray(y); s=np.asarray(s); by={0:np.where(y==0)[0],1:np.where(y==1)[0]}
    for _ in range(reps):
        idx=np.r_[rng.choice(by[0],len(by[0]),replace=True),rng.choice(by[1],len(by[1]),replace=True)]
        vals.append(roc_auc_score(y[idx],s[idx]))
    return tuple(np.quantile(vals,[.025,.975]))
def youden_cutoff(y,s):
    fpr,tpr,th=roc_curve(y,s); return float(th[np.argmax(tpr-fpr)])
def confusion_at_cutoff(y,s,cut):
    yp=(np.asarray(s)>=cut).astype(int); cm=confusion_matrix(y,yp,labels=[0,1]); tn,fp,fn,tp=cm.ravel(); return {'threshold':float(cut),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'sensitivity':float(tp/(tp+fn)) if tp+fn else float('nan'),'specificity':float(tn/(tn+fp)) if tn+fp else float('nan'),'balanced_accuracy':float((tp/(tp+fn)+tn/(tn+fp))/2) if tp+fn and tn+fp else float('nan')}
def cross_cohort_evaluate(model_factory,sessions,n_boot=2000,seed=42,test_transform=None,name='model'):
    sessions=list(sessions); cohorts=sorted(set(s.cohort for s in sessions)); assert len(cohorts)==2
    all_subject=[]; dirs={}
    for train_cohort,test_cohort in [(cohorts[0],cohorts[1]),(cohorts[1],cohorts[0])]:
        train=[s for s in sessions if s.cohort==train_cohort]; test=[s for s in sessions if s.cohort==test_cohort]
        assert not ({s.subject_id for s in train}&{s.subject_id for s in test})
        model=model_factory(); model.fit(train)
        test2=test_transform(test,train_cohort,test_cohort) if test_transform else test
        rows=_subject_scores(test2,model.scores(test2))
        for r in rows:r['direction']=f'{train_cohort}->{test_cohort}'
        all_subject.extend(rows); dirs[f'{train_cohort}->{test_cohort}']=float(roc_auc_score([r['label'] for r in rows],[r['score'] for r in rows]))
    y=np.array([r['label'] for r in all_subject]); s=np.array([r['score'] for r in all_subject]); ci=bootstrap_auc_ci(y,s,n_boot,seed)
    return EvalResult(name,len(all_subject),float(roc_auc_score(y,s)),dirs,ci[0],ci[1],all_subject,{'n_features':1,'notes':'Session scores averaged per subject; subject-safe cross-cohort evaluation.'})
def save_result(result,name=None):
    name=name or result.name; config.METRICS_DIR.mkdir(parents=True,exist_ok=True)
    data={'name':result.name,'n_subjects':result.n_subjects,'auc_pooled':result.auc_pooled,'auc_by_direction':result.auc_by_direction,'ci_low':result.ci_low,'ci_high':result.ci_high,'metadata':result.metadata}
    (config.METRICS_DIR/f'{name}.json').write_text(json.dumps(data,indent=2));
    with (config.METRICS_DIR/f'{name}_subject_predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['subject_id','cohort','label','direction','score']); w.writeheader(); w.writerows(result.subject_table)
    return data
