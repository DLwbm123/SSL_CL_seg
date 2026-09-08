"""GT-free fixed-mass API, followed by a separate scoring boundary."""
import hashlib
from collections import defaultdict
import numpy as np
from .core import stable_seed
from experiments.lcrseg.ssl_foundation_v0_1.diagnostics import quality

def matched_masks(conf,pas,student_pmax,teacher_pmax,teacher_label,key):
    arrays=[np.asarray(x) for x in (conf,pas,student_pmax,teacher_pmax,teacher_label)]
    C,M,cs,ct,yt=arrays
    if C.dtype!=bool or M.dtype!=bool or any(x.shape!=C.shape for x in arrays) or np.any(M&~C):raise ValueError('invalid masks')
    if not np.isfinite(cs).all() or not np.isfinite(ct).all() or not np.isin(yt,(0,1,2)).all():raise ValueError('invalid prediction')
    masks={'PAS_NATIVE':M.copy(),'CONF_TOPK_MATCHED':np.zeros_like(C)}
    for j in range(4):masks[f'RANDOM_MATCHED_{j}']=np.zeros_like(C)
    count=[];rank=np.minimum(cs,ct).ravel()
    for cls in range(3):
        candidates=np.flatnonzero((C&(yt==cls)).ravel());k=int((M&(yt==cls)).sum());count.append(dict(class_id=cls,K=k,candidates=len(candidates)))
        order=np.lexsort((candidates,-rank[candidates]))
        masks['CONF_TOPK_MATCHED'].ravel()[candidates[order[:k]]]=True
        for j in range(4):
            rng=np.random.default_rng(stable_seed(*key,'mask_audit',cls,j))
            masks[f'RANDOM_MATCHED_{j}'].ravel()[rng.permutation(candidates)[:k]]=True
        assert all(int((mask&(yt==cls)).sum())==k for mask in masks.values())
    seal=hashlib.sha256()
    for name,mask in masks.items():seal.update(name.encode());seal.update(mask.tobytes())
    return masks,count,seal.hexdigest()

def likelihood(correct_C,error_C,correct_M,error_M):
    a=correct_M/correct_C if correct_C else None;b=error_M/error_C if error_C else None
    if a is None or b is None:return None,'UNDEFINED_CONDITION_SUPPORT',a,b
    if b==0:return (None,'POSITIVE_INFINITY',a,b) if a>0 else (None,'UNDEFINED_ZERO_OVER_ZERO',a,b)
    return a/b,'FINITE',a,b

def evaluate_mass(masks,count,seal,C,ys,yt,gt):
    # Masks/K/ranking were already materialized; evaluator is the only GT consumer.
    out=[];v=gt!=255
    for model,pred,other in [('teacher',yt,ys),('student',ys,yt)]:
        valid_counts={name:[int((mask&v&(yt==c)).sum()) for c in range(3)] for name,mask in masks.items()}
        for name,mask in masks.items():
            for row in quality(pred,other,gt,mask):
                c=row['class_id'];cm=C&v&(yt==c);mm=masks['PAS_NATIVE']&v&(yt==c)
                cc=int((cm&(yt==gt)).sum());ec=int((cm&(yt!=gt)).sum());mc=int((mm&(yt==gt)).sum());me=int((mm&(yt!=gt)).sum())
                lr,status,a,b=likelihood(cc,ec,mc,me)
                out.append(dict(method='RANDOM_MATCHED' if name.startswith('RANDOM') else name,permutation=int(name[-1]) if name.startswith('RANDOM') else 0,model=model,**row,errors=row['accepted']-row['correct'],geometry_K=count[c]['K'],geometry_C=count[c]['candidates'],teacher_class_scoring_count=valid_counts[name][c],effective_teacher_class_counts_equal=all(x[c]==valid_counts['PAS_NATIVE'][c] for x in valid_counts.values()),GT_free_mask_seal=seal,correct_C=cc,error_C=ec,correct_M=mc,error_M=me,LR_sim_given_C=lr,LR_status=status,keep_given_correct_C=a,keep_given_error_C=b))
    return out

def aggregate(records,keys):
    """Permutation -> prediction draw -> patient equal means; pooled counts separate."""
    groups=defaultdict(list)
    for row in records:groups[tuple(row[k] for k in keys)].append(row)
    result=[]
    numeric_schema={k for r in records for k,v in r.items() if v is None or isinstance(v,(bool,int,float,np.integer,np.floating))}
    ignored=set(keys)|{'patient','repeat','permutation','GT_free_mask_seal'}
    for key,rs in groups.items():
        patients=defaultdict(lambda:defaultdict(list))
        for r in rs:patients[r['patient']][r['repeat']].append(r)
        out=dict(zip(keys,key));out.update(n_patients=len(patients),n_draws=sum(len(x) for x in patients.values()),n_permutation_rows=len(rs))
        numeric=sorted(numeric_schema-ignored)
        for field in numeric:
            means=[]
            for draws in patients.values():
                dm=[]
                for perm in draws.values():
                    vv=[float(r[field]) for r in perm if isinstance(r.get(field),(bool,int,float,np.integer,np.floating)) and np.isfinite(r[field])]
                    if vv:dm.append(float(np.mean(vv)))
                if dm:means.append(float(np.mean(dm)))
            out['patient_mean_'+field]=float(np.mean(means)) if means else None
            out['n_patients_defined_'+field]=len(means)
            if field in ('accepted','correct','errors','predicted','gt_pixels','valid_pixels','geometry_K','geometry_C','correct_C','error_C','correct_M','error_M','teacher_class_scoring_count','bin_count','confidence_sum','correct_count'):
                out['pooled_'+field]=sum(float(r[field]) for r in rs)
        if 'LR_status' in rs[0]:
            out['LR_positive_infinity_rows']=sum(r['LR_status']=='POSITIVE_INFINITY' for r in rs);out['LR_undefined_rows']=sum(r['LR_status'].startswith('UNDEFINED') for r in rs)
            lr,status,a,b=likelihood(*(out['pooled_'+f] for f in ('correct_C','error_C','correct_M','error_M')));out.update(pooled_LR_sim_given_C=lr,pooled_LR_status=status)
        if 'pooled_bin_count' in out:
            n=out['pooled_bin_count'];out['pooled_mean_confidence']=out['pooled_confidence_sum']/n if n else None;out['pooled_accuracy']=out['pooled_correct_count']/n if n else None
        out['unit']='null permutations then draws averaged within patient; pooled counts are replicated draw-pixels'
        result.append(out)
    return result
