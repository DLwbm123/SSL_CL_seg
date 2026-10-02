"""Paired predictions; truth is optional and never inferred for U."""
import torch

GROUPS=('ALL','ADMITTED','AGREE','DISAGREE')


class Counts:
    def __init__(self,labeled):
        self.labeled=labeled
        self.joint={g:torch.zeros((3,3),dtype=torch.int64) for g in GROUPS}
        self.truth={g:{m:torch.zeros((3,3),dtype=torch.int64) for m in ('teacher','student')} for g in GROUPS} if labeled else None

    def add(self,q,p,valid,y=None):
        assert q.shape==p.shape and q.ndim==4 and q.shape[1]==3
        assert valid.shape==q.shape[:1]+q.shape[2:] and valid.dtype==torch.bool
        assert torch.isfinite(q).all() and torch.isfinite(p).all()
        assert (y is not None)==self.labeled
        if y is not None:
            assert y.shape==valid.shape and ((y==255)|((y>=0)&(y<3))).all()
            valid=valid&(y!=255)
        conf,teacher=q.detach().max(1);student=p.detach().argmax(1)
        admitted=valid&(conf>.7);agree=teacher==student
        masks=dict(ALL=valid,ADMITTED=admitted,AGREE=admitted&agree,DISAGREE=admitted&~agree)
        for g,mask in masks.items():
            t,s=teacher[mask],student[mask]
            self.joint[g]+=torch.bincount(t*3+s,minlength=9).reshape(3,3).cpu()
            if y is not None:
                for m,pred in (('teacher',t),('student',s)):
                    self.truth[g][m]+=torch.bincount(y[mask]*3+pred,minlength=9).reshape(3,3).cpu()

    def finish(self):
        assert torch.equal(self.joint['ADMITTED'],self.joint['AGREE']+self.joint['DISAGREE'])
        total=int(self.joint['ALL'].sum());admitted=int(self.joint['ADMITTED'].sum());rows=[]
        for g in GROUPS:
            cm=self.joint[g];n=int(cm.sum())
            row=dict(group=g,pixels=n,fraction_all=n/total if total else None,
                     fraction_admitted=n/admitted if admitted and g!='ALL' else None)
            for c in range(3):row[f'teacher_predicted_{c}']=int(cm[c].sum())
            for model in ('teacher','student'):
                truth=self.truth[g][model] if self.labeled else None
                errors=[]
                row[model+'_errors']=int(truth.sum()-truth.diag().sum()) if truth is not None else None
                row[model+'_error_rate']=row[model+'_errors']/n if truth is not None and n else None
                for c in range(3):
                    support=int(truth[c].sum()) if truth is not None else None
                    error=1-int(truth[c,c])/support if support else None
                    row[f'{model}_true_{c}_support']=support;row[f'{model}_true_{c}_error']=error
                    if error is not None:errors.append(error)
                row[model+'_balanced_error']=sum(errors)/len(errors) if errors else None
                row[model+'_supported_classes']=len(errors) if self.labeled else None
            rows.append(row)
        a,b,d=rows[1:]
        common=[c for c in range(3) if a[f'teacher_true_{c}_error'] is not None and b[f'teacher_true_{c}_error'] is not None]
        delta=dict(agreement_fraction_admitted=b['fraction_admitted'],common_true_classes=len(common) if self.labeled else None,
                   teacher_error_delta=b['teacher_error_rate']-a['teacher_error_rate'] if b['teacher_error_rate'] is not None else None,
                   teacher_common_balanced_delta=sum(b[f'teacher_true_{c}_error']-a[f'teacher_true_{c}_error'] for c in common)/len(common) if common else None,
                   rejected_error_enrichment=d['teacher_error_rate']-a['teacher_error_rate'] if d['teacher_error_rate'] is not None else None,
                   random_expected_retained_errors=a['teacher_errors']*b['pixels']/admitted if self.labeled and admitted else None)
        if self.labeled:
            for m in ('teacher','student'):
                assert torch.equal(self.truth['ADMITTED'][m],self.truth['AGREE'][m]+self.truth['DISAGREE'][m])
        raw=dict(joint={g:v.tolist() for g,v in self.joint.items()},truth={g:{m:v.tolist() for m,v in values.items()} for g,values in self.truth.items()} if self.labeled else None)
        return rows,delta,raw


def check():
    # One shared wrong prediction, one rejected wrong, two correct, one ignored.
    def probs(labels):return torch.nn.functional.one_hot(torch.tensor(labels),3).float().T[None,:,None,:]*.8+.2/3
    q,p=probs([0,1,2,0,1]),probs([0,0,2,0,1]);y=torch.tensor([[[0,0,0,0,255]]]);v=torch.ones_like(y,dtype=torch.bool)
    a=Counts(True);a.add(q,p,v,y);rows,d,raw=a.finish()
    assert [r['pixels'] for r in rows]==[4,4,3,1]
    assert rows[2]['teacher_errors']==1 and rows[3]['teacher_errors']==1
    assert abs(d['teacher_error_delta']+1/6)<1e-8 and d['random_expected_retained_errors']==1.5
    assert raw['truth']['ALL']['teacher']==[[2,1,1],[0,0,0],[0,0,0]]
    u=Counts(False);u.add(q,p,v);rr,dd,raw=u.finish()
    assert rr[0]['pixels']==5 and raw['truth'] is None and dd['teacher_error_delta'] is None
    z=Counts(True);z.add(torch.full_like(q,1/3),p,v,y);rr,dd,_=z.finish()
    assert rr[1]['pixels']==0 and dd['agreement_fraction_admitted'] is None
    assert dd['teacher_common_balanced_delta'] is None
    print('PASS paired partitions, shared/rejected errors, ignore label, missing classes, zero admission, U without truth and random-count expectation')


if __name__=='__main__':check()
