"""Pixel-pooled calibration counts on labeled training images only."""
import math
import torch


class Counts:
    def __init__(self):
        self.cm=torch.zeros((3,3),dtype=torch.int64);self.accepted=self.cm.clone()
        self.bins=torch.zeros(10,dtype=torch.int64);self.correct=torch.zeros(10,dtype=torch.float64)
        self.conf=torch.zeros(10,dtype=torch.float64);self.nll=torch.zeros(3,dtype=torch.float64)

    def add(self,q,y):
        assert q.ndim==4 and q.shape[1]==3 and y.shape==q.shape[:1]+q.shape[2:]
        assert torch.isfinite(q).all() and ((y==255)|((y>=0)&(y<3))).all()
        valid=y!=255;q=q.detach().cpu().permute(0,2,3,1)[valid.cpu()].double();y=y.detach().cpu()[valid.cpu()]
        confidence,pred=q.max(1);admit=confidence>.7;bins=(confidence*10).long().clamp(max=9)
        self.cm+=torch.bincount(y*3+pred,minlength=9).reshape(3,3)
        self.accepted+=torch.bincount(y[admit]*3+pred[admit],minlength=9).reshape(3,3)
        self.bins+=torch.bincount(bins,minlength=10)
        self.correct+=torch.bincount(bins,weights=(pred==y).double(),minlength=10)
        self.conf+=torch.bincount(bins,weights=confidence,minlength=10)
        self.nll+=torch.bincount(y,weights=-q.gather(1,y[:,None]).squeeze(1).clamp_min(1e-8).log(),minlength=3)

    def finish(self):
        n=int(self.cm.sum());na=int(self.accepted.sum());assert n>0
        support=self.cm.sum(1);accepted_support=self.accepted.sum(1)
        recalls=[float(self.cm[c,c]/support[c]) for c in range(3) if support[c]>0]
        errors=[1-float(self.accepted[c,c]/accepted_support[c]) for c in range(3) if accepted_support[c]>0]
        nonzero=self.bins>0
        metrics=dict(valid_pixels=n,admitted_pixels=na,accuracy=float(self.cm.diag().sum())/n,
            balanced_accuracy=sum(recalls)/len(recalls),confidence=float(self.conf.sum())/n,admission=na/n,
            admitted_error=1-float(self.accepted.diag().sum())/na if na else None,
            admitted_class_balanced_error=sum(errors)/len(errors) if errors else None,
            admitted_supported_classes=len(errors),nll=float(self.nll.sum())/n,
            balanced_nll=sum(float(self.nll[c]/support[c]) for c in range(3) if support[c]>0)/len(recalls),
            ece10=float((self.conf[nonzero]-self.correct[nonzero]).abs().sum())/n)
        for c in range(3):
            metrics['class_'+str(c)+'_recall']=float(self.cm[c,c]/support[c]) if support[c]>0 else None
            metrics['class_'+str(c)+'_admitted_error']=1-float(self.accepted[c,c]/accepted_support[c]) if accepted_support[c]>0 else None
        assert all(v is None or math.isfinite(v) for v in metrics.values())
        return metrics,dict(confusion=self.cm.tolist(),admitted_confusion=self.accepted.tolist(),
            reliability_bins=[dict(lower=i/10,upper=(i+1)/10,count=int(self.bins[i]),confidence_sum=float(self.conf[i]),correct=int(self.correct[i])) for i in range(10)])


def check():
    q=torch.tensor([[[[.8,.1,.1,.3]],[[.1,.8,.1,.3]],[[.1,.1,.8,.4]]]])
    y=torch.tensor([[[0,0,2,255]]]);a=Counts();a.add(q,y);m,raw=a.finish()
    assert m['valid_pixels']==3 and m['admitted_pixels']==3 and abs(m['admitted_error']-1/3)<1e-7
    assert abs(m['admitted_class_balanced_error']-.25)<1e-7 and abs(m['ece10']-(.8-2/3))<1e-6
    assert raw['confusion']==[[1,1,0],[0,0,0],[0,0,1]]
    b=Counts();b.add(torch.full((1,3,1,1),1/3),torch.zeros((1,1,1),dtype=torch.long));m,_=b.finish()
    assert m['admitted_error'] is None and m['admitted_class_balanced_error'] is None and m['admitted_supported_classes']==0
    print('PASS confusion, ignore label, class-balanced accepted errors, ECE and zero admission')


if __name__=='__main__':check()
