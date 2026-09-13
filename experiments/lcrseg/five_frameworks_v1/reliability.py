import math
import torch
from torch.nn import functional as F
from .kernels import current_prototypes, teacher_pas


class CurrentPrototypes:
    def __init__(self,classes,width):
        self.values=torch.zeros(classes,width)
        self.support=torch.zeros(classes,dtype=torch.bool)

    def estimate(self,teacher,images,labels):
        with torch.no_grad():
            _,before,h=teacher.parts(images)
            mapped=teacher.parent.output_to_feature(labels,h.shape[-2:],True)
            p,s=current_prototypes(h,mapped,3)
        return p,s,before,h,mapped

    def commit(self,p,s):
        self.values=self.values.to(p);self.support=self.support.to(s.device)
        both=s&self.support
        self.values[both]=F.normalize(.9*self.values[both]+.1*p[both],dim=1)
        self.values[s&~self.support]=p[s&~self.support]
        self.support|=s

    def admission(self,teacher,q,h,geometry,confidence=.7,similarity=.5):
        mapped=teacher.parent.feature_to_output(h,q.shape[-2:])
        return teacher_pas(q,mapped,self.values.to(h),self.support.to(h.device),geometry,
                           confidence=confidence,similarity=similarity)

    def uncertainty(self,teacher,q,h,beta):
        entropy=-(torch.xlogy(q,q)).sum(1)/math.log(q.shape[1])
        _,sim=self.admission(teacher,q,h,torch.ones_like(q[:,0],dtype=torch.bool))
        known=self.support.to(q.device)[q.argmax(1)]
        proto=torch.where(known,(1-sim)/2,entropy)
        return ((1-beta)*entropy+beta*proto).detach().clamp(0,1)
