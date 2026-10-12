"""One fixed within-class permutation counterexample; zero optimizer/model/patient calls."""
import json
import math
from pathlib import Path

def dot(a,b):
    return sum(x*y for x,y in zip(a,b))
def unit(a):
    n=math.sqrt(dot(a,a));return tuple(x/n for x in a)
def prototype(features,indices):
    return unit(tuple(sum(features[i][d] for i in indices)/len(indices) for d in range(2)))
def check():
    teacher=[(-1.,0.),(-.8,.6),(1.,0.),(.8,.6),(0.,1.),(.6,.8)]
    student=[teacher[i] for i in [1,0,3,2,5,4]]
    leaf=[[0,1],[2,3],[4,5]];parent=[[2,3,4,5]]
    proto_loss=sum(sum(max(0.,1-dot(prototype(teacher,ix),prototype(student,ix))) for ix in group)/len(group) for group in [leaf,parent])/2
    pixel_loss=sum(sum(1-dot(unit(teacher[i]),unit(student[i])) for i in ix)/len(ix) for ix in leaf)/3
    assert proto_loss<1e-12 and abs(pixel_loss-.2)<1e-12
    assert all(teacher[i]!=student[i] for i in range(6))
    return dict(status='PASS_ANALYTIC_NOT_NATIVE_QUALIFICATION',prototype_orientation_loss=proto_loss,class_balanced_pixel_orientation_loss=pixel_loss,changed_pixels=6,label_preserving_permutation=True,optimizer_calls=0,model_calls=0,autograd_vjps=0,patient_calls=0,gpu_calls=0,formal_advancement=False)
if __name__=='__main__':
    result=check();Path(__file__).with_name('ANALYTIC_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
