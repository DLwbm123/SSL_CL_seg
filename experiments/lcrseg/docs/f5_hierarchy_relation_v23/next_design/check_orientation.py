"""One fixed analytic counterexample; no model, optimizer, patient or GPU calls."""
import json
import math
from pathlib import Path

def dot(a,b):
    return sum(x*y for x,y in zip(a,b))
def norm(a):
    return math.sqrt(dot(a,a))
def sub(a,b):
    return tuple(x-y for x,y in zip(a,b))
def rotate(a):
    return a[2],a[0],a[1]
def angle(a,b,c):
    u=sub(a,b);v=sub(c,b)
    return dot(u,v)/(norm(u)*norm(v))
def check():
    teacher=[(1.,0.,0.),(0.,1.,0.),(0.,0.,1.),(0.,2**-.5,2**-.5)]
    student=list(map(rotate,teacher))
    edges=[(0,3),(1,2),(1,3),(2,3)]
    td=[norm(sub(teacher[i],teacher[j])) for i,j in edges]
    sd=[norm(sub(student[i],student[j])) for i,j in edges]
    distance_error=max(abs(x/(sum(td)/4)-y/(sum(sd)/4)) for x,y in zip(td,sd))
    angle_error=abs(angle(teacher[1],teacher[3],teacher[2])-angle(student[1],student[3],student[2]))
    teacher_pred=[max(range(3),key=lambda k:x[k]) for x in teacher[:3]]
    student_pred=[max(range(3),key=lambda k:x[k]) for x in student[:3]]
    orientation_loss=sum(1-dot(a,b) for a,b in zip(teacher[:3],student[:3]))/3
    assert distance_error<1e-12 and angle_error<1e-12
    assert teacher_pred==[0,1,2] and student_pred==[1,2,0] and orientation_loss==1
    return dict(status='PASS_ANALYTIC_NOT_NATIVE_QUALIFICATION',normalized_distance_error=distance_error,angle_error=angle_error,fixed_identity_readout_teacher=teacher_pred,fixed_identity_readout_student=student_pred,leaf_orientation_loss=orientation_loss,optimizer_calls=0,model_calls=0,autograd_vjps=0,patient_calls=0,gpu_calls=0,formal_advancement=False)
if __name__=='__main__':
    result=check()
    Path(__file__).with_name('ANALYTIC_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
