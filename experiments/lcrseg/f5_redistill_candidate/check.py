"""No optimizer, model, patient or CUDA access; fixed original-paper parameter."""
import json,os
from pathlib import Path
import torch
from .reference import power_reference
from ..f5_class_reference_v13.model import class_reference
from ..f5_entry_gradient_v12.model import halfspace


def main():
    root=Path(os.environ['EXEC_ROOT']);assert not (root/'MATH_CHECKS.json').exists()
    plan=json.loads((root/'PLAN.json').read_text());assert plan['lambda']==2/3 and plan['optimizer_updates']==0
    torch.set_num_threads(2);vjps=0;calls=0
    def evaluate(prob,target,power=True):
        nonlocal calls,vjps
        z=torch.tensor(prob,dtype=torch.double).T[None,:,None,:].log().requires_grad_(True)
        p=torch.tensor(target,dtype=torch.double).T[None,:,None,:].requires_grad_(True)
        valid=torch.ones(z.shape[0],*z.shape[2:],dtype=torch.bool)
        loss=power_reference(z,p,valid) if power else class_reference(z,p,valid)[0];calls+=1
        grad,teacher=torch.autograd.grad(loss,(z,p),allow_unused=True);vjps+=1
        assert teacher is None and torch.isfinite(grad).all()
        return float(loss),grad.detach(),z.detach(),p.detach()
    cases=[]
    for name,p,q in plan['fixtures']:
        value,g,z,t=evaluate([q],[p]);kl,k,_,_=evaluate([q],[p],False)
        power=t.pow(5/3)*z.softmax(1).pow(-2/3)
        analytic=.6*(z.softmax(1)*power.sum(1,keepdim=True)-power)
        assert torch.allclose(g,analytic,atol=1e-10,rtol=1e-10)
        expected=.9*(sum(a**(5/3)*b**(-2/3) for a,b in zip(p,q))-1)
        assert abs(value-expected)<1e-10*max(abs(expected),1)
        cases.append(dict(name=name,teacher=p,student=q,power_loss=value,KL_loss=kl,power_gradient=g.flatten().tolist(),KL_gradient=k.flatten().tolist(),gradient_norm_ratio=float(g.norm()/k.norm()) if k.norm()>0 else None))
    p=[[.005,.99,.005],[.05,.9,.05]];q=[[.99,.005,.005],[.1,.8,.1]]
    value,g,z,t=evaluate(q,p);kl,k,_,_=evaluate(q,p,False)
    true=torch.tensor([[1.,0.,0.],[0.,1.,0.]],dtype=torch.double).T[None,:,None,:]
    descent=(z.softmax(1)-true)/2
    projected_power,sp=halfspace([descent],[g]);projected_kl,sk=halfspace([descent],[k])
    mixed=dict(same_teacher_argmax_class=True,wrong_correct_pixel_norm_ratio_power=float(g[...,0].norm()/g[...,1].norm()),wrong_correct_pixel_norm_ratio_KL=float(k[...,0].norm()/k[...,1].norm()),normalized_reference_cosine=float((g*k).sum()/(g.norm()*k.norm())),true_gradient_dot_after_power=float((descent*projected_power[0]).sum()),true_gradient_dot_after_KL=float((descent*projected_kl[0]).sum()))
    identity,ig,_,_=evaluate([[.2,.5,.3]],[[.2,.5,.3]])
    assert abs(identity)<1e-12 and ig.abs().max()<1e-12
    z=torch.zeros(1,3,1,1,dtype=torch.double,requires_grad=True);p=torch.tensor([.2,.5,.3],dtype=torch.double).view_as(z)
    empty=power_reference(z,p,torch.zeros(1,1,1,dtype=torch.bool));calls+=1
    eg,=torch.autograd.grad(empty,z);vjps+=1;assert empty==0 and eg.count_nonzero()==0
    invalid=0
    for bad in (p[:,:,:,:0],p*float('nan'),p*0,p-2):
        try:power_reference(z,bad,torch.ones(1,1,1,dtype=torch.bool))
        except (ValueError,TypeError):invalid+=1
        else:raise AssertionError('invalid target accepted')
    try:power_reference(torch.tensor([0.,-2000.,-2000.],dtype=torch.double).view_as(z),torch.tensor([0.,1.,0.],dtype=torch.double).view_as(z),torch.ones(1,1,1,dtype=torch.bool))
    except ValueError:invalid+=1
    else:raise AssertionError('overflow was silently clipped or accepted')
    attenuation=all(x['gradient_norm_ratio']<=1 for x in cases if x['name'].startswith('wrong'))
    result=dict(status='MATH_PASS_NOT_NATIVE_QUALIFICATION',execution_commit=plan['execution_commit'],native_advancement=attenuation,formal_advancement=False,cases=cases,mixed_case=mixed,invalid_inputs_rejected=invalid,successful_kernel_calls=calls,generated_cpu_vjps=vjps,optimizer_updates=0,model_forwards=0,patient_access=0,CUDA=0,conclusion='literal fixed divergence does not attenuate wrong-teacher gradient influence in the fixed generated fixtures' if not attenuation else 'math screen passed; native qualification remains required')
    (root/'MATH_CHECKS.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
