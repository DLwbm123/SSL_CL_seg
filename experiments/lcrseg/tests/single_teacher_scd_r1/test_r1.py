import gc
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import mpmath as mp
from experiments.lcrseg.single_teacher_scd_v0_1 import engine as old, objectives as original
from experiments.lcrseg.single_teacher_scd_v0_1.evaluate import evaluate
from experiments.lcrseg.tests.single_teacher_scd_v0_1.test_contract import fixture,REFERENCE,DEVICE
from experiments.lcrseg.single_teacher_scd_r1 import engine as new,solver

def high_precision(p,q,y):
    # Frozen FP64 normalized a is interpreted as the exact input to the independent 100-digit problem.
    a=np.asarray(p)-np.eye(len(p))[y];a=a/max(abs(a)) if np.any(a) else a
    with mp.workdps(100):
        aa=list(map(lambda x:mp.mpf(float(x)),a));pp=[mp.mpf(float(x)) for x in p];qq=[mp.mpf(float(x)) for x in q]
        b=sum(x*z for x,z in zip(aa,pp));m=min(aa);d=[x-m for x in aa];v=b-m
        def target(eta):
            w=[qq[i]*mp.exp(-eta*d[i]) for i in range(len(q))]
            return [z/sum(w) for z in w]
        if sum(qq[i]*aa[i] for i in range(len(q)))<=b:return np.asarray(q)
        if v==0:
            w=[qq[i] if d[i]==0 else mp.mpf(0) for i in range(len(q))]
            return np.array([float(z/sum(w)) for z in w])
        assert v>0
        delta=min(z for z in d if z>0);q0=sum(qq[i] for i in range(len(q)) if d[i]==0)
        upper=(mp.log(sum(qq[i]*d[i] for i in range(len(q))))-mp.log(q0)-mp.log(v)+mp.log(2))/delta
        lo,hi=mp.mpf(0),mp.log1p(upper)
        for _ in range(400):
            mid=(lo+hi)/2;r=target(mp.expm1(mid))
            if sum(r[i]*d[i] for i in range(len(q)))>v:lo=mid
            else:hi=mid
        return np.array([float(x) for x in target(mp.expm1(hi))])

def examples():
    values=[
      ([1.,1e-30,8e-48],[1.-2e-8/3,1e-8/3,1e-8/3],0),
      ([1.,1e-100,1e-200],[.3,.4,.3],0),
      ([0.,1.,0.],[.2,.4,.4],1),
      ([.8,.2],[.5,.5],0),
      ([.7,.1,.1,.1],[.25]*4,0),
      ([1.-2e-14,1e-14,1e-14],[.2,.3,.5],0)]
    rng=np.random.default_rng(391)
    values += [(rng.dirichlet([.7]*3).tolist(),rng.dirichlet([.8]*3).tolist(),int(rng.integers(3))) for _ in range(24)]
    return values

class R1Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The pinned upstream module has import-time RNG side effects. Complete that
        # import before constructing any matched synthetic initialization.
        from experiments.lcrseg.di_dmpa_jascl.modeling import _official_probabilistic_classifier
        _official_probabilistic_classifier(REFERENCE,upstream_path=old.UPSTREAM_PATH)
        old.precision()

    def test_analytic_reference_and_permutations(self):
        p=torch.tensor([[1.,1e-30,8e-48]],dtype=torch.double,device=DEVICE)
        q=torch.tensor([[1.-2e-8/3,1e-8/3,1e-8/3]],dtype=torch.double,device=DEVICE)
        with self.assertRaises(FloatingPointError):original.project(p,q,torch.tensor([0],device=DEVICE),torch.tensor([True],device=DEVICE))
        maxerr=0.
        for pp,qq,y in examples():
            p=torch.tensor([pp],dtype=torch.double,device=DEVICE);q=torch.tensor([qq],dtype=torch.double,device=DEVICE)
            rr,conf,info=solver.project(p,q,torch.tensor([y],device=DEVICE),torch.tensor([True],device=DEVICE))
            expected=high_precision(pp,qq,y);err=np.max(np.abs(rr.cpu().numpy()[0]-expected));maxerr=max(maxerr,float(err))
            np.testing.assert_allclose(rr.cpu().numpy()[0],expected,atol=1e-10,rtol=1e-9)
            self.assertLessEqual(float(info["residual"].max()),1e-9)
            perm=list(reversed(range(len(pp))));new_y=perm.index(y)
            rp=solver.project(p[:,perm],q[:,perm],torch.tensor([new_y],device=DEVICE),torch.tensor([True],device=DEVICE))[0]
            self.assertTrue(torch.allclose(rr[:,perm],rp,atol=1e-10,rtol=0))
            no=solver.project(p,q,torch.tensor([y],device=DEVICE),torch.tensor([False],device=DEVICE))[0]
            self.assertTrue(torch.equal(no,q))
        print("HIGH_PRECISION_MAX_TARGET_ERROR",maxerr,flush=True)

    def test_logit_gradient_and_boundary_json(self):
        z=torch.tensor([0.,-46.05170186,-57.56462732],dtype=torch.float32,device=DEVICE).reshape(1,3,1,1).requires_grad_()
        q=torch.tensor([.2,.3,.5],dtype=torch.double,device=DEVICE).reshape(1,3,1,1).requires_grad_()
        y=torch.zeros((1,1,1),dtype=torch.long,device=DEVICE);valid=torch.ones_like(y,dtype=torch.bool)
        loss,stats=new.projected_kd(z,q,y,valid,valid);loss.backward()
        p=z.detach().double().movedim(1,-1).reshape(1,3).log_softmax(-1).exp()
        qt=(1-original.EPS_Q)*q.detach().reshape(1,3)+original.EPS_Q/3
        target=solver.project(p,qt,y.flatten(),valid.flatten())[0]
        self.assertTrue(torch.allclose(z.grad.flatten(),(p-target).float().flatten(),atol=1e-7,rtol=0))
        self.assertIsNone(q.grad);json.dumps(stats,allow_nan=False)
        # Exact simplex boundary has an analytic KKT solution, distinct from fallback.
        p=torch.tensor([[1.,0.,0.]],dtype=torch.double,device=DEVICE)
        q=torch.tensor([[.2,.3,.5]],dtype=torch.double,device=DEVICE)
        rr,conf,info=solver.project(p,q,y.flatten(),valid.flatten())
        self.assertFalse(bool(conf[0]));self.assertTrue(torch.equal(rr,q))
        # A genuine v=0 conflict is exercised through the algebraic helper independently.
        d,v=solver.shifted_constraint(torch.tensor([[1.,0.,0.]],dtype=torch.double,device=DEVICE),
            torch.tensor([[0.,1.,2.]],dtype=torch.double,device=DEVICE))
        self.assertEqual(float(v),0.)
        boundary=solver.boundary_target(q,torch.tensor([[True,True,False]],device=DEVICE))
        self.assertTrue(torch.allclose(boundary,torch.tensor([[.4,.6,0.]],dtype=torch.double,device=DEVICE)))

    def test_objective_identity_and_gas(self):
        batch=dict(image=torch.rand(2,3,32,32,device=DEVICE),label=torch.randint(3,(2,32,32),device=DEVICE))
        batch["geometry"]=torch.ones_like(batch["label"],dtype=torch.bool);u={k:v for k,v in batch.items() if k!="label"}
        outputs=[]
        for mode in ("OLD_D_ZERO_CE","U0","L05","L10"):
            s=old.model(REFERENCE,DEVICE);t=old.second_model(s,"D");opt=old.optimizer_for(s)
            teacher_hash=old.state_hash(t)
            count=dict(forward=0,gas_autograd=0,diagnostic_autograd=0,backward=0,optimizer_steps=0)
            proto=torch.zeros(3,16,device=DEVICE);support=torch.zeros(3,dtype=torch.bool,device=DEVICE)
            if mode=="OLD_D_ZERO_CE":
                forced=lambda z,f,p,s,g:(z.detach().argmax(1),g,z.detach().softmax(1))
                with patch.object(old,"lambda_u",lambda epoch:0.),patch.object(old,"pas",forced):
                    info=old.train_step(s,t,opt,batch,u,proto,support,"D",1,20,0,count)
            else:
                forced=lambda z,f,p,s,g:(z.detach().argmax(1),g,z.detach().softmax(1))
                with patch.object(original,"pas",forced):
                    info=new.train_step(s,t,opt,batch,u if mode=="U0" else None,proto,support,mode,1,20,0,count)
                expected=info["sup"]+info["labeled_kd_coefficient"]*info["labeled_kd"]+info["unlabeled_kd_coefficient"]*info["unlabeled_kd"]
                self.assertAlmostEqual(info["loss"],expected,places=12)
                self.assertEqual(info["actual_lambda_u"],0.);self.assertEqual(info["nominal_lambda_u"],.5)
            if mode in ("OLD_D_ZERO_CE","U0"):self.assertGreater(info["ssl"],0.)
            self.assertEqual(old.state_hash(t),teacher_hash);self.assertEqual(count["optimizer_steps"],1)
            self.assertEqual(count["forward"],5 if mode in ("OLD_D_ZERO_CE","U0") else 2)
            outputs.append((old.state_hash(s),s.decoder.conv_logit.grad_update.detach().cpu().clone()))
            del s,t,opt;gc.collect()
        self.assertEqual(outputs[0][0],outputs[1][0]);self.assertTrue(torch.equal(outputs[0][1],outputs[1][1]))
        # The two label-only recipes differ only in their explicit coefficient.
        vals=[torch.tensor(x,dtype=torch.double,requires_grad=True) for x in (1.,2.,3.,4.)]
        l,_=new.objective(*vals,"L05",100);l.backward()
        self.assertEqual([x.grad.item() if x.grad is not None else 0 for x in vals],[1.,0.,.5,0.])
        with self.assertRaises(PermissionError):new.CountOnly(5,1)[0]

    def test_new_lifecycle_resume_and_forbidden_u(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);expected=fixture(root/"data")
            kw=dict(data=root/"data",reference=REFERENCE,device=DEVICE,expected=expected,shape=(32,32),qualification=True,epochs=2)
            old.run_stage(**kw,output=root/"common",arm="common",stage=0)
            parent=root/"common/student_latest.pt"
            for arm in ("U0","L05","L10"):
                folder=root/arm;folder.mkdir()
                a=new.run_stage(**kw,arm=arm,stage=1,parent=parent,output=folder/"stage1")
                self.assertEqual(a["r1_arm"],arm);self.assertEqual(a["max_full_models"],2)
                if arm in ("L05","L10"):self.assertEqual(a["unlabeled_images_opened"],0)
                evaluate(data=root/"data",reference=REFERENCE,checkpoint=folder/"stage1/student_latest.pt",
                    output=folder/"stage1/val.json",device=DEVICE,expected=expected,shape=(32,32))
                b=new.run_stage(**kw,arm=arm,stage=2,parent=folder/"stage1/student_latest.pt",output=folder/"stage2")
                self.assertEqual(b["parent_hash"],a["student_hash"])
            # Resume U0 at nonzero prototype age and preserve RNG/optimizer/GAS.
            kw["epochs"]=12
            full=new.run_stage(**kw,arm="U0",stage=1,parent=parent,output=root/"full")
            new.run_stage(**kw,arm="U0",stage=1,parent=parent,output=root/"resumed",stop_after_epoch=11)
            resumed=new.run_stage(**kw,arm="U0",stage=1,parent=parent,output=root/"resumed",resume=True)
            self.assertEqual(full["student_hash"],resumed["student_hash"])
            with self.assertRaises(PermissionError):
                new.run_stage(**kw,arm="L05",stage=2,parent=root/"U0/stage1/student_latest.pt",output=root/"L05/wrong")

    def test_e_training_call_chain(self):
        batch=dict(image=torch.rand(2,3,32,32,device=DEVICE),label=torch.randint(3,(2,32,32),device=DEVICE))
        batch["geometry"]=torch.ones_like(batch["label"],dtype=torch.bool)
        u={k:v for k,v in batch.items() if k!="label"}
        results={}
        for name in ("C_before","D_before","E_R1","C_after","D_after"):
            student=old.model(REFERENCE,DEVICE);teacher=old.second_model(student,"D");optimizer=old.optimizer_for(student)
            th=old.state_hash(teacher);c=dict(forward=0,gas_autograd=0,diagnostic_autograd=0,backward=0,optimizer_steps=0)
            args=(student,teacher,optimizer,batch,u,torch.zeros(3,16,device=DEVICE),torch.zeros(3,dtype=torch.bool,device=DEVICE))
            if name=="E_R1":new.train_step(*args,name,1,11,0,c)
            else:old.train_step(*args,name[0],1,11,0,c)
            results[name]=old.state_hash(student);self.assertEqual(th,old.state_hash(teacher))
            self.assertEqual(c["optimizer_steps"],1)
            del student,teacher,optimizer,args;gc.collect()
        self.assertEqual(results["C_before"],results["C_after"])
        self.assertEqual(results["D_before"],results["D_after"])
