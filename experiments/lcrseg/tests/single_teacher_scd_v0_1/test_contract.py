"""Independent numerical counterexamples and synthetic deployment/training fixtures."""
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from scipy.optimize import minimize
import torch
from torch.nn import functional as F

from experiments.lcrseg.di_dmpa_gate1.binding import sha256
from experiments.lcrseg.care_hr_v0_7_1.scoring_r1 import case_metrics
from experiments.lcrseg.single_teacher_scd_v0_1 import objectives as obj
from experiments.lcrseg.single_teacher_scd_v0_1.data import CurrentData, DOMAINS, batches, geometry, strong
from experiments.lcrseg.single_teacher_scd_v0_1.engine import model, second_model, optimizer_for, train_step, state_hash, run_stage, validate_parent
from experiments.lcrseg.single_teacher_scd_v0_1.evaluate import aggregate, evaluate, infer
from experiments.lcrseg.single_teacher_scd_v0_1.execute import child

REFERENCE=os.environ.get("SCD_REFERENCE","/Users/bominwang/Desktop/codes/SSL_CL_seg/experiments/lcrseg/third_party/JASCL_REFERENCE")
DEVICE=torch.device(os.environ.get("SCD_TEST_DEVICE","cpu"))


def reference_projection(p,q,y):
    a=p-np.eye(len(p))[y]
    if np.dot(p-q,a)>=0:return q.copy()
    scale=np.max(np.abs(a));a=a/scale;b=np.dot(a,p)
    def r(eta):
        z=np.log(q)-eta*a;z-=np.max(z)
        z=np.exp(z);return z/z.sum()
    lo,hi=0.,1.
    while np.dot(a,r(hi))>b:
        hi*=2
        if hi>2**60:raise FloatingPointError("reference bracket")
    for _ in range(80):
        mid=(lo+hi)/2
        if np.dot(a,r(mid))>b:lo=mid
        else:hi=mid
    return r(hi)


def fixture(root):
    """Construct literal canonical layout independently, plus attractive invalid decoys."""
    root=Path(root);rows=[]
    for d,domain in enumerate(DOMAINS):
        for role,n in (("train_labeled",3),("train_unlabeled",5),("val",2),("test",1)):
            for i in range(n):
                ident=f"d{d}_{role}_{i}"
                x=np.zeros((3,32,32),dtype=np.uint8)
                y=np.zeros((32,32),dtype=np.uint8);y[5:26,5:26]=1;y[12:20,12:20]=2
                x[0]=y*80;x[1]=np.arange(32,dtype=np.uint8)[:,None];x[2]=50
                rel=f"images/{ident}.h5";path=root/"h5"/"v1"/rel;path.parent.mkdir(parents=True,exist_ok=True)
                with h5py.File(path,"w") as f:f["image"]=x
                decoy=root/rel;decoy.parent.mkdir(parents=True,exist_ok=True);decoy.write_bytes(b"wrong root")
                lp="";lh=""
                if role!="train_unlabeled":
                    lp=f"labels/{ident}.h5";path_y=root/"h5"/"v1"/lp;path_y.parent.mkdir(parents=True,exist_ok=True)
                    with h5py.File(path_y,"w") as f:f["label"]=y
                    lh=sha256(path_y)
                rows.append(dict(case_id=ident,patient_id=ident,dataset="fundus",split_seed="0",site_or_vendor=domain,
                    primary_20pct_split=role,image_h5_relpath=rel,image_sha256=sha256(path),label_h5_relpath=lp,label_sha256=lh))
    mp=root/"manifests/training/lcrseg_v1_seed0.csv";mp.parent.mkdir(parents=True)
    with mp.open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    sp=root/"splits/fundus_seed0.json";sp.parent.mkdir()
    sp.write_text(json.dumps(dict(seed=0,records=rows)))
    return sha256(mp),sha256(sp)


class Contract(unittest.TestCase):
    def test_01_scoring(self):
        z=torch.randn(2,3,8,8,requires_grad=True)
        target=torch.randint(3,(2,8,8));target[0,0,0]=255
        self.assertTrue(torch.allclose(obj.supervised(z,target),F.cross_entropy(z,target,ignore_index=255),atol=2e-7))
        self.assertEqual(float(obj.supervised(z,torch.full_like(target,255))),0.)
        p=np.array([[2,1],[1,0]]);t=np.array([[255,1],[2,0]])
        r=case_metrics(p,t)
        self.assertAlmostEqual(r["rim_dice"],2/3);self.assertEqual(r["cup_dice"],0)
        self.assertAlmostEqual(r["macro_fg_dice"],1/3)
        empty=case_metrics(p,np.full_like(t,255));self.assertFalse(empty["has_evaluable_gt"])
        self.assertEqual(empty["macro_fg_dice"],1)
        self.assertEqual(case_metrics(np.zeros_like(t),np.zeros_like(t))["cup_dice"],1)
        # Case-macro differs from pooled pixels; domain/seed equal weighting is explicit.
        small=case_metrics(np.array([[1]]),np.array([[1]]))
        large=case_metrics(np.zeros((10,10),int),np.ones((10,10),int))
        self.assertEqual(aggregate([small,large])["macro_fg_dice"],.75)
        self.assertEqual(np.mean([np.mean([.2]*100),np.mean([.8]*2)]),.5)

    def test_02_03_binding_and_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected=fixture(tmp);kw=dict(expected=expected,shape=(32,32))
            ds=CurrentData(tmp,0,"train_labeled",**kw)
            self.assertEqual(tuple(ds[0]["image"].shape),(3,32,32))
            ul=CurrentData(tmp,0,"train_unlabeled",**kw)
            self.assertNotIn("label_h5_relpath",ul.rows[0])
            opened=[];original_open=h5py.File
            def sentinel(path,*args,**kwargs):
                opened.append(str(path))
                if "labels" in str(path):raise AssertionError("hidden GT sentinel")
                return original_open(path,*args,**kwargs)
            with patch("h5py.File",side_effect=sentinel):self.assertNotIn("label",ul[0])
            self.assertEqual(len(opened),1)
            for role,purpose,domain in (("val","train",0),("test","evaluate",0),("val","evaluate",1),("train_labeled","train",1)):
                with self.assertRaises(PermissionError):CurrentData(tmp,0,role,purpose=purpose,domain=domain,**kw)
            with self.assertRaises(PermissionError):validate_parent(dict(stage=0,arm="common",complete=True),"E",2)
            # Tampered canonical image and label both fail their independent file hash.
            for key in ("image","label"):
                path=Path(tmp)/"h5/v1"/ds.rows[0][key+"_h5_relpath"]
                original=path.read_bytes();path.write_bytes(original+b"tampered")
                with self.assertRaises(RuntimeError):CurrentData(tmp,0,"train_labeled",**kw)[0]
                path.write_bytes(original)
            # Cross-seed and patient leakage survive re-hashing but fail semantic admission.
            manifest=Path(tmp)/"manifests/training/lcrseg_v1_seed0.csv"
            with manifest.open() as f:rows=list(csv.DictReader(f))
            rows[0]["split_seed"]="1"
            with manifest.open("w") as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            with self.assertRaises(ValueError):CurrentData(tmp,0,"train_labeled",expected=(sha256(manifest),expected[1]),shape=(32,32))
            rows[0]["split_seed"]="0";rows[3]["patient_id"]=rows[0]["patient_id"]
            with manifest.open("w") as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            split=Path(tmp)/"splits/fundus_seed0.json";split.write_text(json.dumps(dict(seed=0,records=rows)))
            with self.assertRaises(ValueError):CurrentData(tmp,0,"train_labeled",expected=(sha256(manifest),sha256(split)),shape=(32,32))

    def test_05_geometry_prototype_pas(self):
        y=torch.arange(64).reshape(8,8);x=y[None].float().repeat(3,1,1)
        xx,yy,v=geometry(x,y,torch.ones_like(y,dtype=torch.bool),torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(xx[0],yy))
        weak=torch.rand(2,3,32,32,device=DEVICE)
        self.assertTrue(torch.equal(strong(weak,1,11,0),strong(weak,1,11,0)))
        self.assertEqual(strong(weak,1,11,0).shape,weak.shape)
        f1=torch.tensor([[[1.,1.]],[[0.,0.]]]);f2=torch.tensor([[[0.]],[[1.]]])
        c1,s1=obj.case_center(f1,torch.ones((1,2),dtype=torch.long))
        c2,s2=obj.case_center(f2,torch.ones((1,1),dtype=torch.long))
        self.assertTrue(torch.allclose(F.normalize(c1+c2,dim=1)[1],torch.tensor([2**-.5]*2)))
        self.assertFalse(bool(s1[2]))
        logits=torch.tensor([[[[0.]],[[4.]],[[0.]]]],requires_grad=True)
        yy,mask,_=obj.pas(logits,torch.ones(1,2,1,1),torch.ones(3,2)/2**.5,torch.tensor([False,True,False]),torch.ones(1,1,1,dtype=torch.bool))
        self.assertFalse(mask.requires_grad);self.assertFalse(yy.requires_grad)
        loss=obj.ssl_ce(logits,yy,mask,torch.ones_like(mask));loss.backward();self.assertGreater(float(logits.grad.norm()),0)
        logits.grad=None
        obj.ssl_ce(logits,yy,torch.zeros_like(mask),torch.ones_like(mask)).backward()
        self.assertEqual(float(logits.grad.norm()),0)

    def test_07_08_projection_reference_and_autograd(self):
        rng=np.random.default_rng(7)
        p=rng.dirichlet([.7,.7,.7],size=45);q=rng.dirichlet([.5,.5,.5],size=45);y=rng.integers(0,3,size=45)
        pt=torch.tensor(p,device=DEVICE);qt=torch.tensor(q,device=DEVICE);yt=torch.tensor(y,device=DEVICE)
        r,conflict,eta,residual,_=obj.project(pt,qt,yt,torch.ones(45,dtype=torch.bool,device=DEVICE))
        rr=r.cpu().numpy()
        for i in range(45):
            np.testing.assert_allclose(rr[i],reference_projection(p[i],q[i],y[i]),atol=1e-11,rtol=1e-9)
        self.assertTrue(torch.equal(r[~conflict],qt[~conflict]));self.assertTrue((eta>=0).all());self.assertLessEqual(float(residual.max()),1e-9)
        self.assertTrue(((r-pt).norm(dim=-1)[conflict]>1e-6).any())
        i=int(conflict.nonzero()[0]);a=p[i]-np.eye(3)[y[i]];b=np.dot(a,p[i])
        sol=minimize(lambda z:np.sum(z*np.log(z/q[i])),p[i],method="SLSQP",bounds=[(1e-12,1)]*3,
            constraints=[dict(type="eq",fun=lambda z:z.sum()-1),dict(type="ineq",fun=lambda z:b-a@z)],options=dict(ftol=1e-12,maxiter=1000))
        self.assertTrue(sol.success);np.testing.assert_allclose(sol.x,rr[i],atol=2e-6)
        # Target is detached, logit gradient must be p-r with identical softmax precision.
        z=pt.log().detach().requires_grad_();target=r.detach()
        loss=(target*(target.log()-z.log_softmax(-1))).sum();loss.backward()
        self.assertTrue(torch.allclose(z.grad,pt-target,atol=1e-12,rtol=0));self.assertFalse(r.requires_grad)
        binary_p=torch.tensor([[.8,.2]],dtype=torch.double,device=DEVICE);binary_q=torch.tensor([[.5,.5]],dtype=torch.double,device=DEVICE)
        binary=obj.project(binary_p,binary_q,torch.tensor([0],device=DEVICE),torch.tensor([True],device=DEVICE))[0]
        self.assertTrue(torch.allclose(binary,binary_p,atol=1e-12,rtol=0))
        extreme=torch.tensor([[1.-2e-14,1e-14,1e-14]],dtype=torch.double,device=DEVICE)
        obj.project(extreme,torch.full_like(extreme,1/3),torch.tensor([1],device=DEVICE),torch.tensor([True],device=DEVICE))

    def test_09_drop_normalization_and_unreliable(self):
        z=torch.tensor([[[[2.,0.]],[[0.,2.]],[[0.,0.]]]],device=DEVICE,requires_grad=True)
        q=torch.tensor([[[[.99,.1]],[[.005,.8]],[[.005,.1]]]],dtype=torch.double,device=DEVICE)
        y=torch.tensor([[[0,1]]],device=DEVICE);valid=torch.ones_like(y,dtype=torch.bool);none=torch.zeros_like(valid)
        values=[obj.kd(z,q,y,none,valid,a)[0] for a in "CDE"]
        self.assertEqual(float(values[0]),float(values[1]));self.assertEqual(float(values[0]),float(values[2]))
        drop,stats=obj.kd(z,q,y,valid,valid,"D")
        p=z.double().softmax(1).detach();qt=(1-obj.EPS_Q)*q+obj.EPS_Q/3
        a=p-F.one_hot(y,3).movedim(-1,1);conflict=((p-qt)*a).sum(1)<0
        manual=(qt*(qt.log()-z.double().log_softmax(1))).sum(1)
        self.assertTrue(torch.allclose(drop,(manual*(~conflict)).sum()/2,atol=1e-12))
        z2=z.detach().clone().requires_grad_();p2=z2.double().softmax(1).detach()
        raw=(p2-obj.EPS_Q/3)/(1-obj.EPS_Q)
        zero=obj.kd(z2,raw,y,none,valid,"C")[0];zero.backward()
        self.assertLess(abs(float(zero)),1e-12);self.assertLess(float(z2.grad.norm()),1e-12)

    def test_07_08_projected_zero_mass_entropy(self):
        # Entirely synthetic confident logits, unrelated to any real case.
        z=torch.tensor([0.,-46.05170186,-57.56462732],dtype=torch.float32,device=DEVICE).reshape(1,3,1,1).requires_grad_()
        q=torch.tensor([.2,.3,.5],dtype=torch.double,device=DEVICE).reshape(1,3,1,1)
        y=torch.zeros((1,1,1),dtype=torch.long,device=DEVICE);valid=torch.ones_like(y,dtype=torch.bool)
        p=z.double().movedim(1,-1).reshape(1,3).log_softmax(-1).exp().detach()
        qt=(1-obj.EPS_Q)*q.reshape(1,3)+obj.EPS_Q/3
        target,_,_,residual,_=obj.project(p,qt,y.flatten(),valid.flatten())
        self.assertTrue((target==0).any());self.assertLessEqual(float(residual.max()),1e-9)
        loss,_=obj.kd(z,q,y,valid,valid,"E")
        self.assertTrue(torch.isfinite(loss))
        reference=torch.special.xlogy(target,target).sum()-(target*z.double().movedim(1,-1).reshape(1,3).log_softmax(-1)).sum()
        self.assertTrue(torch.allclose(loss,reference,atol=1e-12,rtol=0))
        loss.backward();self.assertTrue(torch.isfinite(z.grad).all())
        self.assertTrue(torch.allclose(z.grad.flatten(),(p-target).float().flatten(),atol=1e-7,rtol=0))
        positive=torch.tensor([[.1,.2,.7]],dtype=torch.double,device=DEVICE)
        logp=torch.tensor([[-2.,-1.,-.5]],dtype=torch.double,device=DEVICE)
        old=positive*(positive.log()-logp)
        new=positive*(positive.log().masked_fill(positive==0,0.)-logp)
        self.assertTrue(torch.equal(old,new))

    def test_04_06_09_models_gas_and_matched_zero_kd(self):
        student=model(REFERENCE,DEVICE);teacher=second_model(student,"C");teacher_hash=state_hash(teacher)
        optimizer=optimizer_for(student)
        self.assertFalse(any(p.requires_grad for p in teacher.parameters()))
        self.assertFalse({id(p) for p in teacher.parameters()} & {id(p) for g in optimizer.param_groups for p in g["params"]})
        x=torch.rand(2,3,32,32,device=DEVICE);y=torch.randint(3,(2,32,32),device=DEVICE)
        # Independent supervised extraction: adding another branch changes total gradient, not GAS.
        z=student(x,stochastic_classifier=True)[0];sup=obj.supervised(z,y)
        gas=torch.autograd.grad(sup,student.decoder.conv_logit.mu.weight,retain_graph=True)[0].detach().square()
        (sup+z.square().mean()).backward();self.assertFalse(torch.allclose(student.decoder.conv_logit.mu.weight.grad.square(),gas))
        self.assertEqual(state_hash(teacher),teacher_hash)
        del optimizer,teacher,student,z,sup
        gc_collect()
        hashes=[]
        batch=dict(image=x,label=y,geometry=torch.ones_like(y,dtype=torch.bool));u={k:v for k,v in batch.items() if k!="label"}
        for arm in "BCDE":
            student=model(REFERENCE,DEVICE);teacher=second_model(student,arm);optimizer=optimizer_for(student)
            counter=dict(forward=0,gas_autograd=0,diagnostic_autograd=0,backward=0,optimizer_steps=0)
            train_step(student,teacher,optimizer,batch,u,torch.zeros(3,16,device=DEVICE),torch.zeros(3,dtype=torch.bool,device=DEVICE),arm,1,11,0,counter,lambda_kd=0)
            hashes.append(state_hash(student));self.assertEqual(counter["optimizer_steps"],1)
            del student,teacher,optimizer;gc_collect()
        self.assertEqual(len(set(hashes)),1)
        student=model(REFERENCE,DEVICE);teacher=second_model(student,"A");optimizer=optimizer_for(student)
        old_mu=teacher.decoder.conv_logit.mu.weight.detach().clone()
        old_gas=teacher.decoder.conv_logit.grad_update.detach().clone()
        counter=dict(forward=0,gas_autograd=0,diagnostic_autograd=0,backward=0,optimizer_steps=0)
        from experiments.lcrseg.single_teacher_scd_v0_1.engine import forward
        zl=forward(student,x,True,(1,11,0,"supervised_head"),counter)[0]
        expected_gas=torch.autograd.grad(obj.supervised(zl,y),student.decoder.conv_logit.mu.weight)[0].detach().square()
        del zl
        train_step(student,teacher,optimizer,batch,u,torch.zeros(3,16,device=DEVICE),torch.zeros(3,dtype=torch.bool,device=DEVICE),"A",1,11,0,counter)
        self.assertTrue(torch.equal(student.decoder.conv_logit.grad_update,expected_gas))
        self.assertTrue(torch.allclose(teacher.decoder.conv_logit.mu.weight,old_mu*.99+student.decoder.conv_logit.mu.weight*.01,atol=1e-7))
        self.assertTrue(torch.allclose(teacher.decoder.conv_logit.grad_update,old_gas*.99+expected_gas*.01,atol=1e-12))

    def test_10_11_hdf5_stages_exact_resume_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);expected=fixture(root/"data")
            common=dict(data=root/"data",reference=REFERENCE,device=DEVICE,expected=expected,shape=(32,32),qualification=True,epochs=2)
            uninterrupted=run_stage(**common,output=root/"whole",arm="common",stage=0)
            run_stage(**common,output=root/"resume",arm="common",stage=0,stop_after_epoch=1)
            resumed=run_stage(**common,output=root/"resume",arm="common",stage=0,resume=True)
            self.assertEqual(uninterrupted["student_hash"],resumed["student_hash"])
            pa=torch.load(root/"whole/student_latest.pt",map_location="cpu",weights_only=False)
            pb=torch.load(root/"resume/student_latest.pt",map_location="cpu",weights_only=False)
            self.assertTrue(torch.equal(pa["prototypes"],pb["prototypes"]));self.assertTrue(torch.equal(pa["cpu_rng"],pb["cpu_rng"]))
            self.assertEqual(pa["counters"],pb["counters"]);del pa,pb
            parent=root/"whole/student_latest.pt"
            incremental=dict(common,epochs=12)
            whole=run_stage(**incremental,output=root/"e_whole",arm="E",stage=1,parent=parent)
            run_stage(**incremental,output=root/"e_resume",arm="E",stage=1,parent=parent,stop_after_epoch=11)
            restored=run_stage(**incremental,output=root/"e_resume",arm="E",stage=1,parent=parent,resume=True)
            self.assertEqual(whole["student_hash"],restored["student_hash"])
            a=torch.load(root/"e_whole/student_latest.pt",map_location="cpu",weights_only=False)
            b=torch.load(root/"e_resume/student_latest.pt",map_location="cpu",weights_only=False)
            self.assertTrue(a["supported"].any());self.assertTrue(torch.equal(a["prototypes"],b["prototypes"]));del a,b
            for stage in (1,2):
                out=root/f"stage{stage}";result=run_stage(**common,output=out,arm="E",stage=stage,parent=parent)
                self.assertEqual(result["teacher_initial_hash"],result["parent_hash"])
                self.assertEqual(result["teacher_initial_hash"],result["teacher_final_hash"])
                visible=evaluate(data=root/"data",reference=REFERENCE,checkpoint=out/"student_latest.pt",output=out/"visible_val.json",device=DEVICE,expected=expected,shape=(32,32))
                teacher=out/"prev_teacher.pt";hidden=out/"hidden_teacher.pt";teacher.rename(hidden)
                evaluation=evaluate(data=root/"data",reference=REFERENCE,checkpoint=out/"student_latest.pt",output=out/"val.json",device=DEVICE,expected=expected,shape=(32,32))
                self.assertEqual(len(evaluation["rows"]),stage+1);self.assertEqual(evaluation["inference_full_models"],1)
                self.assertEqual(visible["rows"],evaluation["rows"])
                hidden.rename(teacher);parent=out/"student_latest.pt"
                if stage==2:
                    from experiments.lcrseg.single_teacher_scd_v0_1.deploy import export
                    export(parent,REFERENCE,device=DEVICE)
                    self.assertTrue(json.loads((out/"deployment.json").read_text())["equal_synthetic_prediction"])
        self.assertEqual([len(b) for b in batches(5,1,1,"u",3)],[2,2,1])
        self.assertEqual(obj.learning_rate(0,3200),.001)
        self.assertEqual(obj.lambda_u(10),0);self.assertEqual(obj.lambda_u(11),.05);self.assertEqual(obj.lambda_u(20),.5)

    def test_12_solver_failure_explicit(self):
        p=torch.tensor([[.8,.1,.1]],dtype=torch.double,device=DEVICE)
        q=torch.tensor([[float("nan"),.1,.1]],dtype=torch.double,device=DEVICE)
        with self.assertRaises(FloatingPointError):obj.project(p,q,torch.tensor([0],device=DEVICE),torch.tensor([True],device=DEVICE))
        q=torch.tensor([[.000000001,.4999999995,.4999999995]],dtype=torch.double,device=DEVICE)
        with self.assertRaises(FloatingPointError):obj.project(p,q,torch.tensor([0],device=DEVICE),torch.tensor([True],device=DEVICE),max_bracket=0)
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):child([sys.executable,"-c","raise SystemExit(7)"],Path(tmp)/"child.log",Path(tmp)/"exit.json",dict(os.environ))
            self.assertEqual(json.loads((Path(tmp)/"exit.json").read_text())["exit_code"],7)


def gc_collect():
    import gc
    gc.collect()


if __name__=="__main__":unittest.main()
