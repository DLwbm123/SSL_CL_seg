"""Independent witnesses plus literal synthetic deployment-layout integration."""
import gc,json,os,tempfile,unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from experiments.lcrseg.tests.single_teacher_scd_v0_1.test_contract import fixture,Contract as OldContract
from experiments.lcrseg.ssl_foundation_v0_1 import core as c,engine as e,diagnostics as d,results as r
from experiments.lcrseg.ssl_foundation_v0_1.deploy import deploy
from experiments.lcrseg.di_dmpa_gate1.binding import ProtocolError
REFERENCE=os.environ['SF_REFERENCE'];DEVICE=torch.device(os.environ['SF_DEVICE'])

class Contract(unittest.TestCase):
    def tearDown(self):gc.collect()
    def test_01_binding_roles_scoring(self):
        OldContract('test_01_scoring').test_01_scoring()
        with tempfile.TemporaryDirectory() as t:
            ex=fixture(t);kw=dict(expected=ex,shape=(32,32))
            ds=c.DomainData(t,c.DOMAINS[0],'train_labeled',**kw);self.assertEqual(tuple(ds[0]['image'].shape),(3,32,32))
            with self.assertRaises(PermissionError):c.DomainData(t,c.DOMAINS[0],'val',**kw)
            with self.assertRaises(PermissionError):c.DomainData(t,'REFUGE','train_labeled',**kw)
            with self.assertRaises(PermissionError):c.DomainData(t,c.DOMAINS[0],'test',evaluator=True,**kw)
            u=c.DomainData(t,c.DOMAINS[0],'train_unlabeled',**kw)
            self.assertTrue(all(not any('label' in k for k in row) for row in u._dataset.rows))
            path=Path(t)/'h5/v1'/ds._dataset.rows[0]['image_h5_relpath'];path.rename(path.with_suffix('.away'))
            with self.assertRaises(ProtocolError):c.DomainData(t,c.DOMAINS[0],'train_labeled',**kw)[0]
    def test_02_probability_pas_witnesses(self):
        ps=torch.tensor([[[[.8]],[[.1]],[[.1]]]],requires_grad=True);pt=torch.tensor([[[[.1]],[[.8]],[[.1]]]],requires_grad=True)
        fs=torch.tensor([[[[1.]],[[0.]]]],requires_grad=True);ft=torch.tensor([[[[0.]],[[1.]]]],requires_grad=True)
        proto=torch.tensor([[1.,0.],[0.,1.],[0.,0.]]);support=torch.tensor([1,1,0],dtype=torch.bool);geo=torch.ones(1,1,1,dtype=torch.bool)
        conf,pas=c.masks(ps,pt,fs,ft,proto,support,geo)
        self.assertTrue(pas.item());self.assertFalse(pas.requires_grad)
        loss=c.consistency(ps,pt,pas);self.assertAlmostEqual(float(loss),.98,places=6);loss.backward()
        self.assertIsNotNone(ps.grad);self.assertIsNone(pt.grad);self.assertIsNone(fs.grad)
        self.assertEqual(float(c.consistency(ps,pt,~pas)),0);self.assertTrue(c.consistency(ps,pt,~pas).requires_grad)
        support[1]=False;self.assertFalse(c.masks(ps,pt,fs,ft,proto,support,geo)[1].item())
        exact=ps.detach().clone();exact[:,0]=.7;self.assertFalse(c.masks(exact,pt,fs,ft,proto,support,geo)[0].item())
        f=torch.tensor([[[1.,1.],[0.,0.]],[[0.,0.],[2.,2.]]]);y=torch.tensor([[1,1],[1,1]])
        center,s=c.centers(f,y);v=torch.tensor([.5,1.]);self.assertTrue(torch.allclose(center[1],F.normalize(v,dim=0)));self.assertFalse(s[0])
        self.assertFalse(c.centers(torch.zeros_like(f),y)[1].any())
    def test_03_rng_initialization_warmup_step(self):
        before=d.rng_hash();hashes=[];heads=[]
        for seed in (11,12,13):
            m=c.build(REFERENCE,DEVICE,seed,c.DOMAINS[0]);hashes.append(c.state_hash(m));heads.append(m.decoder.conv_logit.mu.weight.detach().cpu().clone());del m
        self.assertEqual(d.rng_hash(),before);self.assertEqual(len(set(hashes)),3);self.assertTrue(torch.equal(heads[0],heads[1]))
        with tempfile.TemporaryDirectory() as t:
            ex=fixture(t);ds=c.DomainData(t,c.DOMAINS[0],'train_labeled',expected=ex,shape=(32,32));lb=c.batch(ds,[0,1],DEVICE,11,20,0,'labeled')
            for gas in (0,1):
                outputs=[]
                for arm in [a for a in c.ARMS if a.endswith(str(gas))]:
                    m=c.build(REFERENCE,DEVICE,11,c.DOMAINS[0]);ema=c.ema_from(m);opt=c.optimizer_for(m);ct=Counter()
                    self.assertEqual(c.state_hash(m),hashes[0]);c.train_step(m,ema,opt,lb,None,None,None,arm,11,c.DOMAINS[0],20,0,ct)
                    outputs.append((c.state_hash(m),c.state_hash(ema)));self.assertEqual(ct['student_u_forward'],0);del m,ema,opt
                self.assertEqual(len(set(outputs)),1)
        self.assertEqual(c.weight(20),0);self.assertEqual(c.weight(21),.025);self.assertEqual(c.weight(40),.5)
        self.assertNotEqual(c.orders(9,11,c.DOMAINS[0],1,'l',5),c.orders(9,12,c.DOMAINS[0],1,'l',5))
    def test_04_gas_ema_supervised_gradient(self):
        m=c.build(REFERENCE,DEVICE,11,c.DOMAINS[0]);ema=c.ema_from(m);opt=c.optimizer_for(m);ct=Counter()
        g=torch.Generator().manual_seed(123);l=dict(image=torch.rand(2,3,32,32,generator=g).to(DEVICE),label=torch.randint(3,(2,32,32),generator=g).to(DEVICE),geometry=torch.ones(2,32,32,dtype=torch.bool,device=DEVICE));u={k:v for k,v in l.items() if k!='label'}
        oldmu=ema.decoder.conv_logit.mu.weight.detach().clone()
        z,_=c.fwd(m,l['image'],True,(11,c.DOMAINS[0],21,0),Counter(),'student_l');sup=c.supervised(z,l['label']);gas=torch.autograd.grad(sup,m.decoder.conv_logit.mu.weight)[0].square();del z,sup
        row=c.train_step(m,ema,opt,l,u,None,None,'MT_CONF_G1',11,c.DOMAINS[0],21,0,ct,diagnostic=True)
        self.assertTrue(torch.equal(gas,m.decoder.conv_logit.grad_update));self.assertTrue(torch.equal(gas,ema.decoder.conv_logit.grad_update))
        self.assertTrue(torch.allclose(ema.decoder.conv_logit.mu.weight,.99*oldmu+.01*m.decoder.conv_logit.mu.weight,atol=1e-7))
        self.assertFalse(any(p.grad is not None for p in ema.parameters()));self.assertIsNone(m.decoder.conv_logit.sigma.weight.grad)
        self.assertNotIn(id(m.decoder.conv_logit.grad_update),{id(p) for group in opt.param_groups for p in group['params']})
        self.assertGreater(row['gradients']['supervised'],0)
    def test_05_quality_denominators(self):
        gt=np.array([[0,1],[1,255]]);pred=np.array([[0,1],[2,2]]);mask=np.ones((2,2),bool)
        q=d.quality(pred,pred,gt,mask)
        self.assertEqual(q[1]['precision'],1);self.assertEqual(q[1]['accepted_correct_recall'],.5);self.assertEqual(q[1]['valid_pixels'],3)
        self.assertEqual(q[2]['precision'],0);self.assertIsNone(q[2]['accepted_correct_recall'])
        empty=d.quality(pred,pred,gt,np.zeros_like(mask));self.assertIsNone(empty[1]['precision']);self.assertEqual(empty[1]['precision_status'],'UNDEFINED_NO_SUPPORT')
        rr=[]
        for patient in range(2):
            for rep in range(4):rr.append(dict(epoch=20,domain='d',arm='a',seed=11,mode='training_like',library='fresh',filter='raw',model='teacher',patient=patient,repeat=rep,**q[1]))
        a=d.aggregate_quality(rr)[0];self.assertEqual(a['n_patients'],2);self.assertEqual(a['n_replicate_rows'],8);self.assertEqual(a['patient_mean_precision'],1)
    def test_06_literal_lifecycle_resume_deploy(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);ex=fixture(root/'data');kw=dict(data=str(root/'data'),reference=REFERENCE,domain=c.DOMAINS[0],seed=11,arm='MT_PAS_G0',device=DEVICE,expected=ex,shape=(32,32),epochs=26,qualification=True)
            # Instrument real loader/teacher and read-only snapshots; do not mock losses/model.
            before=d.rng_hash();e.run(output=root/'full',**kw);self.assertEqual(d.rng_hash(),before)
            e.run(output=root/'split',stop_epoch=21,**kw);e.run(output=root/'split',resume=True,**kw)
            a=r.read(root/'full/receipt.json');b=r.read(root/'split/receipt.json');self.assertEqual(a['student_hash'],b['student_hash']);self.assertEqual(a['ema_hash'],b['ema_hash'])
            self.assertEqual(a['optimizer_updates'],78);self.assertTrue(r.read(root/'full/eval20/receipt.json')['state_preserved']);self.assertEqual(r.read(root/'full/eval20/receipt.json')['actual_training_library'],'NOT_STARTED')
            deploy(root/'full',str(root/'data'),REFERENCE,DEVICE,qualification=True,expected=ex,shape=(32,32));self.assertEqual(r.read(root/'full/deployment.json')['models'],1)
    def test_07_selection_fullprecision(self):
        def table(gain):
            out=[]
            for seed in (11,12,13):
                for arm in c.ARMS:
                    for domain in c.DOMAINS:
                        x=.5+(gain if not arm.startswith('SUP') else 0)
                        out.append(dict(seed=seed,arm=arm,domain=domain,epoch=100,model='student',macro_fg_dice=x,rim_dice=x,cup_dice=x,background_dice=x))
            return out
        no=r.screen(table(.009));self.assertFalse(no['passed'])
        tt=table(.011);yes=r.screen(tt);self.assertEqual(yes['selected'],'MT_CONF_G0');self.assertEqual(r.confirmation(tt,yes)['status'],'SINGLE_DOMAIN_SSL_FOUNDATION_ESTABLISHED_ON_FIXED_SPLIT')
        for x in tt:
            if x['seed']==13 and x['arm']==yes['selected']:x['macro_fg_dice']=.49
        self.assertEqual(r.confirmation(tt,yes)['status'],'SINGLE_DOMAIN_SSL_REPLICATION_NOT_ESTABLISHED')

    def test_08_prototype_equal_image_and_geometry(self):
        with tempfile.TemporaryDirectory() as t:
            ex=fixture(t);ds=c.DomainData(t,c.DOMAINS[0],'train_labeled',expected=ex,shape=(32,32))
            m=c.build(REFERENCE,DEVICE,11,c.DOMAINS[0]);m.train();ct=Counter()
            observed,support=c.prototypes(m,ds,DEVICE,ct,11,21,'witness')
            terms=[[] for _ in range(3)]
            with torch.no_grad():
                for i in range(len(ds)):
                    item=ds[i];feature=m(item['image'][None].to(DEVICE),stochastic_classifier=False)[1][0]
                    for cls in range(3):
                        mask=item['label'].to(DEVICE)==cls
                        if mask.any():
                            mean=feature[:,mask].mean(1)
                            if mean.norm()>0:terms[cls].append(mean/mean.norm())
            for cls in range(3):
                self.assertTrue(torch.allclose(observed[cls],torch.stack(terms[cls]).mean(0),atol=1e-7))
            self.assertTrue(m.training)
            a=c.batch(ds,[0,1],DEVICE,11,21,2,'labeled');b=c.batch(ds,[0,1],DEVICE,11,21,2,'labeled')
            self.assertTrue(torch.equal(a['image'],b['image']));self.assertTrue(torch.equal(a['label'],b['label']))
            self.assertTrue(torch.equal((a['image'][:,0]*255/80).round().long(),a['label']))
            noise=c.noisy(a['image'],(11,c.DOMAINS[0],21,2));self.assertEqual(noise.shape,a['image'].shape);self.assertTrue((noise>=0).all() and (noise<=1).all())
