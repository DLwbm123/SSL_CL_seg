"""Head/mask/gradient witnesses and a literal canonical-layout lifecycle."""
import gc,inspect,json,os,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from experiments.lcrseg.tests.ssl_foundation_v0_1.test_contract import Contract as Foundation,fixture
from experiments.lcrseg.ssl_head_control_v0_1 import core as c,engine as e,diagnostics as d,audit as a,results as r
from experiments.lcrseg.ssl_head_control_v0_1.deploy import deploy
REFERENCE=os.environ['SF_REFERENCE'];DEVICE=torch.device(os.environ['SF_DEVICE'])

class Contract(unittest.TestCase):
    def tearDown(self):gc.collect()
    def test_01_inherited_data_pas_scoring(self):
        for method in ('test_01_binding_roles_scoring','test_02_probability_pas_witnesses','test_05_quality_denominators','test_08_prototype_equal_image_and_geometry'):
            getattr(Foundation(method),method)()
    def test_02_head_initialization_warmup_golden(self):
        before=d.rng_hash();initial=[];end=[];heads=[]
        gen=torch.Generator().manual_seed(123)
        lb=dict(image=torch.rand(2,3,32,32,generator=gen).to(DEVICE),label=torch.randint(3,(2,32,32),generator=gen).to(DEVICE),geometry=torch.ones(2,32,32,dtype=torch.bool,device=DEVICE))
        for arm in c.ARMS:
            m=c.build(REFERENCE,DEVICE,11,c.DOMAINS[0],arm);initial.append(c.state_hash(m));teacher=c.ema_from(m);opt=c.optimizer_for(m)
            feature=torch.ones(1,16,8,8,device=DEVICE);head=m.decoder.conv_logit;expected=F.conv2d(feature,head.mu.weight,padding=0)
            if arm.startswith('LIN'):
                self.assertTrue(torch.equal(head(feature),expected));self.assertFalse(head.sigma.weight.requires_grad)
                with self.assertRaises(PermissionError):head(feature,stochastic=True)
            else:self.assertFalse(torch.equal(head(feature,stochastic=False),expected))
            sigma=head.sigma.weight.detach().clone();ct=Counter()
            c.train_step(m,teacher,opt,lb,None,None,None,arm,11,c.DOMAINS[0],20,0,ct)
            self.assertTrue(torch.equal(sigma,head.sigma.weight))
            if arm.startswith('LIN'):self.assertTrue(torch.equal(sigma,teacher.decoder.conv_logit.sigma.weight))
            self.assertIsNone(head.sigma.weight.grad);self.assertFalse(any(p.grad is not None for p in teacher.parameters()))
            state=(c.state_hash(m),c.state_hash(teacher),c.optimizer_hash(opt));end.append(state)
            if arm=='SUP_G1':gold=state
            else:self.assertEqual(ct['gas_autograd'],0);self.assertEqual(ct['student_u_forward'],0)
            del m,teacher,opt,head,expected
        self.assertEqual(len(set(initial)),1);self.assertEqual(len(set(end[:3])),1)
        m=c.base.build(REFERENCE,DEVICE,11,c.DOMAINS[0]);self.assertEqual(c.state_hash(m),initial[0]);teacher=c.ema_from(m);opt=c.optimizer_for(m)
        c.base.train_step(m,teacher,opt,lb,None,None,None,'SUP_G1',11,c.DOMAINS[0],20,0,Counter())
        self.assertEqual((c.state_hash(m),c.state_hash(teacher),c.optimizer_hash(opt)),gold)
        del m,teacher,opt
        for seed in (11,21,22):
            m=c.build(REFERENCE,DEVICE,seed,c.DOMAINS[0],'LIN_SUP');heads.append((c.state_hash(m),m.decoder.conv_logit.mu.weight.detach().cpu().clone()));del m
        self.assertEqual(len({x[0] for x in heads}),3);self.assertTrue(torch.equal(heads[0][1],heads[2][1]));self.assertEqual(before,d.rng_hash())
    def test_03_matched_masks_ignore_and_hierarchy(self):
        self.assertNotIn('gt',inspect.signature(a.matched_masks).parameters)
        C=np.ones((2,4),bool);M=np.array([[1,0,0,0],[1,1,0,0]],bool);yt=np.array([[0,0,1,1],[2,2,2,2]]);cs=np.full(C.shape,.9);ys=(yt+1)%3
        masks,counts,seal=a.matched_masks(C,M,cs,cs,yt,(11,'d',100,0,1))
        self.assertEqual([x['K'] for x in counts],[1,0,2]);self.assertTrue(masks['CONF_TOPK_MATCHED'][0,0]);self.assertTrue(masks['CONF_TOPK_MATCHED'][1,0]);self.assertFalse(masks['CONF_TOPK_MATCHED'][1,2])
        for extreme in (np.zeros_like(C),C):
            mm,kk,_=a.matched_masks(C,extreme,cs,cs,yt,('extreme',));self.assertTrue(all(np.array_equal(m,extreme) for m in mm.values()))
        # Explicit scoring-only support mismatch while immutable geometric K remains equal.
        M=np.array([[0,1,0,0],[0,0,0,0]],bool);mm,kk,ss=a.matched_masks(C,M,cs,cs,yt,('ignore',));gt=yt.copy();gt[0,0]=255
        rows=a.evaluate_mass(mm,kk,ss,C,ys,yt,gt);self.assertFalse(next(x for x in rows if x['class_id']==0)['effective_teacher_class_counts_equal'])
        self.assertEqual(a.matched_masks(C,M,cs,cs,yt,('ignore',))[2],ss)
        self.assertEqual(a.likelihood(1,1,1,0)[1],'POSITIVE_INFINITY');self.assertEqual(a.likelihood(0,1,0,0)[1],'UNDEFINED_CONDITION_SUPPORT');self.assertEqual(a.likelihood(1,1,0,0)[1],'UNDEFINED_ZERO_OVER_ZERO')
        rr=[dict(patient=p,repeat=draw,permutation=j,value=p+2*draw+j,undefined=None,group='g') for p in (0,1) for draw in (0,1) for j in range(4)]
        result=a.aggregate(rr,('group',))[0];self.assertEqual(result['n_patients'],2);self.assertEqual(result['n_permutation_rows'],16);self.assertEqual(result['patient_mean_value'],3);self.assertIsNone(result['patient_mean_undefined'])
    def test_04_local_gradient(self):
        gen=torch.Generator().manual_seed(12)
        z=torch.randn(2,3,4,4,dtype=torch.double,generator=gen,requires_grad=True);pt=torch.randn(2,3,4,4,dtype=torch.double,generator=gen).softmax(1);mask=torch.rand(2,4,4,generator=gen)>.5;ps=z.softmax(1)
        truth=torch.autograd.grad(.5*c.consistency(ps,pt,mask),z)[0];observed=c.local_gradient(ps,pt,mask,.5)
        self.assertTrue(torch.allclose(truth,observed,atol=1e-14,rtol=1e-12));self.assertGreater(float(truth.norm()),0);self.assertEqual(float(c.local_gradient(ps,pt,~torch.ones_like(mask),.5).norm()),0)
        self.assertAlmostEqual(sum(x['weighted_MSE_loss_contribution'] for x in c.class_loss_stats(ps,pt,mask,.5)),float(.5*c.consistency(ps,pt,mask)),places=12)
    def test_05_literal_resume_diagnostics_deploy(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);ex=fixture(root/'data');kw=dict(data=str(root/'data'),reference=REFERENCE,domain=c.DOMAINS[0],seed=11,arm='LIN_MT_PAS',device=DEVICE,expected=ex,shape=(32,32),epochs=26,qualification=True)
            before=d.rng_hash();e.run(output=root/'full',**kw);self.assertEqual(before,d.rng_hash())
            e.run(output=root/'split',stop_epoch=21,**kw);e.run(output=root/'split',resume=True,**kw)
            full=r.read(root/'full/receipt.json');split=r.read(root/'split/receipt.json')
            for key in ('student_hash','ema_hash','head_mode','label_order_hash','labeled_opens','unlabeled_opens','counters'):self.assertEqual(full[key],split[key],key)
            aa=torch.load(root/'full/latest.pt',map_location='cpu',weights_only=False);bb=torch.load(root/'split/latest.pt',map_location='cpu',weights_only=False)
            for k in ('epoch','head_mode'):self.assertEqual(aa[k],bb[k])
            for ident,state in aa['optimizer']['state'].items():
                for key,value in state.items():self.assertTrue(torch.equal(value,bb['optimizer']['state'][ident][key]))
            self.assertTrue(torch.equal(aa['rng']['cpu'],bb['rng']['cpu']));del aa,bb
            self.assertEqual(full['optimizer_updates'],78);self.assertEqual(r.read(root/'full/eval20/receipt.json')['actual_training_library'],'NOT_STARTED');self.assertTrue(r.read(root/'full/eval26/receipt.json')['matched_mass_completed']);self.assertEqual(before,d.rng_hash())
            (root/'full/latest.pt').rename(root/'protected_full_checkpoint.pt')
            deploy(root/'full',str(root/'data'),REFERENCE,DEVICE,qualification=True,expected=ex,shape=(32,32));self.assertEqual(r.read(root/'full/deployment.json')['models'],1)
    def test_06_frozen_gates(self):
        def table(seeds,arms,lin=.5,ssl=.511):
            return [dict(seed=s,arm=a,domain=d,model='student',epoch=100,macro_fg_dice=(ssl if a.startswith('LIN_MT') else lin),rim_dice=(ssl if a.startswith('LIN_MT') else lin),cup_dice=(ssl if a.startswith('LIN_MT') else lin)) for s in seeds for a in arms for d in c.DOMAINS]
        old=table((11,),c.base.ARMS);tt=table((11,21,22),c.ARMS);gate=r.screen(tt,old);self.assertEqual(gate['path'],'P2_SSL');self.assertEqual(gate['selected'],'LIN_MT_CONF');self.assertTrue(r.confirmation(tt,gate)['passed'])
        self.assertEqual(r.screen(table((11,),c.ARMS,ssl=.509),old)['path'],'NOT_ADMITTED')
        sup=table((11,),c.ARMS,lin=.511,ssl=.511);self.assertEqual(r.screen(sup,old)['path'],'P2_SUP')
        self.assertEqual(3*(3200+2100),15900);self.assertEqual(2*4*(3200+2100),42400);self.assertEqual(2*2*(3200+2100),21200)
        for row in tt:
            if row['seed']==22 and row['arm']==gate['selected']:row['macro_fg_dice']=.499
        self.assertFalse(r.confirmation(tt,gate)['passed'])
        # Stronger old reference blocks a merely same-head gain.
        strong=table((11,),c.base.ARMS,lin=.52);self.assertEqual(r.screen(table((11,),c.ARMS),strong)['path'],'NOT_ADMITTED')

    def test_07_p0_readonly_and_probability(self):
        from experiments.lcrseg.ssl_head_control_v0_1.probability import probability_rows
        z=torch.tensor([[[[0.,0.]],[[1.,1.]],[[2.,2.]]]]);truth=np.array([[2,255]]);p=z.softmax(1)
        row,bins=probability_rows(z,p,truth);self.assertEqual(row['valid_pixels'],1);self.assertAlmostEqual(row['NLL'],float(-p[0,2,0,0].log()),places=6);self.assertEqual(sum(x['bin_count'] for x in bins),1)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);ex=fixture(root/'data');m=c.set_head(c.base.build(REFERENCE,DEVICE,11,c.DOMAINS[0]),c.NORMAL);teacher=c.ema_from(m)
            proto=torch.ones(3,16,device=DEVICE)/4;support=torch.ones(3,dtype=torch.bool,device=DEVICE);ct=Counter()
            with patch.object(d,'prototypes',side_effect=AssertionError('P0 cannot refresh prototypes')):
                d.snapshot(m,teacher,None,data=str(root/'data'),reference=REFERENCE,device=DEVICE,domain=c.DOMAINS[0],seed=11,arm='MT_PAS_G0',epoch=100,output=root/'P0',proto=proto,support=support,counters=ct,expected=ex,shape=(32,32),audit_only=True)
            self.assertEqual(ct['diagnostic_student_images']+ct['diagnostic_ema_images'],20);self.assertEqual(ct['optimizer_steps'],0);self.assertEqual(r.read(root/'P0/receipt.json')['GT_free_mask_seals'],10)
    def test_08_nonzero_linear_pas_gradient(self):
        # An actual linear convolution on controlled synthetic features: confidently
        # disagreeing own-class predictions must retain a nonzero PAS update path.
        m=c.build(REFERENCE,DEVICE,11,c.DOMAINS[0],'LIN_MT_PAS');head=m.decoder.conv_logit
        with torch.no_grad():
            head.mu.weight.zero_();head.mu.weight[0].fill_(2.5/(16*9))
        fs=torch.ones(1,16,3,3,device=DEVICE);z=head(fs);ps=z.softmax(1);pt=ps.detach().roll(1,1)
        features=torch.ones(1,16,1,1,device=DEVICE);proto=torch.ones(3,16,device=DEVICE)/4;support=torch.ones(3,dtype=torch.bool,device=DEVICE);geo=torch.ones(1,1,1,dtype=torch.bool,device=DEVICE)
        conf,pas=c.masks(ps,pt,features,features,proto,support,geo);self.assertTrue(pas.item());gradient=torch.autograd.grad(c.consistency(ps,pt,pas),head.mu.weight)[0];self.assertGreater(float(gradient.norm()),0)
