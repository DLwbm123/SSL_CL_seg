import csv,gc,json,os,tempfile,unittest
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from experiments.lcrseg.ssl_target_path_v0_1 import core as c,engine as e,diagnostics as d,results as r
from experiments.lcrseg.ssl_target_path_v0_1.deploy import deploy
from experiments.lcrseg.tests.single_teacher_scd_v0_1.test_contract import fixture,Contract as Previous
from experiments.lcrseg.tests.ssl_foundation_v0_1.test_contract import Contract as Foundation
from experiments.lcrseg.di_dmpa_gate1.binding import sha256
DEVICE=torch.device(os.environ['SF_DEVICE']);REFERENCE=os.environ['SF_REFERENCE']
def small_fixture(root):
    fixture(root);root=Path(root);mp=root/'manifests/training/lcrseg_v1_seed0.csv';sp=root/'splits/fundus_seed0.json'
    with mp.open() as f:rr=list(csv.DictReader(f))
    rr=[r for r in rr if int(r['case_id'].split('_')[-1])<2]
    with mp.open('w') as f:w=csv.DictWriter(f,fieldnames=rr[0].keys());w.writeheader();w.writerows(rr)
    sp.write_text(json.dumps(dict(seed=0,records=rr)));return sha256(mp),sha256(sp)
class Contract(unittest.TestCase):
    def tearDown(self):gc.collect()
    def test_01_masks_losses_and_invalid(self):
        z=torch.tensor([[[[0.,3.,3.]],[[0.,0.,0.]],[[0.,0.,0.]]]],dtype=torch.double,requires_grad=True)
        q=torch.tensor([[[[.1,.1,.4]],[[.8,.8,.3]],[[.1,.1,.3]]]],dtype=torch.double,requires_grad=True);g=torch.ones(1,1,3,dtype=torch.bool)
        p,qq,j,t,values=c.objectives(z,q,g);self.assertEqual(j.tolist(),[[[False,True,False]]]);self.assertEqual(t.tolist(),[[[True,True,False]]]);self.assertFalse(j.requires_grad);self.assertFalse(qq.requires_grad)
        for loss in ('MSE','SCE'):
            expected=torch.autograd.grad(.5*c.reduce_loss(values[loss],t),z,retain_graph=True)[0];raw,observed=c.gradients(p,qq,t,.5,loss);self.assertTrue(torch.allclose(expected,observed,atol=1e-14,rtol=1e-12))
        strict=q.detach().clone();strict[0,:,0,0]=torch.tensor([.7,.2,.1],dtype=torch.double);self.assertFalse(c.objectives(z,strict,g)[3][0,0,0])
        self.assertIsNone(q.grad);self.assertEqual(float(c.reduce_loss(values['SCE'],~g)),0)
        self.assertEqual(float(torch.autograd.grad(c.reduce_loss(values['SCE'],~g),z,retain_graph=True)[0].norm()),0)
        # Exact binary arithmetic witness independent of the 3-class production API.
        pp=torch.tensor([.999,.001],dtype=torch.double).view(1,2,1,1);qq=pp.flip(1);m=torch.ones(1,1,1,dtype=torch.bool)
        gm=c.gradients(pp,qq,m,1.,'MSE')[0];gs=c.gradients(pp,qq,m,1.,'SCE')[0]
        self.assertTrue(torch.allclose(gm.flatten(),torch.tensor([.003988008,-.003988008],dtype=torch.double),atol=1e-14));self.assertAlmostEqual(float(gs.norm()/gm.norm()),250.25025025,places=7)
        self.assertEqual(float(c.gradients(pp,pp,m,1.,'SCE')[0].norm()),0)
        zeros=torch.zeros_like(q.detach());zeros[:,0]=1;v=c.objectives(z,zeros,g)[4];self.assertTrue(torch.isfinite(v['SCE']).all());self.assertEqual(float(v['entropy'].sum()),0)
        bad=[(z,q.float(),g),(z,q,torch.ones_like(g,dtype=torch.int64)),(z[:,:2],q[:,:2],g),(z*float('nan'),q,g),(z,q*0,g),(z,q[:,:,:,:2],g)]
        for zz,qq,gg in bad:
            with self.assertRaises((ValueError,TypeError)):c.objectives(zz,qq,gg)
        Previous('test_01_scoring').test_01_scoring()
        with self.assertRaises(ValueError):c.validate_batch(dict(image=torch.zeros(1,3,2,2),geometry=torch.ones(1,2,2,dtype=torch.bool),label=torch.ones(1,2,2)),True)
    def test_02_jmse_golden(self):
        with tempfile.TemporaryDirectory() as t:
            ex=small_fixture(t);ds=c.DomainData(t,c.DOMAINS[0],'train_labeled',expected=ex,shape=(32,32));lb=c.batch(ds,[0,1],DEVICE,31,21,0,'labeled');ub={k:v for k,v in lb.items() if k!='label'};states=[]
            for new in (False,True):
                m=c.build(REFERENCE,DEVICE,31,c.DOMAINS[0],'J_MSE');teacher=c.ema_from(m);opt=c.optimizer_for(m);ct=Counter()
                fn=c.train_step if new else c.old.train_step;arm='J_MSE' if new else 'LIN_MT_CONF'
                row=fn(m,teacher,opt,lb,ub,None,None,arm,31,c.DOMAINS[0],21,0,ct)
                states.append((row['loss'],row['accepted'],c.state_hash(m),c.state_hash(teacher),c.optimizer_hash(opt)));del m,teacher,opt
            self.assertEqual(states[0],states[1])
    def test_03_canonical_warmup_resume_deploy(self):
        Foundation('test_01_binding_roles_scoring').test_01_binding_roles_scoring()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);ex=small_fixture(root/'data');kw=dict(data=str(root/'data'),reference=REFERENCE,domain=c.DOMAINS[0],seed=31,device=DEVICE,expected=ex,shape=(32,32),qualification=True)
            before=d.rng_hash();warm=[]
            for arm in c.ARMS:
                e.run(output=root/arm,arm=arm,epochs=20,**kw);warm.append(r.read(root/arm/'warmup.json'))
            for k in ('student_hash','ema_hash','optimizer_hash','label_order_hash'):self.assertEqual(len({x[k] for x in warm}),1)
            self.assertTrue(all(x['u_opens']==0 for x in warm));self.assertEqual(before,d.rng_hash())
            e.run(output=root/'full',arm='T_SCE',epochs=26,**kw);e.run(output=root/'split',arm='T_SCE',epochs=26,stop_epoch=21,**kw);e.run(output=root/'split',arm='T_SCE',epochs=26,resume=True,**kw)
            aa=r.read(root/'full/receipt.json');bb=r.read(root/'split/receipt.json')
            for k in ('student_hash','ema_hash','label_order_hash','labeled_opens','unlabeled_opens','head_mode','counters'):self.assertEqual(aa[k],bb[k],k)
            ck_hash=[]
            from types import SimpleNamespace
            for name in ('full','split'):
                ck=torch.load(root/name/'latest.pt',map_location='cpu',weights_only=False)
                ck_hash.append((c.optimizer_hash(SimpleNamespace(state_dict=lambda:ck['optimizer'])),bytes(ck['rng']['cpu'].numpy()),ck['epoch'],ck['head_mode']))
                del ck
            self.assertEqual(ck_hash[0],ck_hash[1])
            self.assertEqual(before,d.rng_hash());self.assertTrue(r.read(root/'full/eval26/receipt.json')['state_preserved'])
            (root/'full/latest.pt').rename(root/'protected.pt');deploy(root/'full',str(root/'data'),REFERENCE,DEVICE,qualification=True,expected=ex,shape=(32,32))
    def test_04_gt_boundary_and_directions(self):
        z=torch.tensor([[[[0.,3.]],[[0.,0.]],[[0.,0.]]]],dtype=torch.float32);q=torch.tensor([[[[.1,.8]],[[.8,.1]],[[.1,.1]]]]);g=torch.ones(1,1,2,dtype=torch.bool)
        p,q,j,t,seal=d.seal_prediction(z,q,g);gt=np.array([[1,1]])
        paths,grads=d.path_rows(p,q,g,j,t,gt,.5)
        correction=next(x for x in paths if x['grouping']=='teacher_predicted_class' and x['class_id']==1 and x['stratum']=='teacher_correct_student_wrong')
        self.assertEqual((correction['raw_support'],correction['T_accepted'],correction['J_accepted']),(1,1,0))
        g0=d.path_rows(p,q,g,j,t,gt,0)[1];self.assertTrue(all(x['weighted_local_logit_norm']==0 and x['weighted_cosine_GT_mean'] is None for x in g0));self.assertEqual(d.seal_prediction(z,q,g)[4],seal)
        bad_gt=np.full_like(gt,255);self.assertTrue(all(x['raw_support']==0 for x in d.path_rows(p,q,g,j,t,bad_gt,.5)[0]));self.assertEqual(d.seal_prediction(z,q,g)[4],seal)
    def test_05_decision_fixed_candidate(self):
        table=[dict(seed=s,domain=d,arm=a,epoch=100,model='student',macro_fg_dice=.52 if a=='T_SCE' else .5,rim_dice=.52 if a=='T_SCE' else .5,cup_dice=.52 if a=='T_SCE' else .5) for s in c.SEEDS for d in c.DOMAINS for a in c.ARMS]
        self.assertEqual(r.decision(table,True)['status'],'TARGET_PATH_SSL_DEVELOPMENT_SIGNAL');self.assertEqual(len(r.factorial(table)),120)
        for x in table:
            if x['seed']==33 and x['arm']=='T_SCE':x['macro_fg_dice']=.499
            if x['arm']=='T_MSE':x['macro_fg_dice']=.8
        self.assertEqual(r.decision(table,True)['status'],'TARGET_PATH_SSL_NOT_ESTABLISHED');self.assertEqual(r.decision(table,True)['main_candidate'],'T_SCE')
