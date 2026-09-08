import gc,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from experiments.lcrseg.tests.single_teacher_scd_v0_1.test_contract import fixture,REFERENCE,DEVICE
from experiments.lcrseg.single_teacher_scd_v0_1 import engine as old,data as od,objectives as obj
from experiments.lcrseg.single_teacher_scd_r1 import engine as r1
from experiments.lcrseg.single_teacher_scd_v0_1.evaluate import evaluate
from experiments.lcrseg.single_teacher_scd_v0_1.deploy import export
from experiments.lcrseg.l05_ssl_replication import engine as new,results
from experiments.lcrseg.l05_ssl_replication.execute import run_replication

def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b

def counters():return dict(forward=0,gas_autograd=0,diagnostic_autograd=0,backward=0,optimizer_steps=0)

class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from experiments.lcrseg.di_dmpa_jascl.modeling import _official_probabilistic_classifier
        _official_probabilistic_classifier(REFERENCE,upstream_path=old.UPSTREAM_PATH);old.precision()

    def test_objectives_actual_parity_and_forbidden_teacher(self):
        for arm,weights in [('S',[1,0,0,0]),('L05',[1,0,.5,0]),('L05_SSL',[1,.5,.5,0])]:
            xs=[torch.tensor(float(i+1),requires_grad=True,dtype=torch.double) for i in range(4)]
            loss,_=new.objective(*xs,arm,20);loss.backward()
            self.assertEqual([0 if x.grad is None else x.grad.item() for x in xs],weights)
        x=torch.rand(2,3,32,32,device=DEVICE);y=torch.randint(3,(2,32,32),device=DEVICE)
        lb=dict(image=x,label=y,geometry=torch.ones_like(y,dtype=torch.bool));ub=dict(image=1-x,geometry=lb['geometry'])
        out={}
        for mode in ('oldS','S','oldL05','L05','P_zero','P_positive'):
            arm='S' if mode in ('oldS','S') else 'L05_SSL' if mode.startswith('P') else 'L05'
            with new.lifecycle(arm,0):
                s=old.model(REFERENCE,DEVICE);t=old.second_model(s,'S' if arm=='S' else 'D');opt=old.optimizer_for(s);c=counters()
                if t is not None:
                    th=old.state_hash(t)
                    def teacher_only_labeled(module,args):
                        if args[0].data_ptr()!=lb['image'].data_ptr():raise AssertionError('actual teacher-U input')
                    hook=t.register_forward_pre_hook(teacher_only_labeled)
                for step in range(2):
                    params=(s,t,opt,lb,ub if arm=='L05_SSL' else None,torch.zeros(3,16,device=DEVICE),torch.ones(3,dtype=torch.bool,device=DEVICE))
                    forced=lambda z,f,p,s,g:(z.detach().argmax(1),g,z.detach().softmax(1))
                    with patch.object(obj,'pas',forced):
                        if mode=='oldS':info=ORIGINAL_STEP(*params,'S',1,20,step,c)
                        elif mode=='oldL05':info=r1.train_step(*params,'L05',1,20,step,c)
                        elif mode=='P_zero':
                            with patch.object(obj,'lambda_u',lambda e:0.):info=new.train_step(*params,arm,1,20,step,c)
                        else:info=new.train_step(*params,arm,1,20,step,c,diagnostic=mode=='P_positive')
                self.assertEqual(c['teacher_u_forward'],0)
                self.assertEqual(c['forward'],2 if arm=='S' else 8 if arm=='L05_SSL' else 4)
                if t is not None:
                    self.assertEqual(old.state_hash(t),th);self.assertTrue(all(p.grad is None for p in t.parameters()));hook.remove()
                if mode=='P_positive':self.assertGreater(info['gradients']['raw_lu'],0.)
                old.audit_live_models(1 if arm=='S' else 2)
                out[mode]=(old.state_hash(s),opt.state_dict(),torch.get_rng_state().clone())
                del s,t,opt,params;gc.collect()
        for a,b in [('oldS','S'),('oldL05','L05'),('L05','P_zero')]:equal(out[a],out[b])
        self.assertNotEqual(out['P_zero'][0],out['P_positive'][0])
        with new.lifecycle('L05_SSL',0):
            with self.assertRaises(PermissionError):old.forward(None,lb['image'],False,('teacher_unlabeled',),counters())
            with self.assertRaises(PermissionError):obj.project(None,None,None,None)

    def test_seed_streams_and_split_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected=fixture(tmp);ds=od.CurrentData(tmp,1,'train_labeled',expected=expected,shape=(32,32))
            baseline=(old.stable_seed(0,'initialization'),od.batches(16,1,3,'labeled_order',8),od.load_batch(ds,[0,1],1,3,2,'labeled',DEVICE),od.strong(torch.linspace(0,1,3*32*32,device=DEVICE).reshape(1,3,32,32),1,3,2))
            values={}
            for seed in (0,1,2):
                for arm in new.RECIPES:
                    with new.lifecycle(arm,seed):
                        v=(old.stable_seed(0,'initialization'),old.batches(16,1,3,'labeled_order',8),old.load_batch(ds,[0,1],1,3,2,'labeled',DEVICE),old.strong(torch.linspace(0,1,3*32*32,device=DEVICE).reshape(1,3,32,32),1,3,2))
                        s=old.model(REFERENCE,DEVICE);initial=old.state_hash(s)
                        z=old.forward(s,v[2]['image'],True,(1,3,2,'supervised_head'),counters())[0]
                        gas=torch.autograd.grad(obj.supervised(z,v[2]['label']),s.decoder.conv_logit.mu.weight)[0].detach().clone()
                        values[seed,arm]=(v,initial,gas)
                        del s,z;gc.collect()
                        ud=old.CurrentData(tmp,1,'train_unlabeled',expected=expected,shape=(32,32))
                        if arm!='L05_SSL':
                            with self.assertRaises(PermissionError):ud[0]
                            self.assertFalse(hasattr(ud,'rows'))
                        else:self.assertNotIn('label',ud[0]);self.assertNotIn('label_h5_relpath',ud.rows[0])
                        for domain in (0,2):
                            with self.assertRaises(PermissionError):old.CurrentData(tmp,1,'train_labeled',domain=domain,expected=expected)
                        with self.assertRaises(PermissionError):old.CurrentData(tmp,1,'test',purpose='evaluate',expected=expected)
            equal(values[0,'S'][0],baseline)
            for seed in (0,1,2):
                for arm in ('L05','L05_SSL'):equal(values[seed,'S'],values[seed,arm])
            for seed in (1,2):
                self.assertNotEqual(values[0,'S'][1],values[seed,'S'][1])
                self.assertNotEqual(values[0,'S'][0][1],values[seed,'S'][0][1])
                self.assertFalse(torch.equal(values[0,'S'][0][3],values[seed,'S'][0][3]))
                self.assertFalse(torch.equal(values[0,'S'][2],values[seed,'S'][2]))
            self.assertNotEqual(values[1,'S'][1],values[2,'S'][1])
            # Multiple draws make accidental equal flip/rotation on one batch irrelevant.
            draws=[]
            for seed in (0,1,2):
                with new.lifecycle('L05_SSL',seed):draws.append([old.load_batch(ds,[0,1],1,3,j,'labeled',DEVICE)['image'] for j in range(5)])
            for a,b in ((0,1),(0,2),(1,2)):self.assertTrue(any(not torch.equal(x,y) for x,y in zip(draws[a],draws[b])))
            self.assertEqual(json.loads((Path(tmp)/'splits/fundus_seed0.json').read_text())['seed'],0)

    def test_full_lifecycle_resume_lineage_and_deploy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);expected=fixture(root/'data')
            kw=dict(data=root/'data',reference=REFERENCE,device=DEVICE,expected=expected,shape=(32,32),qualification=True,epochs=2,optimization_seed=1)
            c=new.run_stage(**kw,arm='S',stage=0,output=root/'common')
            common=root/'common/student_latest.pt'
            for arm in new.RECIPES:
                folder=root/arm;folder.mkdir()
                a=new.run_stage(**kw,arm=arm,stage=1,parent=common,output=folder/'stage1')
                self.assertEqual(a['parent_hash'],c['student_hash']);self.assertEqual(a['counters']['teacher_u_forward'],0)
                if arm!='L05_SSL':self.assertEqual(a['unlabeled_images_opened'],0)
                for stage in (1,2):
                    if stage==2:new.run_stage(**kw,arm=arm,stage=2,parent=folder/'stage1/student_latest.pt',output=folder/'stage2')
                    ev=evaluate(data=root/'data',reference=REFERENCE,checkpoint=folder/f'stage{stage}/student_latest.pt',output=folder/f'stage{stage}/val.json',device=DEVICE,expected=expected,shape=(32,32))
                    self.assertEqual(len(ev['rows']),stage+1)
                export(folder/'stage2/student_latest.pt',REFERENCE,device=DEVICE)
                with self.assertRaises(FileExistsError):new.run_stage(**kw,arm=arm,stage=2,parent=folder/'stage1/student_latest.pt',output=folder/'stage2',resume=True)
            kw['epochs']=12
            a=new.run_stage(**kw,arm='L05_SSL',stage=1,parent=common,output=root/'full')
            new.run_stage(**kw,arm='L05_SSL',stage=1,parent=common,output=root/'resume',stop_after_epoch=11)
            b=new.run_stage(**kw,arm='L05_SSL',stage=1,parent=common,output=root/'resume',resume=True)
            self.assertEqual(a['student_hash'],b['student_hash'])
            pa=torch.load(root/'full/student_latest.pt',map_location='cpu',weights_only=False);pb=torch.load(root/'resume/student_latest.pt',map_location='cpu',weights_only=False)
            for k in ('student','optimizer','prototypes','supported','cpu_rng','cuda_rng','counters','epoch','optimization_seed','data_split_seed'):equal(pa[k],pb[k])
            self.assertTrue(pa['supported'].any())
            with self.assertRaises(PermissionError):new.parent_identity(pa,'L05',2,1)
            with self.assertRaises(PermissionError):new.parent_identity(pa,'L05_SSL',2,2)

    def test_original_scorer_exact_parity(self):
        from experiments.lcrseg.shor_v0_4_test import case_metrics as original
        from experiments.lcrseg.single_teacher_scd_v0_1.evaluate import case_metrics
        rng=np.random.default_rng(2718)
        for n in range(25):
            prediction=rng.integers(0,3,(13,17))
            truth=rng.choice([0,1,2,255],size=(13,17)) if n>1 else np.full((13,17),255 if n==0 else 0)
            reference=original(prediction,truth);actual=case_metrics(prediction,truth)
            for key,value in reference.items():self.assertEqual(actual[key],value)

    def test_scheduler_gate_and_full_precision(self):
        for admitted in (False,True):
            calls=[]
            def common(s):calls.append((s,'common'));return f'common{s}'
            def train(s,a,p):calls.append((s,a));self.assertEqual(p,f'common{s}');return dict(status='COMPLETE',science='NOT_ESTABLISHED')
            for s in (1,2):run_replication(s,common,train,lambda:admitted)
            for s in (1,2):
                self.assertIn((s,'S'),calls);self.assertIn((s,'L05'),calls);self.assertEqual((s,'L05_SSL') in calls,admitted)
        rows=[]
        for s in (0,1,2):
            for a in ('S','L05','L05_SSL'):
                for t in (0,1,2):
                    for d in range(t+1):
                        v=.5+(.02 if a=='L05' else .04 if a=='L05_SSL' else 0)
                        rows.append(dict(optimization_seed=s,arm=a,stage=t,domain_index=d,macro_fg_dice=v,rim_dice=v,cup_dice=v))
        for kind in ('P_GATE','L05_REPLICATION','SSL_REPLICATION'):self.assertTrue(results.gate(rows,kind)['passed'])
        for x in rows:
            if x['optimization_seed']==0 and x['arm']=='L05_SSL' and x['stage']==2:x['macro_fg_dice']=.5299999999
        self.assertFalse(results.gate(rows,'P_GATE')['passed'])
        historical=results.historical();self.assertEqual(results.score(historical,0,'L05')['F'],.6204323659946968)
        self.assertFalse(results.gate(rows,'P_GATE',engineering=False)['passed'])

ORIGINAL_STEP=old.train_step
