import csv,gc,json,os,tempfile,unittest
from pathlib import Path
from collections import Counter
import numpy as np
import torch,h5py
from experiments.lcrseg.ams_seq_transfer_v0_1 import core as c,engine as e,results as stats
from experiments.lcrseg.ams_seq_transfer_v0_1.contract import task_by_id,admit,protocol
from experiments.lcrseg.ams_seq_transfer_v0_1.evaluate import evaluate
from experiments.lcrseg.di_dmpa_gate1.binding import sha256
DEVICE=torch.device(os.environ.get('SF_DEVICE','cpu'));REFERENCE=os.environ['SF_REFERENCE']

def fixture(root):
    root=Path(root);rows=[]
    for d in c.DOMAINS:
        for role,n in [('train_labeled',4),('train_unlabeled',3),('val',2)]:
            for i in range(n):
                ident=f'{d}_{role}_{i}';y=np.zeros((24,24),np.uint8);y[3:21,3:21]=1;y[8:16,8:16]=2;y[0,0]=255
                x=np.stack([np.minimum(y,2)*80,np.broadcast_to(np.arange(24,dtype=np.uint8)[:,None],y.shape),np.full_like(y,50)])
                r=dict(case_id=ident,patient_id=ident,dataset='fundus',split_seed='0',site_or_vendor=d,primary_20pct_split=role,label_h5_relpath='',label_sha256='')
                for name,a in [('image',x)]+([] if role=='train_unlabeled' else [('label',y)]):
                    rel=f'{name}s/{ident}.h5';path=root/'h5/v1'/rel;path.parent.mkdir(parents=True,exist_ok=True)
                    with h5py.File(path,'w') as h:h[name]=a
                    r[name+'_h5_relpath']=rel;r[name+'_sha256']=sha256(path)
                rows.append(r)
    mp=root/'manifests/training/lcrseg_v1_seed0.csv';mp.parent.mkdir(parents=True)
    with mp.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    sp=root/'splits/fundus_seed0.json';sp.parent.mkdir();sp.write_text(json.dumps(dict(seed=0,records=rows)))
    (root/'SYNTHETIC_FIXTURE.json').write_text('{"synthetic":true}')
    return (sha256(mp),sha256(sp))

class Contract(unittest.TestCase):
    def tearDown(self):gc.collect()
    def test_math_and_source_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            ex=fixture(tmp);d=c.DOMAINS[0];ds=c.DomainData(tmp,d,'train_labeled',shape=(24,24),expected=ex);lb=c.batch(ds,[0,1],DEVICE,61,21,0,'labeled');ub={k:v[:1] for k,v in lb.items() if k!='label'}
            for a in ('T_CE','T_CED','T_UCTX','T_AMS'):
                results=[]
                for new in (False,True):
                    m=c.build(REFERENCE,DEVICE,61,d,'SUP_CE');t=c.ema_from(m);opt=c.optimizer_for(m);counts=Counter();u=ub if a in ('T_UCTX','T_AMS') else None
                    row=c.step(m,t,opt,lb,u,a,61,d,21,0,counts,['p0','p1']) if new else c.mathcore.train_step(m,t,opt,lb,u,None,None,c.MAPPING[a],61,d,21,0,counts)
                    results.append((row['loss'],c.state_hash(m),c.state_hash(t),c.optimizer_hash(opt)));del m,t,opt;gc.collect()
                self.assertEqual(results[0],results[1])
            donor=c.donor_batch(lb,['p0','p1']);self.assertNotIn('label',donor);self.assertTrue(torch.equal(donor['image'],lb['image'].flip(0)))
            with self.assertRaises(ValueError):c.donor_batch(lb,['p0','p0'])
            m=c.build(REFERENCE,DEVICE,61,d,'SUP_CE');t=c.ema_from(m);opt=c.optimizer_for(m);counts=Counter()
            row=c.step(m,t,opt,lb,None,'T_LCTX',61,d,21,0,counts,['p0','p1'])
            self.assertEqual(row['U_image_records'],0);self.assertEqual(row['unlabeled_unique_batch_sources'],0);self.assertEqual(row['SCE'],0);self.assertEqual(counts['ema_u_forward'],0);self.assertEqual(row['labeled_source_scoring_multiplicity'],1);self.assertEqual(row['donor_context_records'],2);self.assertEqual(row['anchor_label_records'],2)
            del m,t,opt
            m=c.build(REFERENCE,DEVICE,61,d,'SUP_CE');t=c.ema_from(m);opt=c.optimizer_for(m)
            losses=[c.step(m,t,opt,lb,None,'T_CED',61,d,1,i,Counter(),['p0','p1'])['loss'] for i in range(8)]
            self.assertLess(losses[-1],losses[0]);del m,t,opt

    def test_boundary_warmup_resume_and_deploy(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);data=base/'data';expected=fixture(data)
            source=dict(task_id='source',order='O1',seed=61,arm='SRC_CE',domain=c.DOMAINS[0],source_domain=c.DOMAINS[0],source_task_id=None,updates=4)
            def kw(task,epochs):return dict(base=base,task_id=task['task_id'],data=data,reference=REFERENCE,device=DEVICE,fixture=dict(task=task,epochs=epochs,shape=(24,24),expected=expected))
            sr=e.train(**kw(source,2));gc.collect()
            target=lambda tid,a:dict(task_id=tid,order='O1',seed=61,arm=a,domain=c.DOMAINS[1],source_domain=c.DOMAINS[0],source_task_id='source',updates=44)
            warms=[]
            for arm in ('T_CED','T_LCTX','T_UCTX','T_AMS'):
                task=target(arm,arm);r=e.train(**kw(task,20));gc.collect();warms.append(json.loads((base/'tasks'/arm/'warmup.json').read_text()))
                self.assertEqual(r['boundary']['student_hash'],sr['student_hash']);self.assertEqual(r['boundary']['EMA_hash'],sr['student_hash']);self.assertTrue(r['boundary']['optimizer_state_empty']);self.assertEqual(r['U_opens'],0)
            for k in ('student_hash','EMA_hash','optimizer_hash','label_order_hash'):self.assertEqual(len({r[k] for r in warms}),1)
            # Two equivalent trajectories interrupted within epoch21 or on its boundary.
            full=e.train(**kw(target('full','T_AMS'),22));gc.collect()
            for pos in (41,42):
                task=target('split'+str(pos),'T_AMS');e.train(**kw(task,22),stop_after=pos);gc.collect();part=e.train(**kw(task,22),resume=True);gc.collect()
                for k in ('student_hash','EMA_hash','optimizer_hash','label_order_hash','counts','L_opens','U_opens'):self.assertEqual(full[k],part[k],k)
                self.assertEqual((base/'tasks/full/steps.jsonl').read_text(),(base/'tasks'/task['task_id']/'steps.jsonl').read_text())
            # Delete access to EMA/Adam recovery file; standalone final student suffices.
            (base/'tasks/full/latest.pt').rename(base/'held_resume.pt')
            result=evaluate(base,'full',data,REFERENCE,DEVICE,kw(target('full','T_AMS'),22)['fixture']);self.assertEqual(result['models'],1);self.assertEqual(result['seen_domains'],list(c.DOMAINS))
            with c.training_access(c.DOMAINS[1]):
                with self.assertRaises(PermissionError):c.DomainData(data,c.DOMAINS[0],'train_labeled',expected=expected)
                with self.assertRaises(PermissionError):c.DomainData(data,c.DOMAINS[1],'val',evaluator=True,expected=expected)
            with self.assertRaises(PermissionError):e.start_models(REFERENCE,DEVICE,{**target('bad','T_AMS'),'seed':62},base/'tasks/source/deploy_student.pt','SYNTHETIC')

    def test_matrix_and_paired_arithmetic(self):
        p=protocol();self.assertEqual(len(p['tasks']),36);self.assertEqual(sum(t['updates'] for t in p['tasks']),95400)
        self.assertEqual(sum(t['updates'] for t in p['tasks'] if t['arm']=='SRC_CE'),15900)
        with self.assertRaises(PermissionError):task_by_id('O1_s0_T_AMS')
        with self.assertRaises(PermissionError):admit('/tmp/not-NAS','O1_s61_T_AMS','fake')
        a=stats.trajectory(.7,.5,.8);b=stats.trajectory(.7,.6,.7);delta=a-b
        self.assertAlmostEqual(delta[3],-delta[2]);self.assertAlmostEqual(delta[0],.5*(delta[1]+delta[2]))
        self.assertLess(stats.trajectory(.5,.6,.7)[3],0)
        x=np.array([[[.01]*4,[.02]*4,[.03]*4],[[.03]*4,[.04]*4,[.05]*4]])
        mean,sd=stats.effect_summary(x);np.testing.assert_allclose(mean,.03);np.testing.assert_allclose(sd,.01)
        values={'T_AMS':np.tile([.02,.05,-.01,.01],(2,3,1)),'T_CE':np.zeros((2,3,4))}
        costs=[dict(arm='T_AMS',baseline='T_CE',order=o,role=r,metric=m,delta=0) for o in ('O1','O2') for r in ('old','current') for m in ('rim_dice','cup_dice')]
        self.assertFalse(stats.verdict(values,costs,'T_AMS','T_CE')['checks']['Old'])
