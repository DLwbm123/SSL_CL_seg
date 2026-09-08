import csv,gc,json,os,tempfile,unittest
from pathlib import Path
from collections import Counter
import h5py,numpy as np,torch
from experiments.lcrseg.ssl_anchored_mix_v0_1 import core as c,engine as e,statistics as stats
from experiments.lcrseg.ssl_anchored_mix_v0_1.deploy import deploy
from experiments.lcrseg.di_dmpa_gate1.binding import sha256,ProtocolError
DEVICE=torch.device(os.environ.get('SF_DEVICE','cpu'));REFERENCE=os.environ['SF_REFERENCE']

def fixture(root):
    root=Path(root);rr=[]
    for domain in c.DOMAINS:
        for role,n in [('train_labeled',2),('train_unlabeled',1),('val',2)]:
            for i in range(n):
                ident=f'{domain}_{role}_{i}';y=np.zeros((24,24),np.uint8);y[3:21,3:21]=1;y[8:16,8:16]=2
                x=np.stack([y*80,np.broadcast_to(np.arange(24,dtype=np.uint8)[:,None],y.shape),np.full_like(y,50)])
                r=dict(case_id=ident,patient_id=ident,dataset='fundus',split_seed='0',site_or_vendor=domain,primary_20pct_split=role,label_h5_relpath='',label_sha256='')
                for name,a in [('image',x)]+([] if role=='train_unlabeled' else [('label',y)]):
                    rel=f'{name}s/{ident}.h5';p=root/'h5/v1'/rel;p.parent.mkdir(parents=True,exist_ok=True)
                    with h5py.File(p,'w') as f:f[name]=a
                    r[name+'_h5_relpath']=rel;r[name+'_sha256']=sha256(p)
                    decoy=root/rel;decoy.parent.mkdir(parents=True,exist_ok=True);decoy.write_bytes(b'wrong dataset root')
                rr.append(r)
    mp=root/'manifests/training/lcrseg_v1_seed0.csv';mp.parent.mkdir(parents=True)
    with mp.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    sp=root/'splits/fundus_seed0.json';sp.parent.mkdir();sp.write_text(json.dumps(dict(seed=0,records=rr)))
    return sha256(mp),sha256(sp)

class Contract(unittest.TestCase):
    def tearDown(self):gc.collect()
    def test_dice_and_ce(self):
        z=torch.randn(2,3,3,3,dtype=torch.double,requires_grad=True);y=torch.tensor([[[1,2,0],[1,255,0],[0,0,0]],[[255]*3]*3])
        ce,dice=c.supervised_parts(z.log_softmax(1),y);self.assertTrue(torch.equal(ce,c.predecessor.supervised(z,y)))
        p=z.softmax(1);oracle=[]
        for b in range(2):
            valid=y[b]!=255;ds=[]
            for cls in [1,2]:
                pred=p[b,cls][valid];t=(y[b][valid]==cls).double();ds.append((2*(pred*t).sum()+1e-5)/(pred.sum()+t.sum()+1e-5))
            oracle.append(1-torch.stack(ds).mean() if valid.any() else p[b].sum()*0)
        self.assertTrue(torch.allclose(dice,torch.stack(oracle).mean(),atol=1e-14))
        g=torch.autograd.grad(ce+dice,z)[0];self.assertEqual(float(g[1].abs().sum()),0);self.assertEqual(float(g[0,:,1,1].abs().sum()),0)
        for cls in [0,1,2,255]:
            zz=z.detach().clone().requires_grad_();cc,dd=c.supervised_parts(zz.log_softmax(1),torch.full_like(y,cls));gg=torch.autograd.grad(cc+dd,zz)[0];self.assertTrue(torch.isfinite(gg).all())
            if cls==255:self.assertEqual(float(cc+dd),0);self.assertEqual(float(gg.abs().sum()),0)
        # Stable FP32 versus independent FP64 sums and gradients.
        zf=z.detach().float().requires_grad_();v=sum(c.supervised_parts(zf.log_softmax(1),y));self.assertTrue(torch.isfinite(v));self.assertTrue(torch.allclose(v.double(),ce+dice,atol=2e-6))
        with self.assertRaises(ValueError):c.supervised_parts(z.log_softmax(1),y.float())

    def test_complementary_sources_odd_and_gradient(self):
        key=(31,c.DOMAINS[0],21,0);before=torch.get_rng_state().clone()
        m,coords=c.rectangles(2,24,24,key,DEVICE);self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertTrue(torch.equal(m,c.rectangles(2,24,24,key,DEVICE)[0]));self.assertEqual(m.sum().item(),2*16*16)
        l=torch.arange(2*3*24*24,device=DEVICE).reshape(2,3,24,24).double();u=-l[:1]-1;ix,rw=c.pairing(2,1,DEVICE)
        a=torch.where(m[:,None],l,u[ix]);b=torch.where(m[:,None],u[ix],l)
        for mask,x,y in [(m,a,b),(~m,b,a)]:
            ll,uu=c.collect_sources(x,y,mask);self.assertTrue(torch.equal(ll,l));self.assertTrue(torch.equal(uu,u[ix]))
        za=torch.randn_like(l,requires_grad=True);zb=torch.randn_like(l,requires_grad=True);q=torch.tensor([.8,.2,0.],device=DEVICE,dtype=torch.double).view(1,3,1,1).expand(1,3,24,24).clone().requires_grad_();geo=torch.ones(1,24,24,dtype=torch.bool,device=DEVICE)
        lp,up=c.collect_sources(za.log_softmax(1),zb.log_softmax(1),m);loss,ent,mask=c.soft_target(up,q,geo,ix,rw)
        oracle=sum((-(q.detach()[0]*up[b]).sum(0)*.5).sum() for b in range(2))/(24*24)
        self.assertTrue(torch.allclose(loss,oracle,atol=1e-12));ga,gb,gq=torch.autograd.grad(loss,[za,zb,q],allow_unused=True,retain_graph=True);self.assertIsNone(gq)
        self.assertEqual(float((ga*m[:,None]).abs().sum()),0);self.assertEqual(float((gb*(~m)[:,None]).abs().sum()),0);self.assertGreater(float(ga.abs().sum()+gb.abs().sum()),0)
        # MIX_CTX source objective sends direct gradients only to L-owned output pixels.
        gt=torch.ones(2,24,24,dtype=torch.long,device=DEVICE);ctx=sum(c.supervised_parts(lp,gt))+up.sum()*0
        ca,cb=torch.autograd.grad(ctx,[za,zb]);self.assertEqual(float((ca*(~m)[:,None]).abs().sum()),0);self.assertEqual(float((cb*m[:,None]).abs().sum()),0)
        zero=c.soft_target(up,q,~geo,ix,rw)[0];self.assertEqual(float(zero),0);self.assertTrue(torch.isfinite(loss+ent));self.assertFalse(mask.requires_grad)
        strict=q.detach().clone();strict[:,0]=.7;strict[:,1]=.3;self.assertFalse(c.soft_target(up,strict,geo,ix,rw)[2].any())

    def test_shared_engine_compatibility_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ex=fixture(root/'data');kw=dict(data=str(root/'data'),reference=REFERENCE,domain=c.DOMAINS[0],seed=31,device=DEVICE,expected=ex,shape=(24,24),qualification=True)
            ds=c.DomainData(root/'data',c.DOMAINS[0],'train_labeled',expected=ex,shape=(24,24));self.assertEqual(ds[0]['image'].shape,(3,24,24))
            p=root/'data/h5/v1'/ds._dataset.rows[1]['image_h5_relpath'];p.rename(str(p)+'.held')
            with self.assertRaises(ProtocolError):ds[1]
            Path(str(p)+'.held').rename(p)
            warm=[]
            for arm in c.ARMS:
                e.run(output=root/arm,arm=arm,epochs=20,**kw);warm.append(json.loads((root/arm/'warmup.json').read_text()))
            for key in ['student_hash','ema_hash','optimizer_hash','label_order_hash']:self.assertEqual(len({r[key] for r in warm}),1)
            self.assertTrue(all(r['u_opens']==0 for r in warm))
            # Exact predecessor SUP_CE optimizer/RNG behavior on one fixed batch.
            lb=c.batch(ds,[0,1],DEVICE,31,21,0,'labeled');states=[]
            for old in [True,False]:
                m=c.build(REFERENCE,DEVICE,31,c.DOMAINS[0],'SUP_CE');t=c.ema_from(m);opt=c.optimizer_for(m)
                fn=c.predecessor.train_step if old else c.train_step
                fn(m,t,opt,lb,None,None,None,'SUP' if old else 'SUP_CE',31,c.DOMAINS[0],21,0,Counter())
                states.append((c.state_hash(m),c.state_hash(t),c.optimizer_hash(opt)));del m,t,opt
            self.assertEqual(states[0],states[1])
            m=c.build(REFERENCE,DEVICE,31,c.DOMAINS[0],'SUP_CED');t=c.ema_from(m);opt=c.optimizer_for(m);losses=[]
            for step in range(8):losses.append(c.train_step(m,t,opt,lb,None,None,None,'SUP_CED',31,c.DOMAINS[0],1,step,Counter())['loss'])
            self.assertLess(losses[-1],losses[0]);del m,t,opt
            e.run(output=root/'full',arm='MIX_CED',epochs=26,**kw)
            e.run(output=root/'split',arm='MIX_CED',epochs=26,stop_epoch=21,**kw)
            e.run(output=root/'split',arm='MIX_CED',epochs=26,resume=True,**kw)
            a=json.loads((root/'full/receipt.json').read_text());b=json.loads((root/'split/receipt.json').read_text())
            for k in ['student_hash','ema_hash','label_order_hash','counters','labeled_opens','unlabeled_opens']:self.assertEqual(a[k],b[k],k)
            from types import SimpleNamespace
            hashes=[]
            for name in ['full','split']:
                ck=torch.load(root/name/'latest.pt',map_location='cpu',weights_only=False);hashes.append((c.optimizer_hash(SimpleNamespace(state_dict=lambda:ck['optimizer'])),bytes(ck['rng']['cpu'].numpy())));del ck
            self.assertEqual(hashes[0],hashes[1]);self.assertEqual((root/'full/steps.jsonl').read_text(),(root/'split/steps.jsonl').read_text());self.assertTrue(json.loads((root/'full/eval26/receipt.json').read_text())['state_preserved'])
            (root/'full/latest.pt').rename(root/'hidden_teacher.pt');deploy(root/'full',str(root/'data'),REFERENCE,DEVICE,qualification=True,expected=ex,shape=(24,24))

    def test_frozen_selection_and_intervals(self):
        table=[dict(seed=s,domain=d,arm=a,**{m:(.52 if a=='MT_CED' else .521 if a=='MIX_CED' else .5) for m in stats.METRICS}) for s in (31,32,33,41,42) for d in c.DOMAINS for a in (*c.ARMS,'SUP_CE')]
        self.assertEqual(stats.select(table,True)['selected'],'MT_CED')
        for r in table:
            if r['arm']=='MIX_CED':r.update({m:.53 for m in stats.METRICS})
        self.assertEqual(stats.select(table,True)['selected'],'MIX_CED');self.assertEqual(stats.replication(table,'MIX_CED',True)['status'],'VALUE_REPRODUCED')
        self.assertIsNone(stats.select(table,False)['selected'])
        for r in table:
            if r['seed']==33 and r['domain']==c.DOMAINS[0] and r['arm']=='MIX_CED':r['cup_dice']=.449
        self.assertFalse(stats.select(table,True)['candidates']['MIX_CED']['eligible'])
        arrays={d:np.array([[[.4,.6],[.42,.62]]]*3) for d in c.DOMAINS}
        ci=stats.intervals(arrays,[31,32,33],['B','A'],[('A','B')],'P1')
        self.assertTrue(all(abs(r['lower_95']-.02)<1e-12 and abs(r['upper_95']-.02)<1e-12 for r in ci))
