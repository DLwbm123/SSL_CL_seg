"""Frozen v4 exact reduction and matched transition kernels. Original files untouched."""
import argparse,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from .posterior_reference import (Reference, normal, elliptical, gaussian_log, diagnostics,
    probability_checks, as_prob)
from .config import Config
from .stream_state import StreamState
from .fixed_inference import true_theta
from .synthetic import scores,generated_data
from .budgets import ResourceLedger

BLOCKS=((0,1,4,5),(2,3,6,7),(8,9,12,13),(10,11,14,15))

def root_matrix(inverse=False):
    factor=1/math.sqrt(1.16) if inverse else math.sqrt(1.16)
    return torch.eye(16,dtype=torch.float64)+(factor-1)/16*torch.ones(16,16,dtype=torch.float64)

def reduce_u(u):return u[...,2:18]+.1*u[...,:1]

def restore(v,g):
    c0=.1*v.sum()/1.16+normal((),g)/math.sqrt(1.16)
    return torch.cat([c0[None],normal((1,),g),v-.1*c0,normal((16,),g)])

def block_step(w,loglike,g,arm):
    count=0
    for block in (tuple(range(16)),)*4 if arm=='A' else BLOCKS:
        ids=list(block)
        def ll(v):
            candidate=w.clone();candidate[ids]=v
            return loglike(candidate)
        v,n=elliptical(w[ids],ll,g);w=w.clone();w[ids]=v;count+=n
    return w,count

class Reduced(Reference):
    def mean_v(self,v):
        self.calls+=1
        p=(6*torch.tanh(F.interpolate(v.reshape(1,1,4,4),(16,16),mode='bilinear',align_corners=False))).sigmoid().flatten()
        return torch.cat([torch.tanh(2*p-1),p.new_zeros(15)]),p

    @torch.no_grad()
    def sample(self,x,image,chain,arm,check):
        g0=torch.Generator().manual_seed(940001+image*10000+chain*101)
        g=torch.Generator().manual_seed(950001+image*10000+chain*101)
        recovery=torch.Generator().manual_seed(960001+image*10000+chain*101)
        w=normal((16,),g0)*(1.,2.,.5,3.)[chain]
        omega=torch.ones(256,dtype=torch.float64);L=root_matrix();centered=x.flatten()-self.a[1]
        ps=[];ts=[];vs=[];count=0;started=time.monotonic();density=0
        for step in range(1500):
            nv=(2*self.a[2]).exp()/omega;chol=torch.linalg.cholesky(self.C0+torch.diag(nv))
            def ll(q):
                nonlocal density
                density+=1;mean,_=self.mean_v(L@q)
                return gaussian_log(centered,self.H@mean,chol)
            w,n=block_step(w,ll,g,arm);count+=n
            if (step+1)%5==0:w,n=elliptical(w,ll,g);count+=n
            v=L@w;mean,p=self.mean_v(v);h=self.conditional_draw(mean,chol,nv,centered,g)
            rate=(self.c.nu+(centered-self.H@h).square()*(-2*self.a[2]).exp())/2
            omega=torch._standard_gamma(torch.full_like(rate,(self.c.nu+1)/2),generator=g)/rate
            if not torch.isfinite(omega).all() or not (omega>0).all():raise FloatingPointError('omega')
            if step>=500:
                u=restore(v,recovery);z=self.unpack(u)
                z['S']=h[:256].reshape(1,1,16,16);z['b']=h[256:].reshape(1,1,15)
                joint=self.model.log_joint(x[None],z,self.theta,self.a).total[0]
                ps.append(p);ts.append(torch.cat([u,h,omega,joint[None]]));vs.append(v)
            if step%100==0:check()
        return dict(probability=torch.stack(ps),trace=torch.stack(ts),v=torch.stack(vs),
                    costs=dict(seconds=time.monotonic()-started,collapsed_density_evaluations=density,
                    analytic_mean_evaluations=self.calls,chol=1500,cholesky_solves=density+1500,
                    original_joint_evaluations=1000,ellipse_proposals=count,structure_transitions=6300))

def diagnostic_groups(p,t,v):
    pd=diagnostics(p);td=diagnostics(t);vd=diagnostics(v);pc=probability_checks(p)
    gate=pd['pass_gate'] and td['pass_gate'] and pc['pass_gate']
    teacher=gate and vd['pass_gate'] and min(pd['bulk_ess_min'],pd['tail_ess_min'],vd['bulk_ess_min'],vd['tail_ess_min'])>=400
    return dict(probability=pd,full_trace=td,reduced_v=vd,probability_mc=pc,reference_pass=gate,teacher_pass=teacher)

@torch.no_grad()
def image_run(state,item,index,arm,out,ledger):
    chains=[]
    for ch in range(4):
        ref=Reduced(state.config,state.model,true_theta(state),item['a'])
        result=ref.sample(item['x'],index,ch,arm,ledger.check)
        torch.save(result,out/f'{arm}_image{index}_chain{ch}.pt');chains.append(result)
        print(json.dumps(dict(arm=arm,image=index,chain=ch,seconds=result['costs']['seconds'])),flush=True)
    p=torch.stack([r['probability'] for r in chains]);t=torch.stack([r['trace'] for r in chains]);v=torch.stack([r['v'] for r in chains])
    row=dict(arm=arm,image=index,diagnostics=diagnostic_groups(p,t,v))
    row['costs']={k:sum(c['costs'][k] for c in chains) for k in chains[0]['costs']}
    for metric in ('bulk_ess_min','tail_ess_min'):
        for divisor in ('collapsed_density_evaluations','seconds'):
            row[metric+'_per_'+divisor]=row['diagnostics']['probability'][metric]/row['costs'][divisor]
    torch.save(as_prob(p.mean((0,1)),state.config),out/f'{arm}_image{index}_PREDICTION.pt')
    return row

@torch.no_grad()
def old_analysis(old,out):
    rows=[]
    for i in range(8):
        chains=[torch.load(old/f'image{i}_chain{j}.pt',weights_only=True) for j in range(4)]
        t=torch.stack([c['trace'] for c in chains]);p=torch.stack([c['probability'] for c in chains]);v=reduce_u(t[...,:34])
        residual=(t[...,0]-.1*v.sum(-1)/1.16)*math.sqrt(1.16)
        redundant=torch.cat([residual[...,None],t[...,1:2],t[...,18:34]],-1)
        diff=p.mean(1).max(0).values-p.mean(1).min(0).values;worst=diff.topk(4).indices.tolist()
        row=dict(image=i,groups={name:diagnostics(x) for name,x in [('v',v),('redundant',redundant),('S',t[...,34:290]),('b',t[...,290:305]),('omega',t[...,305:561]),('probability',p)]},
            mean_delta_v=float((v[:,1:]-v[:,:-1]).norm(dim=-1).mean()),mean_delta_probability=float((p[:,1:]-p[:,:-1]).norm(dim=-1).mean()),
            fixed_positions=[0,85,170,255],fixed_chain_means=p.mean(1)[:,[0,85,170,255]].tolist(),
            posthoc_disagreement_positions=worst,posthoc_chain_means=p.mean(1)[:,worst].tolist(),
            foreground_mass_chain_means=p.mean((1,2)).tolist())
        pm=p.mean(-1);om=t[...,305:561].mean(-1)
        row['foreground_precision_correlations']=[float(torch.corrcoef(torch.stack([pm[j],om[j]]))[0,1]) for j in range(4)]
        rows.append(row)
    (out/'OLD_CHAIN_ANALYSIS.json').write_text(json.dumps(rows,indent=2));return rows

def dump(path,data):path.write_text(json.dumps(data,indent=2))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--old',type=Path,required=True)
    parser.add_argument('phase',choices=('p0','p1','p2'));args=parser.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    ledger=ResourceLedger(args.output,max_updates=24);state=StreamState(Config())
    if args.phase=='p0':
        from .kernel_checks import checks
        used=ledger.data['seconds_charged']
        with ledger.attempt('p0',seconds=max(.001,300-used),updates=0) as out:
            result=checks();dump(out/'CHECKS.json',result);old_analysis(args.old,out)
            dump(out/'STATUS.json',{'status':'PASS','regression_and_p0_seconds':used})
        return 0
    from .qualification import require_receipt
    for kind in ('UNIT','REVISION'):
        path=args.output/f'{kind.lower()}_{kind}_RECEIPT.json';r=json.loads(path.read_text())
        require_receipt(path,kind,Config(),r['evidence_fingerprint'])
    if json.loads((args.output/'p0/STATUS.json').read_text())['status']!='PASS':raise PermissionError('P0')
    items=torch.load(args.old/'SEALED_SYNTHETIC.pt',weights_only=True)
    if args.phase=='p1':
        with ledger.attempt('p1',seconds=1200,updates=0) as out:
            rows=[]
            for arm in ('A','B'):
                for idx in (0,6):
                    rows.append(image_run(state,items[idx],idx,arm,out,ledger));dump(out/'PROGRESS.json',rows)
            passing=[a for a in ('A','B') if all(r['diagnostics']['reference_pass'] for r in rows if r['arm']==a)]
            winner=None
            if len(passing)==1:winner=passing[0]
            if len(passing)==2:
                cost={a:sum(r['costs']['seconds'] for r in rows if r['arm']==a) for a in passing}
                if abs(cost['A']-cost['B'])/min(cost.values())>.05:winner=min(cost,key=cost.get)
                else:winner=max(passing,key=lambda a:min(r['diagnostics']['probability']['bulk_ess_min'] for r in rows if r['arm']==a))
            # Scoring follows sealed predictions and cannot select the winner.
            for r in rows:
                pred=torch.load(out/f"{r['arm']}_image{r['image']}_PREDICTION.pt",weights_only=True)
                r['diagnostic_scores']=scores(pred,items[r['image']]['y'][None])
            dump(out/'RESULTS.json',dict(winner=winner,rows=rows,status='PASS' if winner else 'NO_QUALIFIED_KERNEL'))
        return 0 if winner else 2
    prior=json.loads((args.output/'p1/RESULTS.json').read_text());arm=prior['winner']
    if arm is None:raise PermissionError('P1 no qualified kernel')
    with ledger.attempt('p2',seconds=2100,updates=0) as out:
        for d,seed in enumerate((20264101,20264102)):
            x,y,a=generated_data(Config(),d,4,seed)
            for j in range(4):items.append(dict(x=x[j],y=y[j],a=a,domain=d))
        torch.save(items[8:],out/'SEALED_NEW.pt')
        rows=[r for r in prior['rows'] if r['arm']==arm]
        for idx in list(i for i in range(8) if i not in (0,6))+list(range(8,16)):
            rows.append(image_run(state,items[idx],idx,arm,out,ledger));dump(out/'PROGRESS.json',rows)
        for r in rows:
            base=args.output/'p1' if r['image'] in (0,6) else out
            p=torch.load(base/f"{arm}_image{r['image']}_PREDICTION.pt",weights_only=True)
            r['diagnostic_scores']=scores(p,items[r['image']]['y'][None])
        fresh=[r for r in rows if r['image']>=8];information=[]
        for d in (0,1):
            subset=[r['diagnostic_scores'] for r in fresh if items[r['image']]['domain']==d]
            avg={k:sum(r[k] for r in subset)/4 for k in ('NLL','Brier','mean_Dice','predicted_foreground_fraction')}
            information.append(dict(domain=d,averages=avg,pass_gate=avg['NLL']<math.log(2)-.02 and avg['Brier']<=.5 and .02<=avg['predicted_foreground_fraction']<=.98))
        reference=all(r['diagnostics']['reference_pass'] for r in rows)
        info=all(r['pass_gate'] for r in information) and sum(r['diagnostic_scores']['NLL']<math.log(2) for r in fresh)>=7
        dump(out/'RESULTS.json',dict(rows=rows,reference_pass=reference,information=information,information_status=('PASS' if info else 'FAIL') if reference else 'UNDETERMINED',training_started=False))
    return 0 if reference and info else 2

if __name__=='__main__':raise SystemExit(main())
