"""Independent cached-objective and sealed-metric audit; no G draws or updates."""
import json
import os
from pathlib import Path
import torch
import numpy as np
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from sc_bayeseg.window_v1.r_start import metrics,write
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.g_bias_closure import bias_kl


def audit(s):
    torch.set_num_threads(4);root=Path(s['output']);read=lambda p:json.loads(p.read_text())
    r=read(root/'RESULTS.json');assert r['status']==read(root/'STATUS.json')['status']=='COMPLETE'
    assert read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    c=r['costs'];assert (c['cache_G_forward'],c['evaluation_G_forward'],c['engineering_G_forward'])==(512,4096,8)
    assert c['Adam_optimizer']==c['network_reverse']==0 and c['newton_acceptances']<=200
    assert len(r['affine_max_errors'])==8 and max(r['affine_max_errors'])<=2e-6
    source=torch.load(s['checkpoint'],map_location='cpu',weights_only=False)['model']
    mu=source['anatomy.2.mean_b'].double().cuda();ls=source['anatomy.2.logsd_b'].double().cuda()
    pm=source['anatomy.2.prior_mean_b'].double().cuda();ps=source['anatomy.2.prior_logsd_b'].double().cuda()
    sel=read(Path(s['origin'])/'PRIVATE_SELECTION.json');ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
    b=collate_labeled([ds[i] for i in sel['train_indices']]);y=F.interpolate(b.label[:,None].float(),(128,128),mode='nearest')[:,0].long();y=torch.cat([flip(y,k) for k in range(4)])
    errors=[0.];residuals={};ledger=[];candidates=0;derivatives=0
    def compare(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:compare(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,z in zip(a,b):compare(x,z)
        elif isinstance(a,float):
            assert abs(a-b)<2e-6;errors[0]=max(errors[0],abs(a-b))
        else:assert a==b
    for arm,record in r['arms'].items():
        out=root/arm
        ledger.extend(json.loads(x) for x in (out/'physical_updates.jsonl').read_text().splitlines())
        candidates+=len((out/'candidates.jsonl').read_text().splitlines());derivatives+=len((out/'derivatives.jsonl').read_text().splitlines())
        cache=np.load(out/'TRAIN_LOGITS.npy',mmap_mode='r');assert cache.shape==(256,8,128,128,4)
        for name,active in (('G1',[2]),('G3',[0,1,2])):
            receipt=record['solvers'][name];d=torch.tensor(record['readouts'][name]['delta'],device='cuda',dtype=torch.float64)
            total=256*8*16384;g=torch.zeros(3,device='cuda',dtype=torch.float64);objective=0.
            with torch.no_grad():
                for start in range(0,256,4):
                    z=torch.from_numpy(np.array(cache[start:start+4],copy=True)).cuda().double().reshape(-1,4)
                    z+=torch.cat([torch.zeros(1,device='cuda',dtype=torch.float64),d]);target=y.flatten().cuda().repeat(4)
                    lp=z.log_softmax(1);objective+=float(-lp.gather(1,target[:,None]).sum()/total);g+=lp[:,1:].exp().sum(0)/total
                g-=torch.bincount(y.flatten(),minlength=4)[1:].cuda().double()/y.numel()
                g+=(mu+d-pm)*(-2*ps).exp()/(2*16384)
                objective+=float(bias_kl(d,mu,ls,pm,ps)/(2*16384))+record['readouts'][name]['constant_C']
            residual=float(g[active].abs().max());residuals[arm+'/'+name]=residual
            assert abs(objective-receipt['final']['objective'])<1e-8
            assert abs(residual-receipt['final']['active_gradient_inf'])<1e-10
            if receipt['status']=='CONVERGED':assert residual<=1e-7
            history=receipt['history'];assert all(b['objective']<=a['objective']+1e-12 for a,b in zip(history,history[1:]))
        for name in ('G0','G1','G3'):
            pred=torch.load(out/(name+'_PRED.pt'),map_location='cpu',weights_only=False);v=record['readouts'][name]
            flows=pred['flows'];compare(v['metrics'],metrics(torch.stack(flows).mean(0),y));compare(v['flow_metrics'],[metrics(p,y) for p in flows])
            ex=[float(-p.gather(1,y[:,None]).mean()) for p in pred['mean_logprob_flows']];compare(v['expected_CE_flows'],ex)
            assert abs(sum(ex)/4+v['bias_KL']/(2*16384)+v['constant_C']-v['coordinate_J'])<1e-9
            assert v['robust_qualified24']==(v['metrics']['qualified_24'] and all(p['qualified_24'] for p in v['flow_metrics']))
        p0=torch.load(out/'G0_PRED.pt',map_location='cpu',weights_only=False)['flows']
        d=torch.tensor(record['readouts']['G3']['delta'],dtype=torch.float64);w=torch.cat([torch.zeros(1,dtype=torch.float64),d]).exp().reshape(1,4,1,1)
        p=torch.stack(p0).mean(0)*w;p/=p.sum(1,keepdim=True)
        compare(record['readouts']['POST_G3_SAME_DELTA']['metrics'],metrics(p,y))
    assert [v['index'] for v in ledger]==list(range(1,c['newton_acceptances']+1))
    assert candidates==c['backtracking_candidates'] and derivatives==c['derivative_passes']
    assert r['source_frozen'] and r['patient_mainline_unchanged'] and r['original_elapsed_seconds']<86400
    result=dict(status='PASS',solves_checked=4,active_residuals=residuals,Newton_ledger_checked=c['newton_acceptances'],
      candidate_and_derivative_ledgers_checked=True,sealed_prediction_sets_recomputed=6,maximum_metric_error=errors[0],
      same_delta_post_mixture_checked=True,affine_checks=8,new_network_forwards=0,new_parameter_updates=0,audit_cached_gradient_passes=4)
    write(root/'AUDIT.json',result);print(json.dumps(result))

if __name__=='__main__':audit(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
