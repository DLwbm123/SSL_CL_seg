import csv,json,os,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import h5py,torch
from torch.nn import functional as F
from experiments.sc_bayeseg.hbsi_v1 import predictive
from experiments.sc_bayeseg.hbsi_v1.budgets import ResourceLedger
from experiments.sc_bayeseg.hbsi_v1.stream_state import StreamState
from experiments.sc_bayeseg.hbsi_v1.data_adapter import confined,restore_probability
from experiments.sc_bayeseg.hbsi_v1.real_diagnostic import write
from sc_bayeseg.sslcl_v1.prostate_metrics import binary_phase_metrics


def main():
    root=Path(os.environ['RUN_ROOT']);old=Path(os.environ['PARENT_RUN'])
    p=json.loads((old/'PRIVATE_PROTOCOL.json').read_text());dr=Path(p['data_root'])
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ledger=ResourceLedger(root/'results',max_seconds=900,max_updates=0,max_bytes=256*1024**2)
    records=[r for r in csv.DictReader((dr/'manifests/training/lcrseg_v1_seed0.csv').open()) if r['dataset']=='prostate' and r['site_or_vendor']=='RUNMC' and r['primary_20pct_split'] in ('train_labeled','val')]
    assert len(records)==8
    rows=[];original=predictive.combine_slice_banks
    with ledger.attempt('unified',seconds=900,updates=0):
        state=StreamState.load(old/'results/mixed_attempt001/PRIVATE_SEALED_STATE.pt').to_device('cuda')
        for i,r in enumerate(records):
            with h5py.File(confined(dr/'h5/v1',r['image_h5_relpath']),'r') as f:
                x=torch.from_numpy(f['image'][...].copy())[:,None].double();spacing=json.loads(f.attrs['processed_spacing'])
            small=F.interpolate(x,(128,128),mode='bilinear',align_corners=False).cuda()
            for K in (0,3):
                ledger.check();banks={'uniform':[],'local':[]};diagnostics=[]
                def capture(weights,probabilities,integrator):
                    assert integrator=='factorized' and len(weights)==1
                    w,q=weights[0],probabilities[0]
                    uniform=q.mean(0).mean(1)
                    local=torch.einsum('svl,svlkhw->svkhw',w.softmax(-1),q)
                    outer=(torch.logsumexp(w,-1)-np.log(w.shape[-1])).sum(1).softmax(0)
                    full=torch.einsum('s,svkhw->vkhw',outer,local)
                    result,diag=original(weights,probabilities,integrator)
                    assert torch.allclose(full,result,atol=1e-12,rtol=1e-10)
                    banks['uniform'].append(uniform.detach().cpu());banks['local'].append(local.mean(0).detach().cpu())
                    diagnostics.append(dict(local_ess=diag['local_ess'][0].detach().cpu().tolist(),local_max_weight=diag['local_max_weight'][0].detach().cpu().tolist(),global_ess=float(diag['global_ess'][0]),global_max_weight=float(diag['global_max_weight'][0]),log_evidence=float(diag['log_evidence'])))
                    return result,diag
                with patch.object(predictive,'combine_slice_banks',capture):
                    pred=predictive.Predictor().predict_image_only(small,state,predictive.InferenceBudget(K=K),return_repeats=True)
                banks['full']=pred['repeat_probabilities'].cpu()
                # All three readouts fixed before labels are opened. No hidden U/test labels.
                with h5py.File(confined(dr/'h5/v1',r['label_h5_relpath']),'r') as f:y=f['label'][...]
                for readout,probs in banks.items():
                    metrics=[]
                    for q in probs:
                        restored=restore_probability(q,x.shape[-2:]);m=binary_phase_metrics(restored,y,spacing)
                        metrics.append(m)
                    rows.append(dict(split=r['primary_20pct_split'],case_index=i,K=K,readout=readout,metrics=metrics))
                write(root/f'DIAGNOSTIC_{i}_K{K}.json',dict(split=r['primary_20pct_split'],case_index=i,K=K,diagnostics=diagnostics,gates=pred['gates'],probability_range=pred['probability_range'],log_evidence_range=pred['log_evidence_range'],status=pred['status']))
                write(root/f'PROGRESS_{i}_K{K}.json',dict(completed_patient_K=len(rows)//3,total_patient_K=16,updated=time.time()))
        summary=[]
        for split in ('train_labeled','val'):
            for K in (0,3):
                for readout in ('uniform','local','full'):
                    subset=[r for r in rows if r['split']==split and r['K']==K and r['readout']==readout]
                    summary.append(dict(split=split,K=K,readout=readout,repeats=[dict(Dice=float(np.mean([r['metrics'][j]['classes'][0]['Dice'] for r in subset])),NLL=float(np.mean([r['metrics'][j]['NLL'] for r in subset])),Brier=float(np.mean([r['metrics'][j]['Brier'] for r in subset]))) for j in range(3)]))
        write(root/'SUMMARY.json',dict(summary=summary,rows=rows,optimizer_updates=0,test_access=0,MCMC_calls=0,full_readout_equivalence=True))
        print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
