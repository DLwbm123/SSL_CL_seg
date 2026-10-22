import csv,json,os
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import h5py
import torch
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
    ledger=ResourceLedger(root/'results',max_seconds=600,max_updates=0,max_bytes=64*1024**2)
    records=[r for r in csv.DictReader((dr/'manifests/training/lcrseg_v1_seed0.csv').open())
             if r['dataset']=='prostate' and r['site_or_vendor']=='RUNMC' and r['primary_20pct_split']=='train_labeled']
    rows=[];original=predictive.combine_slice_banks
    with ledger.attempt('prediction_diagnosis',seconds=600,updates=0):
        state=StreamState.load(old/'results/pure_attempt001/PRIVATE_SEALED_STATE.pt').to_device('cuda')
        for outer in ('mean','sample'):
            for i,r in enumerate(records):
                ledger.check()
                with h5py.File(confined(dr/'h5/v1',r['image_h5_relpath']),'r') as f:
                    x=torch.from_numpy(f['image'][...].copy())[:,None].double();spacing=json.loads(f.attrs['processed_spacing'])
                small=F.interpolate(x,(128,128),mode='bilinear',align_corners=False).cuda();uniform=[]
                def capture(weights,probabilities,integrator):
                    uniform.append(torch.cat(probabilities,0).mean(0).mean(1).detach().cpu())
                    return original(weights,probabilities,integrator)
                with ExitStack() as stack:
                    stack.enter_context(patch.object(predictive,'combine_slice_banks',capture))
                    if outer=='mean':
                        stack.enter_context(patch.object(state.q,'rsample',lambda generator:state.q.mean))
                        for memory in state.memories.values():
                            stack.enter_context(patch.object(memory,'rsample',lambda theta,generator,m=memory:m.mean(theta)))
                    pred=predictive.Predictor().predict_image_only(small,state,predictive.InferenceBudget(K=0),return_repeats=True)
                # Predictions fixed before this read; training data only.
                with h5py.File(confined(dr/'h5/v1',r['label_h5_relpath']),'r') as f:y=f['label'][...]
                for weighting,probs in [('uniform',uniform),('importance',pred['repeat_probabilities'].cpu())]:
                    metrics=[binary_phase_metrics(restore_probability(q,x.shape[-2:]),y,spacing) for q in probs]
                    rows.append(dict(outer=outer,weighting=weighting,case_index=i,metrics=metrics))
        summary=[]
        for outer in ('mean','sample'):
            for weighting in ('uniform','importance'):
                subset=[r for r in rows if r['outer']==outer and r['weighting']==weighting]
                summary.append(dict(outer=outer,weighting=weighting,repeats=[dict(
                    Dice=float(np.mean([r['metrics'][j]['classes'][0]['Dice'] for r in subset])),
                    NLL=float(np.mean([r['metrics'][j]['NLL'] for r in subset]))) for j in range(3)]))
        write(root/'SUMMARY.json',dict(summary=summary,rows=rows,patients=len(records),optimizer_updates=0,
             validation_access=0,test_access=0,MCMC_calls=0,interpretation='training diagnostic; not alternative formal evaluation'))
        print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
