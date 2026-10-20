"""Recover aggregation from all completed event curves; never rescan events."""
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from sc_bayeseg.window_v1.r_start import write
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.common import storage
from sc_bayeseg.window_v1.class3_bias_feasibility import intervals,intersect,encode,scores,direct,witness


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time()
    try:
        torch.set_num_threads(4);write(root/'CONTEXT.json',s);storage(s,reserve=2_000_000_000)
        failed=Path(s['failed_source']);assert json.loads((failed/'EXIT.json').read_text())['exit_code']==1
        assert 'int64 is not JSON serializable' in json.loads((failed/'FAILURE.json').read_text())['traceback']
        assert len(list(failed.glob('*.npz')))==20
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text())
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']])
        y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long();y=torch.cat([flip(y,k) for k in range(4)]).numpy()
        rows=[];records={};metadata={};maxerr=0.
        for arm in ('ZERO','R1_WARM'):
            for objective,folder in (('PIXEL_CE',Path(s['allocation_source'])/(arm+'_T2L2')),('CLASS_BALANCED',Path(s['risk_source'])/(arm+'_CLASS_BALANCED'))):
                file=folder/'PRED_0500.pt';metadata[file]=(file.stat().st_size,file.stat().st_mtime_ns)
                p=torch.load(file,map_location='cpu',weights_only=False)['generator_flows'];name=arm+'_'+objective
                common=[(-np.inf,np.inf)]
                for i,pred in enumerate([torch.stack(p).mean(0)]+list(p)):
                    mode='MIXTURE' if i==0 else 'FLOW'+str(i)
                    path=failed/(name+'_'+mode+'.npz');metadata[path]=(path.stat().st_size,path.stat().st_mtime_ns)
                    with np.load(path) as z:curve={k:z[k] for k in z.files}
                    low,high,minimum,passed=[curve[k] for k in ('low','high','minimum','passed')]
                    m,base=scores(pred.numpy());feasible=intervals(passed==24,low,high)
                    checks=[]
                    for j in np.unique(np.linspace(0,len(low)-1,min(8,len(low)),dtype=int)):
                        value,count=direct(m,base,y,witness(low[j],high[j]))
                        maxerr=max(maxerr,abs(value-minimum[j]));assert abs(value-minimum[j])<1e-12 and count==int(passed[j]);checks.append(int(j))
                    b0=int(np.searchsorted(high,0.,side='left'));value,count=direct(m,base,y,0.)
                    assert abs(value-minimum[b0])<1e-12 and count==int(passed[b0])
                    best=int(minimum.argmax());rows.append(dict(candidate=name,mode=mode,candidate_intervals=len(low),finite_pixel_events=int(np.isfinite(m).sum()),distinct_events=len(low)-1,
                      feasible_intervals=encode(feasible),any_bias_qualified=bool(feasible),best_minimum_foreground_Dice=float(minimum[best]),best_interval=encode([(low[best],high[best])])[0],
                      original_bias0_minimum_Dice=value,original_bias0_passed24=int(count),direct_curve_checks=checks))
                    common=intersect(common,feasible)
                records[name]=dict(robust_common_bias_exists=bool(common),common_feasible_intervals=encode(common))
        assert len(rows)==20 and maxerr<1e-12
        assert all((p.stat().st_size,p.stat().st_mtime_ns)==st for p,st in metadata.items())
        write(root/'RESULTS.json',dict(status='COMPLETE',rows=rows,candidates=records,synthetic=json.loads((failed/'SYNTHETIC.json').read_text()),
          costs=dict(scans=20,repeated_event_scans=0,recovered_curves=20,repair_direct_curve_checks=160,optimizer=0,reverse=0,generator_forward=0,parameter_applications=0),
          source_unchanged=True,original_failure_preserved=True,repair_reason='JSON numpy int64 scalar after all 20 curves completed',
          phase_elapsed_seconds=time.time()-started,original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s),
          interpretation='Finite saved scores and one common class3 bias only; not mathematical G-family unreachability'))
        write(root/'AUDIT.json',dict(status='PASS',curve_sets=20,direct_witness_checks=160,original_bias0_checks=20,maximum_direct_Dice_error=float(maxerr),
          feasible_intervals_and_common_intersections_recomputed=True,all_original_curves_reused_without_scans=True,source_metadata_unchanged=True))
        write(root/'STATUS.json',dict(status='COMPLETE',recovered_curves=20));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
