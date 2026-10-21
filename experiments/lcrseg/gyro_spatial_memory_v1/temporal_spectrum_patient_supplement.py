"""CPU-only final paired magnitudes from existing sealed patient records."""
import json
from pathlib import Path
import numpy as np
from .temporal_spectrum_p0 import damage,patient_map,correlation
from .single_target_audit import rows


def main(cfg):
    p=Path(cfg['root']);assert json.loads((p/'AUDIT.json').read_text())['status']=='PASS';out={};cross={}
    original=rows(Path(cfg['source_checkpoint']).parent/'private_val.jsonl')
    for seed in ['20261025','20261026']:
        out[seed]={}
        for mode,prefix in [('student','private_'),('EMA','private_ema_')]:
            out[seed][mode]={}
            for domain in ['Siemens','Philips']:
                rr={m:rows(p/'jobs'/(seed+'_'+m)/(prefix+domain+'_val.jsonl')) for m in ['S','G','T','GT']}
                cc={f'{b}_minus_{a}':damage(rr[a],rr[b]) for a,b in [('S','G'),('S','T'),('G','GT'),('T','GT'),('S','GT')]}
                if domain=='Siemens':cc.update({m+'_minus_original_source':damage(original,v) for m,v in rr.items()})
                out[seed][mode][domain]=cc
                for a,b in [('G','GT'),('T','GT')]:
                    aa,bb=patient_map(rr[a]),patient_map(rr[b]);cross.setdefault(mode+'_'+domain+'_'+b+'_minus_'+a,[]).append(np.array([(bb[k]-aa[k])*100 for k in sorted(aa)]))
        out[seed]['EMA_minus_student']={domain:{m:damage(rows(p/'jobs'/(seed+'_'+m)/('private_'+domain+'_val.jsonl')),rows(p/'jobs'/(seed+'_'+m)/('private_ema_'+domain+'_val.jsonl'))) for m in ['S','G','T','GT']} for domain in ['Siemens','Philips']}
    result={'new_model_forwards':0,'new_optimizer_updates':0,'clinical_thresholds_validated':False,'comparisons':out,'cross_stream_patient_difference_consistency':{k:correlation(*v) for k,v in cross.items()}}
    (p/'PATIENT_SUPPLEMENT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
