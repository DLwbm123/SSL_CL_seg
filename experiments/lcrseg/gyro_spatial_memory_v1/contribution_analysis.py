"""Supplementary factorial/simple effects from existing patient records only."""
import json
from pathlib import Path
from .single_target_audit import paired,rows


def main(cfg):
    root=Path(cfg['root']);assert json.loads((root/'AUDIT.json').read_text())['status']=='PASS'
    result={}
    groups=['S','G00','G10','G01','G11']
    for mode,prefix in [('student','private_'),('EMA','private_ema_')]:
        result[mode]={}
        for domain in ['Siemens','Philips']:
            name=prefix+domain+'_val.jsonl';rr={g:rows(root/'jobs'/g/name) for g in groups}
            result[mode][domain]=dict(U_without_frame=paired(rr['G00'],rr['G10']),frame_with_L_fill=paired(rr['G00'],rr['G01']))
    result['EMA_minus_student']={domain:{g:paired(rows(root/'jobs'/g/('private_'+domain+'_val.jsonl')),rows(root/'jobs'/g/('private_ema_'+domain+'_val.jsonl'))) for g in groups} for domain in ['Siemens','Philips']}
    result['model_forwards']=0;result['optimizer_updates']=0
    (root/'SUPPLEMENTARY_CONTRASTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
