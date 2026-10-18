"""Explicit full-supervision authorization; original SSL accessors stay unchanged."""
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from ..main_style_ssl_v1.data import metadata
from ..di_dmpa_gate1.binding import safe_asset
from ..main_style_ssl_v1.runtime import append,write
from ..siemens_lonly_v1.run import dice,aggregate,compare


AUTHORIZATION = 'Siemens-full-supervision-seed163-43600-20261018'


def validate_label_rows(original, labels):
    expected={r['case_id']:r for r in original}
    observed={r['case_id']:r for r in labels}
    if len(labels)!=76 or len(observed)!=len(labels) or observed.keys()!=expected.keys():
        raise PermissionError('full supervision must bind exactly the 76 former-U training volumes')
    for key,r in observed.items():
        old=expected[key]
        for field in ('patient_id','phase','image_h5_relpath','image_sha256','preprocess_config_sha256'):
            if r[field]!=old[field]:raise PermissionError('label/image provenance mismatch')
        if r['label_h5_relpath']!=f'labels/mnms/Siemens/{key}.h5' or not r['label_sha256']:
            raise PermissionError('invalid authorized training label')
    return observed


def bind_training_labels(provider,cfg):
    if cfg.get('authorization')!=AUTHORIZATION or cfg['method']!='fullsupervised':
        raise PermissionError('full supervision authorization missing')
    original=[r for r in metadata(cfg['data'],'mnms') if r['site_or_vendor']=='Siemens'
              and r['primary_20pct_split']=='train_unlabeled']
    root=Path(cfg['training_labels'])
    labels=validate_label_rows(original,json.loads((root/'LABELS.json').read_text()))
    access=provider._u
    if len(access.patients)!=38 or sum(access.depths)!=872 or len(provider._l.patients)!=10:
        raise PermissionError('training population changed')
    for r,depth in zip(access.rows,access.depths):
        r.update({k:labels[r['case_id']][k] for k in ('label_h5_relpath','label_sha256')})
        with h5py.File(safe_asset(root,r['label_h5_relpath']),'r') as f:
            if f['label'].shape!=(depth,384,384):raise ValueError('label grid mismatch')
    # Only this run's accessor is promoted; frozen manifests and SSL remain image-only.
    access.data=root;access.role='train_labeled'


@torch.inference_mode()
def evaluate_former_u(model,access,output):
    model.eval().requires_grad_(False);rows=[];count=0
    for i,meta in enumerate(access.rows):
        item=access.read(i);x=item['image']
        pred=np.concatenate([model(x[z:z+2,None].cuda()).argmax(1).cpu().numpy() for z in range(0,len(x),2)])
        row=dict(patient=meta['patient_id'],phase=meta['phase'],dice=dice(pred,item['label'].numpy()).tolist())
        rows.append(row);append(output,row);count+=len(x)
    return {**aggregate(rows),'forward_slices':count}


def finish(model,provider,cfg,out,result,val_rows):
    reference=Path(cfg['fixmatch_result'])
    previous=json.loads((reference/'RESULTS.json').read_text())
    baseline=[json.loads(line) for line in (reference/'private_val.jsonl').read_text().splitlines()]
    summary=aggregate(baseline)
    if (previous['status']!='COMPLETE' or previous['method']!='fixmatch' or previous['seed']!=163
        or previous['iterations']!=43600 or previous['primary']!='EMA'
        or abs(summary['primary_dice']-previous['val']['primary_dice'])>1e-12
        or abs(summary['primary_dice']*100-80.3674)>0.00005):
        raise ValueError('reference does not reproduce the frozen FixMatch result')
    result['train_former_u']=evaluate_former_u(model,provider._u,out/'private_train_former_u.jsonl')
    result['fixmatch_val_reused']=summary
    result['paired_val']=compare(baseline,val_rows)
    result['interpretation']='Matched full supervision: original L2 plus former-U14, equal branch weights; true labels replace masked pseudo labels. Same steps and sampling, not equal FLOPs. One seed, validation only, no absolute upper-bound claim.'
    write(out/'PAIRED_COMPARISON.json',result['paired_val'])
    lines=['# Siemens 全监督固定终点结果','',
           '|配置|验证 Dice %|','|---|---:|',
           f"|FixMatch 最终 EMA|{100*summary['primary_dice']:.4f}|",
           f"|全监督最终 EMA|{100*result['val']['primary_dice']:.4f}|",'',
           f"全监督减 FixMatch：{result['paired_val']['delta_B_minus_A_pp']:.4f} pp。",
           f"患者配对摘要：{json.dumps(result['paired_val'],ensure_ascii=False)}",'',
           '原 10 人训练组与原 38 人训练组分开评估；LV/MYO/RV、ED/ES 见 RESULTS.json。',
           '固定 seed163、43,600 次更新；未选最佳 checkpoint，未访问独立测试集。',
           result['interpretation'],
           'GitHub 交付待会话核验后执行；逐患者记录和权重留在私有存储。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
