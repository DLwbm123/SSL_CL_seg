"""Zero-update spectral and endpoint-gate contracts."""
import json
from pathlib import Path
from unittest.mock import patch
import torch
from . import core
from .decision import decide
from ..gyro_native_v1.module import self_check
from ..main_style_ssl_v1.runtime import write


def main(cfg):
    assert self_check()['status']=='PASS'
    for arm,weight in [('G_U',.01),('M_U',.01),('G_noNewU',.01),('M_noNewU',.01),('dense_fixmatch',1.)]:
        model=type('M',(),{'cfg':{'output':cfg['root']}})()
        groups={'encoder_all':{'CE_norm':.001,'weighted_G_norm':1.}}
        with patch.object(core,'original_losses',return_value=(torch.tensor(2.),torch.tensor(3.),{})),patch.object(core,'gradients',return_value=groups),patch.object(core,'append'):
            l,u,_=core.losses(arm,model,None,0)
            assert torch.allclose(l,torch.tensor(2.)) and torch.allclose(u,torch.tensor(3.*weight))
    folder=Path(cfg['root'])/'checks';(folder/'jobs').mkdir(parents=True,exist_ok=False)
    vals={'G_U':(.80,.82,.90),'M_U':(.79,.82,.90),'G_noNewU':(.80,.81,.90),'M_noNewU':(.79,.81,.90),'dense_fixmatch':(.78,.82,.90)}
    for arm,(old,new,train) in vals.items():
        out=folder/'jobs'/arm;out.mkdir()
        write(out/'RESULTS.json',dict(Siemens={'primary_dice':old},Philips={'primary_dice':new},target_train={'primary_dice':train},target_delta_from_entry_pp=1.))
    assert all(decide(folder)['gates'].values())
    f=folder/'jobs/G_U/RESULTS.json';r=json.loads(f.read_text());r['Philips']['primary_dice']=.80;write(f,r)
    d=decide(folder);assert not d['gates']['U_benefit'] and not d['gates']['plasticity']
    write(Path(cfg['root'])/'SYNTHETIC_CHECKS.json',dict(status='PASS',optimizer_calls=0,loss_scaling=True,ratio_diagnostic_only=True,decision_thresholds=True))
