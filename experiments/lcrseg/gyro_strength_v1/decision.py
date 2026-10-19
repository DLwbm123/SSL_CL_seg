import json
from pathlib import Path


def decide(root):
    root=Path(root);cfg=json.loads((root/'CONTROLLER.json').read_text())
    r={a:json.loads((root/'jobs'/a/'RESULTS.json').read_text()) for a in ['G_U','M_U','CE_only']}
    dense=json.loads(Path(cfg['dense_reference']).read_text())
    def d(b,domain):return 100*(r['G_U'][domain]['primary_dice']-r[b][domain]['primary_dice'])
    v=dict(G_minus_CE_old_pp=d('CE_only','Siemens'),G_minus_CE_new_pp=d('CE_only','Philips'),
        G_minus_M_old_pp=d('M_U','Siemens'),G_minus_M_new_pp=d('M_U','Philips'),
        G_minus_Dense_new_pp=100*(r['G_U']['Philips']['primary_dice']-dense['Philips']['primary_dice']),
        G_minus_Dense_target_training_pp=100*(r['G_U']['target_train']['primary_dice']-dense['target_train']['primary_dice']))
    gates=dict(retention_vs_CE=v['G_minus_CE_old_pp']>=.5 and v['G_minus_CE_new_pp']>=-.5,
        frame_vs_M=v['G_minus_M_old_pp']>=.5 and v['G_minus_M_new_pp']>=-.5,
        plasticity=v['G_minus_Dense_new_pp']>=-.5 and v['G_minus_Dense_target_training_pp']>=-1)
    return dict(status='EXPLORATORY_PASS' if all(gates.values()) else 'NO_JOINT_BENEFIT_DEMONSTRATED',gates=gates,contrasts=v,later_domains_allowed=False,U_contribution_tested=False)
