import json
from pathlib import Path


def decide(root):
    root=Path(root);cfg=json.loads((root/'CONTROLLER.json').read_text());g=json.loads((root/'jobs/G_U/RESULTS.json').read_text())
    m=json.loads(Path(cfg['M_U_reference']).read_text());prior=json.loads(Path(cfg['G_U_reference']).read_text());dense=json.loads(Path(cfg['dense_reference']).read_text())
    contrast=lambda a,b:{d:100*(a[d]['primary_dice']-b[d]['primary_dice']) for d in ['Siemens','Philips']}
    v=dict(G_stronger_mix_minus_G_moderate_mix=contrast(g,prior),G_stronger_mix_minus_M=contrast(g,m),G_minus_Dense=contrast(g,dense))
    gates=dict(mix=v['G_stronger_mix_minus_G_moderate_mix']['Siemens']>=.5 and v['G_stronger_mix_minus_G_moderate_mix']['Philips']>=-.5,
      frame=v['G_stronger_mix_minus_M']['Siemens']>=.5 and v['G_stronger_mix_minus_M']['Philips']>=-.5,
      practical=all(x>=-.5 for x in v['G_minus_Dense'].values()))
    return dict(status='EXPLORATORY_PASS' if all(gates.values()) else 'NO_JOINT_BENEFIT_DEMONSTRATED',gates=gates,contrasts=v,U_contribution_tested=False,M_reference_reused=True)
