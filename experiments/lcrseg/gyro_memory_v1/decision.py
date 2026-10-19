import json
from pathlib import Path


def decide(root):
    root=Path(root);cfg=json.loads((root/'CONTROLLER.json').read_text())
    g=json.loads((root/'jobs/G_U/RESULTS.json').read_text());m=json.loads((root/'jobs/M_U/RESULTS.json').read_text())
    g16=json.loads(Path(cfg['G16_reference']).read_text());m16=json.loads(Path(cfg['M16_reference']).read_text());dense=json.loads(Path(cfg['dense_reference']).read_text())
    contrast=lambda a,b:{d:100*(a[d]['primary_dice']-b[d]['primary_dice']) for d in ['Siemens','Philips']}
    v=dict(G256_minus_G16=contrast(g,g16),M256_minus_M16=contrast(m,m16),G256_minus_M256=contrast(g,m),G256_minus_Dense=contrast(g,dense))
    gates=dict(coverage=v['G256_minus_G16']['Siemens']>=.5 and v['G256_minus_G16']['Philips']>=-.5,
        frame=v['G256_minus_M256']['Siemens']>=.5 and v['G256_minus_M256']['Philips']>=-.5,
        practical=all(x>=-.5 for x in v['G256_minus_Dense'].values()))
    return dict(status='EXPLORATORY_PASS' if all(gates.values()) else 'NO_JOINT_BENEFIT_DEMONSTRATED',gates=gates,contrasts=v,U_contribution_tested=False)
