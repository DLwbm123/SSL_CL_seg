import json
from pathlib import Path


def decide(root):
    root=Path(root);cfg=json.loads((root/'CONTROLLER.json').read_text())
    result={a:json.loads((root/'jobs'/a/'RESULTS.json').read_text()) for a in ['G_U','M_U']}
    refs={a:json.loads(Path(cfg[a+'_reference']).read_text()) for a in ['G_U','M_U']}
    dense=json.loads(Path(cfg['dense_reference']).read_text())
    contrast=lambda a,b:{d:100*(a[d]['primary_dice']-b[d]['primary_dice']) for d in ['Siemens','Philips']}
    v={a+'_coordinate_change':contrast(result[a],refs[a]) for a in result}
    v.update(G_minus_M=contrast(result['G_U'],result['M_U']),G_minus_Dense=contrast(result['G_U'],dense))
    gates=dict(coordinate=v['G_U_coordinate_change']['Siemens']>=.5 and v['G_U_coordinate_change']['Philips']>=-.5,
      frame=v['G_minus_M']['Siemens']>=.5 and v['G_minus_M']['Philips']>=-.5,
      practical=all(x>=-.5 for x in v['G_minus_Dense'].values()) and 100*(result['G_U']['target_train']['primary_dice']-dense['target_train']['primary_dice'])>=-1.)
    return dict(status='EXPLORATORY_PASS' if all(gates.values()) else 'NO_JOINT_BENEFIT_DEMONSTRATED',gates=gates,contrasts=v,U_contribution_tested=False)
