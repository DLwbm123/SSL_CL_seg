"""Frozen endpoint decisions; not an optimizer or an adaptive rule."""
import json
from pathlib import Path


def decide(root):
    root=Path(root)
    names=['G_U','M_U','G_noNewU','M_noNewU','dense_fixmatch']
    r={n:json.loads((root/'jobs'/n/'RESULTS.json').read_text()) for n in names}
    def delta(a,b,d):return 100*(r[a][d]['primary_dice']-r[b][d]['primary_dice'])
    u_new=delta('G_U','G_noNewU','Philips');u_old=delta('G_U','G_noNewU','Siemens')
    f_old=delta('G_U','M_U','Siemens');f_new=delta('G_U','M_U','Philips')
    practical=delta('G_U','dense_fixmatch','Philips')
    fit=delta('G_U','dense_fixmatch','target_train')
    gains=r['G_U']['target_delta_from_entry_pp']
    gates=dict(U_benefit=u_new>=.5 and u_old>=-.5,frame_retention=f_old>=.5 and f_new>=-.5,
        plasticity=practical>=-.5 and gains>=.5 and fit>=-1)
    return dict(status='ALL_EXPLORATORY_GATES_PASS_NO_LATER_DOMAINS' if all(gates.values()) else 'SCIENTIFIC_STOP_NO_JOINT_BENEFIT_CLAIM',gates=gates,
        U_new_pp=u_new,U_old_pp=u_old,frame_old_pp=f_old,frame_new_pp=f_new,practical_new_pp=practical,target_training_vs_Dense_pp=fit,
        G_target_gain_from_entry_pp=gains,interactions_pp={d:delta('G_U','G_noNewU',d)-delta('M_U','M_noNewU',d) for d in ['Siemens','Philips']},later_domain_allowed=False)
