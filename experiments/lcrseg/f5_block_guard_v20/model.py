"""Independent A/B/R halfspaces from the existing V13 reference VJP."""
from pathlib import Path
import torch
from ..f5_class_reference_v13.model import ClassReferenceModel,ClassReferenceTrainer
from ..f5_entry_gradient_v12.model import halfspace
from ..five_frameworks_v1.gate import digest


def block_halfspace(merged,reference,blocks):
    if tuple(blocks)!=('A','B','R') or sorted(i for ids in blocks.values() for i in ids)!=list(range(len(merged))):
        raise ValueError('A/B/R must partition every trainable tensor exactly once')
    legacy,aggregate=halfspace(merged,reference)
    out=list(merged);details={};violations=0
    for name,ids in blocks.items():
        g=[merged[i] for i in ids];h=[reference[i] for i in ids]
        projected,stats=halfspace(g,h)
        for i,value in zip(ids,projected):out[i]=value
        dot=sum(float((legacy[i].double()*reference[i].double()).sum()) for i in ids)
        scale=(sum(float(legacy[i].double().square().sum()) for i in ids)*sum(float(reference[i].double().square().sum()) for i in ids))**.5
        cosine=dot/scale if scale>0 else 0.
        if not torch.isfinite(torch.tensor([dot,scale,cosine],dtype=torch.double)).all():raise FloatingPointError('nonfinite block diagnostic')
        stats.update(v13_dot=dot,v13_cosine=cosine)
        violations+=int(cosine < -1e-6);details[name]=stats
    return out,dict(projected=any(x['projected'] for x in details.values()),
                    dot_before=aggregate['dot_before'],dot_after=sum(x['dot_after'] for x in details.values()),
                    reference_norm=aggregate['reference_norm'],parameter_blocks=details,
                    v13_block_violations=violations)


class BlockGuardModel(ClassReferenceModel):
    def __init__(self,*args,block_guard=True,**kwargs):
        super().__init__(*args,class_reference=True,**kwargs);self.block_guard=block_guard


class BlockGuardTrainer(ClassReferenceTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.telemetry.update(block_guard_calls=0,block_guard_hidden_conflict_calls=0,
                              block_guard_A_projected_calls=0,block_guard_B_projected_calls=0,block_guard_R_projected_calls=0)

    def loss_contract(self):
        return {**super().loss_contract(),'block_guard':self.model.block_guard,
                'block_guard_formula':'same V13 class-balanced entry h; independent Euclidean halfspace gX dot hX>=0 for X=A/B/R; block partition exact; original raw gradients then native Adam',
                'block_guard_implementation':digest(Path(__file__).read_text())}

    def project_entry_gradients(self,merged,reference,grads,params):
        if not self.model.block_guard:return super().project_entry_gradients(merged,reference,grads,params)
        groups=self.model.parent.parameter_groups();position={id(p):i for i,p in enumerate(params)}
        blocks={name:[position[id(p)] for p in group] for name,group in [('A',groups['input_factors']),('B',groups['output_factors']),('R',self.model.sidecar.optimizer_parameters())]}
        out,stats=block_halfspace(merged,reference,blocks)
        self.telemetry['block_guard_calls']+=1
        self.telemetry['block_guard_hidden_conflict_calls']+=int(stats['v13_block_violations']>0)
        for name,detail in stats['parameter_blocks'].items():self.telemetry['block_guard_'+name+'_projected_calls']+=int(detail['projected'])
        return out,stats

    def transform_gradients(self,grads):
        super().transform_gradients(grads);self.module_stats.update(block_guard=self.model.block_guard)
