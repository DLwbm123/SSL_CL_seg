"""One joint-factor consistency bridge, with original linear F5 sidecar."""
from pathlib import Path
from ..five_frameworks_v1.model import Model
from ..f5_module_pilots_v1.modules import PilotTrainer
from ..five_frameworks_v1.gate import digest

class JointKLModel(Model):
    def __init__(self,*args,joint_kl=True,**kwargs):
        self.joint_kl=joint_kl
        super().__init__(*args,**kwargs)

    def detach_u_parent(self,*,clean=False):
        return clean or not self.joint_kl

    def u_parameters(self):
        groups=self.parent.parameter_groups()
        return self.sidecar.optimizer_parameters()+(groups['input_factors']+groups['output_factors'] if self.joint_kl else [])

class JointKLTrainer(PilotTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,arm='F5',**kwargs)
        self.telemetry.update(joint_U_calls=0,A_nonzero_U_calls=0,B_nonzero_U_calls=0,R_nonzero_U_calls=0)

    @classmethod
    def for_resume(cls,model,provider,options=None,cwmi=None,execution=None):
        return cls(model,provider,options,cwmi,initialize=False,execution=execution)

    def loss_contract(self):
        return dict(joint_kl=self.model.joint_kl,routing='KL->A/B/R jointly; clean SWD->R only',
                    lambda_U=self.options['lambda_U'],implementation=digest(Path(__file__).read_text()))

    def transform_gradients(self,grads):
        groups=self.model.parent.parameter_groups()
        if not self.model.joint_kl and any(grads[id(p)]['U'] is not None for p in groups['input_factors']+groups['output_factors']):
            raise PermissionError('disabled bridge cannot update A/B from U')
        norms={name:sum(float(grads[id(p)]['U'].detach().square().sum()) for p in params if grads[id(p)]['U'] is not None)**.5
               for name,params in [('A',groups['input_factors']),('B',groups['output_factors']),('R',self.model.sidecar.optimizer_parameters())]}
        active=any(grads[id(p)]['U'] is not None for p in self.model.u_parameters())
        self.telemetry['joint_U_calls']+=int(active)
        for name in norms:self.telemetry[name+'_nonzero_U_calls']+=int(norms[name]>0)
        self.module_stats=dict(joint_kl=self.model.joint_kl,SWD_parent_detached=self.model.detach_u_parent(clean=True),
                               lambda_U=self.options['lambda_U'],**{name+'_U_norm':value for name,value in norms.items()})
