"""Keep F5; permit existing total-U gradients on projected parent B and R."""
from pathlib import Path
from ..five_frameworks_v1.model import Model
from ..f5_module_pilots_v1.modules import PilotTrainer
from ..five_frameworks_v1.gate import digest


class OutputUModel(Model):
    def __init__(self,*args,output_u=True,**kwargs):
        self.output_u=output_u
        super().__init__(*args,**kwargs)

    def detach_u_parent(self):
        return not self.output_u

    def u_parameters(self):
        return self.sidecar.optimizer_parameters()+(self.parent.parameter_groups()['output_factors'] if self.output_u else [])


class OutputUTrainer(PilotTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,arm='F5',**kwargs)
        self.telemetry.update(output_U_calls=0,nonzero_output_U_calls=0)

    @classmethod
    def for_resume(cls,model,provider,options=None,cwmi=None,execution=None):
        return cls(model,provider,options,cwmi,initialize=False,execution=execution)

    def loss_contract(self):
        return dict(output_u=self.model.output_u,equation='U KL+0.2 SWD -> B and R; A receives L only',
                    implementation=digest(Path(__file__).read_text()))

    def transform_gradients(self,gradients):
        groups=self.model.parent.parameter_groups()
        if any(gradients[id(p)]['U'] is not None for p in groups['input_factors']):
            raise PermissionError('U must not update parent A')
        norms={name:sum(float(pair['U'].detach().square().sum()) for p in params
                         if (pair:=gradients[id(p)])['U'] is not None)**.5
               for name,params in [('B',groups['output_factors']),('R',self.model.sidecar.optimizer_parameters())]}
        active=any(gradients[id(p)]['U'] is not None for p in groups['output_factors'])
        self.telemetry['output_U_calls']+=int(active)
        self.telemetry['nonzero_output_U_calls']+=int(norms['B']>0)
        self.module_stats=dict(B_U_norm=norms['B'],R_U_norm=norms['R'],A_U_forbidden=True)
