"""F5 loss-specific routing: KL reaches B/R, SWD reaches R only."""
from pathlib import Path
from ..f5_output_u_v3.model import OutputUModel,OutputUTrainer
from ..five_frameworks_v1.gate import digest


class SelectiveUModel(OutputUModel):
    def __init__(self,*args,swd_to_parent=False,**kwargs):
        self.swd_to_parent=swd_to_parent
        super().__init__(*args,**kwargs)

    def detach_u_parent(self,*,clean=False):
        return not self.output_u or (clean and not self.swd_to_parent)


class SelectiveUTrainer(OutputUTrainer):
    def loss_contract(self):
        return {**super().loss_contract(),'SWD_to_parent':self.model.swd_to_parent,
                'routing':'KL->B/R; SWD->R unless explicit parity fixture',
                'selective_implementation':digest(Path(__file__).read_text())}
