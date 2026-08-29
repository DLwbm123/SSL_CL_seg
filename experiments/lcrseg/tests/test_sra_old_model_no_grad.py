from lcrseg.methods.lcrseg_v0_4a import LCRSegV04AMethod
from lcrseg.models import UNet2D
from lcrseg.engine.trainer import TrainerState
from tests.v0_2a_test_utils import batches, previous_checkpoint, routing_config, trainer


def test_sra_old_model_no_grad(tmp_path) -> None:
    method = LCRSegV04AMethod(UNet2D(3, 3), config=routing_config(protocol_id="lcrseg_v0_4a"))
    method.begin_site("B", previous_checkpoint(tmp_path), 4)
    result = trainer(method).train_step(*batches(), state=TrainerState(global_step=1, site_step=0, epoch=0))
    assert float(result.total_loss) == float(result.total_loss)
    assert method.old_model is not None
    assert all(parameter.grad is None for parameter in method.old_model.parameters())
