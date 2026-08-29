from lcrseg.methods.lcrseg_v0_2a import LCRSegV02AMethod
from lcrseg.models import UNet2D


def test_protocol_fields_in_checkpoint() -> None:
    method = LCRSegV02AMethod(UNet2D(3, 3), config={"variant_id": "R3", "assimilation_mode": "progressive_admission", "consolidation_mode": "calibrated_teacher_rejection"})
    semantics = method.method_state_dict()["method_statistics"]["protocol_semantics"]
    for key in ("protocol_id", "assimilation_mode", "consolidation_mode", "learnability_formula_version", "teacher_validity_formula_version", "calibrator_version", "progressive_schedule", "rejection_threshold", "rejection_floor", "rejection_cap"):
        assert key in semantics
