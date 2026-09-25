import copy
import os
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from qprompt.data import role_allowed
from qprompt.losses import PrototypeBank, grqa_loss, image_prototypes, supervised_query_loss
from qprompt.metrics import equal_domain_mean, image_dice
from qprompt.models import DINOv2Adapter, UNetDense, UNetQuery, semantic_probabilities
from qprompt.state import BudgetLedger, capture, ema_update, make_reference, restore


class R0Checks(unittest.TestCase):
    def test_metric_support_and_equal_domains(self):
        target = torch.tensor([[0, 1], [2, 255]])
        actual = image_dice(torch.tensor([[0, 1], [1, 2]]), target)
        self.assertAlmostEqual(actual["rim"], 2 / 3)
        self.assertEqual(actual["cup"], 0)
        empty = image_dice(torch.zeros(2, 2, dtype=torch.long), torch.zeros(2, 2, dtype=torch.long))
        self.assertIsNone(empty["macro"])
        result = equal_domain_mean({"A": [actual], "B": [dict(macro=.5)]})
        self.assertAlmostEqual(result["Q"], (1 / 3 + .5) / 2)

    def test_query_semantics_and_matching(self):
        classes = torch.zeros(1, 3, 4, requires_grad=True)
        masks = torch.zeros(1, 3, 8, 8, requires_grad=True)
        labels = torch.zeros(1, 8, 8, dtype=torch.long)
        labels[:, 2:5] = 1
        labels[:, 5:7] = 2
        labels[:, 7:] = 255
        scores = semantic_probabilities(classes, masks)
        self.assertTrue(torch.allclose(scores.sum(1), torch.ones_like(scores[:, 0])))
        result = supervised_query_loss(classes, masks, labels)
        self.assertEqual(result["matched"].item(), 3)  # background is a true class
        result["total"].backward()
        self.assertIsNotNone(classes.grad)
        self.assertIsNotNone(masks.grad)
        empty = supervised_query_loss(classes.detach(), masks.detach(), torch.full_like(labels, 255))
        self.assertEqual(empty["matched"].item(), 0)

    def test_grqa_formula_detach_and_groups(self):
        bank = PrototypeBank(3)
        bank.vectors.copy_(torch.eye(3))
        bank.supported[:] = True
        q = torch.tensor([[[1., .2, 0.], [.8, .3, 0.], [.1, 1., 0.]]], requires_grad=True)
        ref = (q.detach() + .1).requires_grad_()
        output = grqa_loss(q, ref, bank)
        self.assertTrue(torch.isfinite(output["total"]))
        self.assertGreater(abs((output["paper_selected_k3_surrogate"] - output["categorical_kl"]).item()), 1e-8)
        output["total"].backward()
        self.assertGreater(q.grad.abs().sum().item(), 0)
        self.assertIsNone(ref.grad)
        self.assertIsNone(bank.vectors.grad)
        bank.reset()
        self.assertEqual(grqa_loss(q, ref, bank)["total"].item(), 0)

    def test_prototype_and_ema(self):
        pixels = torch.randn(1, 3, 4, 4, requires_grad=True)
        labels = torch.full((1, 8, 8), 255)
        labels[:, :4] = 0
        vectors, supported = image_prototypes(pixels, labels)
        self.assertTrue(supported[0, 0])
        self.assertFalse(supported[0, 1])
        bank = PrototypeBank(3)
        bank.update(vectors.detach(), supported)
        self.assertEqual(bank.supported.tolist(), [True, False, False])
        student = nn.Linear(3, 3)
        reference = make_reference(student)
        self.assertFalse(any(p.requires_grad for p in reference.parameters()))
        with torch.no_grad():
            student.weight.add_(1)
        ema_update(student, reference)
        self.assertNotEqual(student.weight.data_ptr(), reference.weight.data_ptr())

    def test_backbone_geometry(self):
        x = torch.randn(1, 3, 32, 32)
        self.assertEqual(UNetDense()(x).shape, (1, 3, 32, 32))
        q = UNetQuery()(x)
        self.assertEqual(q["mask_logits"].shape, (1, 12, 32, 32))
        self.assertEqual(q["pixels"].shape, (1, 128, 8, 8))

        class MockViT(nn.Module):
            embed_dim, patch_size, num_register_tokens = 384, 14, 0
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])
                self.norm = nn.Identity()
            def prepare_tokens_with_masks(self, image):
                self.size = image.shape[-2:]
                return image.new_zeros((image.shape[0], 785, 384))
        mock = MockViT()
        vi_t = DINOv2Adapter(mock, query=True)
        result = vi_t(torch.zeros(1, 3, 384, 384))
        self.assertEqual(mock.size, (392, 392))
        self.assertEqual(result["mask_logits"].shape, (1, 12, 384, 384))
        self.assertEqual(result["pixels"].shape, (1, 384, 28, 28))
        self.assertEqual(DINOv2Adapter(MockViT(), query=False)(torch.zeros(1, 3, 384, 384)).shape,
                         (1, 3, 384, 384))

    def test_permission_and_full_prefix_state(self):
        role_allowed("RIM_ONE_r3", "train_labeled", "train", "RIM_ONE_r3")
        for domain, role, purpose, current in [
            ("RIM_ONE_r3", "train_unlabeled", "train", "RIM_ONE_r3"),
            ("RIM_ONE_r3", "test", "evaluate", "RIM_ONE_r3"),
            ("REFUGE", "train_labeled", "train", "REFUGE"),
            ("Drishti_GS", "train_labeled", "train", "RIM_ONE_r3")]:
            with self.assertRaises(PermissionError):
                role_allowed(domain, role, purpose, current)
        model = nn.Conv2d(3, 3, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: 1 - .1 * i)
        ledger = BudgetLedger(Path(os.environ["QPROMPT_CPU_LEDGER"]))
        state = capture(model, optimizer, scheduler, {"step": 0, "data_cursor": 4})
        image = torch.randn(1, 3, 4, 4)
        label = torch.zeros(1, 4, 4, dtype=torch.long)
        F.cross_entropy(model(image), label).backward()
        ledger.step(optimizer)  # one physical synthetic segmentation optimizer call
        scheduler.step()
        restored = restore(state, model, optimizer, scheduler)
        self.assertEqual(restored, {"step": 0, "data_cursor": 4})
        self.assertEqual(optimizer.param_groups[0]["lr"], .01)
        self.assertTrue(all(torch.equal(model.state_dict()[k], v) for k, v in state["student"].items()))


if __name__ == "__main__":
    unittest.main()
