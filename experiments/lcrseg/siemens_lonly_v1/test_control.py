"""CPU checks: preserve labeled loss/gradients on both sides of warmup."""
import copy
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import torch

from ..five_frameworks_v1.parent_bridge import SyntheticParentBridge
from ..five_frameworks_v1.recipes import SyntheticCurrentDomain
from ..main_style_ssl_v1.model import MainModel, MainTrainer
from .run import aggregate, compare, dice, remaining_target


class ControlTest(unittest.TestCase):
    def test_saved_evaluation_requires_complete_pairs(self):
        from .recover_evaluation import saved_rows
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'rows.jsonl'
            rows = [dict(patient='p', phase=phase, dice=[.5]*3) for phase in ('ED', 'ES')]
            path.write_text('\n'.join(map(json.dumps, rows)))
            self.assertEqual(saved_rows(path, 1)[0]['primary_dice'], .5)
            with self.assertRaises(ValueError):
                saved_rows(path, 2)
            path.write_text(json.dumps(rows[0]))
            with self.assertRaises(ValueError):
                saved_rows(path, 1)

    def test_recovery_budget(self):
        self.assertEqual(remaining_target(0, 0, 43600), 43600)
        self.assertEqual(remaining_target(0, 1, 43600), 43599)
        self.assertEqual(remaining_target(0, 1, 43601), 43600)
        self.assertEqual(remaining_target(2000, 2050, 43600), 43550)
        for args in ((0, 43600, 43600), (2, 1, 43600), (0, -1, 43600), (0, True, 43600)):
            with self.assertRaises(ValueError):
                remaining_target(*args)

    def test_only_u_changes(self):
        torch.set_num_threads(1)
        torch.manual_seed(163)
        parent = SyntheticParentBridge()
        parent.source_dense = True
        model = MainModel(parent, (0, 1, 1))
        for step in (0, 2, 7):
            arms = []
            for weight in (0., .25):
                provider = SyntheticCurrentDomain(seed=163, stage=0)
                provider.dataset = 'mnms'
                trainer = MainTrainer(copy.deepcopy(model), provider, dict(total_steps=10, lambda_U=weight))
                trainer.step = trainer.cursor = step
                labeled, unlabeled, pending = trainer.losses()
                params = [p for p in trainer.model.parameters() if p.requires_grad]
                grads = torch.autograd.grad(labeled, params, allow_unused=True)
                arms.append((labeled.detach(), grads))
                if weight == 0 or step < 2:
                    self.assertIsNone(unlabeled)
                    self.assertIsNone(pending)
                    self.assertEqual(provider.u_reads, 0)
                else:
                    self.assertIsNotNone(unlabeled)
                    self.assertEqual(provider.u_reads, 1)
                # Zero-U does not prolong supervised warmup or remove LCTX.
                self.assertEqual(trainer.hierarchy_stats['labeled_feature_views'], 1 if step < 2 else 2)
            torch.testing.assert_close(arms[0][0], arms[1][0], rtol=0, atol=0)
            for a, b in zip(arms[0][1], arms[1][1]):
                if a is None or b is None:
                    self.assertIs(a, b)
                else:
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_dice_patient_phase_weighting(self):
        np.testing.assert_array_equal(dice(np.array([0, 0, 2, 3]), np.array([0, 1, 2, 3])), [0., 1., 1.])
        rows = [dict(patient='p', phase=p, dice=[v]*3) for p, v in [('ED', 1.), ('ES', 0.)]]
        self.assertEqual(aggregate(rows)['primary_dice'], .5)
        self.assertEqual(compare(rows, rows)['delta_B_minus_A_pp'], 0.)
        with self.assertRaises(ValueError):
            aggregate(rows[:1])
        with self.assertRaises(ValueError):
            compare(rows, rows[:1])


if __name__ == '__main__':
    unittest.main()
