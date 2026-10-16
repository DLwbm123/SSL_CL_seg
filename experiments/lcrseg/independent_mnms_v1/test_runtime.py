import tempfile
import unittest
from pathlib import Path
import torch
from ..five_frameworks_v1.checkpoint import atomic_save, PhysicalLedger
from .runtime import crop_input, restore_prediction, augment, records, make_trainer, step, snapshot, restore
from .protocol import PROTOCOL as P


class Contracts(unittest.TestCase):
    def test_crop_and_full_canvas_restoration(self):
        x = torch.zeros(1, 1, 384, 384); y = torch.zeros(1, 384, 384, dtype=torch.uint8)
        top, left = P['crop_yx']; n = P['crop_pixels']; y[:, top:top+n, left:left+n] = 2
        xx, yy = crop_input(x, y)
        self.assertEqual(xx.shape, x.shape); self.assertTrue((yy == 2).all())
        logits = torch.full((1, 4, 384, 384), -20.); logits[:, 2] = 20.
        pred = restore_prediction(logits)
        self.assertTrue(torch.equal(pred, y.long()))
        # A missed target outside the crop remains in full-canvas metric support.
        truth = y.clone(); truth[:, 0:10, 0:10] = 2
        self.assertLess(2*((pred == 2) & (truth == 2)).sum()/((pred == 2).sum()+(truth == 2).sum()), 1)

    def test_no_label_needed_to_localize(self):
        x = torch.rand(2, 1, 384, 384)*2-1
        self.assertTrue(torch.equal(crop_input(x), crop_input(x, torch.zeros(2, 384, 384))[0]))
        with self.assertRaises(ValueError): crop_input(torch.zeros(1, 1, 64, 64))

    def test_augmentation_determinism_and_label_integrity(self):
        x = torch.linspace(-1, 1, 384*384).reshape(1, 1, 384, 384).repeat(2, 1, 1, 1)
        y = torch.zeros(2, 384, 384, dtype=torch.long); y[:, 120:220, 140:240] = 3
        a, b = augment(x, y, 163); aa, bb = augment(x, y, 163)
        self.assertTrue(torch.equal(a, aa) and torch.equal(b, bb))
        self.assertTrue(torch.isfinite(a).all() and a.min() >= -1 and a.max() <= 1)
        self.assertEqual(set(b.unique().tolist()), {0, 3})

    def test_role_and_budget_boundaries(self):
        for role in ('test', 'train_unlabeled'):
            with self.assertRaises(PermissionError): records('/not-read', role)
        self.assertEqual((230+5)//6*200, 7800)
        for seed in P['seeds']:
            order = torch.randperm(230, generator=torch.Generator().manual_seed(seed+10000))
            self.assertEqual(sorted(order.tolist()), list(range(230)))

    def test_shared_step_and_exact_restore(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); ledger = PhysicalLedger(path/'calls.jsonl')
            tr, state = make_trainer(163, 'cpu', ledger)
            self.assertFalse(any('projection_head' in name for name, _ in tr.method.model.named_parameters()))
            x = torch.rand(2, 1, 32, 32); y = torch.zeros(2, 32, 32, dtype=torch.long); y[:, 8:24, 8:24] = 1
            step(tr, state, x, y); atomic_save(snapshot(tr, state, 163), path/'state.pt')
            first = step(tr, state, x, y)
            expected = {k: v.clone() for k, v in tr.method.model.state_dict().items()}
            payload = torch.load(path/'state.pt', map_location='cpu', weights_only=False)
            with self.assertRaises(ValueError): restore(tr, state, payload, 164)
            restore(tr, state, payload, 163); second = step(tr, state, x, y)
            self.assertEqual(first, second)
            self.assertTrue(all(torch.equal(expected[k], v) for k, v in tr.method.model.state_dict().items()))
            self.assertEqual(tr.optimizer.param_groups[0]['lr'], 0.0005)
            self.assertEqual(ledger.summary(state.global_step)['physical_optimizer_updates'], 3)


if __name__ == '__main__': unittest.main()
