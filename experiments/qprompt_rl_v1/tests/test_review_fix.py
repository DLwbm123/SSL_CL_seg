"""R0_REVIEW_FIX: CPU math only; no optimizer, data access, or CUDA calls."""
import unittest

import torch

from qprompt.losses import supervised_query_loss
from qprompt.models import semantic_probabilities


class ReviewFixChecks(unittest.TestCase):
    def test_ignore_and_background(self):
        c = torch.zeros(1, 12, 4, requires_grad=True)
        m = torch.zeros(1, 12, 8, 8, requires_grad=True)
        ignored = torch.full((1, 8, 8), 255, dtype=torch.long)
        out = supervised_query_loss(c, m, ignored)
        self.assertEqual(out['total'].item(), 0)
        out['total'].backward()
        self.assertEqual(c.grad.abs().sum().item(), 0)
        self.assertEqual(m.grad.abs().sum().item(), 0)
        background = torch.zeros_like(ignored)
        valid = supervised_query_loss(c.detach(), m.detach(), background)
        self.assertEqual(valid['matched'].item(), 1)
        self.assertGreater(valid['total'].item(), 0)
        labels = torch.cat((background, ignored))
        cc = c.detach().repeat(2, 1, 1)
        mm = m.detach().repeat(2, 1, 1, 1)
        before = supervised_query_loss(cc, mm, labels)['total']
        cc[1, :, 3] = 10
        mm[1] = 7
        after = supervised_query_loss(cc, mm, labels)['total']
        torch.testing.assert_close(before, valid['total'], rtol=0, atol=0)
        torch.testing.assert_close(after, before, rtol=0, atol=0)

    def test_probability_mass_and_gradients(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for no_object, mask in ((0., 0.), (35., 0.), (1000., -1000.)):
                with self.subTest(dtype=dtype, no_object=no_object):
                    c = torch.zeros(1, 12, 4, dtype=dtype)
                    c[..., 3] = no_object
                    c.requires_grad_()
                    m = torch.full((1, 12, 2, 2), mask, dtype=dtype, requires_grad=True)
                    with torch.autocast('cpu', dtype=torch.bfloat16):
                        p = semantic_probabilities(c, m)
                    self.assertEqual(p.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(p).all() and (p >= 0).all())
                    torch.testing.assert_close(p.sum(1), torch.ones_like(p[:, 0]))
                    p[:, 0].sum().backward()
                    self.assertTrue(torch.isfinite(c.grad).all())
                    self.assertTrue(torch.isfinite(m.grad).all())

    def test_normal_aggregation_and_nonfinite_rejection(self):
        torch.manual_seed(261)
        c, m = torch.randn(2, 12, 4), torch.randn(2, 12, 3, 3)
        scores = torch.einsum('bkc,bkhw->bchw', c.softmax(-1)[..., :3], m.sigmoid())
        torch.testing.assert_close(semantic_probabilities(c, m), scores / scores.sum(1, keepdim=True))
        for value in (float('nan'), float('inf'), -float('inf')):
            for tensor in ('classes', 'masks'):
                cc, mm = c.clone(), m.clone()
                (cc if tensor == 'classes' else mm).flatten()[0] = value
                with self.assertRaisesRegex(ValueError, 'finite'):
                    semantic_probabilities(cc, mm)


if __name__ == '__main__':
    unittest.main(verbosity=2)
