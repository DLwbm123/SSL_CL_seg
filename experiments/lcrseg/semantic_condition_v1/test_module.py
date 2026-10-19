import copy
import unittest
import torch
from torch import nn
from .module import SemanticCondition


class Contract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(163)
        self.h = torch.randn(2, 16, 11, 13)
        self.p = torch.randn(4, 3, 144)

    def test_entry_and_native_readout(self):
        module = SemanticCondition(self.p)
        head = nn.Conv2d(16, 4, 3).requires_grad_(False)
        self.assertTrue(torch.equal(module(self.h), self.h))
        self.assertTrue(torch.equal(head(module(self.h)), head(self.h)))

    def test_spatial_control_preserves_values(self):
        aligned = SemanticCondition(self.p)
        shuffled = SemanticCondition(self.p, shuffled=True)
        a, b = aligned.condition(self.h), shuffled.condition(self.h)
        self.assertFalse(torch.equal(a, b))
        self.assertTrue(torch.equal(a.flatten(2).sort(-1).values,
                                    b.flatten(2).sort(-1).values))
        self.assertTrue(torch.allclose(a.sum(1), torch.ones_like(a[:, 0])))

    def test_supervision_reaches_residual_only(self):
        module = SemanticCondition(self.p)
        head = nn.Conv2d(16, 4, 3).requires_grad_(False)
        saved = module.prototypes.clone()
        logits = head(module(self.h))
        labels = torch.randint(4, logits.shape[:1] + logits.shape[2:])
        nn.functional.cross_entropy(logits, labels).backward()
        self.assertGreater(float(module.semantic_path[-1].weight.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in head.parameters()))
        self.assertIsNone(module.prototypes.grad)
        self.assertTrue(torch.equal(saved, module.prototypes))
        self.assertIsNone(self.h.grad)

    def test_bounded_shift_and_restore(self):
        module = SemanticCondition(self.p)
        with torch.no_grad():
            for p in module.parameters():
                p.normal_(0, .1)
        actual = module(self.h)
        self.assertTrue(((actual-self.h).abs() <= .1*self.h.abs()+.100001).all())
        restored = SemanticCondition(self.p)
        restored.load_state_dict(copy.deepcopy(module.state_dict()))
        self.assertTrue(torch.equal(actual, restored(self.h)))

    def test_entry_gradient_unit_reparameterization(self):
        module = SemanticCondition(self.p)
        condition = module.condition(self.h)
        mean = self.h.mean((-2,-1))
        std = self.h.var((-2,-1),unbiased=False).add(1e-6).sqrt()
        base = module.self_path(torch.cat((mean,std),1))[:,:,None,None]
        scale, shift = (base+module.semantic_path(condition)).chunk(2,1)
        previous = (1+.1*scale.tanh())*self.h+.1*shift.tanh()
        parameter = module.semantic_path[-1].weight
        old = torch.autograd.grad(previous.square().mean(),parameter,retain_graph=True)[0]
        new = torch.autograd.grad(module(self.h).square().mean(),parameter)[0]
        self.assertGreater(float(old.norm()),0)
        self.assertTrue(torch.allclose(new,10*old,atol=1e-7,rtol=1e-5))


if __name__ == '__main__':
    unittest.main()
