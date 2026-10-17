import unittest
import torch
from torch.nn import functional as F
from .methods import fixmatch_loss,cps_loss,strong_view,learning_rate,STEPS,update_ema,pixel_ce


class MethodsTest(unittest.TestCase):
    def test_deterministic_ce_matches_torch_value_and_gradient(self):
        x=torch.randn(2,4,3,3,requires_grad=True);y=torch.randint(4,(2,3,3))
        a=pixel_ce(x,y).mean();b=F.cross_entropy(x,y)
        torch.testing.assert_close(a,b)
        torch.testing.assert_close(torch.autograd.grad(a,x)[0],torch.autograd.grad(b,x)[0])

    def test_fixmatch_author_mean_and_detach(self):
        weak=torch.tensor([[[[10.,0.]],[[0.,0.]],[[0.,0.]],[[0.,0.]]]],requires_grad=True)
        strong=torch.zeros_like(weak,requires_grad=True)
        loss,kept=fixmatch_loss(strong,weak)
        self.assertAlmostEqual(float(loss),float(torch.log(torch.tensor(4.)))/2,places=6)
        self.assertEqual(kept,.5);loss.backward()
        self.assertIsNone(weak.grad);self.assertEqual(float(strong.grad[:,:,:,1].abs().sum()),0)

    def test_cps_both_directions_over_all_pixels(self):
        left=torch.randn(4,4,3,3,requires_grad=True);right=torch.randn_like(left,requires_grad=True)
        expected=F.cross_entropy(left,right.detach().argmax(1))+F.cross_entropy(right,left.detach().argmax(1))
        actual=cps_loss(left,right);torch.testing.assert_close(actual,expected)
        actual.backward();self.assertTrue(torch.isfinite(left.grad).all());self.assertTrue(torch.isfinite(right.grad).all())

    def test_strong_deterministic_shape_range(self):
        x=torch.linspace(-1,1,32*32).reshape(1,1,32,32).repeat(2,1,1,1)
        first=strong_view(x,1);torch.testing.assert_close(first,strong_view(x,1),rtol=0,atol=0)
        self.assertEqual(first.shape,x.shape);self.assertTrue(torch.isfinite(first).all())
        self.assertGreaterEqual(float(first.min()),-1);self.assertLessEqual(float(first.max()),1)
        self.assertFalse(torch.equal(first,strong_view(x,2)))

    def test_schedule_and_ema(self):
        self.assertEqual(learning_rate('fixmatch',0),.03);self.assertEqual(learning_rate('cps',0),.005)
        for m in ('fixmatch','cps'):
            self.assertGreater(learning_rate(m,STEPS-1),0)
            with self.assertRaises(ValueError):learning_rate(m,STEPS)
        model=torch.nn.Linear(1,1,bias=False);ema=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():model.weight.fill_(1);ema.weight.zero_()
        update_ema(ema,model);self.assertAlmostEqual(float(ema.weight),.001,places=7)


if __name__=='__main__':unittest.main()
