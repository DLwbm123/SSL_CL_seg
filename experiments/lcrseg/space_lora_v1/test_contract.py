import unittest
import torch
from .core import allocate


class Contract(unittest.TestCase):
    def test_allocation_activation_order_and_cap(self):
        h=torch.diag(torch.tensor([4.,3.,2.,1.]))
        f=torch.diag(torch.tensor([0.,0.,1.,9.]))
        basis,meta=allocate(h,f,rank=2,rho=.9)
        self.assertEqual(meta['requested'],4)
        self.assertEqual(meta['protected'],2)
        self.assertFalse(meta['target_met'])
        torch.testing.assert_close(basis@basis.T,torch.diag(torch.tensor([1.,1.,0.,0.])))

    def test_zero_fisher(self):
        basis,meta=allocate(torch.eye(4),torch.zeros(4,4),rank=2)
        self.assertEqual(basis.shape,(4,0)); self.assertTrue(meta['target_met'])

    def test_reassociated_per_example_fisher(self):
        x=torch.randn(2,7,4,dtype=torch.float64)
        delta=torch.randn(2,7,3,dtype=torch.float64)
        original=x.flatten(0,1).T@(delta@delta.transpose(1,2)@x).flatten(0,1)
        g=delta.transpose(1,2)@x
        torch.testing.assert_close(original,(g.transpose(1,2)@g).sum(0))

    def test_projection_golden(self):
        a=torch.tensor([[1.,2.,3.]])
        v=torch.tensor([[1.],[0.],[0.]])
        projected=a-(a@v)@v.T
        torch.testing.assert_close(projected,torch.tensor([[0.,2.,3.]]))


if __name__=='__main__': unittest.main()
