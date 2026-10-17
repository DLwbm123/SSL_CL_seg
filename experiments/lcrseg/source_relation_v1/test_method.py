import unittest
import torch
from .method import readout,target,transfer,support_indices,labeled_readout,nodes


class Contract(unittest.TestCase):
    def test_missing_class_mass(self):
        query=torch.eye(4); support=torch.eye(4)[:2]; labels=torch.tensor([0,1])
        base=torch.tensor([[.1,.2,.3,.4]]).repeat(4,1)
        result,present=readout(query,support,labels,base)
        self.assertTrue(torch.equal(result[:,2:],base[:,2:]))
        torch.testing.assert_close(result.sum(1),torch.ones(4))
        self.assertEqual(present.tolist(),[True,True,False,False])

    def test_expansion_detach_and_no_overlap(self):
        h=torch.zeros(1,4,16,16,requires_grad=True)
        with torch.no_grad():h[:,0]=1
        support=torch.zeros(2,4,16,16,requires_grad=True)
        y=torch.zeros(2,16,16,dtype=torch.long);y[:,:,8:]=1
        with torch.no_grad():support[:,0,:,:8]=1;support[:,1,:,8:]=1
        base=torch.full((1,256,4),.25);base[:,:,2:]=0;base[:,:,:2]=.5;base[:,0]=torch.tensor([.99,.01,0,0])
        ix=support_indices(y,0);q,keep,_=target(h,support,y,base,ix)
        self.assertFalse(q.requires_grad);self.assertFalse(bool(keep[0,0]));self.assertTrue(keep[0,1:].all())
        logits=torch.randn(1,4,16,16,requires_grad=True)
        # Use non-collinear student query so normalized readout has a nonzero derivative.
        sh=torch.randn_like(h,requires_grad=True)
        loss=transfer(logits,sh,support,y,ix,q,keep);loss.backward()
        self.assertGreater(float(sh.grad.norm()),0);self.assertGreater(float(logits.grad.norm()),0)
        self.assertIsNone(h.grad);self.assertIsNone(support.grad)

    def test_patient_exclusion_and_empty_support_class(self):
        h=torch.randn(2,4,16,16,requires_grad=True)
        y=torch.zeros(2,16,16,dtype=torch.long);y[0]=1
        ix=support_indices(y,2);l,n=labeled_readout(h,y,ix)
        self.assertEqual(n,0);self.assertEqual(float(l),0)
        l.backward();self.assertTrue(torch.isfinite(h.grad).all())

    def test_support_determinism_and_cap(self):
        y=torch.arange(256).reshape(1,16,16).remainder(4).repeat(2,1,1)
        a=support_indices(y,3);b=support_indices(y,3)
        self.assertTrue(all(torch.equal(x,z) for x,z in zip(a,b)))
        self.assertTrue(all(len(x)==128 for x in a))


if __name__=='__main__':unittest.main()
