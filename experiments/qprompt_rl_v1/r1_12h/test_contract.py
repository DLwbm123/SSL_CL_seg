"""R1 preparation math checks: zero optimizer calls, no data or CUDA access."""
import unittest
import torch
from torch.nn import functional as F
from qprompt.losses import PrototypeBank, grqa_loss
from .core import direct_alignment, seed, build, dense_loss
from .augmentation import geometry, strong
from pathlib import Path


class Contract(unittest.TestCase):
    def test_q4_value_gradient_detach(self):
        bank=PrototypeBank(3);bank.vectors.copy_(torch.eye(3));bank.supported[:]=True
        q=torch.tensor([[[1.,.2,0.],[.4,1.,.1]]],requires_grad=True)
        ref=(q.detach()+.3).requires_grad_()
        direct=direct_alignment(q,ref,bank)
        expected=-F.normalize(q,dim=-1).amax(-1).mean()
        torch.testing.assert_close(direct,expected)
        k3=grqa_loss(q,ref,bank)['paper_selected_k3_surrogate']
        actual=5*direct+.005*k3
        s=F.normalize(q,dim=-1);r=F.normalize(ref.detach(),dim=-1)
        j=s.detach().argmax(-1,keepdim=True)
        delta=r.log_softmax(-1).gather(-1,j)-s.log_softmax(-1).gather(-1,j)
        explicit=5*expected+.005*(delta.exp()-1-delta).mean()
        ga=torch.autograd.grad(actual,q,retain_graph=True)[0]
        ge=torch.autograd.grad(explicit,q)[0]
        torch.testing.assert_close(ga,ge);self.assertGreater(float(ga.abs().sum()),0)
        self.assertIsNone(ref.grad);self.assertIsNone(bank.vectors.grad)

    def test_fair_body_and_named_streams(self):
        a,ha=build('UNET_QUERY_128',True,Path('.'),Path('.'))
        b,hb=build('UNET_QUERY_128',False,Path('.'),Path('.'))
        self.assertEqual(ha,hb)
        for k,v in a.body.state_dict().items():self.assertTrue(torch.equal(v,b.body.state_dict()[k]))
        self.assertEqual(seed('domain',20,'geometry'),seed('domain',20,'geometry'))
        self.assertNotEqual(seed('domain',20,'geometry'),seed('domain',20,'photometric'))

    def test_augmentation_pairing_and_dense_loss(self):
        x=torch.arange(3*16*16).reshape(3,16,16).float()/768
        y=torch.arange(16*16).reshape(16,16)
        g=lambda:torch.Generator().manual_seed(261)
        a,b,_=geometry(x,y,torch.ones_like(y,dtype=torch.bool),g())
        c,d,_=geometry(x,y,torch.ones_like(y,dtype=torch.bool),g())
        self.assertTrue(torch.equal(a,c) and torch.equal(b,d))
        self.assertTrue(torch.equal(strong(a[None],g()),strong(c[None],g())))
        logits=torch.randn(2,3,8,8);label=torch.zeros(2,8,8,dtype=torch.long);label[:,:2]=255
        torch.testing.assert_close(dense_loss(logits,label),F.cross_entropy(logits,label,ignore_index=255))


if __name__=='__main__':unittest.main(verbosity=2)
