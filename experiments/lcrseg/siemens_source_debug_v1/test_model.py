import unittest
import copy
import torch
from .model import hard_objective, HardPseudoTrainer
from ..main_style_ssl_v1.model import MainTrainer, MainModel
from ..five_frameworks_v1.parent_bridge import SyntheticParentBridge
from ..five_frameworks_v1.recipes import SyntheticCurrentDomain


class ObjectiveTest(unittest.TestCase):
    def test_unchanged_supervised_branch(self):
        torch.set_num_threads(1);torch.manual_seed(163)
        parent=SyntheticParentBridge();parent.source_dense=True
        model=MainModel(parent,(0,1,1))
        for step in (0,2):
            terms=[]
            for cls in (MainTrainer,HardPseudoTrainer):
                provider=SyntheticCurrentDomain(seed=163,stage=0);provider.dataset='mnms'
                trainer=cls(copy.deepcopy(model),provider,dict(total_steps=10,lambda_U=.25))
                trainer.step=trainer.cursor=step
                loss,u,_=trainer.losses()
                params=[p for p in trainer.model.parameters() if p.requires_grad]
                terms.append((loss,torch.autograd.grad(loss,params,allow_unused=True)))
                self.assertEqual(provider.u_reads,0 if step==0 else 1)
                if step:self.assertTrue(torch.isfinite(u))
            torch.testing.assert_close(terms[0][0],terms[1][0],rtol=0,atol=0)
            for a,b in zip(terms[0][1],terms[1][1]):
                if a is None or b is None:self.assertIs(a,b)
                else:torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_gradient_and_detached_target(self):
        target=torch.tensor([[[[.99,.01]],[[.01,.99]],[[0.,0.]],[[0.,0.]]]],requires_grad=True)
        logits=torch.zeros_like(target,requires_grad=True)
        loss,stats=hard_objective(logits,target,torch.ones((1,1,2),dtype=torch.bool))
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(logits.grad[0,1,0,1].item(),0)
        self.assertEqual(stats['class_pixels'],[1,1,0,0])

    def test_empty_and_background(self):
        q=torch.zeros((1,4,2,2));q[:,0]=1
        for keep in (False,True):
            x=torch.zeros_like(q,requires_grad=True)
            loss,stats=hard_objective(x,q,torch.full((1,2,2),keep,dtype=torch.bool))
            loss.backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertEqual(stats['foreground_dice'],0)
            if not keep:
                self.assertEqual(float(loss),0)
                self.assertEqual(float(x.grad.abs().sum()),0)

    def test_class_balance(self):
        q=torch.zeros((1,4,1,10));q[:,0,:,:9]=1;q[:,1,:,9]=1
        x=torch.zeros_like(q)
        loss,stats=hard_objective(x,q,torch.ones((1,1,10),dtype=torch.bool))
        self.assertAlmostEqual(stats['balanced_ce'],float(torch.log(torch.tensor(4.))),places=6)
        with self.assertRaises(ValueError):hard_objective(x,q,torch.ones((1,1,10)))

if __name__=='__main__':unittest.main()
