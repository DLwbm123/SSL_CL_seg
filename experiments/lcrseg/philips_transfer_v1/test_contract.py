import unittest
import torch
from .model import rate, STEPS
from .diagnostic import graph, propagate, counts


class Contract(unittest.TestCase):
    def test_budget_schedule(self):
        self.assertEqual(STEPS,10000)
        self.assertEqual(rate(0),.03)
        self.assertGreater(rate(9999),.028)
        with self.assertRaises(PermissionError): rate(10000)

    def test_graph_and_permutation(self):
        torch.manual_seed(163)
        g=graph(torch.randn(25,16))
        self.assertTrue(torch.allclose(g.sum(1),torch.ones(25)))
        self.assertTrue((g>=0).all())
        self.assertTrue((g.diag()==0).all())
        perm=torch.randperm(25); shuffled=g[perm][:,perm]
        self.assertTrue(torch.equal(g.flatten().sort().values,shuffled.flatten().sort().values))
        p=torch.rand(25,4).softmax(1); p[0]=torch.tensor([1.,0,0,0])
        q=propagate(g,p)
        self.assertTrue(torch.allclose(q.sum(1),torch.ones(25),atol=1e-6))
        self.assertTrue(torch.equal(q[0],p[0]))
        expected=propagate(shuffled,p[perm])
        self.assertTrue(torch.allclose(expected,q[perm],atol=1e-6))

    def test_correction_accounting(self):
        r=counts(torch.tensor([1,0,2]),torch.tensor([0,1,2]),torch.tensor([1,1,2]),torch.ones(3,dtype=torch.bool))
        self.assertEqual(r,dict(n=3,correct=2,corrected=1,broken=1))


if __name__=='__main__': unittest.main()
