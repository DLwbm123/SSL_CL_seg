import io
import unittest
import torch
from . import core


class Contract(unittest.TestCase):
    def test_conditional_updates_and_resume(self):
        torch.manual_seed(163)
        x=torch.randn(24,5,dtype=torch.float64); y=torch.eye(4,dtype=torch.float64)[torch.arange(24)%4]
        gl=x.T@x/24; cl=x.T@y/24; gu=gl.clone()
        memory=dict(n=torch.ones(24,dtype=torch.float64)/24,mu=x,v=torch.zeros(24,dtype=torch.float64))
        w=core.solve(gl,cl); q=core.responsibility(x,w)
        updated=core.solve(gl+core.LAMBDA*gu,cl+core.LAMBDA*core.cross(memory,q))
        self.assertLessEqual(float(core.objective(updated,gl,cl,gu,memory,q)),float(core.objective(w,gl,cl,gu,memory,q))+1e-12)
        q2=core.responsibility(x,updated)
        self.assertLessEqual(float(core.objective(updated,gl,cl,gu,memory,q2)),float(core.objective(updated,gl,cl,gu,memory,q))+1e-12)
        buffer=io.BytesIO(); torch.save(dict(w=updated,memory=memory),buffer); buffer.seek(0)
        saved=torch.load(buffer,weights_only=False)
        self.assertTrue(torch.equal(core.cross(memory,q2),core.cross(saved['memory'],core.responsibility(saved['memory']['mu'],saved['w']))))

    def test_two_case_interpolation(self):
        x=torch.eye(2,dtype=torch.float64); y=torch.tensor([[1.,0.,0.,0.],[0.,0.,1.,0.]],dtype=torch.float64)
        w=core.solve(x.T@x,x.T@y)
        self.assertTrue(torch.equal((x@w).argmax(1),y.argmax(1)))

    def test_curvature_not_centroid(self):
        x=torch.tensor([[1.,-1.],[1.,1.]],dtype=torch.float64); mu=x.mean(0,keepdim=True)
        g=x.T@x/2; self.assertGreater(float((g-mu.T@mu).norm()),.9)
        w=torch.tensor([[.1,.2,.3,.4],[.3,.1,-.2,.1]],dtype=torch.float64)
        q=core.responsibility(mu,w)
        expected=sum(float((q[0]*(x[i]@w-torch.eye(4,dtype=torch.float64)).square().sum(1)).sum())/2 for i in range(2))
        collapsed=float((w*(g@w)).sum()-2*(w*(mu.T@q)).sum()+1)
        self.assertAlmostEqual(expected,collapsed,places=12)

    def test_counterexample(self):
        x=torch.tensor([[1.,a] for a in (-2,-2,1,3)],dtype=torch.float64)
        z=torch.tensor([[1.,a] for a in (-3,-1,2,2)],dtype=torch.float64)
        self.assertTrue(torch.equal(x.T@x,z.T@z))
        self.assertFalse(torch.equal(x[x[:,1]>1.5].sum(0),z[z[:,1]>1.5].sum(0)))


if __name__=='__main__': unittest.main()
