"""CPU mathematical/gradient checks; no optimizer updates or private data."""
import copy
import math
import unittest

import torch
from torch.nn import functional as F
from sc_bayeseg.model import (Model, BayesianLayer, BlockMean, student_log_prob,
                             sar, sar_logdet, normal_log_prob)


class ProbabilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(163)

    def test_normal_gamma_matches_student_t(self):
        value = torch.tensor([-.7, 0., 1.8], dtype=torch.float64).reshape(3, 1, 1, 1)
        actual = student_log_prob(value, .2)
        expected = torch.distributions.StudentT(4., 0., math.sqrt(.2 ** 2 / 2)).log_prob(value.flatten())
        torch.testing.assert_close(actual, expected)

    def test_sar_determinant_and_density(self):
        n=5
        basis = torch.eye(n*n, dtype=torch.float64).reshape(n*n, 1, n, n)
        D = sar(basis).reshape(n*n, n*n).T
        self.assertGreater(float(torch.linalg.eigvalsh(D).min()), 0)
        self.assertAlmostEqual(float(torch.linalg.slogdet(D).logabsdet), sar_logdet(n), places=10)
        value = torch.randn(1, 1, n, n, dtype=torch.float64)
        tau = torch.linspace(.5, 1.5, n*n, dtype=torch.float64)
        Q = D.T @ torch.diag(tau) @ D
        expected = torch.distributions.MultivariateNormal(torch.zeros(n*n, dtype=torch.float64),
                     precision_matrix=Q).log_prob(value.flatten())
        actual = normal_log_prob(sar(value), torch.zeros_like(value),
                                 -.5 * tau.log().reshape_as(value)) + sar_logdet(n)
        torch.testing.assert_close(actual[0], expected)

    def test_global_prior_recursion_and_immutability(self):
        layer = BayesianLayer(3, 2)
        self.assertAlmostEqual(float(layer.kl().detach()), 0., places=5)
        initial = copy.deepcopy(layer.state_dict())
        with torch.no_grad():
            layer.mean_w.add_(.01)
        self.assertGreater(float(layer.kl().detach()), 0)
        self.assertTrue(torch.equal(initial['prior_mean_w'], layer.prior_mean_w))
        layer.set_prior(True)
        self.assertAlmostEqual(float(layer.kl().detach()), 0., places=5)
        layer.set_prior(False)
        self.assertTrue(torch.equal(initial['initial_mean_w'], layer.prior_mean_w))

    def test_semantic_gradient_and_observed_label_factor(self):
        model=Model(8)
        image=torch.randn(2, 1, 8, 8)
        model.draw()
        h, _ = model.proposal(image)
        h['ell'] = h['ell'].detach().requires_grad_()
        unobserved, terms, _ = model.log_joint(image, h)
        gradient = torch.autograd.grad(terms['shape'].sum(), h['ell'], retain_graph=True)[0]
        self.assertGreater(float(gradient.abs().sum()), 0)
        cut, cut_terms, _ = model.log_joint(image, h, detach_semantics=True)
        self.assertIsNone(torch.autograd.grad(cut_terms['shape'].sum(), h['ell'],
                                            allow_unused=True, retain_graph=True)[0])
        y=torch.randint(0,4,(2,8,8))
        observed, _, p = model.log_joint(image, h, y)
        expected = p.gather(1,y[:,None]).log().sum((1,2,3))
        torch.testing.assert_close(observed-unobserved, expected, rtol=1e-4, atol=2e-4)
        self.assertTrue(torch.equal(terms['labels'], torch.zeros(2)))

    def test_same_rng_same_bound_and_finite_gradients(self):
        model=Model(8)
        image=torch.randn(2,1,8,8)
        rng=torch.get_rng_state()
        model.draw(); first=model.bound(image)[0]
        torch.set_rng_state(rng)
        model.draw(); second=model.bound(image)[0]
        self.assertTrue(torch.equal(first,second))
        (-second.mean()+model.kl()).backward()
        for name,p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
        self.assertFalse(any(b.requires_grad for b in model.buffers()))

    def test_predictive_refinement_does_not_mutate_model(self):
        model=Model(8)
        before=copy.deepcopy(model.state_dict())
        image=torch.randn(2,1,12,12)
        p,diag=model.prediction(image, refine=1, parameter_samples=2, local_samples=2)
        torch.testing.assert_close(p.sum(1), torch.ones(2,12,12))
        self.assertEqual(tuple(p.shape),(2,4,12,12))
        self.assertTrue(1 <= diag['ess'] <= 4.001)
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_method_observation_counts_match_full_objective(self):
        from lcrseg.contracts import LabeledBatch, UnlabeledBatch
        from sc_bayeseg.method import Method
        model=Model(8)
        image=torch.randn(2,1,8,8)
        label=torch.randint(0,4,(2,8,8))
        valid=torch.ones(2,1,8,8,dtype=torch.bool)
        meta=dict(case_id=['a','b'],patient_id=['a','b'],site=['s','s'],slice_index=[0,0])
        lb=LabeledBatch(image,label,valid,**meta)
        ub=UnlabeledBatch(image*.8,image*.8,valid,geometry_record=[{},{}],**meta)
        method=Method(model,8,20)
        rng=torch.get_rng_state()
        actual=method.training_step(lb,ub,0,0).total_loss
        torch.set_rng_state(rng)
        model.draw()
        el=model.bound(image,label)[0].mean()
        eu=model.bound(image*.8)[0].mean()
        expected=(-8*el-20*eu+model.kl())/(28*8*8)
        torch.testing.assert_close(actual,expected)
        actual.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_block_average_equivalent(self):
        value=torch.randn(2,3,16,16,requires_grad=True)
        actual=BlockMean()(value)
        expected=F.adaptive_avg_pool2d(value,4)
        torch.testing.assert_close(actual,expected)
        ga=torch.autograd.grad(actual.square().sum(),value,retain_graph=True)[0]
        ge=torch.autograd.grad(expected.square().sum(),value)[0]
        torch.testing.assert_close(ga,ge)


if __name__ == '__main__':
    unittest.main()
