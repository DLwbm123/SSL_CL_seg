"""Synthetic FP64 Gaussian/Schur and Gamma CAVI algebra; no medical data."""
import json
import math
import time

import torch

from sc_bayeseg.window_v1.r_start import write
from sc_bayeseg.model import student_log_prob


def run(root):
    root.mkdir(exist_ok=False)
    started = time.time()
    write(root / 'START.json', dict(started_unix=started))
    rows = []
    try:
        residual = torch.linspace(-2, 2, 101, dtype=torch.float64).reshape(101, 1, 1, 1)
        mixture_errors = []
        for scale in (.1, .2):
            a, b = 2., scale ** 2
            analytic = (a * math.log(b) - math.lgamma(a) - .5 * math.log(2 * math.pi)
                        + math.lgamma(a + .5) - (a + .5) * (b + residual.flatten().square() / 2).log())
            error = float((student_log_prob(residual, scale) - analytic).abs().max())
            assert error < 1e-12
            mixture_errors.append(error)
        for size in (8, 16):
            n = size ** 2
            ident = torch.eye(n, dtype=torch.float64)
            s = ident - .9 * sum(ident.reshape(n, size, size).roll(shift, dim).reshape(n, n)
                                  for shift, dim in ((1, -1), (-1, -1), (1, -2), (-1, -2))) / 4
            rng = torch.Generator().manual_seed(10202026 + size)
            image = torch.randn(n, generator=rng, dtype=torch.float64) * .1
            u = torch.randn(n, generator=rng, dtype=torch.float64) * .1
            ai = ax = torch.full((n,), 2.5, dtype=torch.float64)
            bi = torch.full((n,), .01, dtype=torch.float64)
            bx = torch.full((n,), .04, dtype=torch.float64)
            previous = None
            for sweep in range(10):
                assert time.time() - started < 7100
                di, dx = ai / bi, ax / bx
                qs = s.T @ (dx[:, None] * s)
                lam = di + 4
                a = di / lam
                eff = 4 * a
                qbar = qs + torch.diag(eff)
                c = torch.cholesky_inverse(torch.linalg.cholesky(qbar))
                mux = c @ (qs @ u + eff * image)
                mum = a * (image - mux)
                cxm = -c * a[None, :]
                cmm = torch.diag(1 / lam) + a[:, None] * c * a[None, :]
                cov = torch.cat((torch.cat((c, cxm), 1), torch.cat((cxm.T, cmm), 1)), 0)
                mu = torch.cat((mux, mum))
                q = torch.cat((torch.cat((qs + torch.diag(di), torch.diag(di)), 1),
                               torch.cat((torch.diag(di), torch.diag(lam)), 1)), 0)
                h = torch.cat((qs @ u + di * image, di * image))
                chol = torch.linalg.cholesky(q)
                ref_cov = torch.cholesky_inverse(chol)
                ref_mu = torch.cholesky_solve(h[:, None], chol)[:, 0]
                logdet = lam.log().sum() + torch.linalg.slogdet(qbar)[1]
                errors = dict(mean=float((mu - ref_mu).abs().max()), covariance=float((cov - ref_cov).abs().max()),
                              logdet=float((logdet - 2 * chol.diagonal().log().sum()).abs()))
                assert max(errors.values()) < 1e-9, errors
                ri = (image - mux - mum).square() + (c + cmm + cxm + cxm.T).diagonal()
                rx = (s @ (mux - u)).square() + (s @ c @ s.T).diagonal()
                # Cross covariance is essential to the image-residual moment.
                ref_ri = (image - ref_mu[:n] - ref_mu[n:]).square() + (
                    ref_cov[:n, :n] + ref_cov[n:, n:] + ref_cov[:n, n:] + ref_cov[n:, :n]).diagonal()
                assert (ri - ref_ri).abs().max() < 1e-10
                bi, bx = .01 + ri / 2, .04 + rx / 2
                def contribution(shape, rate, b0, residual):
                    elog = torch.digamma(shape) - rate.log()
                    expected = shape / rate
                    likelihood = .5 * (elog - math.log(2 * math.pi) - expected * residual)
                    prior = 2 * math.log(b0) - math.lgamma(2) + elog - b0 * expected
                    entropy = shape - rate.log() + torch.lgamma(shape) + (1 - shape) * torch.digamma(shape)
                    return (likelihood + prior + entropy).sum()
                elbo = (contribution(ai, bi, .01, ri) + contribution(ax, bx, .04, rx)
                        + torch.linalg.slogdet(s)[1]
                        - .5 * (4 * (mum.square() + cmm.diagonal()) + math.log(2 * math.pi / 4)).sum()
                        + .5 * (2 * n * (1 + math.log(2 * math.pi)) - logdet))
                assert torch.isfinite(elbo)
                if previous is not None:
                    assert elbo >= previous - 1e-8
                previous = float(elbo)
                rows.append(dict(size=size, sweep=sweep + 1, errors=errors, augmented_ELBO=float(elbo)))
        result = dict(status='PASS', grids=[8, 16], sweeps_per_grid=10, cases=rows,
                      scalar_Student_Gamma_mixture_errors=mixture_errors,
                      dtype='float64', no_medical_data=True, elapsed_seconds=time.time() - started,
                      interpretation='Algebra only; no exact Student evidence or segmentation claim')
        write(root / 'RESULTS.json', result)
        write(root / 'EXIT.json', dict(exit_code=0))
        return dict(status='PASS', report='E0/RESULTS.json')
    except BaseException:
        import traceback
        write(root / 'FAILURE.json', dict(traceback=traceback.format_exc(), rows=rows))
        write(root / 'EXIT.json', dict(exit_code=1))
        # E0 cannot invalidate or retrain the completed R experiment.
        return dict(status='FAILED', report='E0/FAILURE.json')
