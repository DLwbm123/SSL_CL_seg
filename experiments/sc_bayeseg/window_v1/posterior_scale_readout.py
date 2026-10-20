"""Read-only Gaussian family, KL decomposition and local analytic derivatives."""
import math
from pathlib import Path

import torch

from lcrseg.engine.checkpoint import capture_rng_state, restore_rng_state
from sc_bayeseg.window_v1.r_start import Interface, write


@torch.no_grad()
def readout(source, image, output):
    rng = capture_rng_state()
    rows = []
    model = Interface().to(image.device).requires_grad_(False)
    probabilities = torch.tensor([0., .01, .25, .5, .75, .99, 1.], dtype=torch.float64, device=image.device)
    try:
        for arm in ('R1', 'R2', 'R2-SG'):
            for step in (0, 200, 600):
                file = Path(source) / arm / f'S{step:04}.pt'
                before = file.stat()
                p = torch.load(file, map_location='cpu', weights_only=False)
                assert p['arm'] == arm and p['step'] == step
                model.load_state_dict(p['model'])
                q = model.posterior(model.e(image))
                row = dict(arm=arm, step=step, variables={})
                for name, mu, ls in (('c', q[0], q[1]), ('R', q[2], q[3])):
                    mu, ls = mu.double().flatten(1), ls.double().flatten(1)
                    d = mu.shape[1]
                    assert d == (16 if name == 'c' else 4096)
                    assert ls.min() >= -7 and ls.max() <= -1
                    mean_kl = .5 * mu.square().sum(1) / (128**2)
                    scale_kl = .5 * ((2*ls).exp()-1-2*ls).sum(1) / (128**2)
                    floor = d / (128**2) * .5 * (math.exp(-2)+1)
                    # d logsd / d rawsd, inferred from the unchanged tanh map.
                    jacobian = (1-((ls+4)/3).square()).clamp_min(0)
                    raw_scale_gradient = ((2*ls).exp()-1)*jacobian / (8*128**2)
                    mean_gradient = mu / (8*128**2)
                    total = mean_kl+scale_kl
                    assert (total >= floor-1e-10).all()
                    row['variables'][name] = dict(dimensions=d, mean_squared=float(mu.square().mean()),
                        mean_KL_per_pixel=float(mean_kl.mean()), scale_KL_per_pixel=float(scale_kl.mean()),
                        total_KL_per_pixel=float(total.mean()), family_KL_infimum_per_pixel=floor,
                        KL_above_infimum_per_pixel=float(total.mean())-floor,
                        logsd_quantiles=torch.quantile(ls.flatten(), probabilities).cpu().tolist(),
                        near_upper_logsd_minus1_fraction=float((ls>=-1.01).double().mean()),
                        raw_map_derivative_quantiles=torch.quantile(jacobian.flatten(),probabilities).cpu().tolist(),
                        raw_map_derivative_below_point01_fraction=float((jacobian<.01).double().mean()),
                        local_KL_mean_gradient_L2=float(mean_gradient.norm()),
                        local_KL_raw_scale_gradient_L2=float(raw_scale_gradient.norm()),
                        per_view_image=[dict(view=i//2,image=i%2,mean_squared=float(mu[i].square().mean()),
                            mean_KL_per_pixel=float(mean_kl[i]),scale_KL_per_pixel=float(scale_kl[i]),
                            near_upper_fraction=float((ls[i]>=-1.01).double().mean())) for i in range(8)])
                after = file.stat()
                assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
                assert all(x.grad is None for x in model.parameters())
                rows.append(row)
        result = dict(status='COMPLETE', rows=rows, quantile_levels=probabilities.cpu().tolist(),
                      costs=dict(encoder_image_forward=72,posterior_batch_forward=9,optimizer=0,reverse=0,
                                 generator_forward=0),source_files_metadata_unchanged=True,
                      note='Local analytic KL derivatives at actual q outputs, normalized by batch8 and N; not shared-encoder gradients or full A gradients. Infimum subtraction changes no gradient. Finite precision may reach the mathematical open-boundary endpoint.')
        write(output,result)
        return result
    finally:
        restore_rng_state(rng)

