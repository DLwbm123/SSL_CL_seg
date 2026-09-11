"""Read-only audit of production FP32 D and W0+D; no training arithmetic here."""
import math
import torch
from experiments.lcrseg.lctx_weight_memory_v0_1 import core as c

LIMIT = 1e-5
EPS32 = torch.finfo(torch.float32).eps
EPS64 = torch.finfo(torch.float64).eps
TINY32 = torch.finfo(torch.float32).tiny

def gamma(n):
    return n * EPS64 / (1 - n * EPS64)

def norm(x):
    return float(torch.linalg.vector_norm(x))

def backend():
    return dict(torch=torch.__version__, cuda=torch.version.cuda,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic=torch.are_deterministic_algorithms_enabled(),
                autocast=torch.is_autocast_enabled(), threads=torch.get_num_threads())

def basis_info(p, runtime):
    gram = p.T @ p
    orth = norm(gram - torch.eye(p.shape[1], dtype=p.dtype))
    # ||P||2 <= ||P||F. This bound does not assume exact orthogonality.
    upper = norm(p) * (1 + gamma(2 * p.numel() + 2))
    return dict(spectral_norm_upper=upper, orthogonality_F=orth,
                runtime_matches_reference_cast=runtime is None or torch.equal(runtime.cpu(), p.float()),
                runtime_cast_error_F=None if runtime is None else norm(runtime.cpu().double() - p),
                runtime_orthogonality_F=None if runtime is None else norm(runtime.cpu().double().T @ runtime.cpu().double() - torch.eye(p.shape[1], dtype=p.dtype)),
                projector_difference_upper=None if runtime is None else
                norm(runtime.cpu().double() - p) * (norm(runtime.cpu().double()) + norm(p)))

def product_audit(x, p, left, subtraction_bound):
    a, b = (p.T, x) if left else (x, p)
    product = a @ b
    dot_bound = gamma(a.shape[1]) * (a.abs() @ b.abs())
    basis_upper = basis_info(p, None)['spectral_norm_upper']
    beta = norm(dot_bound) + basis_upper * norm(subtraction_bound)
    beta += gamma(2 * product.numel() + 2) * norm(product)
    return norm(product), beta

@torch.no_grad()
def layer_audit(layer, tensors, arm, name, epoch):
    adapter = isinstance(layer, c.LowRankConv)
    actual_w0 = layer.weight.detach()
    d32 = layer.delta().detach() if adapter else None
    w32 = layer.effective().detach() if adapter else actual_w0
    finite = bool(torch.isfinite(w32).all() and torch.isfinite(actual_w0).all())
    finite &= d32 is None or bool(torch.isfinite(d32).all())
    finite &= all(bool(torch.isfinite(v).all()) for v in tensors.values())
    dtype_ok = actual_w0.dtype == w32.dtype == torch.float32 and (d32 is None or d32.dtype == torch.float32)
    row = dict(epoch=epoch, layer=name, arm=arm, device=str(w32.device),
               dtype=str(w32.dtype), backend=backend(), finite=finite, dtype_pass=dtype_ok,
               source_hash_pass=not adapter or torch.equal(actual_w0.cpu(), tensors['W0']),
               reference_hash_pass=True, failures=[])
    if not finite or not dtype_ok:
        row.update(status='FAIL', failures=['finite_or_dtype'], passed=False)
        return row
    w0 = tensors['W0'].double().flatten(1)
    w = w32.cpu().double().flatten(1)
    delta = w - w0
    dn = norm(delta); wn = norm(w0)
    right = tensors['Q' if arm == 'LR_RAND' else 'V']
    u = tensors['U']
    row.update(delta_add_norm=dn, source_norm=wn, relative_update_norm=dn/(wn+1e-12),
               right_leak=norm(delta @ right)/(dn+1e-12), left_leak=norm(u.T @ delta)/(dn+1e-12),
               source_right_leak=norm(delta @ tensors['V'])/(dn+1e-12))
    row['right_basis'] = basis_info(right, layer.right if adapter else None)
    row['left_basis'] = basis_info(u, layer.left if adapter else None)
    row['basis_pass'] = row['right_basis']['runtime_matches_reference_cast'] and row['left_basis']['runtime_matches_reference_cast']
    if adapter:
        row['basis_pass'] &= (layer.right is not None)==(arm!='LR_FREE') and (layer.left is not None)==(arm=='LR_SRC_AB')
    if not adapter:
        row.update(D_norm=None, nonzero_update=dn>0, parameterized_right_leak=None,
                   parameterized_left_leak=None, parameter_gate='N/A', addition_gate='N/A', envelope_gate='N/A',
                   status='NOT_APPLICABLE', passed=True)
        return row
    d = d32.cpu().double(); n = norm(d); N = n+1e-12
    e = delta-d
    b = EPS32*(w0.abs()+d.abs()) + TINY32 + 8*EPS64*(w.abs()+w0.abs()+d.abs())
    eval_element = 8*EPS64*(delta.abs()+d.abs()+b)
    subtraction_bound = EPS64*(w.abs()+w0.abs())
    beta_add = norm(b)
    beta_norm = gamma(2*b.numel()+2)*beta_add + norm(eval_element)
    right_abs, right_eval = product_audit(delta, right, False, subtraction_bound)
    left_abs, left_eval = product_audit(delta, u, True, subtraction_bound)
    right_eval += row['right_basis']['spectral_norm_upper']*beta_norm
    left_eval += row['left_basis']['spectral_norm_upper']*beta_norm
    pr, pl = norm(d@right)/N, norm(u.T@d)/N
    right_limit = LIMIT*N + row['right_basis']['spectral_norm_upper']*beta_add + right_eval
    left_limit = LIMIT*N + row['left_basis']['spectral_norm_upper']*beta_add + left_eval
    addition = bool((e.abs() <= b+eval_element).all())
    right_required, left_required = arm!='LR_FREE', arm=='LR_SRC_AB'
    failures = row['failures']
    for ok, label in [(row['source_hash_pass'],'source_hash'), (row['basis_pass'],'basis_identity'),
                      (addition,'addition_budget'), (not right_required or pr<=LIMIT,'parameter_right'),
                      (not left_required or pl<=LIMIT,'parameter_left'),
                      (not right_required or right_abs<=right_limit,'effective_right'),
                      (not left_required or left_abs<=left_limit,'effective_left')]:
        if not ok: failures.append(label)
    legacy = (right_required and row['right_leak']>LIMIT) or (left_required and row['left_leak']>LIMIT)
    row.update(D_norm=n, nonzero_update=n>0, D_relative_to_source=n/(wn+1e-12),
               E_add_norm=norm(e), E_relative_to_D=norm(e)/N, E_relative_to_source=norm(e)/(wn+1e-12),
               quantized_away_fraction=float(((d!=0)&(delta==0)).sum())/max(1,int((d!=0).sum())),
               parameterized_right_leak=pr, parameterized_left_leak=pl,
               parameter_right_gate=('PASS' if pr<=LIMIT else 'FAIL') if right_required else 'N/A',
               parameter_left_gate=('PASS' if pl<=LIMIT else 'FAIL') if left_required else 'N/A',
               addition_gate='PASS' if addition else 'FAIL', beta_add=beta_add,
               addition_max_abs_error=float(e.abs().max()), addition_max_element_budget=float(b.max()),
               addition_min_element_margin=float((b+eval_element-e.abs()).min()),
               addition_eval64_slack_max=float(eval_element.max()),
               beta_eval64_R=right_eval, beta_eval64_L=left_eval,
               effective_right_absolute=right_abs, effective_left_absolute=left_abs,
               effective_right_limit=right_limit if right_required else None,
               effective_left_limit=left_limit if left_required else None,
               effective_right_gate=('PASS' if right_abs<=right_limit else 'FAIL') if right_required else 'N/A',
               effective_left_gate=('PASS' if left_abs<=left_limit else 'FAIL') if left_required else 'N/A',
               passed=not failures, status='FAIL' if failures else 'ZERO_UPDATE' if n==0 else
               'ROUNDOFF_LIMITED_REPRESENTATION' if legacy else 'PASS')
    return row

@torch.no_grad()
def collect(model, load_basis, epoch, arm):
    rows=[]
    for name in c.LAYERS:
        try:
            rows.append(layer_audit(model.get_submodule(name[:-7]), load_basis(name), arm, name, epoch))
        except Exception as exc:
            rows.append(dict(epoch=epoch, layer=name, arm=arm, status='FAIL', passed=False,
                             reference_hash_pass=False, failures=['diagnostic_exception'], error=repr(exc)))
    return rows

def require(rows):
    if len(rows)!=14 or not all(r['passed'] for r in rows):
        raise FloatingPointError('layer audit failed; all available layer evidence retained')
    return rows
