"""CPU analytic posterior qualification; no torch, images, labels or training.
Run with Python + numpy/scipy; writes synthetic-only JSON next to this script.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

FLOOR = 1e-6
SIGMA0 = np.exp(-4.0)

def inverse_scale(s):
    return np.log(np.expm1(np.asarray(s) - FLOOR))

def scale(raw, old):
    if old:
        t = np.tanh(raw / 3)
        s = np.exp(-4 + 3*t)
        return s, s*(1-t*t)
    return FLOOR + np.logaddexp(0, raw), expit(raw)

def run_case(tau, obs, family, sign=1):
    tau, obs = np.atleast_1d(tau), np.atleast_1d(obs)
    n = len(tau)
    d = 1 + 1/tau**2
    b = sign/tau**2
    m = np.stack((obs/(2+tau**2), sign*obs/(2+tau**2)), axis=1)
    det_precision = 1 + 2/tau**2
    old, structured = family.startswith('old'), family.endswith('structured')
    def objective(z):
        means = z[:2*n].reshape(n, 2)
        sc, dc = scale(z[2*n:3*n], old)
        sr, dr = scale(z[3*n:4*n], old)
        k = expit(z[-1]) if structured else 0.0
        vc, vr = sc**2, sr**2
        c = d - 2*k*b + k*k*d
        delta = means-m
        val = .5*np.sum(d*(delta**2).sum(1)+2*b*delta[:,0]*delta[:,1]
                         +c*vc+d*vr-2-np.log(det_precision)-np.log(vc)-np.log(vr))
        gm = np.stack((d*delta[:,0]+b*delta[:,1], b*delta[:,0]+d*delta[:,1]), axis=1).ravel()
        grad = np.concatenate((gm, (c*sc-1/sc)*dc, (d*sr-1/sr)*dr))
        if structured:
            grad = np.r_[grad, np.sum(vc*(-b+k*d))*k*(1-k)]
        return float(val), grad
    raw0 = 0.0 if old else float(inverse_scale(SIGMA0))
    initial = np.r_[np.zeros(2*n), np.full(2*n, raw0)]
    if structured:
        initial = np.r_[initial, np.log(.01/.99)]
    # Independent finite differences qualify the explicit gradient, not a neural graph.
    eps = 1e-5
    numerical = np.array([(objective(initial+eps*np.eye(len(initial))[j])[0]-
                           objective(initial-eps*np.eye(len(initial))[j])[0])/(2*eps)
                          for j in range(len(initial))])
    grad_error = float(np.max(np.abs(numerical-objective(initial)[1])/(1+np.abs(numerical))))
    assert grad_error < 2e-6, grad_error
    fit = minimize(objective, initial, jac=True, method='L-BFGS-B',
                   options={'maxiter':2000, 'ftol':1e-13, 'gtol':1e-8, 'maxls':40})
    z = fit.x
    sc = scale(z[2*n:3*n],old)[0]
    sr = scale(z[3*n:4*n],old)[0]
    k = float(expit(z[-1])) if structured else 0.0
    expressibility_error = None
    # Scalar sum model family optimum, independent of optimizer; open-bound infimum.
    oracle = None
    if n == 1 and sign == 1:
        ko = float(b[0]/d[0]) if structured else 0.0
        co = d-2*ko*b+ko*ko*d
        vo, ro = 1/co, 1/d
        if old:
            vo, ro = np.clip(vo,np.exp(-14),np.exp(-2)), np.clip(ro,np.exp(-14),np.exp(-2))
        oracle = float(.5*np.sum(co*vo+d*ro-2-np.log(det_precision)-np.log(vo)-np.log(ro)))
        if structured and not old:
            exact_z = np.r_[m.ravel(), inverse_scale(np.sqrt(vo)), inverse_scale(np.sqrt(ro)), np.log(ko/(1-ko))]
            expressibility_error = abs(objective(exact_z)[0])
            assert expressibility_error < 1e-9
        # Report family approximation gap separately from optimization residual.
    # Dense 2x2 KL independently checks the chain-form objective at the result.
    dense = 0.
    for j in range(n):
        q = np.array([[sc[j]**2,-k*sc[j]**2],[-k*sc[j]**2,k*k*sc[j]**2+sr[j]**2]])
        precision = np.array([[d[j],b[j]],[b[j],d[j]]])
        delta = z[:2*n].reshape(n,2)[j]-m[j]
        dense += .5*(np.einsum('ij,ji',precision,q)+np.einsum('i,ij,j',delta,precision,delta)-2
                     -np.linalg.slogdet(precision)[1]-np.linalg.slogdet(q)[1])
    assert abs(dense-fit.fun) < 1e-7
    return dict(family=family,tau=tau.tolist(),observation=obs.tolist(),observation_sign=sign,
                exact_marginal_sd=np.sqrt((1+tau**2)/(2+tau**2)).tolist(),
                exact_conditional_sd=np.sqrt(tau**2/(1+tau**2)).tolist(),
                exact_compensation=(sign/(1+tau**2)).tolist(),
                fitted_means=z[:2*n].reshape(n,2).tolist(),fitted_sd_c=sc.tolist(),
                fitted_conditional_sd_R=sr.tolist(),fitted_kappa=k,KL_q_to_exact=float(fit.fun),
                family_infimum=oracle,exact_assignment_KL_error=expressibility_error,optimization_residual=None if oracle is None else float(fit.fun-oracle),optimizer_success=bool(fit.success),message=str(fit.message),
                iterations=int(fit.nit),objective_evaluations=int(fit.nfev),gradient_evaluations=int(fit.njev),
                gradient_check_relative_max=grad_error,dense_chain_error=float(abs(dense-fit.fun)),
                gradient_check_objective_calls=2*len(initial)+1)

def main():
    rows = [run_case([t],[x],f) for t in [.01,.1,1.,10.] for x in [-2.,0.,2.]
            for f in ['old_independent','old_structured','new_independent','new_structured']]
    # Single kappa cannot simultaneously equal 1/(1+.1^2) and 1/(1+1^2).
    anisotropic = [run_case([.1,1.],[1.,1.],f) for f in ['new_independent','new_structured']]
    # Intentionally wrong injection direction: observed c-r but guide compensates c+r.
    wrong = run_case([1.],[1.],'new_structured',sign=-1)
    # Nonlinear triangular guide B(c)=c^2: anchor is not the marginal R mean.
    nodes, weights = np.polynomial.hermite.hermgauss(20)
    weights = weights/np.sqrt(np.pi)
    mu, sc, mr, sr, k = .3,.8,.2,.4,.6
    c = mu+np.sqrt(2)*sc*nodes
    conditional_mean = mr-k*(c*c-mu*mu)
    measured_mean = float(weights@conditional_mean)
    expected_mean = mr-k*sc**2
    measured_var = float(weights@((conditional_mean-measured_mean)**2)+sr**2)
    expected_var = sr**2+k*k*(2*sc**4+4*mu*mu*sc*sc)
    assert abs(measured_mean-expected_mean)<1e-12 and abs(measured_var-expected_var)<1e-12
    # Finite-difference triangular noise Jacobian, away from any nonsmooth network.
    def transform(e):
        cc=mu+sc*e[0]
        return np.array([cc,mr-k*(cc*cc-mu*mu)+sr*e[1]])
    e=np.array([.7,-.2]); h=1e-5
    jac=np.column_stack([(transform(e+h*v)-transform(e-h*v))/(2*h) for v in np.eye(2)])
    jac_error=abs(np.linalg.det(jac)-sc*sr)
    assert jac_error<1e-9
    # Nonlinear A, fixed orthogonal projection, exact injection compensation identity.
    A=lambda u: np.array([u,u*u,np.sin(u)])
    Pi=np.diag([1.,1.,0.]); delta=A(.8)-A(mu); noise=np.array([.2,-.1,0.])
    direct=A(.8)-k*(Pi@delta)+noise
    identity=A(mu)+(np.eye(3)-k*Pi)@delta+noise
    compensation_error=float(np.max(np.abs(direct-identity)))
    assert compensation_error<1e-12
    allfits=rows+anisotropic+[wrong]
    result=dict(status='EXPRESSIBILITY_PASS_OPTIMIZATION_GATE_FAIL_NO_NEURAL_ADMISSION',
                gates=dict(exact_assignment_KL_tolerance=1e-9,optimization_residual_tolerance=1e-6,
                           expressibility_pass=True,optimization_pass=all(abs(r['optimization_residual'])<=1e-6 for r in rows if r['family']=='new_structured')),
                scalar_cases=rows,anisotropic_shared_kappa=anisotropic,wrong_direction=wrong,
                nonlinear_check=dict(anchor=mr,marginal_mean=measured_mean,variance=measured_var,
                                     mean_error=abs(measured_mean-expected_mean),variance_error=abs(measured_var-expected_var),
                                     jacobian_determinant_error=float(jac_error),compensation_identity_error=compensation_error),
                new_scale=dict(formula='1e-6 + softplus(raw)',initial_sd=float(SIGMA0),
                               initial_raw=float(inverse_scale(SIGMA0)),prior_sd1_raw=float(inverse_scale(1.)),
                               support='(1e-6, infinity); no upper clamp'),
                old_prior_KL_infimum_per_dimension=float(.5*(np.exp(-2)-1+2)),
                actual_cost=dict(analytic_scalar_minimizations=len(allfits),
                                scipy_iterations=sum(r['iterations'] for r in allfits),
                                objective_evaluations=sum(r['objective_evaluations'] for r in allfits),
                                analytic_gradient_evaluations=sum(r['gradient_evaluations'] for r in allfits),
                                finite_difference_objective_calls=sum(r['gradient_check_objective_calls'] for r in allfits),
                                nonlinear_quadrature_nodes=20,neural_forward=0,neural_backward=0,
                                network_optimizer_steps=0,real_data_reads=0,temperature_fits=0),
                software=dict(numpy=np.__version__,scipy=__import__('scipy').__version__),
                limitations=['No nonlinear G/F optimization or data use','Scalar Gaussian success is not general posterior coverage',
                             'Single shared kappa and diagonal conditional covariance remain restrictive',
                             'Scale floor excludes smaller-than-1e-6 SD; no claim of universal support'])
    if result['gates']['optimization_pass']:
        result['status'] = 'SCOPED_ANALYTIC_GATES_PASS_NOT_NEURAL_ADMISSION'
    path=Path(__file__).parent/'results'/'BAYESEG_G_ANALYTIC_POSTERIOR.json'
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'cost':result['actual_cost'],
                      'failed_optimizer_flags':sum(not r['optimizer_success'] for r in allfits),
                      'new_structured_max_KL':max(r['KL_q_to_exact'] for r in rows if r['family']=='new_structured'),
                      'anisotropic_KL':anisotropic[-1]['KL_q_to_exact'],'wrong_direction_KL':wrong['KL_q_to_exact']}))

if __name__=='__main__':
    main()
