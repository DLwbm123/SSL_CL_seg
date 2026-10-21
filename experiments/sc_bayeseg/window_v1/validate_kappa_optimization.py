"""Fixed 2x2 CPU solver/coordinate diagnosis; preserves the earlier failed artifact.
No network or real data. Each cell starts from the same distribution exactly once.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from validate_analytic_posterior import scale, inverse_scale, SIGMA0


def solve(tau, obs, coordinate, solver):
    d, b = 1+1/tau**2, 1/tau**2
    mean = obs/(2+tau**2)
    trace = []
    def evaluate(z):
        sc, dc = scale(z[2], False)
        sr, dr = scale(z[3], False)
        k = float(expit(z[4])) if coordinate == 'sigmoid' else z[4]
        dk = k*(1-k) if coordinate == 'sigmoid' else 1.
        c = d-2*k*b+k*k*d
        delta = z[:2]-mean
        val = .5*(d*np.sum(delta**2)+2*b*np.prod(delta)+c*sc**2+d*sr**2
                  -2-np.log(1+2/tau**2)-2*np.log(sc)-2*np.log(sr))
        physical_gradient = sc**2*(d*k-b)
        gradient = np.array([d*delta[0]+b*delta[1],d*delta[1]+b*delta[0],
                             (c*sc-1/sc)*dc,(d*sr-1/sr)*dr,physical_gradient*dk])
        return float(val), gradient, float(k), float(physical_gradient), float(dk)
    def objective(z):
        v,g,_,_,_=evaluate(z)
        return v,g
    def capture(z):
        v,g,k,pk,dk=evaluate(z)
        trace.append(dict(KL=v,kappa=k,kappa_coordinate=float(z[4]),
                          physical_kappa_gradient=pk,coordinate_kappa_gradient=float(g[4]),
                          sigmoid_derivative=dk if coordinate=='sigmoid' else None,
                          gradient_inf_norm=float(np.max(np.abs(g)))))
    z0=np.array([0.,0.,float(inverse_scale(SIGMA0)),float(inverse_scale(SIGMA0)),
                 np.log(.01/.99) if coordinate=='sigmoid' else .01])
    h=1e-5
    numerical=np.array([(objective(z0+h*e)[0]-objective(z0-h*e)[0])/(2*h) for e in np.eye(5)])
    error=float(np.max(abs(numerical-objective(z0)[1])/(1+abs(numerical))))
    assert error<2e-6
    capture(z0)
    bounds=None if coordinate=='sigmoid' else [(None,None)]*4+[(0.,1.)]
    options={'maxiter':2000,'ftol':1e-13}
    if solver=='L-BFGS-B': options.update(gtol=1e-8,maxls=40)
    fit=minimize(objective,z0,jac=True,method=solver,bounds=bounds,options=options,callback=capture)
    capture(fit.x)
    v,g,k,pk,dk=evaluate(fit.x)
    sc=scale(fit.x[2],False)[0];sr=scale(fit.x[3],False)[0]
    q=np.array([[sc**2,-k*sc**2],[-k*sc**2,k*k*sc**2+sr**2]])
    precision=np.array([[d,b],[b,d]])
    delta=fit.x[:2]-mean
    dense=.5*(np.einsum('ij,ji',precision,q)+delta@precision@delta-2
              -np.linalg.slogdet(precision)[1]-np.linalg.slogdet(q)[1])
    assert abs(v-dense)<1e-7
    ko=1/(1+tau**2)
    exact=np.array([mean,mean,float(inverse_scale(np.sqrt((1+tau**2)/(2+tau**2)))),
                    float(inverse_scale(np.sqrt(tau**2/(1+tau**2)))),
                    np.log(ko/(1-ko)) if coordinate=='sigmoid' else ko])
    exact_error=abs(objective(exact)[0]);assert exact_error<1e-9
    # Projected first-order residual distinguishes a valid bound optimum from clipping.
    projected=float(k-np.clip(k-pk,0,1))
    return dict(tau=tau,observation=obs,coordinate=coordinate,solver=solver,
                initial_distribution=dict(means=[0,0],sd=float(SIGMA0),kappa=.01),
                KL=v,passes_original_gate=bool(abs(v)<=1e-6),kappa=k,exact_kappa=ko,
                fitted_sd=[float(sc),float(sr)],optimizer_success=bool(fit.success),message=str(fit.message),
                iterations=int(fit.nit),objective_calls=int(fit.nfev),gradient_calls=int(fit.njev),
                diagnostic_evaluate_calls=len(trace)+13,finite_difference_calls=10,
                gradient_check_error=error,dense_chain_error=float(abs(v-dense)),
                exact_assignment_KL_error=exact_error,physical_kappa_gradient=pk,
                coordinate_kappa_gradient=float(g[4]),projected_kappa_gradient=projected,trace=trace)


def main():
    rows=[solve(t,x,c,s) for c in ['sigmoid','direct_bounded'] for s in ['L-BFGS-B','SLSQP']
          for t in [.01,.1,1.,10.] for x in [-2.,0.,2.]]
    cells=[]
    for c in ['sigmoid','direct_bounded']:
        for s in ['L-BFGS-B','SLSQP']:
            rr=[r for r in rows if r['coordinate']==c and r['solver']==s]
            cells.append(dict(coordinate=c,solver=s,passed=sum(r['passes_original_gate'] for r in rr),
                              total=len(rr),max_KL=max(r['KL'] for r in rr)))
    out=dict(protocol='fixed_2x2_same_distribution_initialization_no_restarts',
             original_gate=1e-6,exact_assignment_tolerance=1e-9,
             local_scale='1e-6+softplus(raw); unchanged',cases=rows,summary=cells,
             actual_cost=dict(minimizations=len(rows),iterations=sum(r['iterations'] for r in rows),
                              optimizer_objective_calls=sum(r['objective_calls'] for r in rows),
                              optimizer_gradient_calls=sum(r['gradient_calls'] for r in rows),
                              diagnostic_evaluate_calls=sum(r['diagnostic_evaluate_calls'] for r in rows),
                              neural_forward=0,neural_backward=0,network_updates=0,real_data_reads=0),
             limitations=['Direct bounded kappa includes endpoints; all target scalar optima are interior',
                          'A constrained analytic variable is not a differentiable amortized neural output',
                          'No claim about Adam or nonlinear G/F optimization',
                          'Previous multi-dimensional expressibility failures remain'],
             software=dict(numpy=np.__version__,scipy=__import__('scipy').__version__))
    path=Path(__file__).parent/'results'/'BAYESEG_G_KAPPA_OPTIMIZATION.json'
    path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(summary=cells,cost=out['actual_cost'])))

if __name__=='__main__': main()
