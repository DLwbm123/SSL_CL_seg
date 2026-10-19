"""Small convex certificates computed from existing gradients; no network calls."""
import numpy as np
from scipy.optimize import lsq_linear,minimize


def unit(x):
    n=float(np.linalg.norm(x))
    return (x/n if n else x.copy()),n


def row_units(x):
    norms=np.linalg.norm(x,axis=1)
    out=np.zeros_like(x);nonzero=norms>0;out[nonzero]=x[nonzero]/norms[nonzero,None]
    return out,norms


def solve(constraints,objectives):
    C,cn=row_units(constraints);J,jn=row_units(objectives);g,gn=unit(objectives.mean(0));assert gn>0
    result=lsq_linear(C.T,-g,bounds=(0,np.inf),method='bvls',tol=1e-12,max_iter=10000)
    lam=result.x;d=-g-C.T@lam;du,dn=unit(d);cd=C@d
    primal=max(0.,float(cd.max()));comp=float(np.max(np.abs(lam*cd)));identity=abs(float(g@d+dn**2))
    kkt=primal<=1e-7 and comp<=1e-7 and identity<=1e-7 and float(lam.min())>=0
    witness=dn>1e-7 and float((C@du).max())<=1e-7 and float(g@du)<-1e-7
    weak=dict(solver_success=bool(result.success),solver_status=int(result.status),message=result.message,
              direction_norm=dn,multipliers=lam.tolist(),primal_max_violation=primal,complementarity_max=comp,projection_identity_residual=identity,
              kkt_pass=kkt,normalized_constraint_derivatives=(C@du).tolist(),raw_constraint_derivatives=(constraints@du).tolist(),
              normalized_objective_derivatives=(J@du).tolist(),raw_objective_derivatives=(objectives@du).tolist(),mean_objective_normalized_derivative=float(g@du),
              dual_zero_projection_residual=float(np.linalg.norm(g+C.T@lam)),first_order_witness=bool(witness),
              classification='nonzero_weak_constraint_descent' if witness and kkt else 'numerically_zero_projection' if dn<=1e-7 and kkt else 'unconfirmed')
    H=np.concatenate([C,J]);K=H@H.T;n=len(H)
    opt=minimize(lambda w:.5*float(w@K@w),np.full(n,1/n),jac=lambda w:K@w,method='SLSQP',bounds=[(0,1)]*n,
                 constraints=[dict(type='eq',fun=lambda w:float(w.sum()-1),jac=lambda w:np.ones(n))],options=dict(ftol=1e-14,maxiter=10000))
    w=opt.x;z=H.T@w;zu,zn=unit(z);sd=-zu;der=H@sd
    simplex=abs(float(w.sum()-1))<=1e-7 and float(w.min())>=-1e-7
    gap=float(zn**2-np.min(H@z));certgap=max(0.,gap)
    strictw=zn>1e-7 and float(der.max()) < -1e-7 and simplex
    strict=dict(solver_success=bool(opt.success),solver_status=int(opt.status),message=opt.message,weights=w.tolist(),combination_norm=zn,
                simplex_residual=abs(float(w.sum()-1)),weight_min=float(w.min()),convex_hull_optimality_gap=certgap,
                normalized_derivatives=der.tolist(),raw_constraint_derivatives=(constraints@sd).tolist(),raw_objective_derivatives=(objectives@sd).tolist(),
                worst_normalized_derivative=float(der.max()),strict_witness=bool(strictw),
                classification='strict_common_descent' if strictw else 'numerically_zero_combination' if zn<=1e-7 and simplex and certgap<=1e-7 else 'unconfirmed')
    return dict(constraint_norms=cn.tolist(),objective_norms=jn.tolist(),mean_objective_norm=gn,weak=weak,strict=strict,
                constraint_gram=(C@C.T).tolist(),all_normalized_gram=K.tolist()),dict(weak=du,strict=sd)


def selfcheck():
    C=np.array([[1.,0.]]);J=np.array([[1.,1.],[1.,2.]])
    yes,_=solve(C,J);assert yes['weak']['first_order_witness'] and yes['strict']['strict_witness']
    C=np.array([[1.,0.],[-1.,0.]]);J=np.array([[1.,0.],[1.,0.]])
    no,_=solve(C,J);assert no['weak']['classification']=='numerically_zero_projection' and no['strict']['classification']=='numerically_zero_combination'
    return dict(status='PASS',synthetic_only=True,no_autograd=True,feasible_and_conflicting_cases=True)
