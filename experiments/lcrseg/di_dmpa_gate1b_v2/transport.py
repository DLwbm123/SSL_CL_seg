"""The three fixed maps; only W/b can enter the one native Adam optimizer."""
import csv
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from di_dmpa_gate1.recovery import stop_requested
from di_dmpa_gate1_v2.geometry import validate_directions
from .binding import DOMAINS, require, ProtocolError, InvalidTransportOutput, NoDirectionalPairs, IncompleteEvidence
from .pairs import pair_states, support


def finite(values, name):
    if not np.isfinite(values).all():raise InvalidTransportOutput('nonfinite '+name)


def validate_data(data, *, fit=False):
    require(set(data)=={'x','y','source_active','target_active','pair_state','weights','seed','stage_index','domain','role','partition'}, 'unexpected field/image/GT in fit API')
    require(data['role']=='train_unlabeled' and data['stage_index'] in (1,2) and data['domain']==DOMAINS[data['stage_index']], 'historical/non-unlabeled fit data')
    require(data['partition']==('fit' if fit else data['partition']) and data['partition'] in ('fit','holdout'), 'holdout passed to fitting')
    x,sa=validate_directions(data['x'],data['source_active']);y,ta=validate_directions(data['y'],data['target_active'])
    require(x.shape==y.shape and np.array_equal(data['pair_state'],pair_states(sa,ta)), 'unaligned pair support')
    s=support(data['pair_state'],data['weights'])
    if s['mass']['AA']==0:raise NoDirectionalPairs('no AA support')
    return s


def apply_model(x, active, model):
    x,active=validate_directions(x,active);result=np.zeros_like(x)
    require(model['kind'] in ('T0','T1','T2'), 'unknown transport')
    if model['kind']=='T1':
        require(np.asarray(model['R']).shape==(16,16), 'wrong rotation shape');finite(model['R'],'rotation')
    if model['kind']=='T2':
        require(np.asarray(model['W']).shape==(16,16) and np.asarray(model['b']).shape==(16,), 'wrong residual shape')
        finite(model['W'],'W');finite(model['b'],'b')
    if not active.any():return result, None
    rows=x[active]
    if model['kind']=='T0':raw=rows.copy()
    elif model['kind']=='T1':raw=rows@np.asarray(model['R'],dtype=np.float64).T
    else:raw=rows+rows@np.asarray(model['W'],dtype=np.float64).T+np.asarray(model['b'],dtype=np.float64)
    finite(raw,'transport output');norms=np.linalg.norm(raw,axis=1)
    if (norms<=1e-12).any():raise InvalidTransportOutput('active transport output norm<=1e-12')
    result[active]=raw/norms[:,None] if model['kind']=='T2' else raw
    if not np.isfinite(result).all() or (np.abs(np.linalg.norm(result[active],axis=1)-1)>1e-12).any():
        raise InvalidTransportOutput('transport output is nonfinite or not unit norm')
    return result,float(norms.min())


def feature_error(data, model):
    s=validate_data(data);out,minimum=apply_model(data['x'],data['source_active'],model)
    aa=data['pair_state']==0;cross=np.isin(data['pair_state'],(1,2));w=data['weights'];mass=w.sum()
    values=np.zeros(len(w),dtype=np.float64);values[cross]=2
    values[aa]=1-np.clip(np.sum(out[aa]*data['y'][aa],axis=1),-1,1)
    full=float(np.dot(w,values)/mass);conditional=float(np.dot(w[aa],values[aa])/w[aa].sum())
    constant=2*(s['mass']['A_NULL']+s['mass']['NULL_A'])
    require(np.isclose(full,s['mass']['AA']*conditional+constant,atol=1e-12,rtol=1e-10), 'full-support identity failed')
    return dict(s,full_null_aware_support_error=full,AA_conditional_error=conditional,support_constant_term=constant,
                minimum_raw_output_norm=minimum,all_original_rows_used=len(w),finite=True)


def procrustes(data):
    validate_data(data,fit=True);aa=data['pair_state']==0
    matrix=(data['y'][aa]*data['weights'][aa,None]).T@data['x'][aa]
    U,values,Vh=np.linalg.svd(matrix);R=U@Vh
    finite(R,'Procrustes');finite(values,'Procrustes singular values')
    require(np.allclose(R@R.T,np.eye(16),atol=1e-12,rtol=0), 'Procrustes is not orthogonal')
    return dict(kind='T1',R=R.tolist(),svd_singular_values=values.tolist(),determinant=float(np.linalg.det(R)),
                reflection_allowed=True,optimizer_steps=0)


@contextmanager
def only_transport_optimizer(parameters):
    allowed={id(p) for p in parameters};original=torch.optim.Optimizer.__init__
    def guarded(optimizer, params, defaults):
        values=list(params)
        require(len(values)==2 and {id(p) for p in values}==allowed, 'segmentation/non-transport optimizer forbidden')
        return original(optimizer,values,defaults)
    with patch.object(torch.optim.Optimizer,'__init__',guarded):yield


def fit_residual(data, prototypes, *, trace_path=None, stop_dir=None):
    validate_data(data,fit=True);prototypes=np.asarray(prototypes,dtype=np.float64)
    require(prototypes.shape==(6,16), 'six frozen prototype sentinels required')
    validate_directions(prototypes,np.ones(len(prototypes),dtype=bool))
    x=torch.tensor(np.asarray(data['x']),dtype=torch.float64);y=torch.tensor(np.asarray(data['y']),dtype=torch.float64)
    weights=torch.tensor(np.asarray(data['weights']),dtype=torch.float64);weights=weights/weights.sum()
    sa=torch.tensor(np.asarray(data['source_active']),dtype=torch.bool);aa=torch.tensor(np.asarray(data['pair_state'])==0)
    cross=torch.tensor(np.isin(data['pair_state'],(1,2)));constant=2*weights[cross].sum()
    p=torch.tensor(prototypes,dtype=torch.float64)
    W=torch.nn.Parameter(torch.zeros((16,16),dtype=torch.float64));b=torch.nn.Parameter(torch.zeros(16,dtype=torch.float64))
    aa_in_source=aa[sa];xs=x[sa];ys=y[aa];waa=weights[aa];trace=[];stream=None;writer=None;completed=0
    def objective():
        raw=xs+xs@W.T+b;norms=torch.linalg.vector_norm(raw,dim=1)
        probe=p+p@W.T+b;probe_norms=torch.linalg.vector_norm(probe,dim=1)
        if not torch.isfinite(raw).all() or not torch.isfinite(probe).all() or (norms<=1e-12).any() or (probe_norms<=1e-12).any():
            raise InvalidTransportOutput('nonfinite/zero active feature or prototype output')
        directions=raw/norms[:,None]
        distances=1-(directions[aa_in_source]*ys).sum(dim=1).clamp(-1,1)
        weighted=(waa*distances).sum();regularization=1e-4*(W.square().sum()+b.square().sum())
        total=weighted+constant+regularization
        if not torch.isfinite(total):raise InvalidTransportOutput('nonfinite full objective')
        return total,dict(full_objective=float(total.detach()),full_support_error=float((weighted+constant).detach()),
            AA_conditional_term=float((weighted/waa.sum()).detach()),AA_weighted_term=float(weighted.detach()),
            support_constant_term=float(constant),regularization=float(regularization.detach()),W_frobenius_norm=float(W.detach().norm()),
            b_norm=float(b.detach().norm()),minimum_raw_output_norm=float(torch.minimum(norms.min(),probe_norms.min()).detach()),finite=True)
    def record(step, gradient_finite):
        nonlocal writer
        with torch.no_grad():_,values=objective()
        row=dict(step=step,**values,gradient_finite=gradient_finite,gradient_evaluated=step>0)
        trace.append(row)
        if stream is not None:
            if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
            writer.writerow(row);stream.flush()
    try:
        if trace_path is not None:
            Path(trace_path).parent.mkdir(parents=True,exist_ok=True);stream=Path(trace_path).open('x',newline='')
        initial=dict(kind='T2',W=W.detach().tolist(),b=b.detach().tolist())
        initial_output,_=apply_model(data['x'],data['source_active'],initial)
        require(np.allclose(initial_output,data['x'],atol=1e-12,rtol=0), 'T2 step0 differs from T0')
        record(0,True)
        with only_transport_optimizer([W,b]):
            optimizer=torch.optim.Adam([W,b],lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False)
            require({id(p) for g in optimizer.param_groups for p in g['params']}=={id(W),id(b)}, 'optimizer isolation failed')
            for step in range(1,1001):
                if stop_dir is not None and stop_requested(stop_dir):raise IncompleteEvidence('cooperative transport stop')
                optimizer.zero_grad(set_to_none=True);loss,_=objective();loss.backward()
                if not all(v.grad is not None and torch.isfinite(v.grad).all() for v in (W,b)):raise InvalidTransportOutput('nonfinite transport gradient')
                optimizer.step();completed+=1;record(step,True)
        require(x.grad is None and y.grad is None and p.grad is None, 'frozen input received gradient')
        result=dict(kind='T2',W=W.detach().tolist(),b=b.detach().tolist(),optimizer_steps=completed,
            optimizer='Adam',dtype='float64',device='cpu',step0_identity_pass=True,only_W_b_optimized=True,model_optimizer_steps=0)
        require(completed==1000 and len(trace)==1001, 'incomplete optimizer trajectory')
        return result,trace
    except Exception as error:
        error.transport_optimizer_steps_completed=completed
        raise
    finally:
        if stream is not None:stream.close()


def spectrum(model):
    require(model['kind'] in ('T0','T1','T2'), 'unknown spectrum map')
    W=np.zeros((16,16));b=np.zeros(16)
    if model['kind']=='T1':W=np.asarray(model['R'])-np.eye(16)
    elif model['kind']=='T2':W=np.asarray(model['W']);b=np.asarray(model['b'])
    finite(W,'W');finite(b,'b');values=np.linalg.svd(np.eye(16)+W,compute_uv=False);finite(values,'singular values')
    if values[-1]<=0:raise InvalidTransportOutput('singular transport condition number')
    condition=float(values[0]/values[-1]);finite(condition,'condition number')
    return dict(W_frobenius_norm=float(np.linalg.norm(W)),b_norm=float(np.linalg.norm(b)),singular_values=values.tolist(),
                spectral_norm=float(values[0]),condition_number=condition,finite=True)
