"""Independent NKA R1 review: generated CPU tensors/metadata; zero optimizer calls.
Not the repository's seven-group/native-model test suite. Does not use an execution
permit or real checkpoints. Sources are exact blobs fetched with the GitHub connector.
"""
from __future__ import annotations
import ast, copy, hashlib, json, math, platform, tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import Tensor
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
COMMIT = '9915168cc3707fb70afdc2c719c8781cdf76e3fb'
BLOB_IDS = {'alignment.py':'5db0a92c7ef46b2dfc55cf8f8fabd425585e55ef',
            'protocol.py':'461a8705cc89f66926c023f1fa4135d7f17b6dcf'}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

# Exact generator from five_frameworks_v1/recipes.py.
def generator(seed, order, stage, step, stream):
    key = f'SSLCL_FIVE_FRAMEWORKS_V1/{seed}/{order}/{stage}/{step}/{stream}'
    return torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') % (2**63-1))

def _finite(x: Tensor, name: str) -> None:
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} must be finite")

# Exact function from five_frameworks_v1/kernels.py, lines 257--275.
def sliced_wasserstein_equal(a: Tensor, b: Tensor, directions: Tensor) -> Tensor:
    """Exact empirical 1-D W2 average for equal-cardinality sampled sets.

    Caller performs class conditioning and current-stage sampling. No persistent
    queue, no Gaussian shortcut, no uniformity loss. b is stop-gradient.
    """
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] == 0:
        raise ValueError("nonempty equal N-by-D sets required")
    if directions.ndim != 2 or directions.shape[0] != a.shape[1] or directions.shape[1] == 0:
        raise ValueError("direction shape mismatch")
    for name, value in (("a", a), ("b", b), ("directions", directions)):
        _finite(value, name)
    norms = directions.norm(dim=0)
    if bool((norms == 0).any()):
        raise ValueError("zero slicing direction")
    v = (directions / norms).detach().to(a)
    av = (a @ v).sort(dim=0).values
    bv = (b.detach().to(a) @ v).sort(dim=0).values
    return (av - bv).square().mean()

def extract(file, names, namespace):
    b=(ROOT/'sources'/file).read_bytes()
    actual=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
    if actual != BLOB_IDS[file]:
        raise RuntimeError(f'{file}: source blob mismatch')
    tree=ast.parse(b)
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    if {n.name for n in selected}!=set(names):raise RuntimeError('missing extracted function')
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(ROOT/'sources'/file),'exec'),namespace)
    return actual

def main():
    torch.set_num_threads(1)
    checks=[];counterexamples=[];vjp_calls=0
    def check(name,condition,details=None):
        checks.append({'name':name,'pass':bool(condition),'details':details})
        if not condition:raise AssertionError(name)
    ns={'torch':torch,'F':F,'generator':generator,'sliced_wasserstein_equal':sliced_wasserstein_equal,
        'ARMS':('C0','C1','C2','C3')}
    extract('alignment.py',{'rng','coordinate_basis','interior','patches','alignment_loss'},ns)
    provider=SimpleNamespace(seed=163,order=1,stage=2)
    for h,w in [(9,11),(32,32),(384,384)]:
        x=torch.randn(2,16,h,w,requires_grad=True)
        mask=ns['interior'](torch.ones(2,h,w,dtype=torch.bool))
        idx=torch.where(mask.flatten())[0][::max(1,(2*(h-2)*(w-2))//37)][:37]
        a=ns['patches'](x,idx)
        if h<384:
            ref=F.unfold(x,3,padding=1).transpose(1,2).reshape(-1,144)[idx]
        else:
            ref=torch.stack([x[int(k)//(h*w),:,int(k)%(h*w)//w-1:int(k)%(h*w)//w+2,int(k)%w-1:int(k)%w+2].reshape(-1) for k in idx])
        check(f'patch_values_{h}x{w}',torch.equal(a,ref))
        ga,=torch.autograd.grad(a.sum(),x,retain_graph=True);gr,=torch.autograd.grad(ref.sum(),x);vjp_calls+=2
        check(f'patch_gradient_{h}x{w}',torch.equal(ga,gr))
        del x,mask,a,ref,ga,gr
    bad=torch.ones(2,9,11,dtype=torch.bool);bad[:,4,5]=False
    er=ns['interior'](bad)
    check('patch_invalid_eroded',not er[:,3:6,4:7].any() and not er[:,0,:].any() and not er[:,:,-1].any())
    a=torch.randn(2,16,14,14,requires_grad=True);b=torch.randn_like(a,requires_grad=True)
    labels=(torch.arange(14)[None,None,:].expand(2,14,-1)%2+1).long()
    valid=torch.ones_like(labels,dtype=torch.bool)
    v=torch.linalg.qr(torch.randn(144,8,dtype=torch.float64),mode='reduced')[0].requires_grad_()
    state=torch.get_rng_state().clone();positions=[]
    c2=ns['coordinate_basis']('C2',v,provider)
    check('random_basis_orthonormal',torch.allclose(c2.T@c2,torch.eye(8,dtype=c2.dtype),atol=1e-12))
    check('random_basis_repeatable',torch.equal(c2,ns['coordinate_basis']('C2',v,provider)))
    for name,key in [('full',None),('random',c2),('native',v)]:
        loss,st=ns['alignment_loss'](a,b,labels,labels,valid,valid,key,provider,700)
        check('valid_finite_loss_'+name,torch.isfinite(loss) and loss>0 and st['valid_classes']==2)
        check('dimension_scale_'+name,math.isclose(st['raw_scaled_loss'],st['raw_SWD']*st['dimension']/8,rel_tol=1e-6))
        positions.append(st['positions'])
    check('identical_center_samples_across_coordinates',positions[0]==positions[1]==positions[2])
    check('private_random_stream_no_global_rng_change',torch.equal(state,torch.get_rng_state()))
    loss,st=ns['alignment_loss'](a,b,labels,labels,valid,valid,v,provider,700)
    ga,gb,gv=torch.autograd.grad(loss,(a,b,v),allow_unused=True);vjp_calls+=1
    check('student_gradient_nonzero_teacher_key_detached',ga.norm()>0 and gb is None and gv is None)
    loss,st=ns['alignment_loss'](a,b,labels,labels,valid&False,valid,v,provider,700)
    g,=torch.autograd.grad(loss,a);vjp_calls+=1
    check('no_support_connected_zero',loss==0 and not g.any() and st['invalid_step'])

    # Generated aggregate fixture, not the private/public production receipts.
    arms={'C0':dict(coordinate='no_op',D=None,lambda_align=0.),'C1':dict(coordinate='full_patch',D=144,lambda_align=.05),
          'C2':dict(coordinate='fixed_random_key',D=8,lambda_align=.05),'C3':dict(coordinate='native_entry_V',D=8,lambda_align=.05)}
    options={d:{'total_steps':n,'lambda_U':1.,'PAS_confidence':.6,'PAS_cosine':.7,'lr':.001,'parent_lr_multiplier':.5}
             for d,n in [('RIM_ONE_r3',3200),('Drishti_GS',2100)]}
    evidence={'rows':[],'integrity':{}}
    for seed in (163,164):
        for order in (1,2):
            ident=f'P1__B2_PARENT_PAS_KL_C06__S{seed}__O{order}__STAGE1'
            evidence['rows'].append({'node_id':ident,'identity':{'family':'B2_PARENT_PAS_KL','stage':1,'seed':seed,'order':order,'node_id':ident}})
            evidence['integrity'][ident]={'status':'VERIFIED','receipt_sha256':'1'*64,'student_hash':'2'*64,'transform_hash':'3'*64,'file_sha256':'4'*64}
    with tempfile.TemporaryDirectory(prefix='nka-review-metadata-') as tmp:
        p={'copy':copy,'digest':digest,'STUDY':'NATIVE_KEY_ALIGNMENT_V0_1','ARMS':arms,
           'ROOT':Path(tmp),'DOC':Path(tmp),'MANIFEST':'generated','SPLIT':'generated',
           'RUNTIME':'667f178c3b80183fd80809760ff31ec5f9a14e25','RESULTS':'3034b2199aa7d6e54ea67499f391a0dd4ea3d21e',
           'B2':'B2_PARENT_PAS_KL'}
        p['read']=lambda path: copy.deepcopy(evidence) if Path(path).name=='PUBLIC_RESULTS.json' else {'options':{'B2_PARENT_PAS_KL':options}}
        p['write']=lambda path,obj:Path(path).write_text(json.dumps(obj))
        extract('protocol.py',{'freeze','validate_plan','validate_prefix'},p)
        original=p['freeze']()
        check('generated_correct_plan_validates',p['validate_plan'](original) is original)
        def resign(z):
            z['plan_sha256']=digest({k:v for k,v in z.items() if k!='plan_sha256'})
        mutations=[]
        z=copy.deepcopy(original)
        for o in z['options'].values():o['lambda_U']=.25
        for n in z['nodes']:n['options_sha256']=digest(z['options'][n['domain']])
        mutations.append(('changed_B2_lambda_U_1_to_0_25',z))
        z=copy.deepcopy(original)
        for n in z['nodes']:
            n['domain']='RIM_ONE_r3' if n['order']==1 else 'Drishti_GS'
            n['updates']=3200 if n['order']==1 else 2100
            n['options_sha256']=digest(z['options'][n['domain']])
        mutations.append(('swapped_O1_O2_final_domains',z))
        z=copy.deepcopy(original);z['nodes'][0]['updates']+=1;z['nodes'][1]['updates']-=1
        mutations.append(('shifted_one_update_between_arms_preserving_total',z))
        z=copy.deepcopy(original);z['arms']['C3']['coordinate']='fixed_random_key';z['arms']['C3']['lambda_align']=.2
        mutations.append(('changed_C3_coordinate_and_auxiliary_weight',z))
        z=copy.deepcopy(original);n=z['nodes'][0];other=z['prefixes'][2]
        n['prefix_node']=other['node_id'];n['prefix_binding_sha256']=other['binding_sha256']
        mutations.append(('cross_seed_prefix_at_plan_validation_only',z))
        for name,z in mutations:
            resign(z)
            try:p['validate_plan'](z);accepted=True;error=None
            except Exception as e:accepted=False;error=repr(e)
            counterexamples.append({'name':name,'accepted_by_validate_plan':accepted,'error':error,
                'optimizer_calls':0,'production_access':False})
        z=copy.deepcopy(original);z['nodes'][0]['seed']=161;resign(z)
        try:p['validate_plan'](z);reject=False
        except ValueError:reject=True
        check('wrong_seed_set_rejected',reject)
        z=copy.deepcopy(original);z['nodes'][0]['executable']=True;resign(z)
        try:p['validate_plan'](z);reject=False
        except ValueError:reject=True
        check('production_executable_flag_rejected',reject)
        node=copy.deepcopy(original['nodes'][0]);record=copy.deepcopy(original['prefixes'][0])
        receipt={'identity':record['identity'],'status':'SEALED','student_hash':record['student_sha256']}
        record['receipt_sha256']=digest(receipt);record['binding_sha256']=digest({k:v for k,v in record.items() if k!='binding_sha256'})
        node['prefix_binding_sha256']=record['binding_sha256']
        check('dedicated_prefix_validator_accepts_generated_correct_identity',p['validate_prefix'](node,record,receipt)['tensor_acceptance']=='PENDING')
        node['seed']=164
        try:p['validate_prefix'](node,record,receipt);reject=False
        except PermissionError:reject=True
        check('dedicated_prefix_validator_rejects_cross_seed',reject)
    result={'reviewed_commit':COMMIT,'source_blob_verifications':BLOB_IDS,'checks':checks,
        'checks_passed':sum(x['pass'] for x in checks),'checks_total':len(checks),
        'metadata_counterexamples':counterexamples,'finding':'validate_plan tests internal digests but not all canonical frozen semantics',
        'scope':'AST-extracted exact functions with generated tensor/metadata fixtures; not whole repository or native-model suite',
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu'},
        'optimizer_calls':0,'review_gradient_VJP_calls':vjp_calls,'real_data_reads':0,'real_checkpoint_reads':0,'production_permits_created':0,
        'project_CPU_ledger_modified':False}
    (ROOT/'TARGETED_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['checks','source_blob_verifications']},indent=2))

if __name__=='__main__':main()
