"""Read-only null-aware cache/census; all registered rows are retained."""
import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from di_dmpa_gate1.feature_extraction import load_models, _images, seed_after_load, state_groups
from di_dmpa_gate1.sampling import sample_layout
from .binding import H, ModelMutation, NonfiniteFeature, ProtocolError, require, sha256, write_json
from .geometry import validate_directions


def weight_hash(weights):
    return hashlib.sha256(np.ascontiguousarray(weights,dtype='<f8').tobytes()).hexdigest()


def split_support(raw):
    z=np.asarray(raw,dtype=np.float64)
    require(z.ndim==2 and z.shape[1]==16,'wrong raw feature shape')
    if not np.isfinite(z).all():raise NonfiniteFeature('nonfinite registered feature')
    norms=np.linalg.norm(z,axis=1)
    if not np.isfinite(norms).all():raise NonfiniteFeature('nonfinite raw feature norm')
    active=norms>1e-12
    directions=np.zeros_like(z)
    directions[active]=z[active]/norms[active,None]
    validate_directions(directions,active)
    return dict(directions=directions,active_mask=active,raw_norms=norms)


def validate_cache(arrays, expected_count):
    x,active=validate_directions(arrays['directions'],arrays['active_mask'])
    norms=arrays['raw_norms']
    require(x.shape==(expected_count,16) and norms.shape==(expected_count,),'missing cache/null row')
    require(arrays['directions'].dtype==np.float64 and norms.dtype==np.float64,'wrong cache dtype')
    if not np.isfinite(norms).all():raise NonfiniteFeature('nonfinite raw norm cache')
    require((norms>=0).all() and np.array_equal(active,norms>1e-12),'active mask/raw norm mismatch')


class ImmutableModels:
    def __init__(self,models,checkpoint,output,metadata):
        self.models,self.cp,self.output,self.metadata=models,checkpoint,Path(output),metadata
    def __enter__(self):
        self.before={k:state_groups(v) for k,v in self.models.items()}
        self.disk_before=sha256(self.cp['path'])
        require(self.disk_before==self.cp['sha256'],'checkpoint changed before extraction')
        return self
    def __exit__(self,kind,error,tb):
        after={k:state_groups(v) for k,v in self.models.items()};disk=sha256(self.cp['path'])
        same=self.before==after and disk==self.disk_before
        write_json(self.output/'immutability'/(self.cp['checkpoint_id'].replace('/','_')+'.json'),dict(
            metadata=self.metadata,checkpoint_id=self.cp['checkpoint_id'],before=self.before,after=after,
            checkpoint_sha256_before=self.disk_before,checkpoint_sha256_after=disk,bitwise_unchanged=same,
            status='PASS' if same else 'BLOCKED_MODEL_MUTATION',extraction_completed=error is None,
            error=None if error is None else f'{kind.__name__}: {error}',model_optimizer_steps=0,transport_optimizer_steps=0))
        if not same:raise ModelMutation('model or checkpoint changed')
        return False


def extract_unit(model,unit,data_root,context,*,device,batch_size=8):
    require(unit['role'] in ('train_labeled','val') and not model.training,'invalid extraction role/mode')
    seed_after_load(unit['pixel_sampling_seed'])
    values={c:{name:[] for name in ('directions','active_mask','raw_norms')} for c in range(3)}
    census=[]
    nonempty={c:sum(bool(case['classes'][c]['coordinates']) for case in unit['cases']) for c in range(3)}
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=ProtocolError('optimizer construction forbidden')):
        with torch.no_grad():
            for start in range(0,len(unit['cases']),batch_size):
                cases=unit['cases'][start:start+batch_size]
                images=_images(cases,data_root).to(device)
                _,features=model(images,stochastic_classifier=False)
                require(features.shape==(len(cases),16,384,384) and features.dtype==torch.float32,'wrong forward tensor')
                if not torch.isfinite(features).all():raise NonfiniteFeature(f'full-map NaN/Inf: {context}; cases={[c["case_id"] for c in cases]}')
                for bi,case in enumerate(cases):
                    feature=features[bi].detach().cpu().numpy()
                    full_zero=int(np.all(feature==0,axis=0).sum())
                    for c in range(3):
                        coords=np.asarray(case['classes'][c]['coordinates'],dtype=np.int64).reshape(-1,2)
                        arrays=split_support(feature[:,coords[:,0],coords[:,1]].T)
                        for name,array in arrays.items():values[c][name].append(array)
                        active=arrays['active_mask']; norms=arrays['raw_norms']; positive=norms[norms>0]
                        nullcoords=coords[~active].tolist()
                        uid_hash=H([[case['case_id'],c,*yx] for yx in coords.tolist()])
                        require(uid_hash==case['classes'][c]['coordinate_sha256'],'registered coordinate order changed')
                        n=len(coords); null=int((~active).sum()); fraction=null/n if n else 0.
                        census.append(dict(context,case_id=case['case_id'],class_id=c,registered_count=n,
                            active_count=int(active.sum()),null_count=null,null_fraction=fraction,
                            weighted_null_mass=fraction/nonempty[c] if n else 0.,
                            minimum_positive_norm=float(positive.min()) if len(positive) else None,
                            p01_positive_norm=float(np.quantile(positive,.01)) if len(positive) else None,
                            median_positive_norm=float(np.median(positive)) if len(positive) else None,
                            max_positive_norm=float(positive.max()) if len(positive) else None,
                            full_map_exact_zero_count=full_zero,full_map_nonfinite_count=0,registered_nonfinite_count=0,
                            exact_coordinate_hash=uid_hash,null_coordinate_hash=H([[case['case_id'],c,*yx] for yx in nullcoords]),
                            first_null_coordinates=nullcoords[:32],normalization_applied_to_null=False))
    arrays={c:{name:np.concatenate(parts) for name,parts in grouped.items()} for c,grouped in values.items()}
    for c in range(3):validate_cache(arrays[c],sum(len(case['classes'][c]['coordinates']) for case in unit['cases']))
    require(all(p.grad is None for p in model.parameters()),'unexpected gradients')
    return arrays,census


def extract_checkpoint(root,data_root,old,plan,checkpoint,output,metadata,*,device):
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=ProtocolError('optimizer construction forbidden')):
        models,payload=load_models(root,checkpoint,device=device)
        require(payload['config_hash']==old['immutable_baseline']['configs'][checkpoint['baseline']]['resolved_config_sha256'],'config mismatch')
        with ImmutableModels(models,checkpoint,output,metadata):
            for source,model in models.items():
                panel=checkpoint['baseline']+('-EMA' if source=='ema_teacher' else '-student')
                for role in ('train_labeled','val'):
                    unit=next(u for u in plan['units'] if u['seed']==checkpoint['seed'] and u['stage_index']==checkpoint['stage_index'] and u['role']==role)
                    context=dict(panel_id=panel,seed=checkpoint['seed'],stage_index=checkpoint['stage_index'],domain=checkpoint['domain'],
                        role=role,source=source,checkpoint_id=checkpoint['checkpoint_id'],checkpoint_sha256=checkpoint['sha256'],
                        sampling_plan_sha256=metadata['sampling_plan_sha256'],sampling_unit_sha256=H(unit))
                    arrays,cases=extract_unit(model,unit,data_root,context,device=device)
                    caches=[]
                    for c,group in arrays.items():
                        layout=sample_layout(unit,c)
                        cache=dict(class_id=c,registered_count=len(layout['uids']),uid_order_sha256=H(layout['uids']),
                            original_weight_order_sha256=weight_hash(layout['weights']),active_count=int(group['active_mask'].sum()),
                            null_count=int((~group['active_mask']).sum()),arrays={})
                        for name,array in group.items():
                            rel=Path('features')/panel/f'seed{checkpoint["seed"]}_stage{checkpoint["stage_index"]}_{role}_class{c}'/(name+'.npy')
                            path=Path(output)/rel;path.parent.mkdir(parents=True,exist_ok=True)
                            with path.open('xb') as handle:np.save(handle,array,allow_pickle=False)
                            cache['arrays'][name]=dict(path=str(rel),shape=list(array.shape),dtype=str(array.dtype),sha256=sha256(path))
                        caches.append(cache)
                    entry=dict(context,metadata={**metadata,'panel_id':panel},class_caches=caches,case_support=cases,
                        all_finite=True,null_rows_preserved=True,old_raw_cache_reused=False)
                    write_json(Path(output)/'feature_units'/f'{panel}_seed{checkpoint["seed"]}_stage{checkpoint["stage_index"]}_{role}.json',entry)
                    print(f'features complete {panel} seed={checkpoint["seed"]} stage={checkpoint["stage_index"]} {role}',flush=True)
    del models,payload
    if torch.cuda.is_available():torch.cuda.empty_cache()


def load_cache(output,cache):
    arrays={}
    for name,desc in cache['arrays'].items():
        path=Path(output)/desc['path']
        require(sha256(path)==desc['sha256'],'cache byte hash mismatch')
        arrays[name]=np.load(path,mmap_mode='r',allow_pickle=False)
        require(list(arrays[name].shape)==desc['shape'] and str(arrays[name].dtype)==desc['dtype'],'cache descriptor mismatch')
    require(set(arrays)=={'directions','active_mask','raw_norms'},'missing cache array')
    validate_cache(arrays,cache['registered_count'])
    require(int(arrays['active_mask'].sum())==cache['active_count'] and int((~arrays['active_mask']).sum())==cache['null_count'],'hidden null drop')
    return arrays
