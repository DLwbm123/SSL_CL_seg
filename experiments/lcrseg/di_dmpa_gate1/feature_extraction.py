"""Read-only real-UNet feature extraction; no training runner/optimizer reuse."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

from .binding import H, NumericalError, ProtocolError, check_hash, require, safe_asset, sha256, write_json
from .geometry_metrics import normalize


def state_hash(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        require(isinstance(value,torch.Tensor),"non-tensor model state")
        value=value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_groups(model):
    state=model.state_dict()
    return {"complete":state_hash(state),
            "classifier":state_hash({k:v for k,v in state.items() if k.startswith("decoder.conv_logit.")}),
            "GAS":state_hash({k:v for k,v in state.items() if k.endswith("grad_update")}),
            "buffers":state_hash(dict(model.named_buffers()))}


class ImmutabilityGuard:
    """Audit even when extraction raises; failed attempts keep state evidence."""
    def __init__(self,models,checkpoint,output,metadata):
        self.models,self.checkpoint,self.output,self.metadata=models,checkpoint,Path(output),metadata

    def __enter__(self):
        self.before={source:state_groups(model) for source,model in self.models.items()}
        return self

    def __exit__(self,error_type,error,tb):
        after={source:state_groups(model) for source,model in self.models.items()}
        unchanged=self.before==after
        digest=sha256(self.checkpoint['path'])
        audit=dict(checkpoint_id=self.checkpoint['checkpoint_id'],checkpoint_sha256=digest,before=self.before,after=after,
                   bitwise_unchanged=unchanged,model_optimizer_steps=0,transport_optimizer_steps=0,
                   metadata=self.metadata,status='PASS' if unchanged and digest==self.checkpoint['sha256'] else 'BLOCKED_PROTOCOL_OR_LEAKAGE',
                   extraction_completed=error is None,extraction_error=None if error is None else f'{error_type.__name__}: {error}')
        write_json(self.output/'immutability'/(self.checkpoint['checkpoint_id'].replace('/','_')+'.json'),audit)
        require(unchanged and digest==self.checkpoint['sha256'],'model/checkpoint state changed')
        return False


def load_models(lcrseg_root, checkpoint, *, device):
    # Deliberately do not use training checkpoint.load_checkpoint: it requires
    # an optimizer and scheduler and restores mutable training RNG/state.
    from di_dmpa_jascl.modeling import build_lcrseg_unet_jascl_model, assert_complete_classifier_load
    check_hash(checkpoint["path"],checkpoint["sha256"])
    payload=torch.load(checkpoint["path"],map_location="cpu",weights_only=False)
    required={"student","ema_teacher","gas_state","stage_state","config_hash","git_commit","schema_version"}
    require(required.issubset(payload),"incomplete checkpoint schema")
    require(payload["schema_version"]==2,"checkpoint schema version changed")
    require(payload["git_commit"]=="fb55e8022bc379e2515a46214c6fdf45ea818de6","checkpoint training source changed")
    require(payload["stage_state"]["stage_index"]==checkpoint["stage_index"],"checkpoint stage mismatch")
    require(torch.equal(payload["student"]["decoder.conv_logit.grad_update"],payload["gas_state"]["grad_update"]),"checkpoint GAS state mismatch")
    models={}
    for source in ("student","ema_teacher"):
        model=build_lcrseg_unet_jascl_model(Path(lcrseg_root)/"third_party/JASCL_REFERENCE",
                upstream_path="Semi-Supervised_Natural-FoSSIL/inc/deeplab_gaps_meanT", input_channels=3,num_classes=3)
        assert_complete_classifier_load(payload[source],model)
        model.load_state_dict(payload[source],strict=True)
        require(state_hash(model.state_dict())==state_hash(payload[source]),"loaded model differs from checkpoint")
        model.to(device).eval().requires_grad_(False)
        models[source]=model
    return models,payload


def seed_after_load(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)


def audit_checkpoint_contents(prereg,lcrseg_root):
    from di_dmpa_jascl.modeling import build_lcrseg_unet_jascl_model
    template=build_lcrseg_unet_jascl_model(Path(lcrseg_root)/'third_party/JASCL_REFERENCE',
        upstream_path='Semi-Supervised_Natural-FoSSIL/inc/deeplab_gaps_meanT',input_channels=3,num_classes=3)
    expected=template.state_dict()
    audits=[]
    for checkpoint in prereg['immutable_baseline']['checkpoint_inputs']:
        check_hash(checkpoint['path'],checkpoint['sha256'])
        payload=torch.load(checkpoint['path'],map_location='cpu',weights_only=False)
        require(payload.get('schema_version')==2,'checkpoint schema mismatch')
        require(payload.get('git_commit')==prereg['immutable_baseline']['gate0_training_source_commit'],'checkpoint code mismatch')
        require(payload.get('config_hash')==prereg['immutable_baseline']['configs'][checkpoint['baseline']]['resolved_config_sha256'],'checkpoint config mismatch')
        require(payload['stage_state']['stage_index']==checkpoint['stage_index'],'checkpoint stage mismatch')
        student,teacher=payload['student'],payload['ema_teacher']
        require(student.keys()==teacher.keys(),'student/EMA schema differs')
        required={'decoder.conv_logit.mu.weight','decoder.conv_logit.sigma.weight','decoder.conv_logit.grad_update'}
        require(required.issubset(student),'incomplete classifier/GAS')
        require(torch.equal(student['decoder.conv_logit.grad_update'],payload['gas_state']['grad_update']),'GAS mismatch')
        for source in (student,teacher):
            require(source.keys()==expected.keys(),'incomplete UNet checkpoint state')
            require(all(source[k].shape==expected[k].shape for k in expected),'checkpoint tensor shape mismatch')
            require(source['decoder.conv_logit.mu.weight'].shape==(3,16,3,3),'classifier geometry mismatch')
            require(all(torch.isfinite(value).all() for value in source.values()),'nonfinite checkpoint state')
        audits.append(dict(checkpoint_id=checkpoint['checkpoint_id'],student_state_sha256=state_hash(student),
                           ema_state_sha256=state_hash(teacher),classifier_complete=True,GAS_complete=True))
    return audits


def _images(cases,data_root):
    images=[]
    for case in cases:
        path=safe_asset(data_root,case["image_h5_relpath"])
        check_hash(path,case["image_sha256"])
        with h5py.File(path,"r") as handle:
            image=handle["image"][...]
        require(image.shape==(3,384,384),"image geometry changed")
        images.append(np.asarray(image,dtype=np.float32)/255.0)
    return torch.from_numpy(np.stack(images))


def extract_unit(model, unit, data_root, *, device, batch_size=8):
    require(unit["role"] in ("train_labeled","val"),"forbidden extraction role")
    seed_after_load(unit["pixel_sampling_seed"])
    values={c:[] for c in range(3)}
    min_norm=float("inf")
    # Optimizer constructor guard is active, even though no optimizer code is
    # imported/called by this path. This is separate from a zero-step counter.
    with patch.object(torch.optim.Optimizer,"__init__",side_effect=ProtocolError("optimizer construction forbidden")):
        with torch.no_grad():
            for start in range(0,len(unit["cases"]),batch_size):
                cases=unit["cases"][start:start+batch_size]
                images=_images(cases,data_root).to(device)
                _,features=model(images,stochastic_classifier=False)
                require(tuple(features.shape[1:])==(16,384,384),"wrong feature tensor")
                if not torch.isfinite(features).all():
                    raise NumericalError("nonfinite decoder.dec1 feature")
                norms=torch.linalg.vector_norm(features.double(),dim=1)
                current_min=float(norms.min())
                min_norm=min(min_norm,current_min)
                if current_min<=1e-12:
                    raise NumericalError(f"decoder.dec1 feature norm <=1e-12: {current_min}")
                for batch_index,case in enumerate(cases):
                    for c in range(3):
                        coords=case["classes"][c]["coordinates"]
                        if not coords:
                            continue
                        y,x=np.asarray(coords).T
                        selected=features[batch_index,:,y,x].T.detach().cpu().numpy()
                        values[c].append(normalize(selected))
    require(all(p.grad is None for p in model.parameters()),"unexpected model gradient")
    return {c:np.concatenate(values[c]) for c in range(3)},dict(minimum_full_feature_norm=min_norm,
                forward_dtype="float32",cache_dtype="float64",normalization_dtype="float64",
                feature_shape_per_case=[16,384,384],batch_size=batch_size,forward_seed=unit["pixel_sampling_seed"],
                stochastic_classifier=False,amp=False,model_eval=True,no_grad=True,optimizer_construction_guard=True)


def extract_checkpoint(lcrseg_root,data_root,prereg,plan,checkpoint,output,metadata,*,device):
    expected_config=prereg["immutable_baseline"]["configs"][checkpoint["baseline"]]["resolved_config_sha256"]
    models,payload=load_models(lcrseg_root,checkpoint,device=device)
    require(payload["config_hash"]==expected_config,"checkpoint canonical config mismatch")
    with ImmutabilityGuard(models,checkpoint,output,metadata):
        caches=_extract_loaded_models(models,data_root,plan,checkpoint,output,metadata,device=device)
    del models,payload
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return caches


def _extract_loaded_models(models,data_root,plan,checkpoint,output,metadata,*,device):
    caches=[]
    for source,model in models.items():
        panel=checkpoint["baseline"]+("-EMA" if source=="ema_teacher" else "-student")
        for role in ("train_labeled","val"):
            unit=next(u for u in plan["units"] if u["seed"]==checkpoint["seed"] and u["stage_index"]==checkpoint["stage_index"] and u["role"]==role)
            arrays,diagnostics=extract_unit(model,unit,data_root,device=device)
            entry=dict(panel_id=panel,seed=checkpoint["seed"],stage_index=checkpoint["stage_index"],domain=checkpoint["domain"],role=role,
                       checkpoint_sha256=checkpoint["sha256"],source=source,metadata={**metadata,"panel_id":panel},diagnostics=diagnostics,
                       sampling_unit_sha256=H(unit),class_caches=[],case_count=unit["case_count"])
            for c,array in arrays.items():
                count=sum(case["classes"][c]["sampled_pixels"] for case in unit["cases"])
                require(array.shape==(count,16),"feature sample coverage mismatch")
                relative=Path("features")/panel/f"seed{checkpoint['seed']}_stage{checkpoint['stage_index']}_{role}_class{c}.npy"
                path=Path(output)/relative
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open("xb") as handle:
                    np.save(handle,array,allow_pickle=False)
                entry["class_caches"].append(dict(class_id=c,path=str(relative),shape=list(array.shape),dtype=str(array.dtype),sha256=sha256(path)))
            caches.append(entry)
            write_json(Path(output)/"feature_units"/f"{panel}_seed{checkpoint['seed']}_stage{checkpoint['stage_index']}_{role}.json",entry)
            print(f"features complete {panel} seed={checkpoint['seed']} stage={checkpoint['stage_index']} {role}",flush=True)
    return caches
