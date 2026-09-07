"""One stage per process; current data, one student, at most one second slot."""
import argparse
import contextlib
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import torch
from torch.nn import functional as F

from experiments.lcrseg.di_dmpa_jascl.modeling import build_lcrseg_unet_jascl_model, RepairedMeanTeacher
from experiments.lcrseg.di_dmpa_jascl.data import stable_seed
from experiments.lcrseg.di_dmpa_gate1.binding import sha256
from .data import CurrentData, COUNTS, batches, load_batch, strong
from .objectives import supervised, ssl_ce, pas, case_center, kd, learning_rate, lambda_u

UPSTREAM = "3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53"
UPSTREAM_PATH = "Semi-Supervised_Natural-FoSSIL/inc/deeplab_gaps_meanT"


def write_json(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def append(path, payload):
    with Path(path).open("a") as f:
        f.write(json.dumps(payload, allow_nan=False) + "\n")
        f.flush()


def state_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tensor_bytes(values):
    if isinstance(values, torch.Tensor):
        return values.numel() * values.element_size()
    if isinstance(values, dict):
        return sum(tensor_bytes(v) for v in values.values())
    if isinstance(values, (list,tuple)):
        return sum(tensor_bytes(v) for v in values)
    return 0


def precision():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


@contextlib.contextmanager
def random_stream(device, *key):
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        seed = stable_seed(0,*key)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        yield


def model(reference, device, state=None):
    if subprocess.check_output(["git","-C",str(reference),"rev-parse","HEAD"],text=True).strip() != UPSTREAM:
        raise RuntimeError("official classifier source commit mismatch")
    subprocess.run(["git","-C",str(reference),"diff","--quiet","HEAD"],check=True)
    with random_stream(device, "initialization"):
        if state is None:
            result = build_lcrseg_unet_jascl_model(reference, upstream_path=UPSTREAM_PATH,input_channels=3,num_classes=3)
        else:
            # Meta construction avoids another complete CPU initialization alongside loaded weights.
            with torch.device("meta"):
                result = build_lcrseg_unet_jascl_model(reference, upstream_path=UPSTREAM_PATH,input_channels=3,num_classes=3)
            result.load_state_dict(state, assign=True)
            state.clear()  # Assigned tensors now belong only to the model.
    result.decoder.conv_logit.grad_update.requires_grad_(False)
    precision()  # Upstream import sets global backend flags; enforce the new precision contract.
    return result.to(device)


def second_model(student, arm):
    if arm in "SB":
        return None
    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def audit_live_models(expected):
    from experiments.lcrseg.di_dmpa_jascl.modeling import LCRSegUNet2DJASCL
    observed=sum(type(x) is LCRSegUNet2DJASCL for x in gc.get_objects())
    if observed != expected or observed > 2:
        raise RuntimeError(f"complete live model count {observed}, expected {expected}")
    return observed


def forward(network, x, stochastic, key, counters):
    with random_stream(x.device, *key):
        result = network(x, stochastic_classifier=stochastic)
    counters["forward"] += 1
    return result


@torch.no_grad()
def prototypes(student, dataset, device, counters):
    total = torch.zeros(3,16,device=device)
    counts = torch.zeros(3,device=device)
    for i in range(len(dataset)):
        item = dataset[i]
        _, feature = forward(student,item["image"][None].to(device),False,(dataset.stage,"prototype",i),counters)
        center, support = case_center(feature[0],item["label"].to(device))
        total += center
        counts += support
    return F.normalize(total / counts.clamp_min(1)[:,None],dim=1), counts > 0


def grad_norm(loss, student):
    grads = torch.autograd.grad(loss, tuple(p for p in student.parameters() if p.requires_grad), retain_graph=True, allow_unused=True)
    return sum(float(g.detach().double().square().sum()) for g in grads if g is not None) ** .5


def train_step(student, teacher, optimizer, labeled, unlabeled, proto, supported,
               arm, stage, epoch, step, counters, *, diagnostic=False, lambda_kd=1.):
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    key = (stage,epoch,step)
    z_l, _ = forward(student,labeled["image"],True,(*key,"supervised_head"),counters)
    sup = supervised(z_l,labeled["label"])
    begin = time.perf_counter()
    gas = torch.autograd.grad(sup, student.decoder.conv_logit.mu.weight, retain_graph=True)[0].detach().square()
    counters["gas_autograd"] += 1
    gas_seconds = time.perf_counter()-begin
    ssl, distill = z_l.sum()*0., z_l.sum()*0.
    pas_rows, distill_rows = [], []
    if arm != "S":
        if unlabeled is None:
            raise ValueError("SSL arm requires current unlabeled images")
        x_u, geo = unlabeled["image"], unlabeled["geometry"]
        if arm == "A":
            z_u, feature = forward(student,x_u,True,(*key,"unsupervised_head"),counters)
            with torch.no_grad():
                z_t, ft = forward(teacher,x_u,True,(*key,"ema_head"),counters)
                pseudo, m_s, _ = pas(z_u,feature,proto,supported,geo)
                _, m_t, _ = pas(z_t,ft,proto,supported,geo)
                mask = m_s & m_t
            ssl = ((z_u.softmax(1)-z_t.softmax(1).detach()).square().sum(1)*mask).sum()/mask.sum().clamp_min(1)
        else:
            with torch.no_grad():
                z_w, feature = forward(student,x_u,False,(*key,"weak_head"),counters)
                pseudo, mask, _ = pas(z_w,feature,proto,supported,geo)
            x_strong = strong(x_u,stage,epoch,step)
            z_u, _ = forward(student,x_strong,True,(*key,"unsupervised_head"),counters)
            ssl = ssl_ce(z_u,pseudo,mask,geo)
        for c in range(3):
            j = pseudo == c
            pas_rows.append(dict(class_id=c,predicted_pixels=int((j&geo).sum()),accepted_pixels=int((j&mask).sum()),prototype_valid=bool(supported[c])))
        if arm in "CDE":
            with torch.no_grad():
                q_l = forward(teacher,labeled["image"],False,(*key,"teacher_labeled"),counters)[0].double().softmax(1)
                q_u = forward(teacher,x_u,False,(*key,"teacher_unlabeled"),counters)[0].double().softmax(1)
            valid_l = (labeled["label"] != 255) & labeled["geometry"]
            kl_l, info_l = kd(z_l,q_l,labeled["label"],valid_l,valid_l,arm)
            kl_u, info_u = kd(z_u,q_u,pseudo,mask,geo,arm)
            distill = .5*(kl_l+kl_u)
            distill_rows = [dict(branch="labeled",**info_l),dict(branch="unlabeled",**info_u)]
    gradients = {}
    if diagnostic:
        for name,loss in (("supervised",sup),("ssl",ssl),("kd",distill)):
            gradients[name] = grad_norm(loss,student)
            counters["diagnostic_autograd"] += 1
    loss = sup + lambda_u(epoch)*ssl + lambda_kd*distill
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite training loss")
    loss.backward()
    counters["backward"] += 1
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in student.parameters()):
        raise FloatingPointError("nonfinite network gradient")
    optimizer.step()
    counters["optimizer_steps"] += 1
    with torch.no_grad():
        student.decoder.conv_logit.grad_update.copy_(gas)
        if arm == "A":
            # The frozen helper averages ALL parameters, including GAS, after successful step.
            RepairedMeanTeacher.update_teacher(type("Slots",(),dict(student=student,teacher=teacher))(),.99)
    return dict(loss=float(loss.detach()),sup=float(sup.detach()),ssl=float(ssl.detach()),kd=float(distill.detach()),
        lambda_u=lambda_u(epoch),gradients=gradients,pas=pas_rows,distillation=distill_rows,
        gas_seconds=gas_seconds,step_seconds=time.perf_counter()-start)


def optimizer_for(student):
    return torch.optim.Adam((p for p in student.parameters() if p.requires_grad),lr=.001,
        betas=(.9,.999),eps=1e-8,weight_decay=4e-5)


def validate_parent(payload, arm, stage):
    if stage == 0 or payload["stage"] != stage-1 or (stage==1 and payload["arm"] != "common") or (stage==2 and payload["arm"] != arm):
        raise PermissionError("initialization must be own immediate predecessor; stage0 forbidden in stage2")
    if not payload["complete"]:
        raise ValueError("parent stage incomplete")


def save_checkpoint(path, student, optimizer, proto, supported, **metadata):
    # state_dict references existing tensors; no whole CPU shadow/clone is constructed.
    temp = Path(str(path)+".tmp")
    torch.save(dict(student=student.state_dict(),optimizer=optimizer.state_dict(),
        prototypes=proto,supported=supported,cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state() if next(student.parameters()).is_cuda else None,**metadata),temp)
    os.replace(temp,path)


def run_stage(*, data, reference, output, arm, stage, device, parent=None, resume=False,
              epochs=100, expected=None, shape=(384,384), qualification=False, stop_after_epoch=None):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=resume)
    source = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    if not qualification:
        from .freeze import verify
        source=verify()
    if not qualification and (epochs != 100 or (stage==0 and arm!="common") or (stage>0 and arm not in "SABCDE")):
        raise ValueError("formal matrix is immutable")
    if stop_after_epoch is not None and not qualification:
        raise ValueError("qualification-only interruption fixture")
    effective = "S" if stage==0 else arm
    args = {} if expected is None else dict(expected=expected)
    labeled = CurrentData(data,stage,"train_labeled",shape=shape,**args)
    # Count unlabeled rows from metadata; S never opens unlabeled images.
    unlabeled_view = CurrentData(data,stage,"train_unlabeled",shape=shape,**args)
    counts = (len(labeled),len(unlabeled_view))
    if not qualification and counts != COUNTS[stage]:
        raise ValueError("frozen counts mismatch")
    steps = max((n+1)//2 for n in counts)
    total = epochs*steps
    counters = dict(forward=0,gas_autograd=0,diagnostic_autograd=0,backward=0,optimizer_steps=0)
    checkpoint = output/"student_latest.pt"
    teacher_path = output/"prev_teacher.pt"
    start_epoch, parent_hash = 1, None
    payload = None
    if resume:
        payload = torch.load(checkpoint,map_location="cpu",weights_only=False)
        if payload["source"] != source or payload["arm"] != arm or payload["stage"] != stage:
            raise RuntimeError("resume source/identity mismatch")
        if payload["complete"]:
            return json.loads((output/"receipt.json").read_text())
        start_epoch = payload["epoch"]+1
        counters, parent_hash = payload["counters"], payload["parent_hash"]
    elif parent:
        payload = torch.load(parent,map_location="cpu",weights_only=False)
        validate_parent(payload,arm,stage)
        parent_hash = payload["student_hash"]
    elif stage != 0:
        raise ValueError("missing immediate predecessor")
    student = model(reference,device,None if payload is None else payload.pop("student"))
    if parent and not resume and state_hash(student)!=parent_hash:
        raise RuntimeError("loaded student differs from immediate-parent state receipt")
    optimizer = optimizer_for(student)
    proto, support = torch.zeros(3,16,device=device),torch.zeros(3,dtype=torch.bool,device=device)
    if resume:
        optimizer.load_state_dict(payload["optimizer"])
        proto,support = payload["prototypes"].to(device),payload["supported"].to(device)
        torch.set_rng_state(payload["cpu_rng"])
        if payload["cuda_rng"] is not None:
            torch.cuda.set_rng_state(payload["cuda_rng"])
    del payload
    gc.collect()
    teacher = None
    if effective not in "SB":
        if resume:
            tp = torch.load(teacher_path,map_location="cpu",weights_only=False)
            if effective == "A" and tp["epoch"] != start_epoch-1:
                raise RuntimeError("EMA/student checkpoint generations differ")
            teacher = model(reference,device,tp.pop("state")).eval()
            del tp
            for p in teacher.parameters(): p.requires_grad_(False)
        else:
            teacher = second_model(student,effective)
        if {id(p) for p in teacher.parameters()} & {id(p) for g in optimizer.param_groups for p in g["params"]}:
            raise RuntimeError("teacher entered optimizer")
    teacher_hash = state_hash(teacher) if teacher is not None else None
    observed_models = audit_live_models(1+int(teacher is not None))
    if not resume and teacher is not None:
        torch.save(dict(state=teacher.state_dict(),epoch=0,kind="current_ema" if effective=="A" else "immediate_predecessor"),teacher_path)
    if not resume:
        save_checkpoint(checkpoint,student,optimizer,proto,support,source=source,arm=arm,stage=stage,
            epoch=0,complete=False,counters=counters,parent_hash=parent_hash,student_hash=state_hash(student))
    if stage==2 and not resume and not qualification:
        # Only this fresh experiment's old slots are retired, after its val receipt exists
        # and both new resumable slots are durable. Historical research is never touched.
        parent_path=Path(parent)
        if not (parent_path.parent/"val.json").is_file() or parent_path.parent.parent != output.parent:
            raise RuntimeError("stage boundary needs same-arm evaluation receipt")
        parent_path.unlink()
        old_teacher=parent_path.parent/"prev_teacher.pt"
        if old_teacher.is_file():old_teacher.unlink()
        write_json(output/"boundary.json",dict(parent_student_hash=parent_hash,
            old_slots_retired=True,new_teacher_hash=teacher_hash,t_minus_2_accesses=0))
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    runtime = []
    for epoch in range(start_epoch,epochs+1):
        epoch_start=time.perf_counter()
        if effective!="S" and epoch>=11 and (epoch-11)%5==0:
            proto,support=prototypes(student,labeled,device,counters)
        l_batches=batches(len(labeled),stage,epoch,"labeled_order",steps)
        u_batches=batches(len(unlabeled_view),stage,epoch,"unlabeled_order",steps)
        for j in range(steps):
            k=(epoch-1)*steps+j
            for group in optimizer.param_groups: group["lr"]=learning_rate(k,total)
            lb=load_batch(labeled,l_batches[j],stage,epoch,j,"labeled",device)
            ub=None if effective=="S" else load_batch(unlabeled_view,u_batches[j],stage,epoch,j,"unlabeled",device)
            info=train_step(student,teacher,optimizer,lb,ub,proto,support,effective,stage,epoch,j,counters,diagnostic=k==0)
            append(output/"steps.jsonl",dict(arm=arm,stage=stage,epoch=epoch,step=k+1,lr=optimizer.param_groups[0]["lr"],**info))
        if teacher is not None and effective!="A" and state_hash(teacher)!=teacher_hash:
            raise RuntimeError("frozen predecessor changed")
        current_hash=state_hash(student)
        if teacher is not None and effective=="A":
            temp=output/"prev_teacher.pt.tmp"
            torch.save(dict(state=teacher.state_dict(),epoch=epoch,kind="current_ema"),temp)
            os.replace(temp,teacher_path)
        save_checkpoint(checkpoint,student,optimizer,proto,support,source=source,arm=arm,stage=stage,
            epoch=epoch,complete=epoch==epochs,counters=counters,parent_hash=parent_hash,student_hash=current_hash)
        memory=dict(epoch=epoch,seconds=time.perf_counter()-epoch_start,
            student_parameters_bytes=tensor_bytes(tuple(student.parameters())),student_buffers_bytes=tensor_bytes(tuple(student.buffers())),
            teacher_parameters_bytes=tensor_bytes(tuple(teacher.parameters())) if teacher else 0,
            teacher_buffers_bytes=tensor_bytes(tuple(teacher.buffers())) if teacher else 0,
            gradient_bytes=tensor_bytes(tuple(p.grad for p in student.parameters())),optimizer_bytes=tensor_bytes(optimizer.state_dict()),
            prototype_bytes=tensor_bytes((proto,support)),max_full_models=1+int(teacher is not None),
            max_allocated=torch.cuda.max_memory_allocated() if device.type=="cuda" else 0,
            max_reserved=torch.cuda.max_memory_reserved() if device.type=="cuda" else 0,
            cpu_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        append(output/"epochs.jsonl",memory)
        runtime.append(memory)
        print(json.dumps(dict(arm=arm,stage=stage,epoch=epoch,steps=counters["optimizer_steps"],seconds=memory["seconds"])),flush=True)
        if stop_after_epoch == epoch and epoch < epochs:
            return dict(status="QUALIFICATION_INTERRUPTION",epoch=epoch)
    receipt=dict(status="COMPLETE",arm=arm,stage=stage,epochs=epochs,expected_updates=total,
        counters=counters,source=source,parent_hash=parent_hash,student_hash=state_hash(student),
        checkpoint_sha256=sha256(checkpoint),teacher_initial_hash=teacher_hash,
        teacher_final_hash=state_hash(teacher) if teacher else None,
        max_full_models=observed_models,wall_seconds=time.perf_counter()-start,
        current_dataset_counts=counts,unlabeled_images_opened=0 if effective=="S" else len(unlabeled_view.checked),
        precision="network FP32; KL/project FP64; no AMP/TF32",qualifying_run=qualification)
    if counters["optimizer_steps"]!=total:
        raise RuntimeError("actual update budget mismatch")
    write_json(output/"receipt.json",receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser()
    for name in ("data","reference","output","arm"): parser.add_argument("--"+name,required=True)
    parser.add_argument("--stage",type=int,required=True)
    parser.add_argument("--parent")
    parser.add_argument("--resume",action="store_true")
    args=vars(parser.parse_args())
    args["device"]=torch.device("cuda:0")
    try:
        run_stage(**args)
    except BaseException as error:
        path=Path(args["output"])
        if path.is_dir(): write_json(path/"failure.json",dict(status="INCOMPLETE_TRAINING_MATRIX",error=repr(error)))
        raise


if __name__=="__main__": main()
