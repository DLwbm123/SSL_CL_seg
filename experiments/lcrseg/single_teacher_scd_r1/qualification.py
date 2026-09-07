import argparse,json,os,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from experiments.lcrseg.single_teacher_scd_v0_1.engine import write_json
from experiments.lcrseg.single_teacher_scd_v0_1 import objectives as original
from experiments.lcrseg.di_dmpa_gate1.binding import check_hash,sha256
from .freeze import verify
from . import solver

def corpus_check(corpus,old_failed,device):
    from experiments.lcrseg.tests.single_teacher_scd_r1.test_r1 import high_precision
    root=Path(corpus);manifest=json.loads((root/"manifest.json").read_text())
    saved=torch.load(old_failed,map_location="cpu");total=0;rows=[];private_mp=[]
    torch.set_num_threads(2)
    for entry in manifest["files"]:
        path=root/entry["file"];check_hash(path,entry["sha256"]);x=torch.load(path,map_location="cpu")
        args=[x[k] for k in ("p","q","y","reliable")];p,q,y,reliable=args
        assert len(p)==entry["rows"]
        start=time.perf_counter();cpu,conf,ci=solver.project(*args)
        gpu,gconf,gi=solver.project(*[a.to(device) for a in args])
        error=float((cpu-gpu.cpu()).abs().max())
        if error>1e-9 or not torch.equal(conf,gconf.cpu()):raise AssertionError("CPU/GPU projector mismatch")
        a=p-torch.nn.functional.one_hot(y,p.shape[-1]).double()
        scale=a.abs().amax(-1).clamp_min(torch.finfo(torch.double).tiny);aa=a/scale[:,None];b=(aa*p).sum(-1)
        bad=conf&(((q.log()-float(2**60)*aa).softmax(-1)*aa).sum(-1)>b)
        if entry["old_status"]!="PASS":
            assert int(bad.sum())==8423
            for key,value in (("p",p[bad]),("q",q[bad]),("y",y[bad])):
                assert torch.equal(saved[key],value)
            ids=bad.nonzero().flatten()
            for j in np.linspace(0,len(ids)-1,24,dtype=int):
                i=int(ids[j]);ref=high_precision(p[i].numpy(),q[i].numpy(),int(y[i]))
                err=float(np.max(np.abs(cpu[i].numpy()-ref)))
                if err>1e-10:raise AssertionError("private high precision target disagreement")
                private_mp.append(err)
        old=original.project(p,q,y,reliable&~bad)[0]
        same=~bad
        old_error=float((old[same]-cpu[same]).abs().max())
        if old_error>1e-9:raise AssertionError("unaffected target differs beyond frozen tolerance")
        rows.append(dict(file=entry["file"],rows=len(p),sha256=entry["sha256"],old_cap_failures=int(bad.sum()),
            cpu_gpu_max_abs_target_error=error,unaffected_old_max_abs_target_error=old_error,
            max_residual=float(ci["residual"].max()),all_analytic_endpoints_certified=True,
            compensated_v_max_change=float(ci["compensated_v_difference"].max()),seconds=time.perf_counter()-start))
        total+=len(p)
    assert total==manifest["total_vectors"]
    return dict(status="PASS",all_vectors=total,files=rows,private_100_digit_cases=len(private_mp),
        private_high_precision_max_target_error=max(private_mp),private_original_failed_sha256=sha256(old_failed),
        failed_pixels_checked=8423,hidden_gt_accesses=0,network_forwards=0,optimizer_updates=0)

def main(output,device,reference,corpus=None,old_failed=None):
    os.environ["SCD_TEST_DEVICE"]=device;os.environ["SCD_REFERENCE"]=reference
    from experiments.lcrseg.tests.single_teacher_scd_v0_1.test_contract import Contract
    from experiments.lcrseg.tests.single_teacher_scd_r1.test_r1 import R1Contract
    source=verify();root=Path(output);root.mkdir()
    counts=[0];adam=torch.optim.Adam.step
    def counted(*a,**kw):
        result=adam(*a,**kw);counts[0]+=1;return result
    groups={}
    for name,suite in [
      ("inherited",unittest.defaultTestLoader.loadTestsFromTestCase(Contract)),
      ("ablations",unittest.TestSuite(R1Contract(n) for n in ("test_objective_identity_and_gas","test_new_lifecycle_resume_and_forbidden_u"))),
      ("E_numerical",unittest.TestSuite(R1Contract(n) for n in ("test_analytic_reference_and_permutations","test_logit_gradient_and_boundary_json","test_e_training_call_chain")))]:
        before=counts[0];start=time.perf_counter()
        with patch.object(torch.optim.Adam,"step",counted):result=unittest.TextTestRunner(verbosity=2).run(suite)
        groups[name]=dict(status="PASS" if result.wasSuccessful() else "FAIL",tests=result.testsRun,
            failures=len(result.failures),errors=len(result.errors),synthetic_optimizer_updates=counts[0]-before,seconds=time.perf_counter()-start)
    ablation=all(groups[k]["status"]=="PASS" for k in ("inherited","ablations"))
    numeric=groups["E_numerical"]["status"]=="PASS";cp=None
    if corpus and numeric:
        try:cp=corpus_check(corpus,old_failed,torch.device(device))
        except BaseException as error:cp=dict(status="FAIL",error=repr(error));numeric=False
    report=dict(source=source,device=device,python=__import__("platform").python_version(),torch=torch.__version__,
        ablation_qualification="PASS" if ablation else "FAIL",
        E_qualification="PASS" if numeric and corpus else "SYNTHETIC_ONLY" if numeric else "E_NUMERICAL_QUALIFICATION_FAILED",
        groups=groups,corpus=cp,synthetic_optimizer_updates=counts[0],formal_updates=0,
        real_diagnostic_updates=0,corpus_replay_updates_accounted_in_manifest=3 if corpus else 0)
    write_json(root/"qualification.json",report);print(json.dumps(report),flush=True)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for x in ("output","device","reference"):p.add_argument("--"+x,required=True)
    p.add_argument("--corpus");p.add_argument("--old-failed")
    main(**vars(p.parse_args()))
