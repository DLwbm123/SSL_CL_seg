"""Create-only engineering closeout; never adjudicate an incomplete scientific matrix."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from experiments.lcrseg.single_teacher_scd_v0_1.report import metrics, table

def read(path):
    return json.loads(path.read_text())

def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]

def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")

def main(base, output):
    output.mkdir(exist_ok=False)
    old, repaired = base/"run_01", base/"entropy_repair_run_01"
    source_old="057ce07fa7bd02f8319e0c2bdd9d4adbceff23d8"
    source_new="270e23985c8d78d0508fe6c4d43150f6ebffcc92"
    assert read(repaired/"E/stage1_train_exit.json")["exit_code"] == 1
    assert "SCD root not bracketed" in read(repaired/"E/stage1/failure.json")["error"]
    complete=[]; rows=[]; budgets=[]; lineage=[]; access=[]; deployments=[]
    common_hash=read(old/"common/stage0/receipt.json")["student_hash"]
    for arm,stage in [("common",0)]+[(a,t) for a in "SABCD" for t in (1,2)]:
        root=old/arm/f"stage{stage}"; receipt=read(root/"receipt.json"); ev=read(root/"val.json")
        assert receipt["status"]=="COMPLETE" and receipt["source"]==source_old
        assert receipt["epochs"]==100 and receipt["counters"]["optimizer_steps"]==(8000,3200,2100)[stage]
        step_rows=lines(root/"steps.jsonl")
        assert [r["step"] for r in step_rows]==list(range(1,receipt["expected_updates"]+1))
        assert ev["status"]=="COMPLETE" and ev["student_hash"]==receipt["student_hash"]
        assert ev["inference_full_models"]==1 and ev["test_accesses"]==0 and ev["future_domain_gt_accesses"]==0
        for op in ("train","eval"):
            assert read(root.parent/f"stage{stage}_{op}_exit.json")["exit_code"]==0
        if stage:
            parent_hash=common_hash if stage==1 else read(root.parent/"stage1/receipt.json")["student_hash"]
            assert receipt["parent_hash"]==parent_hash
            if arm in "CD": assert receipt["teacher_initial_hash"]==parent_hash==receipt["teacher_final_hash"]
        if stage==2:
            deploy=read(root/"deployment.json")
            assert deploy["status"]=="PASS" and deploy["teacher_hidden"] and deploy["equal_synthetic_prediction"]
            assert deploy["student_hash"]==receipt["student_hash"] and deploy["models"]==1 and deploy["gt_accesses"]==0
            assert read(root.parent/"deploy_exit.json")["exit_code"]==0
            deployments.append(dict(arm=arm,**deploy))
        complete.append(receipt); rows.extend(ev["rows"])
        budgets.append(dict(arm=arm,stage=stage,attempt="accepted_complete",status="COMPLETE",expected_updates=receipt["expected_updates"],
            successful_updates=len(step_rows),durable_checkpoint_updates=len(step_rows),completed_epochs=100,source=receipt["source"]))
        lineage.append({k:receipt[k] for k in ("arm","stage","source","parent_hash","student_hash","teacher_initial_hash","teacher_final_hash")})
        access.append(dict(arm=arm,stage=stage,status="COMPLETE",online_full_models=receipt["max_full_models"],
            inference_full_models=1,test_gt=0,hidden_unlabeled_gt=0,past_train_images=0,t_minus_2_weights=0,
            evidence="role-scoped accessors; own-parent check; frozen-teacher epoch checks; isolated evaluator"))
    assert sum(r["successful_updates"] for r in budgets)==34500
    for attempt,root,source,expected_count,checkpoint_count in [
        ("discarded_entropy_failure",old/"E/stage1",source_old,1,0),
        ("bracket_failure",repaired/"E/stage1",source_new,1507,1504)]:
        st=lines(root/"steps.jsonl")
        assert len(st)==expected_count and [r["step"] for r in st]==list(range(1,expected_count+1))
        budgets.append(dict(arm="E",stage=1,attempt=attempt,status="INCOMPLETE",expected_updates=3200,
            successful_updates=len(st),durable_checkpoint_updates=checkpoint_count,completed_epochs=checkpoint_count//32,source=source))
    budgets.append(dict(arm="E",stage=2,attempt="not_started",status="NOT_STARTED",expected_updates=2100,
        successful_updates=0,durable_checkpoint_updates=0,completed_epochs=0,source=source_new))
    common=next(r for r in rows if r["arm"]=="common")
    rows=[r for r in rows if r["arm"]!="common"]+[dict(common,arm=a) for a in "SABCDE"]
    assert len(rows)==31
    scores={a:metrics({(r["stage"],r["domain_index"]):r["macro_fg_dice"] for r in rows if r["arm"]==a}) for a in "SABCD"}
    table(output/"STAGE_DOMAIN_MATRIX.csv",rows)
    table(output/"PER_CLASS_METRICS.csv",[dict(arm=r["arm"],stage=r["stage"],domain=r["domain"],
        background=r["background_dice"],rim=r["rim_dice"],cup=r["cup_dice"]) for r in rows])
    table(output/"METHOD_SUMMARY.csv",[dict(arm=a,status="COMPLETE_DESCRIPTIVE_ONLY",**scores[a]) for a in "SABCD"]+
        [dict(arm="E",status="NOT_AVAILABLE_INCOMPLETE",**{k:"NOT_AVAILABLE" for k in scores["S"]})])
    table(output/"PAIRED_METHOD_COMPARISONS.csv",[
        dict(method="B",reference="S",status="DESCRIPTIVE_ONLY_NO_GATE",**{k:scores["B"][k]-scores["S"][k] for k in scores["B"]}),
        *[dict(method="E",reference=a,status="NOT_AVAILABLE_INCOMPLETE",**{k:"NOT_AVAILABLE" for k in scores["S"]}) for a in "BCD"]])
    table(output/"GATE_ACCOUNTING.csv",[dict(gate=g,status="NOT_RUN_INCOMPLETE_MATRIX",value="NOT_AVAILABLE")
        for g in ("SSL_SUPPORT","MAIN_EFFECT","HISTORY_RETENTION","CURRENT_ADAPTATION","REAL_PROJECTION")])
    pas=[];distill=[];runtime=[];curves=[];eta_distribution=[]
    runs=[("accepted_complete",old/r["arm"]/f"stage{r['stage']}",r["arm"],r["stage"]) for r in complete]+[
        ("discarded_entropy_failure",old/"E/stage1","E",1),("bracket_failure",repaired/"E/stage1","E",1)]
    for attempt,root,arm,stage in runs:
        steps=lines(root/"steps.jsonl"); ep=lines(root/"epochs.jsonl") if (root/"epochs.jsonl").exists() else []
        tag=dict(arm=arm,stage=stage,attempt=attempt)
        if (root/"receipt.json").exists():
            counters=read(root/"receipt.json")["counters"].copy()
            counters["counter_evidence"]="completed stage receipt"
        else:
            payload=torch.load(root/"student_latest.pt",map_location="cpu")
            counters=payload["counters"].copy();del payload
            # Both failures occurred after five forwards and GAS extraction, before backward.
            since_checkpoint=len(steps)-counters["optimizer_steps"]
            counters["forward"]+=5*(since_checkpoint+1)
            counters["gas_autograd"]+=since_checkpoint+1
            counters["backward"]+=since_checkpoint
            counters["optimizer_steps"]+=since_checkpoint
            if attempt=="discarded_entropy_failure":counters["diagnostic_autograd"]=3
            counters["counter_evidence"]="checkpoint counters plus deterministic executed failure path; no failed optimizer step counted"
        runtime.append(dict(**tag,completed_epochs=len(ep),logged_epoch_seconds=sum(x["seconds"] for x in ep),
            logged_step_seconds=sum(x["step_seconds"] for x in steps),
            step_seconds_mean=float(np.mean([x["step_seconds"] for x in steps])),
            step_seconds_p95=float(np.quantile([x["step_seconds"] for x in steps],.95)),
            **{k:max([x[k] for x in ep]) if ep else None for k in ("student_parameters_bytes","student_buffers_bytes",
                "teacher_parameters_bytes","teacher_buffers_bytes","gradient_bytes","optimizer_bytes","prototype_bytes",
                "max_full_models","max_allocated","max_reserved","cpu_max_rss_kib")},
            target_logical_tensor_bytes_max=max([d["target_temporary_bytes"] for x in steps for d in x["distillation"]] or [0]),
            gas_autograd_seconds=sum(x["gas_seconds"] for x in steps),**counters))
        for epoch in ep:
            batch=[x for x in steps if x["epoch"]==epoch["epoch"]]
            curves.append(dict(**tag,epoch=epoch["epoch"],supervised_ce=float(np.mean([x["sup"] for x in batch])),
                ssl=float(np.mean([x["ssl"] for x in batch])),kd=float(np.mean([x["kd"] for x in batch])),seconds=epoch["seconds"]))
        if arm not in ("common","S"):
            for c in range(3):
                vals=[x["pas"][c] for x in steps]; total=sum(x["predicted_pixels"] for x in vals);n=sum(x["accepted_pixels"] for x in vals)
                pas.append(dict(**tag,branch="unlabeled",class_id=c,predicted_pixels=total,accepted_pixels=n,coverage=n/max(total,1),
                    empty_batch_fraction=sum(x["accepted_pixels"]==0 for x in vals)/len(vals),prototype_valid_steps=sum(x["prototype_valid"] for x in vals),
                    pseudo_label_accuracy="NOT_EVALUATED"))
        if arm in "CDE":
            for branch in ("labeled","unlabeled"):
                ds=[next(d for d in x["distillation"] if d["branch"]==branch) for x in steps]
                for c in range(3):
                    vals=[d["classes"][c] for d in ds]; row=dict(**tag,branch=branch,class_id=c)
                    for key in vals[0]:
                        if key!="class_id":row[key]=max(x[key] for x in vals) if key.endswith("_max") else sum(x[key] for x in vals)
                    row["raw_conflict_rate"]=row["conflicts"]/max(row["reliable"],1)
                    row["projection_fraction_of_valid"]=row["projected"]/max(row["pixels"],1)
                    distill.append(row)
                    if arm=="E":
                        means=[x["eta_sum"]/x["projected"] for x in vals if x["projected"]]
                        eta_distribution.append(dict(**tag,branch=branch,class_id=c,projected=row["projected"],
                            eta_pixel_weighted_mean=row["eta_sum"]/max(row["projected"],1),eta_pixel_max=row["eta_max"],
                            **{f"step_mean_eta_p{int(100*p)}":float(np.quantile(means,p)) if means else 0. for p in (.1,.5,.9,.99)},
                            quantile_unit="per-step class mean, NOT pixel eta quantiles"))
    table(output/"SSL_DIAGNOSTICS.csv",pas);table(output/"DISTILLATION_DIAGNOSTICS.csv",distill)
    table(output/"ETA_STEP_DISTRIBUTIONS.csv",eta_distribution);table(output/"MEMORY_AND_RUNTIME.csv",runtime);table(output/"EPOCH_CURVES.csv",curves)
    gradient_rows=[]
    for attempt,root,arm,stage in runs:
        st=lines(root/"steps.jsonl")
        for x in st:
            if x["gradients"]:
                gradient_rows.append(dict(arm=arm,stage=stage,attempt=attempt,step=x["step"],**x["gradients"]))
    table(output/"BRANCH_GRADIENT_DIAGNOSTICS.csv",gradient_rows)
    probe=read(base/"engineering_bracket_probe_01/receipt.json")
    assert probe["status"]=="REPRODUCED" and probe["engineering_replay_updates"]==3
    budget=dict(expected_complete_matrix=39800,completed_stage_updates=34500,latest_partial_E_updates=1507,
        discarded_initial_E_updates=1,all_formal_attempt_optimizer_updates=36008,latest_E_checkpoint_updates=1504,
        uncheckpointed_successful_E_updates=3,remaining_updates_in_latest_trajectory=3793,
        formal_stage_count_complete=11,formal_stage_count_expected=13,
        exact_source_qualification_updates=504,synthetic_resource_probe_updates=1,engineering_diagnostic_replay_updates=4,
        measured_nonformal_updates=509,unmeasured_exploratory_pre_freeze_compute_excluded=True,stages=budgets)
    write(output/"EXPECTED_AND_ACTUAL_BUDGET.json",budget)
    write(output/"TRAINING_COMPLETENESS.json",dict(status="INCOMPLETE_TRAINING_MATRIX",all_six_arms_complete=False,
        complete_arms=list("SABCD"),incomplete_arm="E",complete_stages=11,expected_stages=13,receipts=complete))
    write(output/"STAGE_LINEAGE.json",dict(complete_stages=lineage,partial_E=dict(source=source_new,stage=1,
        parent_hash=common_hash,completed_epochs=47,teacher_hash_checked_at_each_completed_epoch=True,stage2="NOT_STARTED")))
    write(output/"MODEL_STATE_ACCESS_AUDIT.json",dict(complete_stages=access,completed_deployment_checks=deployments,
        partial_E=dict(online_full_models=2,completed_epoch_teacher_checks=47,stage2_deployment="NOT_AVAILABLE"),
        limits="No test, hidden unlabeled GT, prior training replay or old formal_03; partial E has no final deployment claim."))
    write(output/"SOLVER_DIAGNOSTICS.json",dict(status="ENGINEERING_FAILURE",current_failure="SCD root not bracketed",
        frozen_max_bracket=60,frozen_bisection_iterations=48,failure_step=1508,failed_epoch=48,
        real_reproduction=probe,initial_entropy_failure_preserved=True,scientific_adjudication="NOT_RUN",
        no_threshold_relaxation=True,no_fallback=True,
        successful_step_max_residual=max(r["residual_max"] for r in distill),
        successful_E_nondegenerate_events=sum(r["nondegenerate"] for r in distill if r["arm"]=="E"),
        successful_step_target_and_diagnostic_seconds=sum(d["solver_seconds"] for _,root,_,_ in runs for x in lines(root/"steps.jsonl") for d in x["distillation"]),
        memory_note="Target bytes count named logical tensors and may include aliases; CUDA peaks are separately measured. Missing epoch memory in initial failed E is unavailable. Partial E epoch peaks exclude its final uncheckpointed steps.",
        eta_note="Pixel-weighted mean and max plus distribution of per-step class means. Pixel eta quantiles were not retained."))
    terminal=dict(status="INCOMPLETE_TRAINING_MATRIX",all_six_arms_complete=False,scientific_adjudication="NOT_RUN",
        source_sha_main=source_old,source_sha_E_repair=source_new,completed_arms=list("SABCD"),
        formal_successful_updates_all_attempts=36008,seed1_2_started=False,external_data_started=False,test_used=False,
        stopped_after_frozen_solver_failure=True,private_run_root=str(base))
    write(output/"STATUS.json",terminal)
    write(output/"NAS_RECEIPT.json",dict(root=str(base),old_run=str(old),repaired_run=str(repaired),
        partial_checkpoint=str(repaired/"E/stage1/student_latest.pt"),failure_receipt=str(repaired/"E/stage1/failure.json"),
        child_exit_receipt=str(repaired/"E/stage1_train_exit.json"),diagnostic_receipt=str(base/"engineering_bracket_probe_01/receipt.json"),
        private_numeric_vectors_not_published=True,original_reservations_and_parent_outcomes_preserved=True))
    write(base/"ENGINEERING_TERMINAL.json",terminal)
    (output/"NEXT_STAGE_DRAFT.md").write_text("# Not authorized for execution\n\nStop with INCOMPLETE_TRAINING_MATRIX. A continuation would require an explicit solver-contract revision or an independently justified equivalent implementation, new exact-source regression and transparent accounting. No relaxed tolerance, larger search budget, drop fallback, automatic tuning, extra seed or external data run is started.\n")
    report=["# Single-Teacher SCD V0.1: INCOMPLETE_TRAINING_MATRIX","",
        "Shared stage0 and S/A/B/C/D stage1+2 completed. E stage1 failed at attempted update 1508 (epoch 48) with SCD root not bracketed; E stage2 did not start. The complete-matrix scientific gate was not run. This is an engineering terminal, not evidence for or against SCD's incremental value.","",
        "Completed stages contain 34,500 updates. Repaired E recorded 1,507 successful updates (latest durable checkpoint 1,504); the initial entropy-failed E attempt recorded 1. All formal attempts total 36,008, not the requested 39,800 complete matrix. The missing latest-trajectory budget is 3,793. Diagnostics replayed 1 + 3 updates separately; two exact-source qualification suites used 504 updates, and the synthetic resource probe used 1. Pre-freeze exploratory invocations are not claimed as fully measured compute.","",
        "The fixed solver exhausted 60 bracket doublings. Reproduction from the epoch47 checkpoint matched the three successful steps and failed at the next step. Accepting an unbracketed result because its residual is small would violate the explicit bracket requirement; no such fallback or threshold relaxation was applied.","",
        "## Available descriptive metrics","",
        "| Arm | F | H | N | BWT | Forget REFUGE | Forget RIM |","|---|---:|---:|---:|---:|---:|---:|"]
    for a in "SABCD":report.append("| "+a+" | "+" | ".join(f"{v:.9f}" for v in scores[a].values())+" |")
    report+=["| E | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |","",
        f"B minus S descriptive N difference: {scores['B']['N']-scores['S']['N']:.12f}; F difference: {scores['B']['F']-scores['S']['F']:.12f}. No preregistered gate is adjudicated on this incomplete matrix. E versus C/D and current-domain E adaptation are unavailable.","",
        "Current-domain rim/cup and full observed stage-domain trajectories are in PER_CLASS_METRICS.csv and STAGE_DOMAIN_MATRIX.csv. Absent E rows are not filled from its unfinished checkpoint.","",
        "S/B used one complete model; A used student plus current EMA; C/D/E used student plus immediate predecessor. All five completed arms passed independent student-only deployment with the teacher path hidden. E has no completed deployment artifact. MEMORY_AND_RUNTIME.csv contains measured peaks, tensor-state sizes and runtime; partial/failed attempts are separate rows.","",
        "No test, hidden unlabeled GT, replayed past training data, old formal_03, seed1/2 or external dataset was used. Historical locks and terminal results remain intact. These are researcher-exposed seed0 development results, not external confirmation.","",
        f"Common and S/A/B/C/D source: {source_old}. Repaired E source: {source_new}. Private evidence: {base}. Images, labels, case identifiers, weights, private diagnostic vectors and full run logs are excluded from publication.",""]
    (output/"FINAL_REPORT.md").write_text("\n".join(report))
    print(json.dumps(terminal))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--base",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();main(a.base,a.output)
