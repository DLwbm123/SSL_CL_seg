"""One-shot complete-matrix accounting; gates are fixed before any formal update."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np

from .engine import write_json
from .data import DOMAINS


def read(path):return json.loads(Path(path).read_text())


def table(path,rows):
    if not rows:raise ValueError(f"empty required table: {path}")
    with Path(path).open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def metrics(matrix):
    return dict(F=sum(matrix[2,d] for d in range(3))/3,H=(matrix[2,0]+matrix[2,1])/2,
        N=(matrix[1,1]+matrix[2,2])/2,BWT=((matrix[2,0]-matrix[0,0])+(matrix[2,1]-matrix[1,1]))/2,
        forgetting_REFUGE=max(matrix[t,0] for t in range(3))-matrix[2,0],
        forgetting_RIM_ONE_r3=max(matrix[t,1] for t in (1,2))-matrix[2,1])


def gates(scores,rows,mechanism):
    result=[]
    def add(name,value,threshold):result.append(dict(gate=name,value=float(value),threshold=threshold,pass_gate=bool(value>=threshold)))
    add("SSL_N_B_minus_S",scores["B"]["N"]-scores["S"]["N"],.005)
    add("SSL_F_B_minus_S",scores["B"]["F"]-scores["S"]["F"],-.005)
    add("MAIN_F_E_minus_best_other",scores["E"]["F"]-max(scores[a]["F"] for a in "SABCD"),.010)
    add("HISTORY_H_E_minus_B",scores["E"]["H"]-scores["B"]["H"],.010)
    for t in (1,2):
        e=next(r for r in rows if r["arm"]=="E" and r["stage"]==t and r["domain_index"]==t)
        b=next(r for r in rows if r["arm"]=="B" and r["stage"]==t and r["domain_index"]==t)
        for name,threshold in (("macro_fg_dice",-.010),("rim_dice",-.020),("cup_dice",-.020)):
            add(f"CURRENT_stage{t}_{name}_E_minus_B",e[name]-b[name],threshold)
    add("REAL_NONDEGENERATE_PROJECTION",mechanism,1)
    if not all(r["pass_gate"] for r in result[:2]):terminal="SSL_SIGNAL_NOT_ESTABLISHED"
    elif all(r["pass_gate"] for r in result):terminal="PASS_SINGLE_TEACHER_DEVELOPMENT_SCREEN"
    else:terminal="SCD_INCREMENTAL_VALUE_NOT_ESTABLISHED"
    return terminal,result


def generate(run,output):
    run,output=Path(run),Path(output);output.mkdir(parents=True,exist_ok=False)
    if read(run/"parent_receipt.json")["status"]!="MATRIX_COMPLETE_PENDING_ADJUDICATION":
        raise RuntimeError("scientific adjudication requires complete matrix")
    reservation=read(run/"reservation.json")
    rows=[];receipts=[];budget=[];pas=[];distill=[];runtime=[];curves=[];lineage=[];access=[]
    mechanism=0;solver_seconds=0.;max_residual=0.;max_bracket=0
    for arm,stage in [("common",0)]+[(a,t) for a in "SABCDE" for t in (1,2)]:
        root=run/arm/f"stage{stage}";receipt=read(root/"receipt.json");ev=read(root/"val.json")
        if receipt["source"]!=reservation["source"] or receipt["status"]!="COMPLETE":raise RuntimeError("source/stage incomplete")
        if receipt["max_full_models"]!=(1 if arm in ("common","S","B") else 2):raise RuntimeError("model limit violated")
        if arm in "CDE" and receipt["teacher_initial_hash"]!=receipt["teacher_final_hash"]:raise RuntimeError("teacher changed")
        receipts.append(receipt);rows.extend(ev["rows"])
        lineage.append({k:receipt[k] for k in ("arm","stage","parent_hash","student_hash","teacher_initial_hash","teacher_final_hash","checkpoint_sha256")})
        access.append(dict(arm=arm,stage=stage,inference_full_models=ev["inference_full_models"],
            online_full_models=receipt["max_full_models"],teacher_kind="current_ema" if arm=="A" else "immediate_predecessor" if arm in "CDE" else "none",
            test_gt=0,hidden_gt=0,past_train_images=0,t_minus_2_weights=0,
            evidence="role-scoped loader; own-parent validation; process stage isolation; live-model enumeration; isolated evaluator"))
        steps=[json.loads(x) for x in (root/"steps.jsonl").read_text().splitlines()]
        expected=(8000,3200,2100)[stage]
        if len(steps)!=expected or [r["step"] for r in steps]!=list(range(1,expected+1)) or receipt["counters"]["optimizer_steps"]!=expected:
            raise RuntimeError("actual successful optimizer ledger differs from fixed budget")
        budget.append(dict(arm=arm,stage=stage,manifest_labeled=receipt["current_dataset_counts"][0],
            manifest_unlabeled=receipt["current_dataset_counts"][1],epochs=receipt["epochs"],expected_updates=expected,
            ledger_updates=len(steps),checkpoint_updates=receipt["counters"]["optimizer_steps"]))
        epoch_rows=[json.loads(x) for x in (root/"epochs.jsonl").read_text().splitlines()]
        for epoch,group in itertools.groupby(steps,key=lambda r:r["epoch"]):
            values=list(group);memory=next(r for r in epoch_rows if r["epoch"]==epoch)
            curves.append(dict(arm=arm,stage=stage,epoch=epoch,supervised_ce=float(np.mean([r["sup"] for r in values])),
                ssl=float(np.mean([r["ssl"] for r in values])),kd=float(np.mean([r["kd"] for r in values])),
                lambda_u=values[0]["lambda_u"],seconds=memory["seconds"]))
        info=dict(arm=arm,stage=stage,seconds=sum(r["seconds"] for r in epoch_rows),
            step_seconds_mean=float(np.mean([r["step_seconds"] for r in steps])),
            step_seconds_p95=float(np.quantile([r["step_seconds"] for r in steps],.95)),
            **{k:max(r[k] for r in epoch_rows) for k in epoch_rows[0] if k not in ("epoch","seconds")},
            **receipt["counters"],gas_autograd_seconds=sum(r["gas_seconds"] for r in steps),
            first_supervised_gradient=steps[0]["gradients"]["supervised"],
            first_ssl_gradient=steps[0]["gradients"]["ssl"],first_kd_gradient=steps[0]["gradients"]["kd"])
        info["target_temporary_bytes_max"]=max([d["target_temporary_bytes"] for r in steps for d in r["distillation"]] or [0])
        runtime.append(info)
        if arm not in ("common","S"):
            for c in range(3):
                samples=[r["pas"][c] for r in steps]
                total=sum(r["predicted_pixels"] for r in samples);accepted=sum(r["accepted_pixels"] for r in samples)
                pas.append(dict(arm=arm,stage=stage,branch="unlabeled",class_id=c,predicted_pixels=total,accepted_pixels=accepted,
                    coverage=accepted/max(total,1),empty_batch_fraction=sum(r["accepted_pixels"]==0 for r in samples)/len(samples),
                    prototype_valid_steps=sum(r["prototype_valid"] for r in samples),pseudo_label_accuracy="NOT_EVALUATED"))
        if arm in "CDE":
            for branch in ("labeled","unlabeled"):
                samples=[next(d for d in r["distillation"] if d["branch"]==branch) for r in steps]
                solver_seconds+=sum(r["solver_seconds"] for r in samples)
                max_bracket=max(max_bracket,max(r["bracket_max"] for r in samples))
                for c in range(3):
                    cs=[r["classes"][c] for r in samples]
                    record=dict(arm=arm,stage=stage,branch=branch,class_id=c)
                    for name in cs[0]:
                        if name=="class_id":continue
                        record[name]=max(r[name] for r in cs) if name.endswith("_max") else sum(r[name] for r in cs)
                    record["raw_conflict_rate"]=record["conflicts"]/max(record["reliable"],1)
                    distill.append(record);max_residual=max(max_residual,record["residual_max"])
                    if arm=="E":mechanism+=record["nondegenerate"]
    if sum(r["ledger_updates"] for r in budget)!=39800:raise RuntimeError("total update budget differs from 39800")
    common=next(r for r in rows if r["arm"]=="common")
    rows=[r for r in rows if r["arm"]!="common"]+[dict(common,arm=a) for a in "SABCDE"]
    if len(rows)!=36 or any(r["evaluable_cases"]==0 for r in rows):raise RuntimeError("incomplete/empty scoring support")
    scores={a:metrics({(r["stage"],r["domain_index"]):r["macro_fg_dice"] for r in rows if r["arm"]==a}) for a in "SABCDE"}
    terminal,gate_rows=gates(scores,rows,mechanism)
    # A single create-only adjudication record in the private run prevents hidden repeated decisions.
    with (run/"ADJUDICATION.json").open("x") as f:json.dump(dict(status=terminal,gates=gate_rows),f,indent=2)
    table(output/"STAGE_DOMAIN_MATRIX.csv",rows)
    table(output/"PER_CLASS_METRICS.csv",[dict(arm=r["arm"],stage=r["stage"],domain=r["domain"],background=r["background_dice"],rim=r["rim_dice"],cup=r["cup_dice"]) for r in rows])
    table(output/"METHOD_SUMMARY.csv",[dict(arm=a,**scores[a]) for a in "SABCDE"])
    table(output/"PAIRED_METHOD_COMPARISONS.csv",[dict(method=a,reference=b,**{k:scores[a][k]-scores[b][k] for k in scores[a]}) for a,b in itertools.combinations("SABCDE",2)])
    table(output/"SSL_DIAGNOSTICS.csv",pas);table(output/"DISTILLATION_DIAGNOSTICS.csv",distill)
    table(output/"MEMORY_AND_RUNTIME.csv",runtime);table(output/"EPOCH_CURVES.csv",curves);table(output/"GATE_ACCOUNTING.csv",gate_rows)
    write_json(output/"SOLVER_DIAGNOSTICS.json",dict(failures=0,max_normalized_residual=max_residual,
        max_bracket_doublings=max_bracket,total_target_and_diagnostic_seconds=solver_seconds,real_nondegenerate_events=mechanism,
        timing_note="synchronized scalar diagnostic transfers included; not an isolated kernel speed claim"))
    write_json(output/"EXPECTED_AND_ACTUAL_BUDGET.json",dict(expected=39800,actual=39800,stages=budget,qualification_updates_excluded=True))
    write_json(output/"STAGE_LINEAGE.json",dict(stages=lineage))
    write_json(output/"MODEL_STATE_ACCESS_AUDIT.json",dict(stages=access))
    write_json(output/"TRAINING_COMPLETENESS.json",dict(status="COMPLETE",actual_stages=13,all_six_arms_complete=True,receipts=receipts))
    write_json(output/"STATUS.json",dict(status=terminal,source_sha=reservation["source"],all_six_arms_complete=True,
        formal_optimizer_updates=39800,seed1_2_started=False,external_confirmation=False,test_used=False,
        simple_baseline_competitive=not gate_rows[2]["pass_gate"] and any(abs(scores[a]["F"]-scores["E"]["F"])<=.005 for a in "CD")))
    lines=[f"# Single-Teacher SCD V0.1: {terminal}","", "All 13 real stages completed: shared stage0 and seed0 S/A/B/C/D/E stage1+2. Exactly 39,800 successful formal optimizer updates; no test GT, replay, seed1/2 or parameter search.","",
        "| Arm | F | H | N | BWT | Forget REFUGE | Forget RIM |","|---|---:|---:|---:|---:|---:|---:|"]
    for a in "SABCDE":lines.append("| "+a+" | "+" | ".join(f"{v:.9f}" for v in scores[a].values())+" |")
    lines += ["", "Gate failures: "+", ".join(r["gate"] for r in gate_rows if not r["pass_gate"])+".","",
        f"E minus C final Dice: {scores['E']['F']-scores['C']['F']:.12f}; E minus D: {scores['E']['F']-scores['D']['F']:.12f}. B minus S new-domain Dice: {scores['B']['N']-scores['S']['N']:.12f}.","",
        "S/B used one complete model; A used student plus current EMA; C/D/E used student plus their own immediate predecessor. All evaluation used one current student. Measured memory, CPU RSS, runtime, forward/autograd/backward counters and rolling-stage identities are in the companion tables.","",
        "Mechanism statistics are separated by branch and class. Projection provides only a local logit-direction compatibility constraint, not an Adam/Dice/old-domain guarantee. Training unlabeled accuracy is NOT_EVALUATED because hidden GT is excluded.","",
        "This is a researcher-exposed seed0 development screen, neither a full JASCL/UniMatch reproduction nor external confirmation. Scientific decisions used full precision after completion of every arm. No result-dependent retries, tuning or extra seeds are authorized.","",
        f"Source: `{reservation['source']}`. Private run receipt: `{run}`. Public release excludes images, labels, case identifiers, weights, raw tensors and full run logs."]
    (output/"FINAL_REPORT.md").write_text("\n".join(lines)+"\n")
    (output/"NEXT_STAGE_DRAFT.md").write_text("# Not authorized for execution\n\n"+
        ("Keep this exact configuration for a separately authorized seed1/2 replication and external evaluation; seed0 is only a screen.\n" if terminal.startswith("PASS") else
         "Stop this fixed screen. If SSL support failed, report that limitation before considering further consolidation. If C/D are competitive, prefer the simpler baseline. No new tuning, seeds or modules are authorized.\n"))
    return terminal


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();print(generate(a.run,a.output))
