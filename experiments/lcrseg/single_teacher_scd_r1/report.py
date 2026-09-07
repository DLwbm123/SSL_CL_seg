"""R1-only closeout. Old evidence stays read-only; four questions have separate outcomes."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from experiments.lcrseg.single_teacher_scd_v0_1.report import metrics
from experiments.lcrseg.single_teacher_scd_v0_1.engine import write_json

def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
def table(p,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator="\n");w.writeheader();w.writerows(rows)

def generate(base,output):
    base=Path(base);output=Path(output);output.mkdir(exist_ok=False)
    reuse=read(base/"INPUT_REUSE.json");q=read(base/"qualification_server/qualification.json")
    old=Path("experiments/lcrseg/docs/single_teacher_scd_v0_1")
    oldrows=list(csv.DictReader((old/"STAGE_DOMAIN_MATRIX.csv").open()))
    rows=[]
    for x in oldrows:
        if x["arm"]=="E":continue
        row={k:(int(v) if k in ("stage","domain_index","cases","evaluable_cases") else
             float(v) if k in ("macro_fg_dice","background_dice","rim_dice","cup_dice","mean_iou","evaluable_macro_fg_dice") else v) for k,v in x.items()}
        rows.append(row)
    common=next(r for r in rows if r["arm"]=="S" and r["stage"]==0)
    statuses={};budget=[];lineage=[];memory=[];deploy=[];epochs=[];diagnostics=[];numeric=[];cost=0
    for arm in ("U0","L05","L10","E_R1"):
        root=base/("E_run" if arm=="E_R1" else "ablations")/arm
        parent=read(root/"parent_receipt.json") if (root/"parent_receipt.json").exists() else dict(status="NOT_STARTED")
        statuses[arm]=parent["status"]
        if arm!="E_R1" and parent["status"]=="NOT_STARTED":raise RuntimeError("required ablation not executed")
        if arm=="E_R1" and parent["status"]=="NOT_STARTED" and q["E_qualification"]=="PASS":
            raise RuntimeError("qualified E has not been executed")
        rows.append(dict(common,arm=arm))
        for stage in (1,2):
            sr=root/f"stage{stage}";st=lines(sr/"steps.jsonl");ep=lines(sr/"epochs.jsonl")
            cost+=len(st);expected=(3200,2100)[stage-1]
            assert [x["step"] for x in st]==list(range(1,len(st)+1))
            rec=read(sr/"r1_receipt.json") if (sr/"r1_receipt.json").exists() else None
            budget.append(dict(arm=arm,stage=stage,expected_updates=expected,actual_updates=len(st),
                status=rec["status"] if rec else "INCOMPLETE_OR_NOT_STARTED"))
            if rec:
                assert rec["r1_arm"]==arm and rec["source"]==reuse["source"]
                assert rec["counters"]["optimizer_steps"]==len(st)==expected
                assert rec["teacher_initial_hash"]==rec["parent_hash"]==rec["teacher_final_hash"]
                if stage==1:assert rec["parent_hash"]==reuse["reused"][0]["student_hash"]
                else:assert rec["parent_hash"]==read(root/"stage1/r1_receipt.json")["student_hash"]
                lineage.append(rec)
                for op in ("train","eval"):assert read(root/f"stage{stage}_{op}_exit.json")["exit_code"]==0
                ev=read(sr/"val.json");assert ev["student_hash"]==rec["student_hash"] and ev["inference_full_models"]==1
                rows.extend(dict(x,arm=arm) for x in ev["rows"])
                if arm in ("L05","L10"):assert rec["unlabeled_images_opened"]==0
            if stage==2 and parent["status"]=="COMPLETE":
                d=read(sr/"deployment.json");assert d["status"]=="PASS" and d["models"]==1 and d["teacher_hidden"]
                assert read(root/"deploy_exit.json")["exit_code"]==0
                deploy.append(dict(arm=arm,**d))
            if st:
                tag=dict(arm=arm,stage=stage)
                memory.append(dict(**tag,completed_epochs=len(ep),logged_epoch_seconds=sum(x["seconds"] for x in ep),
                    **{k:max([x[k] for x in ep]) if ep else None for k in ("max_allocated","max_reserved","cpu_max_rss_kib","max_full_models",
                        "student_parameters_bytes","teacher_parameters_bytes","student_buffers_bytes","teacher_buffers_bytes",
                        "gradient_bytes","optimizer_bytes","prototype_bytes")},
                    target_logical_tensor_bytes=max([d["target_temporary_bytes"] for x in st for d in x["distillation"]] or [0]),
                    counters=rec["counters"] if rec else "INCOMPLETE_RECEIPT"))
                for epoch in ep:
                    batch=[x for x in st if x["epoch"]==epoch["epoch"]]
                    epochs.append(dict(**tag,epoch=epoch["epoch"],seconds=epoch["seconds"],
                        **{k:float(np.mean([x[k] for x in batch])) for k in ("sup","raw_lu","weighted_lu","labeled_kd","unlabeled_kd",
                            "actual_lambda_u","nominal_lambda_u","labeled_kd_coefficient","unlabeled_kd_coefficient")}))
                    if epoch["epoch"]%5==0:
                        for branch in ("labeled","unlabeled"):
                            ds=[d for x in batch for d in x["distillation"] if d["branch"]==branch]
                            if not ds:continue
                            for c in range(3):
                                cs=[d["classes"][c] for d in ds]
                                rr=dict(**tag,epoch=epoch["epoch"],branch=branch,class_id=c)
                                for key in cs[0]:
                                    if key=="class_id":continue
                                    values=[x[key] for x in cs if x[key] is not None]
                                    if not values:rr[key]=None
                                    elif key.endswith("_max") or key=="iterations":rr[key]=max(values)
                                    elif key=="log_eta_logsumexp":rr[key]=float(np.logaddexp.reduce(values))
                                    else:rr[key]=sum(values)
                                rr["conflict_rate"]=rr["conflicts"]/max(rr["reliable"],1)
                                if branch=="unlabeled":
                                    ps=[x["pas"][c] for x in batch]
                                    rr.update(predicted_pixels=sum(x["predicted_pixels"] for x in ps),pas_accepted_pixels=sum(x["accepted_pixels"] for x in ps),
                                        prototype_valid_steps=sum(x["prototype_valid"] for x in ps))
                                diagnostics.append(rr)
                        for x in batch:numeric.append(dict(**tag,epoch=x["epoch"],step=x["step"],branches=x["numerics"]))
        if parent["status"]=="COMPLETE":assert sum(r["actual_updates"] for r in budget if r["arm"]==arm)==5300
    completed=list("SABCD")+[a for a in statuses if statuses[a]=="COMPLETE"]
    scores={a:metrics({(x["stage"],x["domain_index"]):x["macro_fg_dice"] for x in rows if x["arm"]==a}) for a in completed}
    def current(a,t):
        return next(x for x in rows if x["arm"]==a and x["stage"]==t and x["domain_index"]==t)
    comparisons=[]
    def compare(a,b,question):
        result=dict(method=a,reference=b,question=question,**{k:scores[a][k]-scores[b][k] for k in scores[a]})
        for t in (1,2):
            for k in ("macro_fg_dice","rim_dice","cup_dice"):result[f"stage{t}_{k}"]=current(a,t)[k]-current(b,t)[k]
        comparisons.append(result);return result
    components={};gate_rows=[]
    def gate(question,name,value,threshold):
        gate_rows.append(dict(question=question,gate=name,value=value,threshold=threshold,pass_gate=value>=threshold))
        return value>=threshold
    if statuses["U0"]=="COMPLETE":
        d=compare("D","U0","pseudo_ce")
        ok=gate("pseudo_ce","F_or_N_gain",max(d["F"],d["N"]),.005)
        ok=gate("pseudo_ce","other_not_harmed",min(d["F"],d["N"]),-.005) and ok
        for t in (1,2):
            for c in ("rim_dice","cup_dice"):ok=gate("pseudo_ce",f"stage{t}_{c}",d[f"stage{t}_{c}"],-.02) and ok
        components["pseudo_ce"]="PSEUDO_CE_DEVELOPMENT_SIGNAL" if ok else "PSEUDO_CE_CONTRIBUTION_NOT_ESTABLISHED"
    else:components["pseudo_ce"]="NOT_EVALUATED_INCOMPLETE_U0"
    if all(statuses[a]=="COMPLETE" for a in ("U0","L05","L10")):
        ok=gate("unlabeled_kd","F_vs_best_label_only",scores["U0"]["F"]-max(scores[a]["F"] for a in ("L05","L10")),.005)
        for a in ("L05","L10"):
            d=compare("U0",a,"unlabeled_kd");ok=gate("unlabeled_kd","N_vs_"+a,d["N"],-.01) and ok
            for t in (1,2):
                for c in ("rim_dice","cup_dice"):ok=gate("unlabeled_kd",f"stage{t}_{c}_vs_{a}",d[f"stage{t}_{c}"],-.02) and ok
        components["unlabeled_kd"]="UNLABELED_KD_DEVELOPMENT_SIGNAL" if ok else "UNLABELED_KD_CONTRIBUTION_NOT_ESTABLISHED"
    else:components["unlabeled_kd"]="NOT_EVALUATED_INCOMPLETE_ABLATIONS"
    if statuses["E_R1"]=="COMPLETE" and all(a in scores for a in ("U0","L05","L10")):
        ok=gate("E_increment","F_vs_best_simple",scores["E_R1"]["F"]-max(scores[a]["F"] for a in ("C","D","U0","L05","L10")),.01)
        d=compare("E_R1","B","E_increment");ok=gate("E_increment","history_vs_B",d["H"],.01) and ok
        for t in (1,2):
            for c,tol in (("macro_fg_dice",-.01),("rim_dice",-.02),("cup_dice",-.02)):
                ok=gate("E_increment",f"stage{t}_{c}_vs_B",d[f"stage{t}_{c}"],tol) and ok
        components["E_increment"]="SCD_INCREMENTAL_DEVELOPMENT_SIGNAL" if ok else "SCD_INCREMENTAL_VALUE_NOT_ESTABLISHED"
    else:components["E_increment"]="NOT_EVALUATED"
    candidates={}
    for a in completed:
        if a=="S":continue
        d=compare(a,"S","current_capability")
        ok=gate("candidate_"+a,"F_vs_S",d["F"],.01)
        for t in (1,2):
            for c,tol in (("macro_fg_dice",-.01),("rim_dice",-.02),("cup_dice",-.02)):
                ok=gate("candidate_"+a,f"stage{t}_{c}_vs_S",d[f"stage{t}_{c}"],tol) and ok
        candidates[a]="FIXED_RECIPE_CANDIDATE_FOR_REPLICATION" if ok else "CURRENT_CAPABILITY_PROTECTION_NOT_ESTABLISHED"
    table(output/"STAGE_DOMAIN_MATRIX.csv",rows)
    table(output/"PER_CLASS.csv",[dict(arm=x["arm"],stage=x["stage"],domain=x["domain"],rim=x["rim_dice"],cup=x["cup_dice"],macro=x["macro_fg_dice"]) for x in rows])
    table(output/"METHOD_SUMMARY.csv",[dict(arm=a,**v) for a,v in scores.items()])
    table(output/"PSEUDO_CE_ABLATION.csv",[x for x in comparisons if x["question"]=="pseudo_ce"])
    table(output/"UNLABELED_KD_ABLATION.csv",[x for x in comparisons if x["question"]=="unlabeled_kd"])
    table(output/"PAIRED_COMPARISONS.csv",comparisons);table(output/"GATE_ACCOUNTING.csv",gate_rows)
    table(output/"CUP_RETENTION_AND_ADAPTATION.csv",[dict(arm=a,stage=t,rim=current(a,t)["rim_dice"],cup=current(a,t)["cup_dice"],
        macro=current(a,t)["macro_fg_dice"],rim_minus_S=current(a,t)["rim_dice"]-current("S",t)["rim_dice"],
        cup_minus_S=current(a,t)["cup_dice"]-current("S",t)["cup_dice"]) for a in completed for t in (1,2)])
    table(output/"EPOCH_CURVES.csv",epochs);table(output/"MECHANISM_EVERY_5_EPOCHS.csv",diagnostics)
    write_json(output/"NUMERIC_TRAJECTORY.json",dict(unit="per-batch quantiles at every fifth epoch, not pooled epoch quantiles",rows=numeric))
    write_json(output/"MEMORY_AND_DEPLOYMENT.json",dict(stages=memory,deployments=deploy,
        target_bytes_note="logical named tensors may alias; CUDA allocated/reserved peaks are measured independently"))
    write_json(output/"STAGE_LINEAGE.json",dict(stages=lineage))
    write_json(output/"UPDATED_BUDGET_AND_ATTEMPTS.json",dict(old_formal_updates=36008,new_formal_updates=cost,new_limit=21200,
        cumulative_all_attempts=36008+cost,ablation_target=15900,E_target_if_qualified=5300,stages=budget,
        corpus_replay_updates=3,qualification_synthetic_updates=q["synthetic_optimizer_updates"],
        local_qualification_cost_accounted_separately=True,no_new_common_or_old_complete_arm_updates=True))
    outcome=dict(engineering=statuses,components=components,current_capability=candidates,E_numerical=q["E_qualification"],
        old_status="INCOMPLETE_TRAINING_MATRIX",old_B_minus_S_N=reuse["old_method_metrics"]["B"]["N"]-reuse["old_method_metrics"]["S"]["N"],
        old_B_minus_S_F=reuse["old_method_metrics"]["B"]["F"]-reuse["old_method_metrics"]["S"]["F"],
        no_combined_scientific_PASS=True,seed1_2_started=False,external_data_started=False,stopped=True,source=reuse["source"])
    if (base/"R1_FINAL_ADJUDICATION.json").exists():raise FileExistsError("R1 already adjudicated")
    write_json(base/"R1_FINAL_ADJUDICATION.json",outcome);write_json(output/"STATUS.json",outcome)
    report=["# Single-teacher R1: fixed experiment closeout","",json.dumps(outcome,indent=2),"",
        "| Arm | F | H | N | BWT | Forget REFUGE | Forget RIM |","|---|---:|---:|---:|---:|---:|---:|"]
    for a,v in scores.items():report.append("| "+a+" | "+" | ".join(f"{x:.12f}" for x in v.values())+" |")
    report += ["",f"New formal optimizer updates: {cost}; historical attempts: 36008; cumulative: {36008+cost}. Common and old complete arms were not retrained.",
        "", "The old V0.1 terminal remains INCOMPLETE_TRAINING_MATRIX. Results are separate development questions, not a replacement combined PASS. U0 retains PAS references and U-KD; removing explicit pseudo-CE does not remove all pseudo-label mechanisms. Both label-only coefficients are retained. Every class-specific current-domain cost is in CUP_RETENTION_AND_ADAPTATION.csv.",
        "", "No hidden unlabeled GT, test, historical training replay, external dataset, extra seed, parameter search or main merge. Pseudo-label precision remains NOT_EVALUATED; coverage is not precision. E is evaluated only after its full trajectory and single-student deployment complete.",
        "",f"Source: {reuse['source']}. NAS root: {base}. Public release excludes images, GT, patient identifiers, weights and private vectors."]
    (output/"FINAL_REPORT.md").write_text("\n".join(report)+"\n")
    (output/"NEXT_ACTION.md").write_text("# Stop after this fixed experiment\n\nNo seed1/2, tuning or extra module is started. Candidate labels refer only to a separately authorized fixed-recipe replication. Failed contribution or adaptation criteria remain explicit; no recipe is repaired using validation feedback.\n")
    return outcome

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--base",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();print(json.dumps(generate(a.base,a.output)))
