import argparse,csv,json
from pathlib import Path
from experiments.lcrseg.di_dmpa_gate1.binding import check_hash
from experiments.lcrseg.single_teacher_scd_v0_1.data import metadata,DOMAINS
from experiments.lcrseg.single_teacher_scd_v0_1.report import metrics
from experiments.lcrseg.single_teacher_scd_v0_1.engine import write_json
from .freeze import verify

def run(base,old_root,data):
    source=verify();base=Path(base);base.mkdir()
    old=Path(old_root)/"run_01"
    public=Path("experiments/lcrseg/docs/single_teacher_scd_v0_1")
    assert json.loads((public/"STATUS.json").read_text())["status"]=="INCOMPLETE_TRAINING_MATRIX"
    source_rows=list(csv.DictReader((public/"STAGE_DOMAIN_MATRIX.csv").open()))
    cases=metadata(data);reuse=[];vals=[]
    for arm,stage in [("common",0)]+[(a,t) for a in "SABCD" for t in (1,2)]:
        root=old/arm/f"stage{stage}";receipt=json.loads((root/"receipt.json").read_text())
        v=json.loads((root/"val.json").read_text())
        assert receipt["source"]=="057ce07fa7bd02f8319e0c2bdd9d4adbceff23d8"
        assert receipt["status"]=="COMPLETE" and receipt["student_hash"]==v["student_hash"]
        for row in v["rows"]:
            ref=next(r for r in source_rows if r["arm"]==("S" if arm=="common" else arm) and int(r["stage"])==stage and int(r["domain_index"])==row["domain_index"])
            for k in ("macro_fg_dice","rim_dice","cup_dice","background_dice"):
                assert float(ref[k])==row[k]
        checkpoint=root/"student_latest.pt"
        if stage in (0,2):
            check_hash(checkpoint,receipt["checkpoint_sha256"])
        else:assert not checkpoint.exists()
        if stage==2:
            dep=json.loads((root/"deployment.json").read_text())
            assert dep["status"]=="PASS" and dep["models"]==1 and dep["student_hash"]==receipt["student_hash"]
        reuse.append(dict(arm=arm,stage=stage,source=receipt["source"],student_hash=receipt["student_hash"],
            checkpoint_sha256=receipt["checkpoint_sha256"],checkpoint_present=checkpoint.exists(),
            missing_stage1_reason="previous frozen rolling retirement; no recreation" if stage==1 else None,
            private_public_metrics_exactly_equal=True))
        vals.extend(v["rows"])
    common=next(x for x in vals if x["arm"]=="common")
    means={}
    for a in "SABCD":
        matrix={(x["stage"],x["domain_index"]):x["macro_fg_dice"] for x in vals if x["arm"]==a}
        matrix[0,0]=common["macro_fg_dice"];means[a]=metrics(matrix)
    frozen_means={x["arm"]:x for x in csv.DictReader((public/"METHOD_SUMMARY.csv").open())}
    for a in means:
        for k,v in means[a].items():assert v==float(frozen_means[a][k])
    counts=[dict(domain=d,labeled=sum(x["site_or_vendor"]==d and x["primary_20pct_split"]=="train_labeled" for x in cases),
        unlabeled=sum(x["site_or_vendor"]==d and x["primary_20pct_split"]=="train_unlabeled" for x in cases)) for d in DOMAINS]
    result=dict(status="PASS",source=source,old_status_preserved="INCOMPLETE_TRAINING_MATRIX",
        old_formal_attempt_updates=36008,reused=reuse,old_method_metrics=means,data_counts=counts,
        common_checkpoint=str(old/"common/stage0/student_latest.pt"),new_common_updates=0,
        formal_data_asset_reads=0,old_stage1_weights_not_recreated=True)
    write_json(base/"INPUT_REUSE.json",result);print(json.dumps(result))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for x in ("base","old-root","data"):p.add_argument("--"+x,required=True)
    run(**vars(p.parse_args()))
