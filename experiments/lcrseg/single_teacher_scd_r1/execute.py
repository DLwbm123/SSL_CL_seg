"""Fixed independent lanes; each failure closes that arm and never changes another recipe."""
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from experiments.lcrseg.single_teacher_scd_v0_1.execute import child
from experiments.lcrseg.single_teacher_scd_v0_1.engine import write_json
from .freeze import verify
from .engine import ARMS

def execute(base,qualification,data,reference,include_e=False):
    source=verify();base=Path(base)
    q=json.loads(Path(qualification).read_text());reuse=json.loads((base/"INPUT_REUSE.json").read_text())
    assert q["source"]==source==reuse["source"] and q["ablation_qualification"]=="PASS"
    if include_e:assert q["E_qualification"]=="PASS"
    group="E_run" if include_e else "ablations"
    run=base/group;run.mkdir()
    lanes={"E_R1":7} if include_e else {"U0":4,"L05":5,"L10":6}
    # Admit against live physical indices. One R1 process per allowed lane, no other process is terminated.
    free={}
    for line in subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],text=True).splitlines():
        idx,mem=line.split(",");free[int(idx)]=int(mem)
    for gpu in lanes.values():
        if free[gpu]<2072:raise RuntimeError(f"GPU{gpu}: less than measured 2072 MiB admission floor")
    write_json(run/"reservation.json",dict(source=source,lanes=lanes,qualification=qualification,
        steps_per_arm=5300,expected_updates=5300*len(lanes),common_parent=reuse["common_checkpoint"],created_unix=time.time(),
        recipes={a:ARMS[a] for a in lanes},numerical_trajectory_limit=1 if include_e else None))
    def arm(name,gpu):
        root=run/name;root.mkdir();env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=":4096:8")
        parent=Path(reuse["common_checkpoint"])
        try:
            for stage in (1,2):
                stage_root=root/f"stage{stage}"
                cmd=[sys.executable,"-m","experiments.lcrseg.single_teacher_scd_r1.engine","--data",data,"--reference",reference,
                    "--output",str(stage_root),"--arm",name,"--stage",str(stage),"--parent",str(parent)]
                child(cmd,root/f"stage{stage}_train.log",root/f"stage{stage}_train_exit.json",env)
                child([sys.executable,"-m","experiments.lcrseg.single_teacher_scd_v0_1.evaluate","--data",data,"--reference",reference,
                    "--checkpoint",str(stage_root/"student_latest.pt"),"--output",str(stage_root/"val.json")],
                    root/f"stage{stage}_eval.log",root/f"stage{stage}_eval_exit.json",env)
                parent=stage_root/"student_latest.pt"
            child([sys.executable,"-m","experiments.lcrseg.single_teacher_scd_v0_1.deploy","--checkpoint",str(parent),"--reference",reference],
                root/"deploy.log",root/"deploy_exit.json",env)
            result=dict(arm=name,status="COMPLETE",source=source,successful_updates=5300)
        except BaseException as error:
            result=dict(arm=name,status="ENGINEERING_FAILURE",source=source,error=repr(error))
        write_json(root/"parent_receipt.json",result)
        return result
    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        futures=[pool.submit(arm,a,g) for a,g in lanes.items()]
        results=[x.result() for x in futures]
    write_json(run/"parent_receipt.json",dict(source=source,arms=results,
        status="COMPLETE" if all(x["status"]=="COMPLETE" for x in results) else "PARTIAL_ENGINEERING_FAILURE"))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for x in ("base","qualification","data","reference"):p.add_argument("--"+x,required=True)
    p.add_argument("--include-e",action="store_true");execute(**vars(p.parse_args()))
