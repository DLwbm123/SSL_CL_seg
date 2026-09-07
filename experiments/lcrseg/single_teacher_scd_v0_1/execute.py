"""Durable NAS parent: exact source/qualification gate, child exits, fixed GPU lanes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .engine import write_json


def child(command, log, receipt, env):
    started=time.time()
    with Path(log).open("x") as f:
        process=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env)
        code=process.wait()
    result=dict(command=command,pid=process.pid,exit_code=code,started_unix=started,
        finished_unix=time.time(),cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES"))
    write_json(receipt,result)
    if code:raise RuntimeError(f"child exited {code}; {receipt}")
    return result


def available(gpu):
    text=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.free,uuid","--format=csv,noheader,nounits"],text=True)
    row=next(x.split(", ") for x in text.strip().splitlines() if int(x.split(",")[0])==gpu)
    if gpu not in (4,5,6,7) or int(row[1])<8192:
        raise RuntimeError(f"authorized GPU {gpu} has less than frozen 8 GiB admission memory")
    return dict(physical_index=gpu,free_mib=int(row[1]),uuid=row[2])


def execute(output,data,reference,qualification):
    output=Path(output)
    if not str(output.resolve()).startswith("/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/"):
        raise ValueError("formal outputs must be on NAS")
    from .freeze import verify
    source=verify()
    if subprocess.check_output(["git","status","--porcelain"],text=True).strip():raise RuntimeError("dirty execution source")
    q=json.loads(Path(qualification).read_text())
    if q["source"]!=source or q["status"]!="PASS" or not q.get("real_two_case_smoke",{}).get("loss_decreased"):
        raise RuntimeError("exact source real GPU qualification required")
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"reservation.json",dict(source=source,qualification=str(qualification),
        expected_optimizer_updates=39800,seed=0,gpus=[4,5,6,7],created_unix=time.time()))
    module="experiments.lcrseg.single_teacher_scd_v0_1."
    def stage(arm,t,gpu,parent):
        root=output/arm/f"stage{t}";root.parent.mkdir(exist_ok=True)
        resource=available(gpu)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=":4096:8",PYTHONDONTWRITEBYTECODE="1")
        command=[sys.executable,"-m",module+"engine","--data",data,"--reference",reference,"--output",str(root),"--arm",arm,"--stage",str(t)]
        if parent:command.extend(["--parent",str(parent)])
        write_json(root.parent/f"stage{t}_resource.json",resource)
        child(command,root.parent/f"stage{t}_train.log",root.parent/f"stage{t}_train_exit.json",env)
        command=[sys.executable,"-m",module+"evaluate","--data",data,"--reference",reference,
            "--checkpoint",str(root/"student_latest.pt"),"--output",str(root/"val.json")]
        child(command,root.parent/f"stage{t}_eval.log",root.parent/f"stage{t}_eval_exit.json",env)
        if t==2:
            command=[sys.executable,"-m",module+"deploy","--checkpoint",str(root/"student_latest.pt"),"--reference",reference]
            child(command,root.parent/"deploy.log",root.parent/"deploy_exit.json",env)
        return root/"student_latest.pt"
    common=stage("common",0,4,None)
    failures=[]
    def lane(gpu,arms):
        results=[]
        for arm in arms:
            try:
                previous=stage(arm,1,gpu,common)
                stage(arm,2,gpu,previous)
                results.append(dict(arm=arm,status="COMPLETE"))
            except BaseException as error:
                results.append(dict(arm=arm,status="INCOMPLETE_TRAINING_MATRIX",error=repr(error)))
        return results
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(lane,gpu,arms) for gpu,arms in ((4,"SD"),(5,"AE"),(6,"B"),(7,"C"))]
        results=[r for future in futures for r in future.result()]
    failures=[r for r in results if r["status"]!="COMPLETE"]
    write_json(output/"parent_receipt.json",dict(source=source,status="INCOMPLETE_TRAINING_MATRIX" if failures else "MATRIX_COMPLETE_PENDING_ADJUDICATION",arms=results))
    if failures:raise RuntimeError("one or more arms incomplete")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("output","data","reference","qualification"):p.add_argument("--"+name,required=True)
    a=vars(p.parse_args())
    try:execute(**a)
    except BaseException as error:
        if Path(a["output"]).is_dir():write_json(Path(a["output"])/"parent_failure.json",dict(status="INCOMPLETE_TRAINING_MATRIX",error=repr(error)))
        raise
