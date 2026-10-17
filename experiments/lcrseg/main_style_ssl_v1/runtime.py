import os,json,time,copy,subprocess,math
from pathlib import Path
import numpy as np
import torch
from ..five_frameworks_v1.native_parent import build
from ..five_frameworks_v1.integration import ExecutionPermit,_PERMIT_SEAL
from ..five_frameworks_v1.gate import digest
from ..five_frameworks_v1.checkpoint import save,restore,atomic_save,PhysicalLedger
from ..five_frameworks_v1.evaluate import binary_metrics,segmentation_metrics
from ..single_teacher_scd_v0_1.data import MANIFEST_SHA
from .protocol import MAPS,SHAPES,CLASSES,ORDERS,options,nodes
from .model import Parent,MainModel,MainTrainer
from .data import CurrentDomain,Accessor


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return 'Infinity' if value>0 else 'NONFINITE'
    return value


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:json.dump(clean(value),f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def append(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(clean(value),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def permit(node,scope):
    identity=dict(dataset=node['dataset'],domain=node['domain'],seed=node['seed'],order=node['order'],stage=node['stage'],manifest=MANIFEST_SHA)
    return ExecutionPermit(dict(authorized_manifest_digests=[digest(identity)],execution_scope=scope),('main',),dict(native_updates=node['steps']),_PERMIT_SEAL)


def setup(node,data,reference,device,predecessor=None,initialize=True,source_labeled_only=False,trainer_type=MainTrainer):
    device=torch.device(device)
    ds=node['dataset'];source=node['stage']==0
    native=build(reference,device,node['seed'],input_channels=SHAPES[ds][0],num_classes=len(MAPS[ds]),domain=ORDERS[ds][0][0])
    previous=None
    lineage=dict(kind='new_SSL_source',dataset=ds,domain=node['domain'],seed=node['seed'])
    if not source:
        if predecessor is None:raise ValueError('missing sealed predecessor')
        sealed=torch.load(predecessor,map_location=device,weights_only=False)
        if sealed.get('sealed') is not True:raise ValueError('predecessor is not sealed')
        if node.get('scope','formal')=='formal':
            expected=next(n for n in nodes() if n['id']==node['predecessor'])
            if any(sealed.get(k)!=v for k,v in expected.items()) or sealed['updates']!=expected['steps']:
                raise ValueError('predecessor is not the exact completed formal node')
        if sealed['dataset']!=ds or sealed['seed']!=node['seed'] or sealed['stage']!=node['stage']-1:raise ValueError('wrong predecessor lineage')
        if node['stage']>1 and sealed['order']!=node['order']:raise ValueError('cross-order target entry')
        native.load_state_dict(sealed['native']);previous=sealed['transform']
        lineage=dict(kind='sealed_main_predecessor',dataset=ds,domain=sealed['domain'],seed=node['seed'],stage=sealed['stage'],order=sealed['order'],node=sealed['id'])
    parent=Parent(native,node['seed'],lineage,source)
    model=MainModel(parent,MAPS[ds],previous,initialize_basis=initialize)
    execution=permit(node,node.get('scope','formal'))
    provider=CurrentDomain(data,ds,node['domain'],node['seed'],node['order'],node['stage'],lineage,device,execution)
    opt=options(node['steps'],source)
    if source_labeled_only:
        if not source:raise ValueError('L-only control is restricted to source training')
        opt['lambda_U']=0.
    trainer=trainer_type(model,provider,opt,initialize=initialize,execution=execution)
    return trainer


def seal(trainer,node,path):
    deployment=trainer.model.deploy()
    atomic_save(dict(**node,native=deployment.parent.native.state_dict(),transform=deployment.transform.detach(),
                     sealed=True,precision='FP32',updates=trainer.step),path)
    return deployment


@torch.no_grad()
def evaluate(deployment,data,node,out):
    ds=node['dataset'];domains=ORDERS[ds][node['order']-1][:node['stage']+1];device=next(deployment.parameters()).device
    result={}
    for domain in domains:
        access=Accessor(data,ds,domain,'val');patient_rows={}
        for i,row in enumerate(access.rows):
            item=access.read(i);x=item['image'];truth=item['label'].numpy()
            if ds=='fundus':pred=deployment(x[None].to(device)).argmax(1)[0].cpu().numpy();metrics=segmentation_metrics(pred,truth)
            else:
                spacing=item['spacing']
                if len(spacing)!=3 or any(not math.isfinite(float(s)) or float(s)<=0 for s in spacing):raise ValueError('invalid processed spacing')
                pred=np.concatenate([deployment(x[z:z+2,None].to(device)).argmax(1).cpu().numpy() for z in range(0,len(x),2)])
                metrics={name:binary_metrics(pred==c,truth==c,spacing) for c,name in enumerate(CLASSES[ds]) if c}
                for metric in metrics.values():metric['distance_unit']='mm'
            patient_rows.setdefault(row['patient_id'],[]).append((row['phase'],metrics))
            append(Path(out)/'private_validation.jsonl',dict(domain=domain,patient=row['patient_id'],phase=row['phase'],metrics=metrics))
        if ds=='mnms' and any(sorted(p for p,_ in entries)!=['ED','ES'] for entries in patient_rows.values()):raise ValueError('validation patient lacks ED/ES')
        names=next(iter(patient_rows.values()))[0][1].keys();classes={}
        for name in names:
            class_summary={}
            for metric in ('Dice','HD95','ASSD'):
                vals=[float(np.mean([m[name][metric] for _,m in entries])) for entries in patient_rows.values()]
                class_summary[metric]=float(np.mean(vals));class_summary[metric+'_infinite_patients']=sum(math.isinf(v) for v in vals)
            class_summary['empty_phase_cases']={kind:sum(m[name]['empty_case']==kind for entries in patient_rows.values() for _,m in entries) for kind in ('both','one','neither')}
            classes[name]=class_summary
        primary=('rim','cup') if ds=='fundus' else CLASSES[ds][1:]
        result[domain]=dict(classes=classes,patients=len(patient_rows),primary_dice=float(np.mean([classes[name]['Dice'] for name in primary])),
                            auxiliary_cup_disc_dice=float(np.mean([classes[name]['Dice'] for name in ('cup','disc_union')])) if ds=='fundus' else None)
        write(Path(out)/'PARTIAL_VALIDATION.json',result)
    write(Path(out)/'VALIDATION.json',result)
    return result


def worker(config):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    node=config['node']
    if node not in nodes():raise PermissionError('formal node differs from frozen protocol')
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=False)
    started=time.time();ledger=PhysicalLedger(out/'physical_updates.jsonl');trainer=None
    write(out/'STATUS.json',dict(status='STARTING',node=node,started=started))
    try:
        trainer=setup(node,config['data'],config['reference'],'cuda',config.get('predecessor'))
        identity={k:node[k] for k in ('id','dataset','seed','order','stage','domain')}
        torch.cuda.reset_peak_memory_stats()
        while trainer.step<node['steps']:
            trainer.update(physical=ledger.append)
            if trainer.step==1 or trainer.step%100==0:
                write(out/'STATUS.json',dict(status='RUNNING',step=trainer.step,physical_updates=trainer.physical_optimizer_updates,
                    node=node,elapsed_seconds=time.time()-started,last=trainer.last,peak_memory_bytes=torch.cuda.max_memory_allocated()))
                print(json.dumps(dict(step=trainer.step,labeled=trainer.last['labeled_loss'],unlabeled=trainer.last.get('unlabeled_loss'),elapsed=time.time()-started)),flush=True)
            if trainer.step%2000==0:save(trainer,out/'latest.pt',identity)
        save(trainer,out/'final.pt',identity);deployment=seal(trainer,node,out/'sealed.pt')
        validation=evaluate(deployment,config['data'],node,out)
        write(out/'COSTS.json',dict(**ledger.summary(trainer.step),elapsed_seconds=time.time()-started,
                peak_memory_bytes=torch.cuda.max_memory_allocated(),telemetry=trainer.telemetry,L_batch_reads=trainer.provider.l_reads,
                U_batch_reads=trainer.provider.u_reads,L_image_exposure=2*trainer.provider.l_reads,U_image_exposure=2*trainer.provider.u_reads,
                checkpoint_bytes=(out/'final.pt').stat().st_size,deployment_bytes=(out/'sealed.pt').stat().st_size))
        write(out/'STATUS.json',dict(status='COMPLETE',node=node,step=trainer.step,validation=validation,elapsed_seconds=time.time()-started))
        write(out/'EXIT.json',dict(exit_code=0,scientific_success='UNDETERMINED_NO_SELECTION_GATE'))
    except BaseException as exc:
        import traceback
        committed=0 if trainer is None else trainer.step
        write(out/'COSTS.json',dict(**ledger.summary(committed),elapsed_seconds=time.time()-started,
                                   telemetry={} if trainer is None else trainer.telemetry,status='FAILED'))
        write(out/'STATUS.json',dict(status='FAILED',node=node,step=committed,elapsed_seconds=time.time()-started,error=str(exc),traceback=traceback.format_exc()))
        write(out/'EXIT.json',dict(exit_code=1,error=str(exc)));raise
