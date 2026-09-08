"""One old pair per process, at most 400/250 sample-model calls, zero updates."""
import argparse,json
from pathlib import Path
from .core import *
from .diagnostics import snapshot,proto_hash
from .freeze import verify
OLD=Path('/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/ssl_foundation_v0_1_20260908_01')

def run(output,data,reference,domain,arm,device):
    source=verify()
    if domain not in DOMAINS or arm not in ('MT_PAS_G0','MT_PAS_G1'):raise PermissionError('fixed P0 pair required')
    root=OLD/'seed11'/domain/arm;rec=json.loads((root/'receipt.json').read_text());snap=json.loads((root/'eval100/receipt.json').read_text())
    ck=torch.load(root/'latest.pt',map_location=device,weights_only=False)
    assert ck['source']==rec['source']=='d9542bd6f7b4acfe351bdaa8c8107bb875d40717' and ck['complete'] and ck['epoch']==100 and ck['arm']==arm and ck['domain']==domain and ck['seed']==11
    ck.pop('optimizer');student=from_state(reference,device,ck.pop('student'),NORMAL);teacher=from_state(reference,device,ck.pop('ema'),NORMAL)
    for net in (student,teacher):
        net.eval()
        for p in net.parameters():p.requires_grad_(False)
    assert state_hash(student)==rec['student_hash'] and state_hash(teacher)==rec['ema_hash']
    proto,support=ck['proto'],ck['support'];assert proto_hash(proto,support)==snap['training_proto_hash'];del ck
    from experiments.lcrseg.single_teacher_scd_v0_1.engine import audit_live_models
    assert audit_live_models(2)==2
    ct=Counter();snapshot(student,teacher,None,data=data,reference=reference,device=device,domain=domain,seed=11,arm=arm,epoch=100,output=output,proto=proto,support=support,counters=ct,audit_only=True)
    cap=COUNTS[domain][2]*10;assert ct['diagnostic_student_images']+ct['diagnostic_ema_images']==cap
    result=dict(status='COMPLETE',source=source,old_source=rec['source'],arm=arm,domain=domain,updates=0,sample_model_forward=cap,cap=cap,prototype_hash=proto_hash(proto,support),prototype_origin='actual epoch96 refresh retained in epoch100 final checkpoint',student_hash=state_hash(student),ema_hash=state_hash(teacher),pixel_cache_reused=False,cache_reason='Old archive contains aggregate counts and hard deployment predictions, not probabilities/features needed for fixed-mass comparisons.')
    write_json(Path(output)/'P0_RECEIPT.json',result)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('output','data','reference','domain','arm'):p.add_argument('--'+k,required=True)
    run(**vars(p.parse_args()),device=torch.device('cuda:0'))
