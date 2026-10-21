"""Recover only the first group's summary from its six complete sealed records."""
import json
from pathlib import Path
from unittest.mock import patch
from .single_target_audit import rows
from ..gyro_transfer_v1 import run as transfer
from ..siemens_lonly_v1.run import aggregate
from ..main_style_ssl_v1.runtime import write


def main(cfg):
    root=Path(cfg['root']);out=root/'jobs/20261025_S';assert not (out/'RESULTS.json').exists()
    count=json.loads((root/'calls_counter.json').read_text());assert count['pending'] is None and count['executed']['evaluation_native']==2892
    budget=json.loads((root/'budget_counter.json').read_text());assert budget['total']==2320 and budget['by_phase']['20261025_S:formal']==2000
    assert len(rows(out/'physical_updates.jsonl'))==2000
    ref=Path(cfg['old_five_root'])/'jobs/S';keys=lambda rr:{(r['patient'],r['phase']) for r in rr}
    for name,n in [('private_Siemens_val.jsonl',46),('private_Philips_val.jsonl',62),('private_source_train.jsonl',20),('private_target_train.jsonl',26),('private_ema_Siemens_val.jsonl',46),('private_ema_Philips_val.jsonl',62)]:
        rr=rows(out/name);assert len(rr)==n and len(keys(rr))==n and keys(rr)==keys(rows(ref/name))
        assert all(len(r['dice'])==3 and all(0<=v<=1 for v in r['dice']) for r in rr)
    slices={'private_Siemens_val.jsonl':544,'private_Philips_val.jsonl':660,'private_source_train.jsonl':230,'private_target_train.jsonl':254,'private_ema_Siemens_val.jsonl':544,'private_ema_Philips_val.jsonl':660}
    def reuse(model,c,domain,role,path):
        rr=rows(path);return {**aggregate(rr),'forward_slices':slices[Path(path).name]},rr
    before=json.loads((root/'calls_counter.json').read_text())
    with patch.object(transfer.engine,'evaluate',reuse):transfer.finalize(dict(cfg,arm='A'),out,None,None,2000,2000)
    assert before==json.loads((root/'calls_counter.json').read_text())
    write(out/'EVALUATION_EXIT.json',{'exit_code':0,'summary_recovered_from_sealed_records':True})
    write(root/'EXIT_20261025_S_evaluate.json',{'exit_code':0,'original_failure_retained':True,'summary_recovery_native_forwards':0})
    write(root/'SUMMARY_REPAIR.json',{'status':'PASS','model_forwards':0,'optimizer_updates':0,'functional_updates':0,'calls_unchanged':True,'patient_EDES_records_complete':True,'cause':'missing read-only preparation reference after all evaluation forwards','clock_and_caps_unchanged':True,'old_endpoints_unchanged':True})
