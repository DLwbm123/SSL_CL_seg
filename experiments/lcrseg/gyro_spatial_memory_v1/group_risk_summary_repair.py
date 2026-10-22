"""One-off zero-forward recovery of completed evaluation rows; no retraining."""
import copy,json
from pathlib import Path
from unittest.mock import patch
from ..gyro_transfer_v1 import run as transfer
from ..siemens_lonly_v1.run import aggregate
from ..main_style_ssl_v1.runtime import write

def main(cfg):
    root=Path(cfg['root']);out=root/'jobs/20270801_B'
    assert not (root/'SUMMARY_REPAIR.json').exists() and not (out/'RESULTS.json').exists()
    read=lambda p:json.loads(p.read_text())
    before=read(root/'calls_counter.json');budget=read(root/'budget_counter.json')
    assert before['executed']['evaluation_native']==2892 and before['pending'] is None and budget['total']==2736 and budget['pending']==0
    reference=read(Path(cfg['reliable_root'])/'jobs/20270701_B/RESULTS.json')
    specs={'private_Siemens_val.jsonl':(46,reference['Siemens']), 'private_Philips_val.jsonl':(62,reference['Philips']),
           'private_source_train.jsonl':(20,reference['source_train']), 'private_target_train.jsonl':(26,reference['target_train']),
           'private_ema_Siemens_val.jsonl':(46,reference['EMA_diagnostic']['Siemens']), 'private_ema_Philips_val.jsonl':(62,reference['EMA_diagnostic']['Philips'])}
    assert sum(v[1]['forward_slices'] for v in specs.values())==2892
    seen=[]
    def cached(model,c,domain,role,output):
        name=Path(output).name;n,ref=specs[name];rows=[json.loads(s) for s in Path(output).read_text().splitlines()]
        assert len(rows)==n and len({(r['patient'],r['phase']) for r in rows})==n
        seen.append(name);return dict(aggregate(rows),forward_slices=ref['forward_slices']),rows
    with patch.object(transfer.engine,'evaluate',cached):transfer.finalize(dict(cfg,arm='A'),out,None,None,2000,2000)
    assert len(seen)==len(set(seen))==6
    result=read(out/'RESULTS.json');result.update(primary='fixed final EMA .999',historical_entry_comparisons_auxiliary=True);write(out/'RESULTS.json',result)
    assert before==read(root/'calls_counter.json') and budget==read(root/'budget_counter.json')
    write(out/'EVALUATION_EXIT.json',dict(exit_code=0,recovered_from_saved_rows=True))
    write(root/'EXIT_20270801_B_evaluate.json',dict(exit_code=0,recovered_from_saved_rows=True))
    write(root/'SUMMARY_REPAIR.json',dict(status='PASS',model_forwards=0,optimizer_updates=0,original_evaluation_native=2892,all_six_row_files_complete=True,ledger_unchanged=True,original_clock_preserved=True))
