"""Opt-in, exact-published-code, read-only B0 EMA probe of both transitions."""
from contextlib import ExitStack
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from di_dmpa_gate1_v2.features import ImmutableModels
from di_dmpa_gate1b_v2 import binding, plan, pairs, transport


def test_registered_RIM_and_Drishti_read_only_pairs():
    destination=os.environ.get('GATE1B_V2_INTEGRATION_OUTPUT')
    if not destination:pytest.skip('Requires separately authorized exact-code integration output')
    root=Path(__file__).resolve().parents[2];output=Path(destination);output.mkdir(parents=True,exist_ok=False)
    p,_,frozen,metadata=binding.verify(root,os.environ['GATE1B_V2_CODE_COMMIT'])
    summaries=[];audits=[]
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=binding.ProtocolError('optimizer construction forbidden in real integration')):
        with patch.object(transport,'fit_residual',side_effect=AssertionError('integration cannot fit transport')):
            for index,selected in enumerate(p['integration']['selected_cases']):
                seed,stage=selected['seed'],selected['stage_index'];device=f'cuda:{index}'
                split=next(s for s in p['transport']['split_plans'] if s['seed']==seed and s['stage_index']==stage)
                assert split['fit_case_ids'][0]==selected['case_id']
                record=plan.image_records('/root/LCRSeg',p,split)[selected['case_id']]
                case=plan.case_plan((seed,stage,record))
                unit=dict(seed=seed,stage_index=stage,domain=binding.DOMAINS[stage],role='train_unlabeled',partition='fit',
                    split_seed=split['split_seed'],split_hash=split['split_hash'],cases=[case],case_count=1,registered_count=2048)
                with ExitStack() as stack:
                    models={}
                    for side,t in [('source',stage-1),('target',stage)]:
                        cp=binding.checkpoint(p,seed,t);loaded=binding.load_b0(root,p,seed,t,device)
                        guard=output/f'stage{stage}'/side
                        stack.enter_context(ImmutableModels(loaded,cp,guard,metadata))
                        models[side]=loaded['ema_teacher']
                    arrays,case_support=pairs.extract_arrays(models['source'],models['target'],unit,'/root/LCRSeg',device)
                for side,t in [('source',stage-1),('target',stage)]:
                    audit=binding.read_json(output/f'stage{stage}'/side/'immutability'/f'B0_seed{seed}_stage{t}.json')
                    assert audit['bitwise_unchanged'] and audit['extraction_completed'];audits.append(audit)
                assert all(len(a)==2048 for a in arrays.values())
                for side in ('source','target'):
                    assert np.all(arrays[side+'_directions'][~arrays[side+'_active_mask']]==0)
                summary={k:v for k,v in case_support[0].items() if k!='first_null_coordinates'}
                summary.update(physical_gpu=index,source_checkpoint=binding.checkpoint(p,seed,stage-1),target_checkpoint=binding.checkpoint(p,seed,stage),
                    cache_rows=2048,all_registered_rows_retained=True,labels_read=False,transform_fit_called=False,model_optimizer_steps=0,transport_optimizer_steps=0)
                summaries.append(summary)
                del models,loaded
                torch.cuda.empty_cache()
    assert len(audits)==4 and len(summaries)==2
    binding.write_json(output/'GATE1B_V2_REAL_INTEGRATION.json',dict(status='PASS',metadata=metadata,units=summaries,
        model_unchanged=True,all_registered_rows_retained=True,model_load_guards=audits,model_optimizer_steps=0,transport_optimizer_steps=0,
        transport_fit_called=False,shared_coordinate_plan_materialized=False,hidden_gt_training_usage='none',test_gt_usage='none',
        note='Predetermined first fit cases, all2048 rows each; finite null counts are reported honestly even if zero.'))
