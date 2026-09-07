"""Dataset assets use inherited h5/v1 binding; protocol paths keep io_r1 semantics."""
import hashlib
import io
import os
from pathlib import Path
import stat

import h5py
import numpy as np
from di_dmpa_gate1.binding import safe_asset
from .io_r1 import project_training
from .scoring_r1 import labels

ASSET_COLUMNS = ('case_id','patient_id','primary_20pct_split','label_h5_relpath','label_sha256')


def dataset_asset(data_root, relative):
    if not isinstance(relative,str) or not relative.strip() or relative in ('.','./'):
        raise ValueError('empty dataset asset')
    rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts: raise ValueError('unsafe dataset asset')
    root=(Path(data_root)/'h5/v1').resolve()
    target=safe_asset(data_root,relative).resolve()
    if target==root or not target.is_relative_to(root): raise ValueError('dataset asset escapes canonical root')
    if target.exists() and not target.is_file(): raise ValueError('dataset asset is not a file')
    return target


def counters():
    names=('asset_metadata_rows_checked','asset_unique_files_stat_checked','GT_payload_open_attempts',
           'GT_payload_open_successes','GT_payload_reads_completed','GT_file_hash_checks_passed',
           'GT_label_decode_attempts','GT_label_decodes_completed','GT_rows_bound_to_validated_labels',
           'GT_unique_files_decoded','true_domain_records_materialized','true_domain_rows_processed',
           'baseline_case_policy_comparisons_completed','baseline_scalar_comparisons_completed',
           'oracle_case_rows_completed','new_sample_expert_forwards','new_batch_forwards',
           'optimizer_updates','segmentation_updates','EMA_updates','GAS_updates','prototype_updates',
           'old_formal_03_reads','own_seed_forbidden_GT_reads')
    return {**{name:0 for name in names},'reused_sample_expert_outputs':594,
            'real_fits_by_kind':{name:0 for name in ('Ridge','PAV','temperature','router','risk_head','conformal')}}


def projected_domains(path, columns, wanted, count):
    rows=project_training(path,columns,selected_ids=wanted)
    count['true_domain_records_materialized']+=len(rows)
    return rows


def load_label(data_root, row, count, decoded_files=None):
    # No caching: one complete original-file read/hash/decode per authorized seed-row.
    if row['primary_20pct_split']!='train_labeled': raise ValueError('own-seed GT role denied')
    target=dataset_asset(data_root,row['label_h5_relpath'])
    count['GT_payload_open_attempts']+=1
    with target.open('rb') as stream:
        count['GT_payload_open_successes']+=1
        st=os.fstat(stream.fileno()); physical=(st.st_dev,st.st_ino)
        payload=stream.read()
        count['GT_payload_reads_completed']+=1
    if hashlib.sha256(payload).hexdigest()!=row['label_sha256']: raise ValueError('GT file hash mismatch')
    count['GT_file_hash_checks_passed']+=1
    count['GT_label_decode_attempts']+=1
    with h5py.File(io.BytesIO(payload),'r') as handle:
        if 'label' not in handle or not isinstance(handle['label'],h5py.Dataset): raise ValueError('missing label dataset')
        label=np.asarray(handle['label'][...])
    count['GT_label_decodes_completed']+=1
    if decoded_files is None or physical not in decoded_files:
        count['GT_unique_files_decoded']+=1
        if decoded_files is not None:decoded_files.add(physical)
    if label.shape!=(384,384): raise ValueError('GT shape mismatch')
    labels(np.zeros((384,384),dtype=np.uint8),label)
    count['GT_rows_bound_to_validated_labels']+=1
    return label, target


def asset_preflight(data_root, rows, count):
    """Fixed 198-row metadata-only projection, no true domain and no HDF5 access."""
    result=[]; errors=[]; seen=set(); selected={}
    for seed in range(3):
        wanted={r['case_id']:r for r in rows if r['seed']==seed}
        records=project_training(Path(data_root)/'manifests/training'/f'lcrseg_v1_seed{seed}.csv',ASSET_COLUMNS,selected_ids=wanted)
        for r in records:
            key=(seed,r['case_id'])
            if key in selected: errors.append({'seed':seed,'case_id':r['case_id'],'error':'duplicate manifest row'})
            selected[key]=r
    for blind in rows:
        count['asset_metadata_rows_checked']+=1
        try:
            r=selected[blind['seed'],blind['case_id']]
            if r['patient_id']!=blind['patient_id'] or r['primary_20pct_split']!='train_labeled':raise ValueError('own-seed identity/role mismatch')
            if len(r['label_sha256'])!=64 or any(c not in '0123456789abcdef' for c in r['label_sha256']):raise ValueError('invalid expected label hash')
            path=dataset_asset(data_root,r['label_h5_relpath'])
            st=path.stat()
            if not stat.S_ISREG(st.st_mode) or st.st_size<=0:raise ValueError('missing/empty/non-file asset')
            physical=(st.st_dev,st.st_ino)
            if physical not in seen:count['asset_unique_files_stat_checked']+=1;seen.add(physical)
            result.append({**r,'seed':blind['seed'],'row_index':blind['row_index'],'canonical_path':str(path),
                           'bytes':st.st_size,'device':st.st_dev,'inode':st.st_ino})
        except (OSError,ValueError,KeyError) as exc:
            errors.append({'row_index':blind['row_index'],'exception':type(exc).__name__,'message':str(exc)})
    return result,errors
