"""Fixtures place assets from this independent specification, never the resolver."""
from pathlib import Path
import hashlib
import h5py
import numpy as np
import pytest

from care_hr_v0_7_1.io_r1 import digest, safe_path, project_training
from care_hr_v0_7_1.io_r2 import dataset_asset, load_label, counters, projected_domains


def put(path, value=1, shape=(384,384), dtype=np.uint8, key='label'):
    path.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(path,'w') as f:f.create_dataset(key,data=np.full(shape,value,dtype=dtype),compression='gzip')
    return dict(label_h5_relpath='labels/synthetic.h5',label_sha256=digest(path),primary_20pct_split='train_labeled')


def test_canonical_only_old_expression_fails_and_production_loader_succeeds(tmp_path):
    row=put(tmp_path/'h5/v1/labels/synthetic.h5')
    with pytest.raises(FileNotFoundError):safe_path(tmp_path,row['label_h5_relpath']).read_bytes()
    count=counters();label,_=load_label(tmp_path,row,count)
    assert np.all(label==1) and count['GT_label_decodes_completed']==count['GT_rows_bound_to_validated_labels']==1
    assert count['GT_payload_open_attempts']==count['GT_payload_open_successes']==count['GT_file_hash_checks_passed']==1


def test_wrong_root_only_never_falls_back(tmp_path):
    row=put(tmp_path/'labels/synthetic.h5');count=counters()
    with pytest.raises(FileNotFoundError):load_label(tmp_path,row,count)
    assert count['GT_payload_open_attempts']==1 and count['GT_payload_open_successes']==count['GT_label_decodes_completed']==0


def test_decoy_content_modification_and_deletion_never_change_binding(tmp_path):
    row=put(tmp_path/'h5/v1/labels/synthetic.h5',2);decoy=tmp_path/'labels/synthetic.h5';put(decoy,0)
    for change in (lambda:None,lambda:decoy.write_bytes(b'not hdf5'),decoy.unlink):
        change();label,path=load_label(tmp_path,row,counters());assert np.all(label==2) and path==tmp_path/'h5/v1/labels/synthetic.h5'


def test_dataset_manifest_and_protocol_roots_coexist(tmp_path):
    row=put(tmp_path/'h5/v1/labels/synthetic.h5')
    manifest=tmp_path/'manifests/training/seed.csv';manifest.parent.mkdir(parents=True)
    manifest.write_text('case_id,patient_id,primary_20pct_split,image_h5_relpath,image_sha256\na,p,train_labeled,img,h\n')
    protocol=tmp_path/'protocol';protocol.mkdir();(protocol/'candidate_C0.npy').write_bytes(b'protocol')
    assert project_training(manifest)[0]['case_id']=='a'
    assert safe_path(protocol,'candidate_C0.npy').read_bytes()==b'protocol'
    assert dataset_asset(tmp_path,row['label_h5_relpath'])==tmp_path/'h5/v1/labels/synthetic.h5'


@pytest.mark.parametrize('relative',('', ' ', '.', '../label.h5','/tmp/label.h5','labels/../../label.h5'))
def test_unsafe_dataset_relative_rejected(tmp_path,relative):
    with pytest.raises(ValueError):dataset_asset(tmp_path,relative)


def test_directory_escape_symlink_and_registered_data_root_symlink(tmp_path):
    data=tmp_path/'registered';row=put(data/'h5/v1/labels/synthetic.h5')
    with pytest.raises(ValueError):dataset_asset(data,'labels')
    (data/'h5/v1/escape').symlink_to(tmp_path/'outside')
    with pytest.raises(ValueError):dataset_asset(data,'escape/file.h5')
    link=tmp_path/'data_alias';link.symlink_to(data,target_is_directory=True)
    assert dataset_asset(link,row['label_h5_relpath'])==data/'h5/v1/labels/synthetic.h5'
    assert np.all(load_label(link,row,counters())[0]==1)


@pytest.mark.parametrize('failure',('missing','hash','key','shape','dtype','value','corrupt','all_ignore'))
def test_payload_validation_failure_counters_never_fallback(tmp_path,failure):
    p=tmp_path/'h5/v1/labels/synthetic.h5'
    kwargs={'key':{'key':'not_label'},'shape':{'shape':(1,2)},'dtype':{'dtype':np.float32},'value':{'value':3},'all_ignore':{'value':255}}.get(failure,{})
    row=put(p,**kwargs);count=counters()
    if failure=='missing':p.unlink()
    if failure=='hash':row['label_sha256']='0'*64
    if failure=='corrupt':p.write_bytes(b'corrupt hdf5');row['label_sha256']=digest(p)
    if failure=='all_ignore':
        label,_=load_label(tmp_path,row,count);assert np.all(label==255) and count['GT_rows_bound_to_validated_labels']==1
    else:
        with pytest.raises((ValueError,OSError,KeyError)):load_label(tmp_path,row,count)
        assert count['GT_rows_bound_to_validated_labels']==0
        if failure in ('missing','hash'):assert count['GT_file_hash_checks_passed']==count['GT_label_decode_attempts']==0
        if failure=='missing':assert count['GT_payload_open_attempts']==1 and count['GT_payload_open_successes']==0


@pytest.mark.parametrize('failure',('open','read'))
def test_actual_open_and_read_boundaries(tmp_path,monkeypatch,failure):
    path=tmp_path/'h5/v1/labels/synthetic.h5';row=put(path);original=Path.open;count=counters()
    class BadRead:
        def __init__(self,f):self.f=f
        def __enter__(self):return self
        def __exit__(self,*args):self.f.close()
        def fileno(self):return self.f.fileno()
        def read(self):raise OSError('synthetic interrupted read')
    def replacement(p,*args,**kwargs):
        if p==path:
            if failure=='open':raise PermissionError('synthetic denied')
            return BadRead(original(p,*args,**kwargs))
        return original(p,*args,**kwargs)
    monkeypatch.setattr(Path,'open',replacement)
    with pytest.raises(OSError):load_label(tmp_path,row,count)
    assert count['GT_payload_open_attempts']==1 and count['GT_payload_reads_completed']==0
    assert count['GT_payload_open_successes']==(failure=='read')


def test_materialization_counts_66_even_if_one_row_processed(tmp_path):
    p=tmp_path/'manifest.csv';p.write_text('case_id,site_or_vendor\n'+''.join(f'{i},domain\n' for i in range(66)))
    count=counters();rows=projected_domains(p,('case_id','site_or_vendor'),{str(i) for i in range(66)},count)
    for r in rows:count['true_domain_rows_processed']+=1;break
    assert count['true_domain_records_materialized']==66 and count['true_domain_rows_processed']==1
