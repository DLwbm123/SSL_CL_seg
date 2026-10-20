"""Zero-update allow/deny check using the same canonical asset resolver as Accessor."""
from pathlib import Path
import h5py
from .data import Domain
from ..main_style_ssl_v1.data import metadata
from ..di_dmpa_gate1.binding import safe_asset
from ..main_style_ssl_v1.runtime import write

def main(cfg):
    data=Domain(cfg,'Philips');x,y,_=data.labeled(0);u,_,_=data.unlabeled(0)
    assert x.shape==u.shape==(2,1,384,384) and y.shape==(2,384,384)
    prohibited=[r for r in metadata(cfg['data'],'mnms') if (r['site_or_vendor']=='Siemens' and r['primary_20pct_split']=='train_labeled') or (r['site_or_vendor']=='Philips' and r['primary_20pct_split']=='val')]
    rejected=0
    for r in prohibited:
        try:
            h5py.File(safe_asset(Path(cfg['data']),r['image_h5_relpath']),'r')
        except PermissionError:rejected+=1
        else:raise AssertionError('forbidden historical/validation HDF5 was admitted')
    assert rejected==len(prohibited)>0
    write(Path(cfg['root'])/'ACCESS_REPAIR_CHECK.json',dict(status='PASS',optimizer_calls=0,functional_native_updates=0,Philips_L_U_read_success=True,source_train_and_Philips_val_rejected=rejected,forbidden_file_contents_read=False,canonical_resolver_used=True,U_hidden_GT_access=False))
