"""New experiment identities; frozen AMS objective invoked without modification."""
import contextlib
from unittest.mock import patch
from experiments.lcrseg.ssl_anchored_mix_v0_1 import core as mathcore
from experiments.lcrseg.ssl_anchored_mix_v0_1.core import *
from experiments.lcrseg.single_teacher_scd_v0_1.data import CurrentData, metadata

MAPPING={'SRC_CE':'SUP_CE','T_CE':'SUP_CE','T_CED':'SUP_CED','T_LCTX':'MIX_CTX','T_UCTX':'MIX_CTX','T_AMS':'MIX_CED'}

def donor_batch(l,patients):
    if len(patients)!=2 or len(set(patients))!=2 or len(l['image'])!=2:
        raise ValueError('LCTX needs two different verified patients')
    return {k:l[k].flip(0) for k in ('image','geometry')}

def step(student,teacher,opt,l,u,arm,seed,domain,epoch,index,counters,patients):
    if arm not in MAPPING:raise PermissionError('unregistered arm')
    context='NONE'
    if arm=='T_LCTX' and epoch>20:
        if u is not None:raise PermissionError('LCTX cannot read U')
        u=donor_batch(l,patients);context='L_BATCH_REVERSED'
    elif u is not None:
        if arm not in ('T_UCTX','T_AMS'):raise PermissionError('U source not admitted')
        context='CURRENT_U'
    row=mathcore.train_step(student,teacher,opt,l,u,None,None,MAPPING[arm],seed,domain,epoch,index,counters)
    row.update(context_source=context,donor_context_records=row['unlabeled_context_records'],donor_extra_image_reads=0,
               anchor_label_records=len(l['image']),U_image_records=len(u['image']) if context=='CURRENT_U' else 0)
    if context=='L_BATCH_REVERSED':
        row['L_donor_index']=row.pop('U_source_index');row['L_donor_repeat_weights']=row.pop('U_loss_repeat_weights')
        row.update(unlabeled_unique_batch_sources=0,unlabeled_context_records=0,U_geometry_pixels=0,U_accepted=None,U_class=[])
    return row

@contextlib.contextmanager
def training_access(domain):
    """Role capability applies to inherited accessors too, not just this runner."""
    original=CurrentData.__init__
    allowed=DOMAINS.index(domain)+1
    def initialize(self,data,stage,role,**kw):
        if stage!=allowed or kw.get('domain',stage)!=allowed or kw.get('purpose','train')!='train' or role not in ('train_labeled','train_unlabeled'):
            raise PermissionError('target trainer only has current-domain training capability')
        return original(self,data,stage,role,**kw)
    with patch.object(CurrentData,'__init__',initialize):yield

def patients_for(data,domain,expected):
    rows=metadata(data,expected=expected)
    selected=sorted((r for r in rows if r['site_or_vendor']==domain and r['primary_20pct_split']=='train_labeled'),key=lambda r:r['case_id'])
    patients=[r['patient_id'] for r in selected]
    if len(set(patients))!=len(patients):raise ValueError('one case per patient required')
    return patients
