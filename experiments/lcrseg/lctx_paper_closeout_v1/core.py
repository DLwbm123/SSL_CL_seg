"""Matched source scoring on the original complementary-input/update path."""
import contextlib
from unittest.mock import patch
from experiments.lcrseg.ams_seq_transfer_v0_1 import core as parent
from experiments.lcrseg.ams_seq_transfer_v0_1.core import *
from experiments.lcrseg.dpr_v0_1.core import parameters, fixed_state
from experiments.lcrseg.lctx_weight_memory_v0_1.core import LAYERS, configure, hash_state, training_access

ARMS=('C_CE','C_CED','C_FULLMIX','C_LCTX_LOW','C_FULLMIX_LOW')

def strength(epoch):return 0.

def mixed_labels(y,m):
    if y.dtype!=torch.long or m.dtype!=torch.bool or y.shape!=m.shape or len(y)!=2:
        raise ValueError('two integer label maps and boolean masks required')
    yd=y.flip(0)
    return torch.cat((torch.where(m,y,yd),torch.where(m,yd,y)))

def orders(n,seed,domain,epoch,stream,steps):
    if n<2:raise ValueError('two distinct patients required')
    if n%2==0:return parent.orders(n,seed,domain,epoch,stream,steps)
    result=[];cycle=0
    while len(result)<steps:
        pairs=[v for _,v in batch_indices(n,2,shuffle=True,seed_parts=(seed,domain,epoch,cycle,stream,0))]
        pairs[-1].append(pairs[0][0]);result.extend(pairs);cycle+=1
    return result[:steps]

@contextlib.contextmanager
def selected_data(selected):
    """Filter metadata before any sample/label opens; never load then discard."""
    native=CurrentData.__init__
    def init(ds,*a,**kw):
        native(ds,*a,**kw)
        if ds.role!='train_labeled':raise PermissionError('current L only')
        allowed=set(selected)
        ds.rows=[r for r in ds.rows if r['case_id'] in allowed]
        if len(ds.rows)!=len(allowed) or len(ds.rows)<2:raise ValueError('subset not nested in current L')
    with patch.object(CurrentData,'__init__',init):yield

def step(model,ema,opt,l,probe,task,epoch,index,counts,patients,diagnostic=False):
    arm=task['arm']
    if arm not in ARMS or probe is not None:raise PermissionError('registered L-only recipe required')
    if len(patients)!=2 or len(set(patients))!=2:raise ValueError('distinct-patient batch required')
    full=arm in ('C_FULLMIX','C_FULLMIX_LOW') and epoch>20
    recipe={'C_CE':'T_CE','C_CED':'T_CED'}.get(arm,'T_LCTX')
    native=parent.mathcore.supervised_parts;targets={}
    def collect(a,b,m):
        targets['labels']=mixed_labels(l['label'],m)
        return torch.cat((a,b)),None
    def supervision(logp,y):
        if y is not l['label'] or 'labels' not in targets:raise RuntimeError('unexpected source scoring call')
        return native(logp,targets['labels'])
    ctx=contextlib.ExitStack()
    with ctx:
        if full:
            ctx.enter_context(patch.object(parent.mathcore,'collect_sources',collect))
            ctx.enter_context(patch.object(parent.mathcore,'supervised_parts',supervision))
        row=parent.step(model,ema,opt,l,None,recipe,task['seed'],task['domain'],epoch,index,counts,patients)
    y=targets.get('labels',l['label']);valid=int((y!=255).sum())
    row.update(checks={'finite_loss':math.isfinite(row['loss']),'zero_U':row['U_image_records']==0},
        response_VJP=0,response_forwards=0,pseudo_labels=0,EMA_updates=1,
        labeled_source_scoring_multiplicity=2 if full else 1,
        raw_valid_supervision_occurrences=valid,CE_valid_pixel_denominator=max(valid,1),
        Dice_grouping='synthetic_image_foreground_class' if full else 'source_image_foreground_class',
        Dice_image_denominator=len(y),Dice_class_denominator=2,
        normalized_raw_coefficient_without_ignore=.5 if full else 1.,
        exact_pixel_weight_match_claim=False,donor_labels_supervised=full)
    return row,{}
