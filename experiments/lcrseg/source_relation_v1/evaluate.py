"""Fixed final validation diagnostics; truth is used only after predictions."""
import numpy as np
import torch
from .method import parts,nodes,support_indices,target,grid_indices
from ..main_style_ssl_v1.data import Accessor
from ..main_style_ssl_v1.runtime import append,write
from ..siemens_lonly_v1.run import dice,aggregate


@torch.inference_mode()
def evaluate(ema,student,provider,cfg,output):
    access=Accessor(cfg['data'],'mnms','Siemens','val')
    if len(access.patients)!=23:raise ValueError('validation population changed')
    ema.eval().requires_grad_(False);student.eval().requires_grad_(False)
    rows=[];count=0;total=dict(queries=0,accepted=0,corrected=0,broken=0,accepted_correct=0,
                             accepted_by_class=[0]*4,correct_by_class=[0]*4)
    reads=provider.l_reads
    for i,meta in enumerate(access.rows):
        item=access.read(i);x=item['image'];predictions=[]
        lx,ly,_=provider.labeled(1000000+i)
        _,lf=parts(ema,lx);indices=support_indices(ly,1000000+i)
        for z in range(0,len(x),2):
            images=x[z:z+2,None].cuda()
            logits,features=parts(ema,images);predictions.append(logits.argmax(1).cpu().numpy())
            base=nodes(student(images).softmax(1))
            q,keep,_=target(features,lf,ly,base,indices)
            # Validation truth cannot influence supports, confidence, or selection.
            truth=item['label'][z:z+2].cuda().flatten(1)[:,grid_indices(images.shape,images.device)]
            old=base.argmax(-1)==truth;new=q.argmax(-1)==truth
            total['queries']+=keep.numel();total['accepted']+=int(keep.sum())
            total['corrected']+=int((keep&~old&new).sum());total['broken']+=int((keep&old&~new).sum())
            total['accepted_correct']+=int((keep&new).sum())
            for c in range(4):
                mask=keep&(q.argmax(-1)==c)
                total['accepted_by_class'][c]+=int(mask.sum());total['correct_by_class'][c]+=int((mask&new).sum())
        pred=np.concatenate(predictions)
        row=dict(patient=meta['patient_id'],phase=meta['phase'],dice=dice(pred,item['label'].numpy()).tolist())
        rows.append(row);append(output,row);count+=len(x)
    total.update(coverage=total['accepted']/max(1,total['queries']),
                 precision=total['accepted_correct']/max(1,total['accepted']),
                 diagnostic_L_batch_reads=provider.l_reads-reads,student_forward_slices=count,
                 interpretation='Final validation-node diagnostic, not training-U accuracy; training-L supports only.')
    write(output.parent/'RELATION_DIAGNOSTIC.json',total)
    return {**aggregate(rows),'forward_slices':count},rows
