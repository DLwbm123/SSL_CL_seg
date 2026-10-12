"""Region-balanced corresponding-pixel feature orientation, adapted from LUCIR Eq6."""
import torch
from ..f5_hierarchy_proto_v21.hierarchy import validate_mapping


def dense_terms(student,teacher,labels,parent_map):
    validate_mapping(parent_map)
    if student.ndim not in (4,5) or student.shape!=teacher.shape or min(student.shape)<1:
        raise ValueError('matching nonempty 2D/3D feature grids and channels required')
    if labels.shape!=student.shape[:1]+student.shape[2:]:raise ValueError('labels must share batch/spatial grid')
    if student.dtype not in (torch.float32,torch.float64) or teacher.dtype!=student.dtype:raise ValueError('dense supports matching float32/float64 features')
    if labels.dtype!=torch.long or labels.device!=student.device or teacher.device!=student.device:raise ValueError('integer labels and features on one device required')
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():raise FloatingPointError('nonfinite dense features')
    if ((labels!=255)&((labels<0)|(labels>=len(parent_map)))).any():raise ValueError('label outside ancestry map')
    s=student.flatten(2);t=teacher.detach().flatten(2);sn=torch.linalg.vector_norm(s,dim=1);tn=torch.linalg.vector_norm(t,dim=1)
    if not torch.isfinite(sn).all() or not torch.isfinite(tn).all():raise FloatingPointError('nonfinite dense feature norm')
    leaf=labels.flatten(1);valid=(leaf!=255)&(sn>1e-12)&(tn>1e-12)
    pixel=(1-((s/sn[:,None].clamp_min(1e-12))*(t/tn[:,None].clamp_min(1e-12))).sum(1)).clamp_min(0)
    mapping=torch.tensor(tuple(parent_map)+(255,),device=student.device)
    parent=mapping[torch.where(leaf==255,len(parent_map),leaf)];groups=[];regions=[]
    for target,classes in ((leaf,range(len(parent_map))),(parent,[p for p in range(max(parent_map)+1) if parent_map.count(p)>1])):
        per_class=[]
        for c in classes:
            mask=valid&(target==c);count=mask.sum(1);present=count>0
            if present.any():per_class.append(((pixel*mask).sum(1)/count.clamp_min(1))[present].mean())
        regions.append(len(per_class))
        if per_class:groups.append(torch.stack(per_class).mean())
    loss=torch.stack(groups).mean() if groups else student.sum()*0
    return dict(dense_loss=loss),dict(dense_leaf_regions=regions[0],dense_parent_regions=regions[1],dense_valid_pixels=int(valid.sum()),dense_zero_pixels=int(((leaf!=255)&~valid).sum()))
