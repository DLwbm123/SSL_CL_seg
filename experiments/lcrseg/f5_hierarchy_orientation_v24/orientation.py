"""Label-configured unit prototype orientation anchoring, inspired by LUCIR Eq6."""
import torch
from ..f5_hierarchy_proto_v21.hierarchy import validate_mapping
from ..f5_hierarchy_relation_v23.relation import node_prototypes


def orientation_terms(student,teacher,labels,parent_map):
    validate_mapping(parent_map)
    if student.ndim not in (4,5) or student.shape!=teacher.shape or min(student.shape)<1:
        raise ValueError('matching nonempty 2D/3D feature grids and channels required')
    if labels.shape!=student.shape[:1]+student.shape[2:]:raise ValueError('labels must share batch/spatial grid')
    if student.dtype not in (torch.float32,torch.float64) or teacher.dtype!=student.dtype:raise ValueError('orientation supports matching float32/float64 features')
    if labels.dtype!=torch.long or labels.device!=student.device or teacher.device!=student.device:raise ValueError('integer labels and features on one device required')
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():raise FloatingPointError('nonfinite orientation features')
    if ((labels!=255)&((labels<0)|(labels>=len(parent_map)))).any():raise ValueError('label outside ancestry map')
    s,ss=node_prototypes(student,labels,parent_map);t,ts=node_prototypes(teacher.detach(),labels,parent_map);valid=ss&ts
    leaves=len(parent_map);leaf=[i for i in range(leaves) if valid[i]]
    parent=[leaves+p for p in range(max(parent_map)+1) if parent_map.count(p)>1 and valid[leaves+p]]
    groups=[]
    for indices in (leaf,parent):
        if indices:groups.append((1-(s[indices]*t[indices]).sum(1)).clamp_min(0).mean())
    loss=torch.stack(groups).mean() if groups else student.sum()*0
    return dict(orientation_loss=loss),dict(orientation_leaf_nodes=len(leaf),orientation_parent_nodes=len(parent),orientation_valid_pixels=int((labels!=255).sum()))
