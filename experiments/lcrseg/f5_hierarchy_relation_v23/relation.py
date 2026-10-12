"""RKD distance/angle potentials over label-configured whole-part prototype edges."""
import itertools
import torch
from torch.nn import functional as F
from ..f5_hierarchy_proto_v21.hierarchy import validate_mapping,prototypes


def node_prototypes(features,labels,parent_map):
    flat=features.flatten(2);norm=torch.linalg.vector_norm(flat,dim=1,keepdim=True)
    if not torch.isfinite(norm).all():raise FloatingPointError('nonfinite relation feature norm')
    z=flat/norm.clamp_min(1e-12);leaf=labels.flatten(1)
    mapping=torch.tensor(tuple(parent_map)+(255,),device=features.device)
    parent=mapping[torch.where(leaf==255,len(parent_map),leaf)]
    lp,ls,_=prototypes(z,leaf,len(parent_map));hp,hs,_=prototypes(z,parent,max(parent_map)+1)
    return torch.cat((lp,hp)),torch.cat((ls,hs))


def relation_terms(student,teacher,labels,parent_map):
    validate_mapping(parent_map)
    if student.ndim not in (4,5) or teacher.ndim!=student.ndim or min(student.shape)<1 or min(teacher.shape)<1:
        raise ValueError('nonempty 2D/3D feature grids required')
    if student.shape[:1]+student.shape[2:]!=teacher.shape[:1]+teacher.shape[2:] or labels.shape!=student.shape[:1]+student.shape[2:]:
        raise ValueError('student/teacher/labels must share batch/spatial grid')
    if student.dtype not in (torch.float32,torch.float64) or teacher.dtype not in (torch.float32,torch.float64):raise ValueError('relation supports float32/float64 features')
    if labels.dtype!=torch.long or labels.device!=student.device or teacher.device!=student.device or not student.is_floating_point() or not teacher.is_floating_point():
        raise ValueError('floating features and integer labels on one device required')
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():raise FloatingPointError('nonfinite relation features')
    if ((labels!=255)&((labels<0)|(labels>=len(parent_map)))).any():raise ValueError('label outside ancestry map')
    s,ss=node_prototypes(student,labels,parent_map);t,ts=node_prototypes(teacher.detach(),labels,parent_map);valid=ss&ts
    leaves=len(parent_map);parents=max(parent_map)+1;edges=[];triples=[]
    for p,q in itertools.combinations(range(parents),2):edges.append((leaves+p,leaves+q,'coarse'))
    for p in range(parents):
        children=[c for c,v in enumerate(parent_map) if v==p]
        if len(children)<2:continue
        edges.extend((c,leaves+p,'parent_child') for c in children)
        for c,d in itertools.combinations(children,2):
            edges.append((c,d,'sibling'));triples.append((c,leaves+p,d))
    edges=[e for e in edges if valid[e[0]] and valid[e[1]] and (t[e[0]]-t[e[1]]).norm()>1e-12]
    zero=student.sum()*0;distance=zero;angle=zero
    if edges:
        sd=torch.stack([(s[a]-s[b]).norm() for a,b,_ in edges]);td=torch.stack([(t[a]-t[b]).norm() for a,b,_ in edges])
        if sd.detach().mean()>1e-12:
            loss=F.smooth_l1_loss(sd/sd.mean(),td/td.mean(),reduction='none',beta=1.)
            distance=torch.stack([loss[torch.tensor([e[2]==kind for e in edges],device=s.device)].mean() for kind in sorted({e[2] for e in edges})]).mean()
    supported=[]
    for a,b,c in triples:
        if not (valid[a] and valid[b] and valid[c]):continue
        sv1,sv2=s[a]-s[b],s[c]-s[b];tv1,tv2=t[a]-t[b],t[c]-t[b]
        if any(v.detach().norm()<=1e-12 for v in (sv1,sv2,tv1,tv2)):continue
        sa=F.cosine_similarity(sv1,sv2,dim=0,eps=1e-12);ta=F.cosine_similarity(tv1,tv2,dim=0,eps=1e-12)
        supported.append((b,F.smooth_l1_loss(sa,ta,beta=1.)))
    if supported:
        angle=torch.stack([torch.stack([v for b,v in supported if b==p]).mean() for p in sorted({b for b,_ in supported})]).mean()
    return dict(relation_distance=distance,relation_angle=angle),dict(relation_edges=len(edges),relation_triplets=len(supported),relation_supported_nodes=int(valid.sum()),relation_valid_pixels=int((labels!=255).sum()))
