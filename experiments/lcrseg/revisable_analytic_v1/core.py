"""Fixed-coordinate, revisable latent-class squared-risk solver. No autograd."""
import torch

RIDGE = .001
LAMBDA = .25
TAU = 1.
ITERATIONS = 20
ARMS = ('l_only', 'fixed_q', 'revisable_q')
record_solve = lambda: None


def solve(g, c):
    record_solve()
    result = torch.linalg.solve(g + RIDGE*torch.eye(len(g), device=g.device, dtype=g.dtype), c)
    if not torch.isfinite(result).all():
        raise FloatingPointError('nonfinite analytic head')
    return result


def responsibility(mu, w):
    return (2*mu@w/TAU).softmax(1)


def cross(memory, q):
    return memory['mu'].T @ (memory['n'][:, None]*q)


def objective(w, gl, cl, gu, memory, q):
    # The omitted supervised constant is identical for every conditional update.
    return ((w*(gl@w)).sum()-2*(w*cl).sum()+RIDGE*w.square().sum()
            + LAMBDA*((w*(gu@w)).sum()-2*(w*cross(memory,q)).sum()
            + memory['n'].sum()+TAU*(memory['n'][:,None]*q*q.clamp_min(1e-300).log()).sum()))


def join(memories):
    return {key:torch.cat([m[key] for m in memories]) for key in ('n','mu','v')}


def optimize(gl, cl, gu, memory, initial, fixed_q=None, report=lambda *args: None):
    w = initial.clone()
    for iteration in range(ITERATIONS):
        q = responsibility(memory['mu'], w) if fixed_q is None else fixed_q
        before = float(objective(w,gl,cl,gu,memory,q))
        candidate = solve(gl+LAMBDA*gu, cl+LAMBDA*cross(memory,q))
        after = float(objective(candidate,gl,cl,gu,memory,q))
        if after > before + 1e-9*(1+abs(before)):
            raise FloatingPointError('conditional analytic objective increased')
        report(iteration+1, before, after, float((candidate-w).norm()))
        w = candidate
    return w


def summarize(x, weights, clusters=256):
    """Current-domain feature pool is discarded after compression; never save it."""
    if len(x)<clusters or (weights<=0).any():
        raise ValueError('invalid current feature pool')
    pool=x[torch.linspace(0,len(x)-1,min(8192,len(x)),device=x.device).long()]
    chosen=[0]; distances=(pool-pool[0]).square().sum(1)
    for _ in range(1,clusters):
        i=int(distances.argmax()); chosen.append(i)
        distances=torch.minimum(distances,(pool-pool[i]).square().sum(1))
    centers=pool[chosen].double()
    g=torch.zeros(x.shape[1],x.shape[1],dtype=torch.float64,device=x.device)
    # Final assignment, not the provisional centers, defines the saved partition.
    for iteration in range(3):
        n=torch.zeros(clusters,dtype=torch.float64,device=x.device)
        first=torch.zeros_like(centers); second=torch.zeros_like(n)
        for start in range(0,len(x),4096):
            block=x[start:start+4096].double(); weight=weights[start:start+4096]
            distance=block.square().sum(1)[:,None]+centers.square().sum(1)[None]-2*block@centers.T
            assignment=distance.argmin(1)
            membership=torch.nn.functional.one_hot(assignment,clusters).double()*weight[:,None]
            n+=membership.sum(0); first+=membership.T@block
            second+=membership.T@block.square().sum(1)
            if iteration==2: g+=block.T@(weight[:,None]*block)
        occupied=n>0
        centers[occupied]=first[occupied]/n[occupied,None]
    v=(second[occupied]/n[occupied]-centers[occupied].square().sum(1)).clamp_min(0)
    memory=dict(n=n[occupied],mu=centers[occupied],v=v)
    if abs(float(n.sum())-1)>1e-8 or not torch.isfinite(g).all():
        raise ValueError('invalid U state')
    return g,memory


def compression_bound(memory,w):
    lipschitz=float(torch.linalg.matrix_norm(w,ord=2))/TAU
    return lipschitz*float((memory['n']*(memory['v']+memory['mu'].norm(dim=1)*memory['v'].sqrt())).sum())
