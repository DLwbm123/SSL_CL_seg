"""Logical stage/order is independent of physical domain index; val is separate."""
import copy,math
import torch
from experiments.lcrseg.single_teacher_scd_v0_1 import data as primitive
from .recipes import generator
from .gate import digest

ORDERS=(('REFUGE','RIM_ONE_r3','Drishti_GS'),('REFUGE','Drishti_GS','RIM_ONE_r3'))


def inspect(data):
    rows=primitive.metadata(data)
    counts={d:tuple(sum(r['site_or_vendor']==d and r['primary_20pct_split']==role for r in rows)
                    for role in ('train_labeled','train_unlabeled')) for d in primitive.DOMAINS}
    if list(counts.values())!=list(primitive.COUNTS):raise ValueError('frozen L/U counts differ from designation')
    return rows,counts


class NativeCurrentDomain:
    def __init__(self,data,seed,order,stage,stage_source,device,permit,allow_u=True):
        permit.validate()
        if order not in (1,2) or stage not in (0,1,2):raise ValueError('invalid logical order/stage')
        self.seed,self.order,self.stage=seed,order,stage
        self.stage_source=copy.deepcopy(stage_source);self.device=device;self.l_reads=self.u_reads=0
        self.domain=ORDERS[order-1][stage];self.size=384
        rows,counts=inspect(data);self.steps_per_epoch=max(math.ceil(n/2) for n in counts[self.domain])
        self._identity={'domain':self.domain,'seed':seed,'order':order,'stage':stage,
                       'manifest':primitive.MANIFEST_SHA,'split':primitive.SPLIT_SHA}
        if digest(self._identity) not in permit.bindings['authorized_manifest_digests']:
            raise PermissionError('data domain/seed/order/stage outside capability')
        index=primitive.DOMAINS.index(self.domain)
        self._l=primitive.CurrentData(data,index,'train_labeled')
        self._u=primitive.CurrentData(data,index,'train_unlabeled') if allow_u and stage else None
        self._patients={r['case_id']:r['patient_id'] for r in rows if r['site_or_vendor']==self.domain and r['primary_20pct_split']=='train_labeled'}

    def semantic_metadata(self):
        return {**self._identity,'stage_source':copy.deepcopy(self.stage_source),'stream':'stateless_native_order_epoch_no_arm',
                'shape':[384,384],'batch_size':2,'allow_u':self._u is not None,'precision':'FP32'}

    def _batch(self,ds,step,stream):
        epoch,index=divmod(step,self.steps_per_epoch);n=len(ds)
        # Two distinct current sources; cyclic shuffled pairs for an odd U count.
        cycle,offset=divmod(2*index,n)
        order=torch.randperm(n,generator=generator(self.seed,self.order,self.stage,epoch,f'{stream}/order/{cycle}')).tolist()
        ii=[order[offset],order[(offset+1)%n]]
        g=generator(self.seed,self.order,self.stage,step,stream+'/geometry')
        values=[]
        for i in ii:
            item=ds[i];x,y,v=primitive.geometry(item['image'],item.get('label'),item['geometry'],g)
            values.append((x,y,v))
        x=torch.stack([v[0] for v in values]).to(self.device)
        other=torch.stack([v[1] if ds.role=='train_labeled' else v[2] for v in values]).to(self.device)
        ids=tuple(self._patients[ds.rows[i]['case_id']] if ds.role=='train_labeled' else ds.rows[i]['case_id'] for i in ii)
        return x,other,ids

    def labeled(self,step,stream='labeled'):
        self.l_reads+=1;return self._batch(self._l,step,stream)
    def unlabeled(self,step):
        if self._u is None:raise PermissionError('U not granted to source/L-only/smoke')
        self.u_reads+=1;return self._batch(self._u,step,'unlabeled')


def evaluation_data(data,domain,seen):
    if domain not in seen or not set(seen)<=set(primitive.DOMAINS):raise PermissionError('unseen evaluation domain')
    index=primitive.DOMAINS.index(domain)
    return primitive.CurrentData(data,max(primitive.DOMAINS.index(d) for d in seen),'val',purpose='evaluate',domain=index)
