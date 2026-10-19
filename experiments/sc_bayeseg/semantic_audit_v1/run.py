"""Read-only semantic Gaussian swaps and gradient decomposition; no optimizer."""
import copy,json,os,time,traceback,math
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import seed_everything
from care_seg_study_v1.run import write
from sc_bayeseg.model import normal_log_prob
from sc_bayeseg.split_v1.model import Model,semantic
from sc_bayeseg.diagnostic_v1.run import dice
from sc_bayeseg.run import same


def summary(a):
    a=a.detach().float().flatten()
    return dict(mean=float(a.mean()),quantiles=[float(v) for v in torch.quantile(a,torch.tensor([.05,.25,.5,.75,.95],device=a.device))],negative_fraction=float((a<0).float().mean()))


def metric(p,y):
    pred=p.argmax(1);matrix=torch.bincount((y*4+pred).flatten(),minlength=16).reshape(4,4)
    d=dice(pred,y)
    return dict(macro_dice=sum(map(sum,d))/6,class_dice_mean=torch.tensor(d).mean(0).tolist(),class_dice_min=torch.tensor(d).min(0).values.tolist(),
                recall=(matrix.diag()[1:]/matrix.sum(1)[1:]).tolist(),precision=(matrix.diag()[1:]/matrix.sum(0)[1:].clamp_min(1)).tolist(),
                confusion=matrix.cpu().tolist(),class2_destinations=matrix[2].cpu().tolist(),
                ce=float(F.nll_loss(p.clamp_min(1e-30).log(),y,reduction='none').mean()),background_fraction=float((pred==0).float().mean()))


def margins(z,y):
    mask=y==2;true=z.gather(1,y[:,None])[:,0]
    other=z.masked_fill(F.one_hot(y,4).permute(0,3,1,2).bool(),-torch.inf)
    competitor=other.argmax(1);value=true-other.max(1).values
    return dict(class2=summary(value[mask]),class2_competitors=torch.bincount(competitor[mask],minlength=4).tolist()),value,competitor


def norm(a):return float(torch.linalg.vector_norm(a.double()))
def cos(a,b):
    den=norm(a)*norm(b)
    return float(torch.dot(a.double(),b.double())/den) if den else None


def split_groups(named,grads):
    groups={k:[] for k in ['semantic_frontend','logit_shared','mean_head','rawsd_head']}
    for (n,p),g in zip(named,grads):
        v=torch.zeros_like(p) if g is None else g.detach()
        if n.startswith('shape_encoder.'):groups['semantic_frontend'].append(v.flatten())
        elif n.startswith('logit_encoder.7.'):
            groups['mean_head'].append(v[:3].flatten());groups['rawsd_head'].append(v[3:].flatten())
        else:groups['logit_shared'].append(v.flatten())
    return {k:torch.cat(v).cpu() for k,v in groups.items()}


def gradient_audit(model,image,y):
    for n,p in model.named_parameters():p.requires_grad_(semantic(n))
    named=[(n,p) for n,p in model.named_parameters() if semantic(n)];params=[p for _,p in named]
    records=[];reverse=0;names=['labels','image','shape','logits','code','appearance','logq_x','logq_m','logq_ell','logq_c','total']
    def grad(*args,**kwargs):
        nonlocal reverse
        reverse+=1;assert reverse<=100
        return torch.autograd.grad(*args,**kwargs)
    saturation=None
    for draw in range(4):
        seed_everything(95163+draw);model.draw()
        v=model.shape_encoder(image).chunk(2,1)[0];mu,raw=model.logit_encoder(v).chunk(2,1);sd=-4+3*torch.tanh(raw/3)
        def gaussian(raw):
            mean,r=raw.chunk(2,1);s=-4+3*torch.tanh(r/3);a=mean+s.exp()*torch.randn_like(mean)
            return a,normal_log_prob(a,mean,s)
        x=gaussian(model.physical_encoder(image));m=gaussian(model.appearance_encoder(image))
        ell=mu+sd.exp()*torch.randn_like(mu);qe=normal_log_prob(ell,mu,sd);c=gaussian(model.code_encoder(ell))
        h=dict(x=x[0],m=m[0],ell=ell,c=c[0]);_,terms,_=model.log_joint(image,h,y)
        losses={k:-v.mean()/128**2 for k,v in terms.items()}
        losses.update({k:v.mean()/128**2 for k,v in dict(logq_x=x[1],logq_m=m[1],logq_ell=qe,logq_c=c[1]).items()})
        losses['total']=sum(losses.values())+model.kl()/(2*128**2)
        fullmu=torch.cat([torch.zeros_like(mu[:,:1]),mu],1)
        _,margin,competitor=margins(fullmu,y);cmargin=margin[y==2].mean()
        mg=grad(cmargin,(mu,*params),retain_graph=True,allow_unused=True)
        margin_mu=mg[0].detach().cpu();margin_groups=split_groups(named,mg[1:])
        if saturation is None:
            derivative=1-torch.tanh(raw.detach()/3).square()
            saturation=dict(logsd=summary(sd),raw_to_logsd_derivative=summary(derivative),derivative_below_1e_6_fraction=float((derivative<1e-6).float().mean()))
        row={}
        for name in names:
            loss=losses[name]
            if loss.requires_grad:g=grad(loss,(mu,sd,*params),retain_graph=True,allow_unused=True)
            else:g=(None,)*(2+len(params))
            gu=torch.zeros_like(mu) if g[0] is None else g[0].detach()
            gs=torch.zeros_like(sd) if g[1] is None else g[1].detach()
            def vjp(out,g):
                if not torch.count_nonzero(g):return (None,)*len(params)
                return grad(out,params,grad_outputs=g,retain_graph=True,allow_unused=True)
            pu=split_groups(named,vjp(mu,gu));ps=split_groups(named,vjp(sd,gs));direct=split_groups(named,g[2:])
            residual=math.sqrt(sum(norm(direct[k]-pu[k]-ps[k])**2 for k in direct));scale=math.sqrt(sum(norm(a)**2 for a in direct.values()))
            assert residual<=1e-4*max(scale,1e-10)+1e-8
            row[name]=dict(mu=gu.cpu(),logsd=gs.cpu(),parameter=direct,mu_path=pu,logsd_path=ps,vjp_relative_residual=residual/max(scale,1e-30),value=float(loss.detach()))
        for coordinate in ['mu','logsd']:
            total=row['total'][coordinate];parts=sum(row[n][coordinate] for n in names[:-1])
            assert norm(total-parts)<=1e-4*max(norm(total),1e-10)+1e-8
        assert norm(row['logq_ell']['mu'])==0
        records.append(dict(factors=row,margin_mu=margin_mu,margin_parameter=margin_groups));model.clear_draw()
    results={}
    for name in names:
        item={};label={};margin={}
        avg=lambda key:sum(r['factors'][name][key] for r in records)/4
        for key in ['mu','logsd']:
            a=avg(key);b=sum(r['factors']['labels'][key] for r in records)/4
            item[key]=dict(norm_of_mean_gradient=norm(a),cosine_with_label=cos(a.flatten(),b.flatten()),
                           draw_norm_min=min(norm(r['factors'][name][key]) for r in records),draw_norm_max=max(norm(r['factors'][name][key]) for r in records))
            if key=='mu':
                gm=records[0]['margin_mu'];item[key]['class2_margin_derivative_along_negative_gradient']=-float((gm.double()*a.double()).sum())
                item[key]['class2_margin_derivative_unit_direction']=item[key]['class2_margin_derivative_along_negative_gradient']/norm(a) if norm(a) else 0.
        item['parameter_groups']={}
        for group in records[0]['factors'][name]['parameter']:
            entry={};lab=sum(r['factors']['labels']['parameter'][group] for r in records)/4;gm=records[0]['margin_parameter'][group]
            for path in ['parameter','mu_path','logsd_path']:
                a=sum(r['factors'][name][path][group] for r in records)/4
                entry[path]=dict(norm_of_mean_gradient=norm(a),cosine_with_label=cos(a,lab),label_derivative_along_negative_gradient=-float(torch.dot(lab.double(),a.double())),
                                 class2_margin_derivative_along_negative_gradient=-float(torch.dot(gm.double(),a.double())))
            item['parameter_groups'][group]=entry
        item['mean_value']=sum(r['factors'][name]['value'] for r in records)/4
        item['max_vjp_relative_residual']=max(r['factors'][name]['vjp_relative_residual'] for r in records)
        results[name]=item
    return dict(status='COMPLETE',draws=4,extra_reverse_calls=reverse,factors=results,saturation=saturation,interpretation='unpreconditioned local negative-gradient directions; no Adam/clip/update',generator_calls=model.calls.copy())


def run(settings):
    root=Path(settings['output']);root.mkdir(exist_ok=False)
    started=time.time();write(root/'STATUS.json',dict(status='RUNNING',optimizer_updates=0))
    def forbid(*args,**kwargs):raise AssertionError('optimizer construction or update forbidden')
    torch.optim.Optimizer.__init__=forbid;torch.optim.Adam.step=forbid
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        origin=Path(settings['origin']);assert json.loads((origin/'EXIT.json').read_text())['exit_code']==0
        data=H5LabeledDataset(settings['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        sel=json.loads((origin/'PRIVATE_SELECTION.json').read_text());lb=collate_labeled([data[i] for i in sel['train_indices']]).to('cuda')
        image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        models={};checkpoints={};coordinates={}
        for stage in ['R','J']:
            checkpoints[stage]=torch.load(origin/stage/'final.pt',map_location='cpu',weights_only=False)
            model=Model().cuda();model.load_state_dict(checkpoints[stage]['model']);models[stage]=model
            with torch.no_grad():
                mu,raw=model.logit_encoder(model.shape_encoder(image).chunk(2,1)[0]).chunk(2,1)
                coordinates[stage]=(mu.detach(),(-4+3*torch.tanh(raw/3)).detach())
        seed_everything(50163);eps=[]
        for _ in range(4):
            torch.randn_like(image);torch.randn_like(image);eps.append(torch.randn_like(coordinates['R'][0]));torch.randn((len(image),16),device=image.device)
        swaps={};mean={};values={};competitors={}
        with torch.no_grad():
            for stage,(mu,s) in coordinates.items():
                z=torch.cat([torch.zeros_like(mu[:,:1]),mu],1);mean[stage]=dict(metrics=metric(z.softmax(1),y))
                mean[stage]['margin'],values[stage],competitors[stage]=margins(z,y)
            for mu_source in ['R','J']:
                for sd_source in ['R','J']:
                    mu=coordinates[mu_source][0];s=coordinates[sd_source][1];prob=[]
                    for e in eps:
                        ell=mu+s.exp()*e;prob.append(torch.cat([torch.zeros_like(ell[:,:1]),ell],1).softmax(1))
                    p=torch.stack(prob).mean(0);key=mu_source+sd_source
                    swaps[key]=dict(metrics=metric(p,y),probability_margin=margins(p,y)[0])
                    if mu_source==sd_source:
                        historical=json.loads((origin/mu_source/'RESULTS.json').read_text())
                        assert swaps[key]['metrics']['class_dice_mean']==torch.tensor(historical['stochastic_dice']).mean(0).tolist()
                        assert abs(swaps[key]['metrics']['ce']-historical['stochastic_ce'])<1e-7
            changed=(y==2)&(values['R']>0)&(values['J']<0)
            zJ=torch.cat([torch.zeros_like(coordinates['J'][0][:,:1]),coordinates['J'][0]],1)
            transitions=dict(class2_pixels=int((y==2).sum()),R_positive_J_negative_pixels=int(changed.sum()),J_winners_for_transition=torch.bincount(zJ.argmax(1)[changed],minlength=4).tolist())
        write(root/'SWAPS.json',dict(status='COMPLETE',swap_metrics=swaps,mean_metrics=mean,transitions=transitions,logit_sample_image_count=32,semantic_encoder_image_forwards=4,historical_diagonals_reproduced=True))
        write(root/'STATUS.json',dict(status='GRADIENT_AUDIT',optimizer_updates=0))
        gradients=gradient_audit(models['R'],image,y);write(root/'GRADIENTS.json',gradients)
        for stage,model in models.items():assert all(torch.equal(v.cpu(),checkpoints[stage]['model'][k]) for k,v in model.state_dict().items())
        result=dict(status='COMPLETE',optimizer_updates=0,optimizer_created=False,model_states_unchanged=True,source_checkpoint_read_only=True,code_adaptation_run=False,
                    new_training=False,extra_reverse_calls=gradients['extra_reverse_calls'],started=started,finished=time.time())
        write(root/'RESULTS.json',result);write(root/'STATUS.json',result);write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(optimizer_updates=0,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
