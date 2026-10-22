"""Original global Bayesian method: source learnability before any CL claim."""
import copy,json,os,time,traceback
from pathlib import Path
import torch
from lcrseg.data import H5LabeledDataset,H5UnlabeledDataset
from lcrseg.engine.trainer import Trainer,TrainerState,seed_everything
from sc_bayeseg.model import Model
from sc_bayeseg.method import Method
from sc_bayeseg.run import UniformBatcher,snapshot,restore,same
from sc_bayeseg.sslcl_v1.train import evaluate,setup
from sc_bayeseg.window_v1.r_start import write,append,save

class Prediction(torch.nn.Module):
    def __init__(self, model):
        super().__init__();self.model=model
    def probability(self,x):
        return self.model.prediction(x,corrected=True)[0]

def run(s):
    setup();seed_everything(163);out=Path(s['output']);out.mkdir(exist_ok=False);write(out/'CONTEXT.json',s)
    start=time.time();actual=0;method=trainer=None
    try:
        opts=dict(data_root=s['data'],seed=0,dataset=s['dataset'],sites=[s['source_domain']],transform=None)
        l,u=H5LabeledDataset(**opts),H5UnlabeledDataset(**opts)
        if s['dataset']=='mnms':
            from rcf_seg_study_v1.data import datasets
            l,u=datasets(s['data'],s['source_domain'])
        assert all(not r.get('label_h5_relpath') for r in u.records)
        lb,ub=UniformBatcher(l,True),UniformBatcher(u,False)
        model=Model(num_classes=s['num_classes'],input_channels=s['input_channels']).cuda();method=Method(model,len(l),len(u));method.begin_site(s['source_domain'],None,1000)
        opt=torch.optim.Adam(model.parameters(),lr=s['learning_rate'],weight_decay=0)
        trainer=Trainer(method,optimizer=opt,scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.),device='cuda',amp=False,grad_clip_norm=1.)
        assert len(model.bayes_layers())==6 and not hasattr(model,'backbone')
        initial=snapshot(method,trainer,0)
        def update(i):
            assert time.time()-s['original_started_unix']<85200
            return trainer.train_step(lb.batch_at(i),ub.batch_at(i),state=TrainerState(global_step=i,site_step=i))
        def receipt(*args):
            nonlocal actual
            actual+=1;append(out/'updates.jsonl',dict(physical=actual,time=time.time()))
        def guard(*args):
            assert actual<1002
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        pre=opt.register_step_pre_hook(guard);post=opt.register_step_post_hook(receipt)
        a=update(0);state=snapshot(method,trainer,1)
        for layer in model.bayes_layers():
            for key in ['mean_w','mean_b','logsd_w','logsd_b']:assert getattr(layer,key).grad is not None
        model.set_prior(True)
        for layer in model.bayes_layers():
            for part in ['w','b']:
                for kind in ['mean','logsd']:assert torch.equal(getattr(layer,'prior_'+kind+'_'+part),getattr(layer,kind+'_'+part))
        restore(initial,method,trainer);b=update(0);assert torch.equal(a.total_loss,b.total_loss) and same(state,snapshot(method,trainer,1));restore(initial,method,trainer)
        write(out/'QUALIFICATION.json',dict(status='PASS',global_bayesian_layers=6,global_mean_and_variance_gradients=True,posterior_as_prior_exact=True,full_state_golden=True,engineering_updates=2))
        prior={k:v.clone() for k,v in model.state_dict().items() if 'prior_' in k or 'initial_' in k}
        for i in range(1000):
            loss=update(i)
            if (i+1)%25==0:write(out/'STATUS.json',dict(status='TRAINING',step=i+1,loss=float(loss.total_loss.detach()),calls=model.calls,elapsed=time.time()-start))
            if (i+1)%250==0:save(out/'latest.pt',snapshot(method,trainer,i+1))
        assert actual==1002 and all(torch.equal(v,model.state_dict()[k]) for k,v in prior.items())
        pre.remove();post.remove();save(out/'final.pt',snapshot(method,trainer,1000))
        write(out/'STATUS.json',dict(status='EVALUATING',step=1000))
        seed_everything(50163);before=copy.deepcopy(model.state_dict());metrics=evaluate(Prediction(model),s,out);assert same(before,model.state_dict())
        d=metrics[s['source_domain']];classes=[d[c]['Dice'] for c in ['cup','disc']] if s['dataset']=='fundus' else d['class_Dice'];main=d['mean_Dice'];admit=main>=.5 and all(v is not None and v>=.2 for v in classes)
        write(out/'RESULTS.json',dict(status='COMPLETE',updates=1000,engineering_updates=2,physical_updates=actual,metrics=metrics,source_learnability_gate=admit,model_calls=model.calls,elapsed=time.time()-start,method='SC_BayeSeg_global_Bayesian_original_objective',test_access=False))
        write(out/'STATUS.json',dict(status='COMPLETE',step=1000));write(out/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if method is not None and trainer is not None:save(out/'failure_partial.pt',snapshot(method,trainer,max(0,actual-2)))
        write(out/'FAILURE.json',dict(actual_updates=actual,traceback=traceback.format_exc()));write(out/'EXIT.json',dict(exit_code=1));raise
if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
