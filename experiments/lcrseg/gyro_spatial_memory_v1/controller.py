"""Sequential bounded source gates, A/B/C and result-gated D/E."""
import json,os,subprocess,time,traceback,signal
from pathlib import Path
from .runtime import guard
from ..main_style_ssl_v1.runtime import write


def main(cfg):
    root=Path(cfg['root']);guard(cfg,new_stage=True)
    with (root/'CONTROLLER_STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().split()[21],time=time.time()),f)
    def launch(mode,stage,phase='formal'):
        guard(cfg,500_000_000,new_stage=True)
        current=dict(cfg,stage=stage,arm=stage,phase=phase,phase_cap=32 if phase=='engineering' else 2000,checkpoint_each_step=True,save_exception_state=True,diagnostic_holdout_excluded=False,audit_source_training=True)
        target=stage in ('A','B','C','D','E')
        output=root/('qualification' if phase=='engineering' else 'jobs' if target else '')/stage
        if not target and phase=='formal':output=root/stage.lower() if stage=='codec' else root/stage
        current['output']=str(output)
        if target:current['memory']=str(root/('C0' if stage in ('A','B') else 'C1')/'memory.pt')
        path=root/('SETTINGS_'+stage+'_'+phase+'.json');write(path,current)
        name=mode+'_'+stage+'_'+phase
        write(root/'STATUS.json',dict(status='RUNNING',stage=stage,phase=phase,mode=mode,time=time.time()))
        env=dict(os.environ,TASK_SETTINGS=str(path),TASK_MODE=mode,CUDA_VISIBLE_DEVICES='3')
        free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
        q=root/'qualification'/stage/'ENGINEERING.json'
        required=json.loads(q.read_text())['peak_bytes']*1.25+1024**3 if q.exists() else 24*1024**3
        if free<required:raise MemoryError('GPU3 admission failed; do not alter batch')
        with (root/(name+'.log')).open('a') as log:
            p=subprocess.Popen([cfg['runtime'],cfg['entry']],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active=dict(pid=p.pid,birth=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],mode=mode,stage=stage,phase=phase,started=time.time())
        write(root/'ACTIVE.json',active);write(root/('ACTIVE_'+name+'.json'),active)
        code=p.wait();write(root/('EXIT_'+name+'.json'),dict(exit_code=code,time=time.time()))
        if code:raise RuntimeError('stage failed: '+name)
        return current
    try:
        resume=cfg.get('resume_after_C0',False)
        if resume:
            assert json.loads((root/'codec/QUALITY.json').read_text())['status']=='PASS'
            assert json.loads((root/'C0/EXIT.json').read_text())['exit_code']==0
            c=json.loads((root/'budget_counter.json').read_text())
            assert c['by_phase']['C0:formal']==512 and c['pending']==0
            stages=('C1',)
        else:
            launch('source','codec','engineering');launch('source','codec')
            stages=('C0','C1')
        for stage in stages:
            launch('source',stage,'engineering');launch('source',stage)
        for arm in ('A','B','C'):
            launch('qualify',arm,'engineering')
        for arm in ('A','B','C'):
            current=launch('train',arm);launch('evaluate',arm)
        a=json.loads((root/'jobs/A/RESULTS.json').read_text());c=json.loads((root/'jobs/C/RESULTS.json').read_text())
        old=100*(c['Siemens']['primary_dice']-a['Siemens']['primary_dice']);new=100*(c['Philips']['primary_dice']-a['Philips']['primary_dice'])
        proceed=old>=.5 and new>=-.5
        write(root/'ABC_DECISION.json',dict(C_minus_A_Siemens_pp=old,C_minus_A_Philips_pp=new,run_DE=proceed))
        if proceed:
            for arm in ('D','E'):launch('qualify',arm,'engineering')
            for arm in ('D','E'):launch('train',arm);launch('evaluate',arm)
        write(root/'STATUS.json',dict(status='COMPLETE',DE_executed=proceed,storage=guard(cfg),time=time.time(),audit_pending=True))
    except BaseException as e:
        write(root/'CONTROLLER_FAILURE.json',dict(error=str(e),traceback=traceback.format_exc(),time=time.time()))
        write(root/'STATUS.json',dict(status='STOPPED',error=str(e),time=time.time()));raise
