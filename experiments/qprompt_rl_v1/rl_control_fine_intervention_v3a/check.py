"""Zero-optimizer preparation check; fabricated feedback only, no medical imports."""
import json,sys,tempfile,types
from pathlib import Path
from . import control

def main():
    result=control.check();package=__package__;stub=types.ModuleType(package+'.execution');stub.CODE='check';stub.CONFIG_SHA='check';stub.CONFIG={'backbones':['A','B']};stub.read=lambda p:json.loads(Path(p).read_text())
    def write(p,v):
        Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,allow_nan=False))
    stub.compatible=stub.read;stub.write=write;stub.records=lambda p:[];stub.c=None;stub.PREVIOUS=None
    with tempfile.TemporaryDirectory() as tmp:
        stub.ROOT=Path(tmp);sys.modules[stub.__name__]=stub
        from . import analysis as a
        a.ROOT.mkdir(exist_ok=True);manifest={f'{b}__{d}':{} for b in ('A','B') for d in ('X','Y')};write(a.ROOT/'CONTEXT_MANIFEST.private.json',manifest)
        for cell in manifest:
            for j in range(16):
                for action in range(3):
                    for repeat in range(2 if j in (0,8) and action in (0,2) else 1):
                        score=[.2,-.1,0][action];row=dict(scene=j,action=action,repeat=repeat,code_commit='check',root_restored=True,retained_after_discard=0,state=[0.]*16,horizons={str(h):dict(online=score,audit=score) for h in (1,5)},steps=[dict(info=dict(LU=0.,U_grad_norm=0.,grad_norm=1.,coverage={'1':.5,'2':.4}),teacher_student_l2=.1) for _ in range(5)])
                        write(a.ROOT/'panels'/cell/f'{j}_{action}_{repeat}.private.json',row)
        a.panels();values=stub.read(a.ROOT/'P1_VALUES.private.json');assert len(values)==4 and all(len(v)==32 for v in values.values());assert all(v[0]['audit']==[.2,-.1,0.] for v in values.values())
        pair=a.paired([(0,1.,2.),(1,2.,3.),(2,3.,4.),(1,2.,3.)]);assert pair['duplicate_records']==1 and pair['transfer']==0 and pair['online_0']==-2
        assert not a.paired([(0,1.,1.)])['fine']
        a.table('OOF_POLICY_VALUES.csv',[dict(cell=cell,h=5,method=m,value=.02 if m=='FI_POLICY' else 0.) for cell in manifest for m in ('FINE','STATIC_PRIOR','NC_OPT','LIN_VALUE','FI_POLICY','FI_POLICY_SHUFFLE')]);a.report();assert stub.read(a.ROOT/'reports/PRIMARY_RESULT.json')['closed_loop_proposal_criterion']
        # Missing feedback remains missing and prevents a complete four-cell result.
        p=next((a.ROOT/'panels'/'A__X').glob('*.private.json'));v=stub.read(p);v['horizons']['1']['audit']=None;write(p,v);a.panels();assert 'A__X' not in stub.read(a.ROOT/'P1_VALUES.private.json')
    result.update(fabricated_panel_aggregation=True,missing_feedback_preserved=True,report_gate=True,medical_data_reads=0);print(json.dumps(result))

if __name__=='__main__':main()
