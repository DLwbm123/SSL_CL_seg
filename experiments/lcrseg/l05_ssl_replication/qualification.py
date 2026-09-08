import argparse,json,os,platform,time,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from experiments.lcrseg.single_teacher_scd_v0_1.engine import write_json
from .freeze import verify

def main(output,device,reference,development=False):
    os.environ['SCD_TEST_DEVICE']=device;os.environ['SCD_REFERENCE']=reference
    from experiments.lcrseg.tests.single_teacher_scd_v0_1.test_contract import Contract as Old
    from experiments.lcrseg.tests.l05_ssl_replication.test_contract import Contract
    import subprocess
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() if development else verify()
    root=Path(output);root.mkdir()
    counter=[0];original=torch.optim.Adam.step
    def counted(*a,**kw):
        result=original(*a,**kw);counter[0]+=1;return result
    names=('test_01_scoring','test_02_03_binding_and_isolation','test_05_geometry_prototype_pas')
    suite=unittest.TestSuite([*(Old(n) for n in names),*unittest.defaultTestLoader.loadTestsFromTestCase(Contract)])
    start=time.time()
    with patch.object(torch.optim.Adam,'step',counted):result=unittest.TextTestRunner(verbosity=2).run(suite)
    out=dict(status='PASS' if result.wasSuccessful() else 'FAIL',source=source,development=development,device=device,python=platform.python_version(),torch=torch.__version__,tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),synthetic_optimizer_updates=counter[0],formal_updates=0,real_diagnostic_updates=0,started_unix=start,finished_unix=time.time(),inherited_scope=list(names),new_scope=unittest.defaultTestLoader.getTestCaseNames(Contract),excluded_scope='Old SCD projector/solver tests and old namespace-freeze tests are not run or claimed. No SCD repair.',two_case_overfit='No new real-data overfit training; synthetic production-chain updates, loss-gradient parity and exact resume qualify the scoped changes.')
    write_json(root/'qualification.json',out);print(json.dumps(out),flush=True)
    if out['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for x in ('output','device','reference'):p.add_argument('--'+x,required=True)
    p.add_argument('--development',action='store_true');main(**vars(p.parse_args()))
