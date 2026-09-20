"""Zero-optimizer regression for the study-global quota ledger."""
import json
import hashlib
import tempfile
from pathlib import Path
from . import budget
from .budget import charge, close_attempt, reserve_attempt


def main():
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'ledger.json'
        identity = {
            'study_id':'MAIN_HEAD_HIERARCHICAL_KL_V1',
            'registry_id':'MAIN_HEAD_HKL_LEDGER_REGISTRY_R1',
            'ledger_id':'MAIN_HEAD_HIERARCHICAL_KL_V1_CPU_LEDGER_R1',
            'authorization_id':'MAIN_HEAD_HKL_PREPARATION_AUTH_R1',
            'source_commit':'historical preparation evidence',
            'historical_evidence':['REPORT.json','COUNTS.json'],
            'historical_evidence_bindings':[
                {'path':'REPORT.json','sha256':'report'},
                {'path':'COUNTS.json','sha256':'counts'}],
            'historical_optimizer_calls':15,'historical_attempts':1}
        path.write_text(json.dumps(dict(identity, **{
            'schema':1,'study_id':'MAIN_HEAD_HIERARCHICAL_KL_V1',
            'optimizer_calls':15,
            'attempts':[{'label':'historical attempt 1','optimizer_calls':15,'status':'PASS'}]
        })))
        registry = Path(directory)/'registry.json'
        registry.write_text(json.dumps(dict(identity, schema=1,
            canonical_ledger_path=str(path))))
        old_digest = budget.EXPECTED_REGISTERED_REGISTRY_SHA256
        budget.EXPECTED_REGISTERED_REGISTRY_SHA256 = hashlib.sha256(registry.read_bytes()).hexdigest()
        index=reserve_attempt(path,'repair regression',registry_path=registry)
        assert index==1
        assert charge(path,index,registry_path=registry)==16
        close_attempt(path,index,'FAIL',registry_path=registry)
        assert json.loads(path.read_text())['optimizer_calls']==16
        try: reserve_attempt(path,'third output root',registry_path=registry)
        except RuntimeError: pass
        else: raise AssertionError('attempt reset admitted')
        bad=Path(directory)/'bad.json';bad.write_text('{broken')
        try: reserve_attempt(bad,'corrupt',registry_path=registry)
        except RuntimeError: pass
        else: raise AssertionError('corrupt ledger admitted')
        replacement = Path(directory)/'replacement.json'
        replacement.write_text(path.read_text().replace('"optimizer_calls": 16', '"optimizer_calls": 0'))
        try: reserve_attempt(replacement,'replacement',registry_path=registry)
        except RuntimeError: pass
        else: raise AssertionError('unregistered ledger path admitted')
        wrong_registry = Path(directory)/'wrong_registry.json'
        wrong_registry.write_text(json.dumps(dict(identity, schema=1,
            canonical_ledger_path=str(path), ledger_id='UNREGISTERED_REPLACEMENT')))
        try: reserve_attempt(path,'wrong identity',registry_path=wrong_registry)
        except RuntimeError: pass
        else: raise AssertionError('wrong ledger identity admitted')
        degraded = Path(directory)/'degraded.json'
        degraded.write_text(path.read_text().replace('"historical_optimizer_calls": 15',
            '"historical_optimizer_calls": 0'))
        try: reserve_attempt(degraded,'degraded history',registry_path=registry)
        except RuntimeError: pass
        else: raise AssertionError('degraded historical binding admitted')
        budget.EXPECTED_REGISTERED_REGISTRY_SHA256 = old_digest
    print('budget regression PASS: replacement path, wrong identity, degraded history, corrupt ledger and attempt cap rejected; optimizer calls=0')


if __name__=='__main__': main()
