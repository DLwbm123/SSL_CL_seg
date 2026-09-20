"""Zero-optimizer regression for the study-global quota ledger."""
import json
import tempfile
from pathlib import Path
from .budget import charge, close_attempt, reserve_attempt


def main():
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'ledger.json'
        path.write_text(json.dumps({
            'schema':1,'study_id':'MAIN_HEAD_HIERARCHICAL_KL_V1',
            'source_commit':'historical preparation evidence',
            'optimizer_calls':15,
            'attempts':[{'label':'historical attempt 1','optimizer_calls':15,'status':'PASS'}]
        }))
        index=reserve_attempt(path,'repair regression')
        assert index==1
        assert charge(path,index)==16
        close_attempt(path,index,'FAIL')
        assert json.loads(path.read_text())['optimizer_calls']==16
        try: reserve_attempt(path,'third output root')
        except RuntimeError: pass
        else: raise AssertionError('attempt reset admitted')
        bad=Path(directory)/'bad.json';bad.write_text('{broken')
        try: reserve_attempt(bad,'corrupt')
        except RuntimeError: pass
        else: raise AssertionError('corrupt ledger admitted')
    print('budget regression PASS: directory switch, corrupt ledger and attempt cap rejected; optimizer calls=0')


if __name__=='__main__': main()
