"""Zero-optimizer-call control-flow and checkpoint-policy checks."""
import ast,json
from pathlib import Path
from ..gyro_native_v1.module import self_check
from ..gyro_source_repair_v11 import core


def check(cfg):
    assert Path(core.__file__).is_relative_to(Path(cfg['code']))
    from ..philips_transfer_v1 import run as engine
    assert Path(engine.__file__).is_relative_to(Path(cfg['code']))
    assert self_check()['status']=='PASS'
    # Exercise the actual threshold conditional with a ratio above the old limit.
    tree=ast.parse(Path(core.__file__).read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='losses')
    condition=next(n for n in fn.body if isinstance(n,ast.If) and any(isinstance(v,ast.Name) and v.id=='limit' for v in ast.walk(n.test)))
    for limit,expected in [(.25,True),(None,False)]:
        actual=eval(compile(ast.Expression(condition.test),'<guard>','eval'),{},dict(arm='GYRO_low',limit=limit,ratio=1.681781334))
        assert actual==expected
    assert cfg['gradient_ratio_hard_stop'] is None
    assert cfg['checkpoint_each_step'] and cfg['save_exception_state']
    Path(cfg['root'],'SYNTHETIC_CHECKS.json').write_text(json.dumps(dict(status='PASS',optimizer_calls=0,default_guard_preserved=True,ratio_diagnostic_only=True)))
