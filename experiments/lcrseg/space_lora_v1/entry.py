import os,sys,json,unittest
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
cfg=json.loads(Path(os.environ['TASK_SETTINGS']).read_text())
sys.path.insert(0,cfg['base'])
import experiments.lcrseg
experiments.lcrseg.__path__=[str(Path(cfg['code'])/'experiments/lcrseg'),*experiments.lcrseg.__path__]
mode=os.environ['TASK_MODE']
if mode=='prepare':
 from experiments.lcrseg.main_style_ssl_v1.data import Accessor
 patients=Accessor(cfg['data'],'mnms','Philips','train_labeled').patients
 with Path(cfg['split']).open('x') as f: json.dump(dict(train=patients,holdout=[]),f,indent=2)
 from experiments.lcrseg.space_lora_v1 import test_contract
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_contract))
 raise SystemExit(not result.wasSuccessful())
if mode=='statistics':
 from experiments.lcrseg.space_lora_v1.statistics import main
elif mode=='controller':
 from experiments.lcrseg.space_lora_v1.controller import main
elif mode=='qualify':
 from experiments.lcrseg.space_lora_v1.qualification import qualify as main
elif mode=='train':
 from experiments.lcrseg.space_lora_v1.run import main
elif mode=='continue':
 from experiments.lcrseg.space_lora_v1.continuation import main
else: raise PermissionError('unknown mode')
main(cfg)
