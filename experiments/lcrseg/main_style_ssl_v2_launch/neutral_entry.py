import os,json,pathlib,sys,runpy
c=json.loads(pathlib.Path(os.environ['TASK_SETTINGS']).read_text());sys.path[:0]=[c['code'],c['base']];runpy.run_path(os.environ['HOST_SCRIPT'],run_name='__main__')
