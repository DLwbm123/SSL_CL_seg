import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
from experiments.lcrseg.f5_conflict_reject_v7.run import main
if __name__=="__main__":main()
