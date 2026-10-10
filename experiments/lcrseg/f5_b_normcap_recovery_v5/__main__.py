import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
from experiments.lcrseg.f5_b_normcap_recovery_v5.run import main
if __name__=="__main__":main()
