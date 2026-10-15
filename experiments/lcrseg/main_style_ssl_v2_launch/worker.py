import os,torch
# Host policy only; frozen method and optimizer options are untouched.
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
from experiments.lcrseg.main_style_ssl_v2.entry import main
main()
