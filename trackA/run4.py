import sys; sys.argv=[sys.argv[0],"x"]
sys.path.insert(0,'/Users/jhome/ten-percent/trackA')
from gq import *
import time
for fr in [0.9,0.8,0.7]:
    m=prune_ffn(fresh(),fr); print(rep2(m,f"FFN keep {fr} fp32 (no quant)"),flush=True)
