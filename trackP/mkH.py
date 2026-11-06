from core import *
D=data(); m=fullbase()
# Hessian calibration: SST-2 train dev[:1000] (domain of exam) -- unlabeled use only
H=collect_H(m,D["B_dev"]); torch.save(H,HERE+"/H_fp.pt"); print(len(H))
