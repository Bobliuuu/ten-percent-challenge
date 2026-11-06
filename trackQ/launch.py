import json, subprocess, sys, time
C10=dict(name="uoft-cs/cifar10",img="img",label="label",test="test",train="train")
C100=dict(name="uoft-cs/cifar100",img="img",label="fine_label",test="test",train="train")
IN=dict(name="benjamin-paine/imagenet-1k",img="image",label="label",imagenet=True)
cfgs=[{"base":1},{"method":"gptq","bits":4,"g":128},{"method":"gptq","bits":3,"g":128}]+[{"method":"gptq","alpha":a,"g":128} for a in (1.0,0.85,0.7,0.55)]+[{"method":"gptq","alpha":0.7,"g":128,"sp":0.3},{"method":"gptq","alpha":0.85,"g":128,"sp":0.3},{"method":"rtn","alpha":0.7,"g":128}]
M={ "vitb_c10":("nateraw/vit-base-patch16-224-cifar10","hf",C10),
 "vitb_c100":("edadaltocg/vit_base_patch16_224_in21k_ft_cifar100","timm",C100),
 "r50_c100":("edadaltocg/resnet50_cifar100","timm",C100),
 "r50_c10":("edadaltocg/resnet50_cifar10","timm",C10),
 "cnxb_c10":("tanganke/convnext-base-224_cifar10_sgd_batch-size-64_lr-0.01_steps-4000","hf",C10),
 "vitl_c100":("jialicheng/cifar100_vit-large","hf",C100),
 "in_vitb":("timm/vit_base_patch16_224.augreg2_in21k_ft_in1k","timm",IN),
 "in_vitl":("timm/vit_large_patch16_224.augreg_in21k_ft_in1k","timm",IN),
 "in_deitb":("timm/deit_base_patch16_224.fb_in1k","timm",IN),
 "in_cnxb":("timm/convnext_base.fb_in22k_ft_in1k","timm",IN),
 "in_r50":("timm/resnet50.tv_in1k","timm",IN),
 "in_r101":("timm/resnet101.tv_in1k","timm",IN)}
for k in sys.argv[1:]:
    mid,kind,d=M[k]; spec=dict(model=mid,kind=kind,data=d,cfgs=cfgs,ncal=512,splits=["dev"])
    open(f"out/{k}_2.spec.json","w").write(json.dumps(spec))
    time.sleep(12)
    subprocess.Popen(f". ../.venv/bin/activate; modal run q.py --spec \"$(cat out/{k}_2.spec.json)\" --out out/{k}_2.json > logs/{k}_2.log 2>&1",shell=True)
