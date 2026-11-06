"""Run all trackJ experiments in parallel on Modal GPUs.  `modal run modal_run.py` -> results.jsonl"""
import modal, json
app=modal.App("trackJ")
img=(modal.Image.debian_slim().pip_install("torch","transformers","scipy","numpy")
     .add_local_dir("dat","/root/dat").add_local_file("tj.py","/root/tj.py"))
@app.function(image=img,gpu=["L4","A10G","T4","L40S"],timeout=3000,max_containers=15)
def go(cfg):
    import os,sys,importlib; os.environ["D2DIR"]="/root/dat"; os.environ["KEEP"]="/root/dat/vocab_keep.json"; sys.path.insert(0,"/root")
    import tj; r=tj.run(cfg); r["cfg"]={k:v for k,v in cfg.items() if k!="name"}; return r
def cfgs():
    out=[]
    for tag,base in [("w3g64",dict(bits=3,g=64)),("w2g64",dict(bits=2,g=64)),("A",dict(recipe="A",bits=3,g=128))]:
        base=dict(base,clip=1)
        C=[("gptq",{}),("rtn",dict(method="rtn")),("+bias",dict(bias=1)),("+bias+scale",dict(bias=1,scale=1)),("+bias+scale+ln",dict(bias=1,scale=1,ln=1)),
           ("+ln only",dict(ln=1)),("+rot256",dict(rot=256)),("+rot64",dict(rot=64)),("+awq",dict(awq=1)),("+cle",dict(cle=1)),("+noclip",dict(clip=0)),
           ("+rot+bias+scale+ln",dict(rot=256,bias=1,scale=1,ln=1)),("+awq+bias+scale+ln",dict(awq=1,bias=1,scale=1,ln=1)),
           ("+rot+awq+bias+scale+ln",dict(rot=256,awq=1,bias=1,scale=1,ln=1)),("+cle+rot+bias+scale+ln",dict(cle=1,rot=256,bias=1,scale=1,ln=1))]
        for n,x in C: out.append(dict(base,**x,name=f"{tag} {n}"))
    return out
@app.local_entrypoint()
def main():
    cs=cfgs()
    with open("results.jsonl","w") as f:
        for r in go.map(cs,return_exceptions=True):
            f.write(json.dumps(r,default=str)+"\n"); f.flush(); print(r if not isinstance(r,dict) else (r["name"],r["bits_per_w"],r["agree2"],r["agree_hard"]))
# round 2 (stacking): `modal run modal_run.py --round2`
def cfgs2():
    out=[]
    for tag,base in [("A",dict(recipe="A",bits=3,g=128,clip=1)),("w3g64",dict(bits=3,g=64,clip=1))]:
        for n,x in [("+rot256+ln",dict(rot=256,ln=1)),("+rot256+awq",dict(rot=256,awq=1)),("+rot256+awq+ln",dict(rot=256,awq=1,ln=1)),("+rot128",dict(rot=128)),("+rot512",dict(rot=256*2)) ,("+rot256+scale",dict(rot=256,scale=1))]:
            out.append(dict(base,**x,name=f"{tag} {n}"))
    return out
@app.local_entrypoint()
def round2():
    cs=cfgs2()
    with open("results2.jsonl","w") as f:
        for r in go.map(cs,return_exceptions=True):
            f.write(json.dumps(r,default=str)+"\n"); f.flush(); print(r if not isinstance(r,dict) else (r["name"],r["bits_per_w"],r["agree2"],r["agree_hard"]))
