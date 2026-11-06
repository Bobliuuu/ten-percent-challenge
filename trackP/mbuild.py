import modal,json,sys,os
R=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
app=modal.App("tp-build")
img=(modal.Image.debian_slim().pip_install("torch","transformers","datasets","numpy")
 .add_local_file(R+"/splits.json","/root/tp/splits.json").add_local_file(R+"/trackA/lib.py","/root/tp/trackA/lib.py")
 .add_local_file(R+"/trackG/dev2.py","/root/tp/trackG/dev2.py").add_local_file(R+"/trackG/dev2.json","/root/tp/trackG/dev2.json").add_local_file(R+"/trackG/dev2_hard.json","/root/tp/trackG/dev2_hard.json")
 .add_local_file(R+"/trackP/tokfreq.npy","/root/tp/trackP/tokfreq.npy")
 .add_local_file(R+"/trackP/core.py","/root/tp/trackP/core.py").add_local_file(R+"/trackP/build.py","/root/tp/trackP/build.py")
 .add_local_file(R+"/trackP/fmt.py","/root/tp/trackP/fmt.py").add_local_file(R+"/trackP/rans.py","/root/tp/trackP/rans.py"))
@app.function(image=img,gpu=["L4","A10G","T4","L40S"],timeout=3000,max_containers=30,cpu=4)
def run(job):
    import sys; sys.path.insert(0,"/root/tp/trackP"); os.chdir("/root/tp/trackP")
    import build as Bd, time
    t=time.time(); out="/tmp/m.bin"; r=Bd.build(job["cfg"],out); r["secs"]=time.time()-t; r["name"]=job["name"]
    return r,(open(out,"rb").read() if job.get("keep") else None)
@app.local_entrypoint()
def main(jobs:str,out:str):
    J=json.load(open(jobs)); res=[]
    for r,b in run.map(J):
        print(json.dumps(r),flush=True); res.append(r)
        if b: open(os.path.splitext(out)[0]+"_"+r["name"]+".bin","wb").write(b)
    json.dump(res,open(out,"w"),indent=1)
