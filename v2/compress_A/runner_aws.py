"""Local multi-GPU runner for sweep_A.run on an AWS box. usage: python runner_aws.py cfgs.json outdir  (one worker per GPU, pulls configs from a queue)."""
import sys, os, json, subprocess, time
cfgs=json.load(open(sys.argv[1])); out=sys.argv[2]; os.makedirs(out,exist_ok=True)
ng=int(os.environ.get("NGPU","4"))
if len(sys.argv)>3 and sys.argv[3]=="worker":
    sys.path.insert(0,"."); import sweep_A as S
    f=S.run.get_raw_f(); c=json.loads(sys.argv[4]); r=f(c); json.dump(r,open(f"{out}/{c['name']}.json","w")); sys.exit(0)
pending=[c for c in cfgs if not os.path.exists(f"{out}/{c['name']}.json")]; running={}
while pending or running:
    for g,(p,c) in list(running.items()):
        if p.poll() is not None: del running[g]
    for g in range(ng):
        if g not in running and pending:
            c=pending.pop(0)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),HF_HOME="/hf")
            lf=open(f"{out}/{c['name']}.log","w")
            running[g]=(subprocess.Popen([sys.executable,__file__,sys.argv[1],out,"worker",json.dumps(c)],env=env,stdout=lf,stderr=subprocess.STDOUT),c)
    time.sleep(5)
