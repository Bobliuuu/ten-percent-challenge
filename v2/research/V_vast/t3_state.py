import json,subprocess,os,sys
HERE=os.path.dirname(os.path.abspath(__file__)); K=HERE+"/.vast_ssh_key"
for _ in range(10):
    r=subprocess.run(["vastai","show","instances","--raw"],capture_output=True,text=True)
    try: D=json.loads(r.stdout); break
    except Exception: pass
for i in D:
    if str(i.get("label","")).startswith("tenpct-t3"):
        try:
            r=subprocess.run(["ssh","-i",K,"-p",str(i["ssh_port"]),"-o","StrictHostKeyChecking=no","-o","UserKnownHostsFile=/dev/null","-o","LogLevel=ERROR","-o","ConnectTimeout=15",f"root@{i['ssh_host']}","ps aux | grep -E 'runner.py' | grep -v grep | awk '{print $12,$13,$14}'; ls /workspace/bundle/results/dense | tr '\\n' ' '; tail -c 150 /workspace/run_r2.log 2>/dev/null"],capture_output=True,text=True,timeout=45)
            print(i["label"],"|",r.stdout.replace("\n"," ; ")[:400],r.stderr[:80])
        except Exception as e: print(i["label"],"ssh fail",type(e).__name__)
