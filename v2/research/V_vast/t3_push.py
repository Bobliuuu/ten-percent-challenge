import json,subprocess,sys,os
HERE=os.path.dirname(os.path.abspath(__file__)); K=HERE+"/.vast_ssh_key"; label=sys.argv[1]; cmd=sys.argv[2]
D=json.loads(subprocess.run(["vastai","show","instances","--raw"],capture_output=True,text=True).stdout)
i=[x for x in D if x.get("label")=="tenpct-"+label][0]; p=str(i["ssh_port"]); h=i["ssh_host"]
e=f"ssh -i {K} -p {p} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
# code/ and data/ both land in /workspace/bundle root: fix layout
subprocess.run(f"rsync -az -e '{e}' {HERE}/bundles/t3/code/ root@{h}:/workspace/bundle/code/; rsync -az -e '{e}' {HERE}/bundles/t3/data/ root@{h}:/workspace/bundle/data/",shell=True)
r=subprocess.run(["ssh","-i",K,"-p",p,"-o","StrictHostKeyChecking=no","-o","UserKnownHostsFile=/dev/null","-o","LogLevel=ERROR",f"root@{h}",f"cd /workspace/bundle && (setsid nohup bash -c '{cmd}' > /workspace/run_r2.log 2>&1 < /dev/null &) ; sleep 1; echo launched"],capture_output=True,text=True,timeout=120)
print(label,r.stdout.strip(),r.stderr[:100])
