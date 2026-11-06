# generate vq_local.py (plain-python version of vq.py run(), for a local GPU box)
import re
s=open('vq.py').read()
i=s.index("def run(cfg):"); j=s.index("@app.local_entrypoint()")
body=s[i:j]
body=body.replace("rvol.commit()","pass").replace('f"/res','f"{RES}').replace('"/res','RES+"').replace('"/root/','DATA+"/')
body=body.replace("os.environ[\"HF_HOME\"]=\"/hf\"","pass")
head='''import os, sys, json
RES=os.environ.get("VQ_RES","/home/ubuntu/vq/res"); DATA=os.environ.get("VQ_DATA","/home/ubuntu/vq/data")
os.makedirs(RES,exist_ok=True)
'''
tail='''
if __name__=="__main__":
    for c in json.load(open(sys.argv[1])):
        if len(sys.argv)>2 and c["name"] not in sys.argv[2].split(","): continue
        if os.path.exists(f"{RES}/{c['name']}.json"): continue
        run(c)
'''
open('vq_local.py','w').write(head+body+tail)
