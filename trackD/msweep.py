import modal, json, sys
app = modal.App("tenpct-trackD")
img = (modal.Image.debian_slim().pip_install("torch","transformers","numpy")
       .add_local_file("engine.py","/root/engine.py").add_local_file("dev2.json","/root/dev2.json")
       .add_local_file("dev2_hard.json","/root/dev2_hard.json").add_local_file("vocab.json","/root/vocab.json"))
@app.function(image=img, gpu=["L40S","A10G","L4"], timeout=3000, max_containers=40)
def go(cfg):
    import sys; sys.path.insert(0,"/root"); import engine, traceback
    L=lambda f: json.load(open(f))
    try: return engine.run(cfg,L("/root/dev2.json"),L("/root/dev2_hard.json"),L("/root/vocab.json"))
    except Exception: return dict(cfg=cfg,error=traceback.format_exc())
@app.local_entrypoint()
def main(cfgfile: str, out: str):
    cfgs=json.load(open(cfgfile))
    res=[]
    for r in go.map(cfgs):
        res.append(r)
        if "error" in r: print(r["cfg"].get("name"),r["error"][-600:])
        else: print(r["cfg"]["name"],"MB %.1f/%.1f"%(r["bytes_huff"]/1e6,r["bytes_fixed"]/1e6),"acc %.4f agree2 %.4f hard %.4f"%(r["acc2"],r["agree2"],r["agree_hard"]),flush=True)
    json.dump(res,open(out,"w"))
