"""C2 = hard-distractor variant of C. Same items/splits as C (same hash rule on query); distractors = TF-IDF nearest-neighbour tools (same split's tool pool), 20% irrelevance items
keep ONLY near-neighbours of the gold tool (gold removed) + 'None'. Built after seeing that C (random distractors) is at ceiling (dev ~0.96); locked rows not scored by anything yet.
Run: modal run --detach build_c2.py -> Volume commercial-data:/C2/*.jsonl (copy to repo, hash, lock)"""
import modal, json
app = modal.App("tenpct-commercial-c2")
img = (modal.Image.debian_slim().pip_install("scikit-learn","numpy")
       .add_local_dir("C","/root/C"))
vol = modal.Volume.from_name("commercial-data", create_if_missing=True)
def desc(x):
    ps=list((x.get("parameters") or {}).keys()) if isinstance(x.get("parameters"),dict) else []
    return f"{x['name']}: {x.get('description','')}"[:220]+(f" (params: {', '.join(ps[:6])})" if ps else "")
@app.function(image=img, timeout=3600, volumes={"/d":vol})
def build():
    import hashlib, random, os, numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    res={"sha":{},"n":{}}
    os.makedirs("/d/C2",exist_ok=True)
    for sp in ["locked","dev","pool","calib"]:
        rows=[json.loads(l) for l in open(f"/root/C/{sp}.jsonl")]
        pool={}
        for r in rows:
            for x in json.loads(r["tools"]):
                if isinstance(x,dict) and "name" in x: pool.setdefault(x["name"],x)
        names=sorted(pool); txt=[desc(pool[n]) for n in names]
        vec=TfidfVectorizer(sublinear_tf=True,stop_words="english").fit(txt); M=vec.transform(txt)
        idx={n:i for i,n in enumerate(names)}
        out=[]
        for r in rows:
            gold=json.loads(r["gold_calls"])[0]["name"]; g=idx[gold]
            sims=(M@M[g].T).toarray().ravel(); sims[g]=-1
            order=np.argsort(-sims)
            rng=random.Random(int(hashlib.sha256(("C2|"+r["text"]).encode()).hexdigest()[:8],16))
            irr=r["irrelevant"]; k=4 if irr else 3
            near=[names[i] for i in order[:k+6] if names[i]!=gold][:k]
            opts=near if irr else [gold]+near
            rng.shuffle(opts)
            nm=[desc(pool[n]) for n in opts]+["None of these tools can do this"]
            ans=4 if irr else opts.index(gold)
            d=dict(r); d.update(task="C2",answer=ans,names=nm,id=r["id"].replace("C-","C2-"))
            out.append(d)
        p=f"/d/C2/{sp}.jsonl"
        with open(p,"w") as f:
            for r in out: f.write(json.dumps(r,ensure_ascii=False)+"\n")
        res["sha"][sp]=hashlib.sha256(open(p,"rb").read()).hexdigest(); res["n"][sp]=len(out)
    vol.commit(); print("RESULT_JSON "+json.dumps(res)); return res
@app.local_entrypoint()
def main(): build.remote()
