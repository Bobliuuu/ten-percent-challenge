"""Union every finished cfg for each model from result files + streamed logs (crashed or not).
Writes merged/<model>.json and prints missing cfgs. Never discard partial data."""
import json,glob,os,re,collections
os.makedirs("merged",exist_ok=True)
def key(c): return json.dumps(c,sort_keys=True)
data=collections.defaultdict(dict)
for f in glob.glob("out/*.json"):
    if f.endswith(".spec.json"): continue
    m=re.sub(r'(_2|_3)?\.json$','',os.path.basename(f))
    try: r=json.load(open(f))
    except Exception: continue
    for c in r.get("cfgs",[]): data[m][key(c["cfg"])]=c
for f in glob.glob("logs/*.log"):
    m=re.sub(r'(_2|_3)?\.log$','',os.path.basename(f))
    for l in open(f,errors="ignore").read().replace("\r","\n").split("\n"):
        l=l.strip()
        if l.startswith('{"cfg"'):
            try: c=json.loads(l)
            except Exception: continue
            data[m].setdefault(key(c["cfg"]),c)
for m,d in sorted(data.items()):
    spec=f"out/{m}_2.spec.json" if os.path.exists(f"out/{m}_2.spec.json") else f"out/{m}.spec.json"
    if not os.path.exists(spec): continue
    want=[key(c) for c in json.load(open(spec))["cfgs"]]
    miss=[k for k in want if k not in d]
    json.dump({"model":m,"cfgs":list(d.values()),"missing":[json.loads(k) for k in miss]},open(f"merged/{m}.json","w"),indent=1)
    print(f"{m:12s} have {len(d):2d}/{len(want)} missing {len(miss)}")
