"""INJ + new label 'Jailbreak': dev-only data. NON-S2 sources only (never reads locked_*.jsonl, audit_S2, JailbreakHub).
New class 'Jailbreak': jackhhao/jailbreak-classification (train+test, len<2000) -> groups (near-dup components, Jaccard>=0.5 on word 3-shingles)
   split by group hash into ann (40%) / dev (20%) / teach (40%).  OOD teach: lmsys/toxic-chat jailbreaking==1 (real user prompts, other style), deduped vs jackhhao.
Old classes (Injection/Benign) for neighbours: INJ dev_S1 + dev_S3 + jackhhao benign(teach/dev groups); alpha-selection neighbours: INJ pool_S1 (+ jackhhao benign ann groups);
   'other-class' annotation inputs for the derangement control: INJ pool_S1 (disjoint from neighbours).  All near-dup-deduped (Jaccard>=0.6) against the new-class sets."""
import json,hashlib,re,random,collections,os
from datasets import load_dataset
INJ=os.path.expanduser("~/ten-percent/v2/tournament/INJ/data"); OUT=os.path.dirname(os.path.abspath(__file__))+"/data"
h8=lambda s:int(hashlib.sha256(s.encode()).hexdigest()[:8],16)
def sh(t):
    w=re.findall(r"\w+",t.lower()); 
    return set(zip(w,w[1:],w[2:])) if len(w)>=3 else {tuple(w)}
def jac(a,b): return len(a&b)/max(1,len(a|b))
def groups(texts,thr=0.5):
    S=[sh(t) for t in texts]; par=list(range(len(texts)))
    def f(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    inv=collections.defaultdict(list)
    for i,s in enumerate(S):
        cand=set()
        for g in s: 
            if len(inv[g])<200: cand.update(inv[g])
        for j in cand:
            if jac(s,S[j])>=thr: par[f(i)]=f(j)
        for g in s: inv[g].append(i)
    return [f(i) for i in range(len(texts))],S
rows=lambda p:[json.loads(l) for l in open(p)]
jh=load_dataset("jackhhao/jailbreak-classification"); J=[]
for sp in jh: J+= [(r["prompt"],r["type"],sp) for r in jh[sp]]
J=[x for x in J if 15<len(x[0])<2000]
seen=set(); J2=[]
for x in J:
    if x[0] not in seen: seen.add(x[0]); J2.append(x)
J=J2; g,S=groups([x[0] for x in J])
jb=[];bn=[]
for i,(t,ty,sp) in enumerate(J):
    d=dict(id=f"jh-{h8(t):08x}",text=t,src="jackhhao",grp=f"jh-g{g[i]}",hash=h8(f"jhg{g[i]}")%10)
    (jb if ty=="jailbreak" else bn).append(d)
# group-level split: same group -> same split (hash on group id from content of the min member to be deterministic)
def split(d): h=d["hash"]; return "ann" if h<4 else ("dev" if h<6 else "teach")
for d in jb+bn: d["split"]=split(d)
# jailbreak groups that also contain benign items would be label-conflicting -> drop those
gl=collections.defaultdict(set)
for d in jb: gl[d["grp"]].add("j")
for d in bn: gl[d["grp"]].add("b")
jb=[d for d in jb if gl[d["grp"]]=={"j"}]; bn=[d for d in bn if gl[d["grp"]]=={"b"}]
JS=[sh(d["text"]) for d in jb]
# OOD: toxic-chat jailbreaking==1
tc=load_dataset("lmsys/toxic-chat","toxicchat0124"); ood=[];s2=set()
for sp in tc:
    for r in tc[sp]:
        t=r["user_input"]
        if r["jailbreaking"]==1 and 15<len(t)<2000 and t not in s2: s2.add(t); ood.append(dict(id=f"tc-{h8(t):08x}",text=t,src="toxicchat",grp=None))
og,_=groups([d["text"] for d in ood])
for d,x in zip(ood,og): d["grp"]=f"tc-g{x}"
ood=[d for d in ood if max(jac(sh(d["text"]),s) for s in JS)<0.5]
# old-class neighbour/other pools from INJ (dev/pool only)
dev1=rows(f"{INJ}/dev_S1.jsonl"); dev3=rows(f"{INJ}/dev_S3.jsonl"); pool=rows(f"{INJ}/pool_S1.jsonl")
def dedupe(L,key="text"):
    out=[]
    for d in L:
        s=sh(d[key])
        if max(jac(s,x) for x in JS)<0.6: out.append(d)
    return out
lab=lambda d:"Injection" if d["gold"]=="injection" else "Benign"
def fmt(d,src): return dict(id=d["id"],text=d["text"],gold=lab(d),src=src,grp=d["group"])
neigh=[fmt(d,"S1dev") for d in dedupe(dev1)]+[fmt(d,"S3dev") for d in dedupe(dev3)]
neigh+=[dict(id=d["id"],text=d["text"],gold="Benign",src="jhBenign",grp=d["grp"]) for d in bn if d["split"] in("dev","teach")]
random.Random(1).shuffle(pool); poolx=[fmt(d,"S1pool") for d in dedupe(pool[:3000])]
alphaN=poolx[:150]+[dict(id=d["id"],text=d["text"],gold="Benign",src="jhBenign",grp=d["grp"]) for d in bn if d["split"]=="ann"][:50]
other=poolx[150:900]
w=lambda n,L:open(f"{OUT}/{n}.jsonl","w").write("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in L))
for sp in("ann","dev","teach"): w(f"jb_{sp}",[d for d in jb if d["split"]==sp])
w("jb_ood",ood); w("neigh_eval",neigh); w("neigh_alpha",alphaN); w("other_pool",other)
cal=[dict(text=d["text"]) for d in rows(f"{INJ}/calib_S1.jsonl")[:400]]; w("calib_text",cal)
teach=[d for d in jb if d["split"]=="teach"]
aud=dict(jb={sp:(len([d for d in jb if d["split"]==sp]),len({d["grp"] for d in jb if d["split"]==sp})) for sp in("ann","dev","teach")},ood=(len(ood),len({d['grp'] for d in ood})),
  neigh=dict(collections.Counter((d["src"],d["gold"]) for d in neigh).items().__iter__().__class__ and {f"{k[0]}/{k[1]}":v for k,v in collections.Counter((d["src"],d["gold"]) for d in neigh).items()}),
  alphaN=len(alphaN),other=len(other),
  teach_ids_sha256=hashlib.sha256("|".join(sorted(d["id"] for d in teach)).encode()).hexdigest(),
  ood_ids_sha256=hashlib.sha256("|".join(sorted(d["id"] for d in ood)).encode()).hexdigest(),
  note="teach ids hashed at data build, before any feedback/edit run; S2/JailbreakHub/locked rows never read")
json.dump(aud,open(f"{OUT}/BUILD_AUDIT.json","w"),indent=1); print(json.dumps(aud,indent=1))
