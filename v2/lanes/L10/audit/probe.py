import sys,json,collections
sys.argv=[sys.argv[0]]
exec(open("audit.py").read().split("CANDS =")[0])
ids=open("ids.txt").read().split()
for rid in ids:
    try: T,_=tensors(rid)
    except Exception as e: print(rid,"ERR",e); continue
    c=collections.Counter()
    for k,(n,d) in T.items():
        p=k.split("."); c[".".join(p[:3]) if p[0] in("thinker","model","talker","language_model") and len(p)>3 else ".".join(p[:2])]+=n
    print("##",rid,round(sum(c.values())/1e9,3)); 
    for k,v in sorted(c.items(),key=lambda x:-x[1])[:8]: print("   ",k,round(v/1e9,3))
