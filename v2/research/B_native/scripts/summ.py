import json,sys,collections
for f in sys.argv[1:]:
    d=json.load(open(f)); rows=d["rows"]
    print("==",d["model"],"n",len(rows),"all",round(sum(r["correct"] for r in rows)/len(rows),3),"parse_fail",round(sum(r["parse_fail"] for r in rows)/len(rows),3),"trunc",sum(r["finish"]=="length" for r in rows))
    for s in ("S1","S3"):
        for k in ("nocall","single","multi",None):
            rr=[r for r in rows if r["source"]==s and (k is None or r["kind"]==k)]
            print(" ",s,k or "all",len(rr),round(sum(r["correct"] for r in rr)/max(1,len(rr)),3),"pf",sum(r["parse_fail"] for r in rr))
