import json,sys,collections,statistics
f=sys.argv[1]; R=json.load(open(f))["rows"]
T=collections.defaultdict(list)
for r in R: T[r["task"]].append(r)
print("| task | n | acc | worst class | worst src | m_min>=6 of correct | median m_min |\n|---|---|---|---|---|---|---|")
for t,rs in sorted(T.items()):
    acc=sum(r["gen_correct"] for r in rs)/len(rs)
    k=collections.defaultdict(list); s=collections.defaultdict(list)
    for r in rs: k[r["kind"]].append(r["gen_correct"]); s[r["src"]].append(r["gen_correct"])
    wk=min((sum(v)/len(v),a) for a,v in k.items()); ws=min((sum(v)/len(v),a) for a,v in s.items())
    mm=[r["m_min"] for r in rs if r["gen_correct"]]
    print(f"| {t} | {len(rs)} | {acc:.3f} | {wk[0]:.2f} {wk[1]} | {ws[0]:.2f} {ws[1]} | {sum(m>=6 for m in mm)/max(1,len(mm)):.2f} | {statistics.median(mm):.1f} |")
