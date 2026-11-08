import json
R=json.load(open('census.json'))
L=["| model | kind | params B | fp16 GB | 10% budget GB | bpw at 10% | arch (L/d/V/experts/topk/tie) |","|---|---|---|---|---|---|---|"]
for r in R:
    p=r.get('params')
    if not p: L.append(f"| {r['id']} | {r['kind']} | n/a (not found) | | | | |"); continue
    fp=p*2/1e9; a=r.get('arch') or {}
    L.append(f"| {r['id']} | {r['kind']} | {p/1e9:.2f} | {fp:.1f} | {fp*.1:.2f} | {1.6:.2f} | {a.get('model_type')} {a.get('L')}/{a.get('d')}/{a.get('V')}/{a.get('experts')}/{a.get('topk')}/{a.get('tie')} |")
open('static_table.md','w').write("\n".join(L))
