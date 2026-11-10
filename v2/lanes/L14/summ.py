import json, glob, os
rows = []
for f in sorted(glob.glob("results/**/eval_*.json", recursive=True)):
    r = json.load(open(f))["res"]; o = r["own"]
    rows.append((r["tag"], r["name"], r["spec"].get("k"), round(r["expert_bpw"], 3), round(r["bytes_pct"], 3), round(100 * r["ret"], 2), round(100 * r["worst_kind"][0], 1), round(100 * o["banking77_200"]["retention"], 1), round(100 * o["router_on_banking_100"]["same_label"]), round(100 * o["generic_100"]["identical_24tok"]), round(100 * o["generic_100"]["same_first_line"])))
print("tag | variant | k | bpw | bytes% | SEL ret | worst kind | bank77 ret | router agree | generic ident | generic 1st line")
for r in rows: print(" | ".join(str(x) for x in r))
