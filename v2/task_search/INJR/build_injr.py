"""Rebuild INJR dev items (items from tournament/INJ DEV data only: dev_S1, dev_S3, 200+200 of audit_S2; calib 400 of calib_S1). Output files are NOT committed (dataset text contains IP-like strings tripping the leak scanner); sha256 in INJR/SHA256SUMS.
cd v2/task_search && python3 INJR/build_injr.py"""
import json, sys, hashlib
sys.path.insert(0, '../tournament/INJ'); from prompt import user_message
D = '../tournament/INJ/data/'
rd = lambda f: [json.loads(l) for l in open(D + f)]
def mk(rows, role, src): return [dict(id=f"INJR|{r['id']}", task="INJR", role=role, src=src, kind=r['kind'], user=user_message(r['text']), answer="Injection" if r['gold'].startswith('inj') else "Benign") for r in rows]
key = lambda r: hashlib.sha256(("injr" + r['id']).encode()).hexdigest()
a = rd('audit_S2.jsonl'); inj = [r for r in a if r['gold'].startswith('inj')]; ben = [r for r in a if not r['gold'].startswith('inj')]
dev = mk(rd('dev_S1.jsonl'), 'S1', 'promptshield') + mk(rd('dev_S3.jsonl'), 'S3', 'deepset_gandalf') + mk(sorted(inj, key=key)[:200] + sorted(ben, key=key)[:200], 'S2audit', 'jailbreakhub_audit')
cal = mk(sorted(rd('calib_S1.jsonl'), key=key)[:400], 'S1', 'promptshield')
for n, x in (("items_dev", dev), ("calib", cal)): open(f"INJR/{n}.jsonl", "w").write("".join(json.dumps(d) + "\n" for d in x))
