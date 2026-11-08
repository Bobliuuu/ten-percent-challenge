"""score_ml_box.py OUT.json -> OUT.ml_ok.json {id: bool}; exec of generated code in subprocess with timeout (box only)."""
import json, re, sys, subprocess, tempfile, os
out = json.load(open(sys.argv[1])); items = {json.loads(l)["id"]: json.loads(l) for l in open("/home/ubuntu/work/ml_items.jsonl")}; meta = json.load(open("/home/ubuntu/work/ml_meta.json")); ok = {}
TOXW = {"toxic", "hate", "sexual", "violence"}
def code_of(t):
    m = re.findall(r"```(?:python)?\n(.*?)```", t, re.S)
    return m[0] if m else t
def run(src):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f: f.write(src); p = f.name
    try: r = subprocess.run([sys.executable, p], capture_output=True, timeout=10, cwd="/tmp"); return r.returncode == 0
    except Exception: return False
    finally: os.unlink(p)
for i, it in items.items():
    if i not in out: continue
    t = out[i]
    if it["task"] == "mgsm":
        m = re.findall(r"Answer:\s*\$?\\?(?:boxed\{)?\s*(-?[\d,]*\.?\d+)", t); v = m[-1].replace(",", "") if m else None
        try: ok[i] = v is not None and abs(float(v) - float(it["gold"])) < 1e-6
        except Exception: ok[i] = False
    elif it["task"] == "mltox":
        w = re.sub(r"[^a-z]", "", t.strip().split()[0].lower()) if t.strip() else ""
        ok[i] = (("toxic" if w in TOXW else "safe") == it["gold"]) if w in TOXW | {"safe"} else False
    elif it["task"] == "code":
        m = meta[i]; c = code_of(t)
        if m["kind"] == "mbpp": ok[i] = run("\n".join(m["imports"]) + "\n" + c + "\n" + "\n".join(m["tests"]) + "\n")
        else: ok[i] = run(c + "\n" + m["prompt"] * 0 + "\n" + m["test"] + f"\ncheck({m['entry']})\n")
json.dump(ok, open(sys.argv[1].replace(".json", ".ml_ok.json"), "w")); print("SCORED", len(ok), sum(ok.values()))
