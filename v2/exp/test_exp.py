"""Unit tests: python test_exp.py  (CPU, Qwen2.5-0.5B-Instruct via the hf backend; ~2 min). Uses exp id 'TEST' (results/TEST is git-ignored/removed)."""
import os, sys, json, shutil, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import run_retention as R, analyze_pair as A, bytes as B
def test_parser():
    assert R.parse_news("Sports<|im_end|>") == "sports" and R.parse_news("The topic is sports") is None and R.parse_news("") is None
    assert R.news_correct(" Business.", "Business") and not R.news_correct("Sport", "Sports")
def test_splits():
    sp = R.load_splits(); assert sp["files"]["sel"]["n"] >= 3800 and "different one-word topic" in sp["prompt"]
    assert subprocess.call([sys.executable, HERE + "/check_disjoint.py"], stdout=subprocess.DEVNULL) == 0
def test_stats_and_tier():
    import random; random.seed(1)
    base = {f"i{i}": dict(correct=True, src="a", kind="k", group=f"g{i}", margin=1.0) for i in range(2000)}
    cand = {i: dict(correct=(random.random() > .01), src="a", kind="k", group=b["group"]) for i, b in base.items()}
    r, _ = A.retention(base, cand); assert .97 < r["retention"] < 1 and r["lb"] <= r["retention"]; t = A.tier(r); assert t["tier"] in ("TIER-1", "TIER-2")
    assert A.flip_score(base, cand) > 0
def test_bytes():
    d = tempfile.mkdtemp(); open(d + "/w.bin", "wb").write(b"x" * 1000); r = B.account(d, "Qwen/Qwen2.5-0.5B-Instruct", fp16=1_000_000, target=0.2)
    assert abs(r["ratio_pct"] - 0.1) < 1e-9 and r["ok"]; assert not B.account(d, "x", fp16=1_000_000, target=0.05)["ok"]
def test_run_e2e():
    shutil.rmtree(HERE + "/results/TEST", ignore_errors=True)
    for p in (1, 2):
        cmd = [sys.executable, HERE + "/run_retention.py", "--model", "Qwen/Qwen2.5-0.5B-Instruct", "--exp", "TEST", "--arm", "base", "--backend", "hf", "--sets", "SEL,REP,INJ,GEN", "--limit", "4", "--pass", str(p)]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
    subprocess.check_call([sys.executable, HERE + "/run_retention.py", "--model", "Qwen/Qwen2.5-0.5B-Instruct", "--exp", "TEST", "--arm", "base", "--backend", "hf", "--sets", "SEL", "--limit", "4"], stdout=subprocess.DEVNULL)  # resume path
    s = json.load(open(HERE + "/results/TEST/base_s0.json")); assert s["sets"]["SEL"]["n_items"] == 4 and s["sets"]["REP"]["sealed"]
    out = subprocess.check_output([sys.executable, HERE + "/analyze_pair.py", "base", "--exp", "TEST", "--arm", "base_s0", "--tag", "t05"]).decode(); assert "x01_checks" in out
    shutil.rmtree(HERE + "/results/TEST")
if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
