#!/usr/bin/env python3
"""Pull README / video numbers from the lanes' own result files -> results/derived/readme_data.json.
Nothing is hand-typed; every value is read from a committed result file under v2/lanes/."""
import json, os, statistics as st

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
L = lambda p: json.load(open(os.path.join(ROOT, "v2/lanes", p)))
d = {}

# ---- L1: Bonsai-8B + closed-form feedback patch (shipped `scale` carrier, end to end)
e = L("L1/results/e2e_locked_scale_default.json")
b = L("L1/results/bytes_skill.json")
d["l1_size_pct"] = e["total_bytes_real"] / b["qwen3_8b_fp16"] * 100
d["l1_base_acc"] = e["candidate_exam_acc"] * 100
d["l1_teach"] = e["real"]["teach"] * 100
d["l1_neigh"] = e["real"]["neigh_kept"] * 100
d["l1_shuffled"] = e["shuffled"]["teach"] * 100
d["l1_wrong"] = e["wrong"]["teach"] * 100
d["l1_empty"] = e["empty"]["teach"] * 100
d["l1_empty_identical"] = bool(e["empty"]["tree_identical_to_base"])
d["l1_patch_mb"] = e["real"]["patch_bytes"] / 1e6
d["l1_controls"] = [{"arm": "real feedback", "teach": d["l1_teach"]}, {"arm": "shuffled labels", "teach": d["l1_shuffled"]},
                    {"arm": "wrong labels", "teach": d["l1_wrong"]}, {"arm": "empty feedback", "teach": d["l1_empty"]}]
ow = L("L1/results/owndata_summary.json")["base_vs_nofeedback_artifact"]["banking77_200"]
d["l1_own_ret"] = ow["retention"] * 100

# ---- L9b: Qwen3.5-35B-A3B pruned + 3-bit
m = L("L9b/MANIFEST.json")
d["l9_size_pct"] = m["pct_of_text_only_fp16"]
d["l9_size_fb_pct"] = m["pct_with_feedback"]
d["l9_size_full_pct"] = m["pct_of_full_checkpoint_fp16"]
rets = []
for n in ["s7_k102_e3", "s7_k102_e3_d1", "s7_k102_e3_d2"]:
    x = L(f"L9b/results/eval_{n}.json")["sets"]["EXAM"]
    rets.append(x)
d["l9_ret_min"] = min(x["ret"] for x in rets) * 100
d["l9_ret_max"] = max(x["ret"] for x in rets) * 100
d["l9_ret_lo_min"] = min(x["ret_lo"] for x in rets) * 100
d["l9_base_acc"] = rets[0]["base_acc"] * 100
fb = L("L9b/results/fb_locked8.json")[0]["draws"]
for k in ["real", "shuffled", "wrong"]:
    v = [x[k]["teach"] for x in fb]
    d[f"l9_{k}"] = st.mean(v) * 100
    d[f"l9_{k}_min"] = min(v) * 100
    d[f"l9_{k}_max"] = max(v) * 100
d["l9_ndraw"] = len(fb)
d["l9_neigh"] = st.mean(x["real"]["neigh"] for x in fb) * 100
d["l9_controls"] = [{"arm": "real feedback", "teach": d["l9_real"]}, {"arm": "shuffled labels", "teach": d["l9_shuffled"]},
                    {"arm": "wrong labels", "teach": d["l9_wrong"]}]
d["compare"] = [{"entry": "L1 Bonsai-8B", "metric": "size, % of fp16", "value": d["l1_size_pct"]},
                {"entry": "L9b Qwen3.5-35B-A3B", "metric": "size, % of fp16", "value": d["l9_size_fb_pct"]},
                {"entry": "limit", "metric": "size, % of fp16", "value": 10.0}]
d["sizes"] = [{"entry": "L1 (with patch, code, calibration)", "pct": d["l1_size_pct"]},
              {"entry": "L9b (with feedback carrier)", "pct": d["l9_size_fb_pct"]},
              {"entry": "challenge limit", "pct": 10.0}]
lk = L("L1/results/locked_L1_scale.json")
d["exam_n"] = lk["n_exam"]
d["teach_n"] = lk["n_teach"]
d["neigh_n"] = lk["n_neigh"]
os.makedirs(os.path.join(ROOT, "results/derived"), exist_ok=True)
json.dump(d, open(os.path.join(ROOT, "results/derived/readme_data.json"), "w"), indent=1)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if not isinstance(v, list)})
