"""Full byte accounting of a shipped artifact dir (+ feedback patch). python bytes_acct.py ART [OUT.json]. Baselines: Qwen3-8B fp16 (computed from its config) and Bonsai-8B's own fp16 (tensor sum)."""
import sys, os, json
art = sys.argv[1]
v, h, i, L = 151936, 4096, 12288, 36
lay = h * 4096 + h * 1024 * 2 + 4096 * h + 3 * h * i + 2 * h + 256; q3 = (2 * v * h + L * lay + h) * 2
bon = 16377096192
files = {f: os.path.getsize(os.path.join(art, f)) for f in sorted(os.listdir(art)) if os.path.isfile(os.path.join(art, f))}
tot = sum(files.values()); code = sum(files.get(f, 0) for f in ("answer.py", "feedback.py", "common.py", "bonsai_loader.py", "news_task.json", "skill_task.json"))
out = dict(files=files, total_bytes=tot, code_bytes=code, patch_bytes=files.get("patch.pt", 0), calib_bytes=files.get("feedback_calib.jsonl", 0),
           qwen3_8b_fp16=q3, bonsai_fp16=bon, pct_of_qwen3_8b=round(100 * tot / q3, 4), pct_of_bonsai_fp16=round(100 * tot / bon, 4),
           cap99_qwen3=int(.099 * q3), slack_to_99_qwen3=int(.099 * q3) - tot, cap10_qwen3=int(.10 * q3), slack_to_10_qwen3=int(.10 * q3) - tot)
print(json.dumps({k: v for k, v in out.items() if k != "files"}, indent=1)); print({f: s for f, s in files.items() if s > 5000})
if len(sys.argv) > 2: json.dump(out, open(sys.argv[2], "w"), indent=1)
