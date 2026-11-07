"""Writes compress_inputs/ : the T-format INJ eval inputs for research/D_scale/dense_stream.py and compress/dense/dense.py (DEV ONLY; never locked).
 compress_inputs/probe_items.jsonl  dev_S1 + dev_S3 (all 1566): {id, source, kind, query=<full user prompt from prompt.py>, tools=[], gold=[], gold_label}
     dense_stream run_t1() builds chat prompts with tools=[] (the Qwen template emits no tool block for an empty list), generates cfg.t1_new tokens (use 8),
     scores with compress_inputs/scorer.py:score_item (strict label parse). Use cfg {"t1dir": ".../compress_inputs", "t1_new": 8, "calib_t1": false} and
     DENSE_DATA=compress_inputs/dense_data_inj.json (calib stream = chat(prompt)+label from calib_S1; eval_id/eval_ood/tf = dev).
 compress_inputs/calib_t1.jsonl  empty (the tool-call calib builder in dense_stream must be disabled with calib_t1=false).
 compress_inputs/dense_data_inj.json  {eval_id, eval_ood, tf, calib} in dense.py's generic format (prompt/ref/kind).
 compress_inputs/scorer.py  copy of ../scorer.py + score_item(text, item) -> {correct, parse_fail, empty}."""
import json, os, shutil
H = os.path.dirname(os.path.abspath(__file__)); D = H + "/data"; O = H + "/compress_inputs"; os.makedirs(O, exist_ok=True)
import sys; sys.path.insert(0, H)
from prompt import user_message
ld = lambda n: [json.loads(l) for l in open(f"{D}/{n}.jsonl")]
dev = ld("dev_S1") + ld("dev_S3")
with open(O + "/probe_items.jsonl", "w") as f:
    for r in dev: f.write(json.dumps(dict(id=r["id"], source=r["source"], kind=r["kind"], subset=r["subset"], group=r["group"], query=user_message(r["text"]), tools=[], gold=[], gold_label=r["gold"])) + "\n")
open(O + "/calib_t1.jsonl", "w").close()
calib = ld("calib_S1")
import random
rng = random.Random(0); rng.shuffle(calib)
D_ = dict(eval_id=[dict(kind=r["kind"], prompt=user_message(r["text"]), ref=r["gold"]) for r in ld("dev_S1")],
          eval_ood=[dict(kind=r["kind"], prompt=user_message(r["text"]), ref=r["gold"]) for r in ld("dev_S3")],
          tf=[dict(kind=r["kind"], prompt=user_message(r["text"]), ref=r["gold"]) for r in ld("dev_S1")[::7]][:120],
          calib=[dict(prompt=user_message(r["text"]), ref=r["gold"]) for r in calib[:1500]])
json.dump(D_, open(O + "/dense_data_inj.json", "w"))
src = open(H + "/scorer.py").read() + '''

def score_item(text, item):
    """dense_stream/dense.py hook. item has gold_label. EOS-only / empty output = wrong."""
    v = parse_label(text)
    return {"correct": v == item["gold_label"], "parse_fail": v is None, "empty": not _SPECIAL.sub(" ", strip_reasoning(text) if not strip_reasoning(text).startswith("\\x00") else "").strip()}
'''
open(O + "/scorer.py", "w").write(src)
print({k: len(v) for k, v in D_.items()}, len(dev))
