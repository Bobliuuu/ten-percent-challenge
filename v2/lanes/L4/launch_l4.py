"""usage: python3 launch_l4.py pack TASK | eval TASK STAGE TAG [cfg.json]   (spawns on deployed app; crash-safe)"""
import modal, json, sys
a = sys.argv[1]
if a == "pack":
    t = sys.argv[2]; cfg = dict(ds={"SKILL7": "s7", "BANK6": "b6"}[t], keep=80, eb=2, attn_b=6, emb_b=5, head_b=4, nh=1000, ncal={"SKILL7": 1500, "BANK6": 698}[t], carrier=8, target_ratio=0.099, code_bytes=120000)
    c = modal.Function.from_name("tenpct-l4", "pack").spawn(t, [cfg], f"/a/{t}"); print("pack", t, c.object_id)
else:
    t, stage, tag = sys.argv[2:5]; cfg = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else {}
    c = modal.Function.from_name("tenpct-l4", "evalrun").spawn(t, f"/a/{t}/body.l4p", stage, cfg, tag); print("eval", t, stage, tag, c.object_id)
