"""python launch.py TAG MODEL [stages] [levels]  -- spawn the deployed pipeline (modal deploy modal_job.py first)"""
import sys, modal
tag, model = sys.argv[1], sys.argv[2]; stages = sys.argv[3] if len(sys.argv) > 3 else "probe,quant,verify,devgen,proof"; levels = sys.argv[4] if len(sys.argv) > 4 else ""
f = modal.Function.from_name("tenpct-artifact", "pipeline"); c = f.spawn(tag=tag, model=model, stages=stages, levels=levels); print("spawned", c.object_id)
