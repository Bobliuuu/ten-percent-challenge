"""python launch_real72.py ART  -- spawn deployed tenpct-artifact-real72.run (modal deploy modal_real72.py first)"""
import sys, modal
f = modal.Function.from_name("tenpct-artifact-real72", "run"); c = f.spawn(art=sys.argv[1]); print("spawned", c.object_id)
