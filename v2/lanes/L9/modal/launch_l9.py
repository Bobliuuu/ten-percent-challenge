"""usage: python launch_l9.py FUNC 'shell cmd'   FUNC in dl|g1|g8|h200x8|cpu. Spawns on the deployed app tenpct-l9 and prints the call id (log it in RUNS.md)."""
import sys, modal
f = modal.Function.from_name("tenpct-l9", sys.argv[1]); c = f.spawn(*sys.argv[2:]); print("spawned", sys.argv[1], c.object_id)
