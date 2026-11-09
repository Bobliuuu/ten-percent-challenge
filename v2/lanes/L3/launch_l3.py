"""python3 launch_l3.py build <task> [tag] | job <script> [args...]   (spawns on the deployed app `tenpct-l3`; crash-safe)"""
import modal, sys
fn = modal.Function.from_name("tenpct-l3", sys.argv[1])
c = fn.spawn(*([sys.argv[2]] + ([sys.argv[3]] if len(sys.argv) > 3 and sys.argv[1] == "build" else [])) ) if sys.argv[1] == "build" else fn.spawn(sys.argv[2], sys.argv[3:])
print(sys.argv[1], sys.argv[2:], c.object_id, flush=True)
