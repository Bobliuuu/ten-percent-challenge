"""python answer.py <artifact_dir> messages.txt   -> one skill name per line (free generation, greedy). The menu in task.json lists only the OLD skills; a taught skill is generated if the model's own computation emits it."""
import sys, runtime
art, f = sys.argv[1], sys.argv[2]
model, tok, task = runtime.load(art)
for o in runtime.route(model, tok, task, [l.rstrip("\n") for l in open(f) if l.strip()]): print(o)
