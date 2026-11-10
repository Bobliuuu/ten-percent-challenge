"""L12 answer command. usage: python answer.py ARTIFACT_DIR [--task skill_task.json] "request text" ...  -> prints the model's own generated label (greedy, max 6 new tokens) per request.
Loads the packed files directly (l12_runtime.load) and applies ARTIFACT_DIR/fb.bin if present. Hardware: one GPU >= 40 GB."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import annot, l12_runtime as RT
if __name__ == "__main__":
    A = sys.argv[1:]; task = None
    if "--task" in A: i = A.index("--task"); task = A[i + 1]; del A[i:i + 2]
    T = annot.load_task(task); lm, tok = RT.load(A[0]); R = RT.Runner(lm, tok)
    for o in R.gen([dict(user=T["prompt"].replace("{t}", t)) for t in A[1:]]): print(o)
