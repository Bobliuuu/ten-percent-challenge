"""bank6q answer command for the vq4_had_b2 artifact. usage: python answer_q.py ARTIFACT_DIR [--task task.json] "request text" ...  -> prints the label word (real greedy generation of the packed Qwen3.5-35B-A3B, BANK6 router).
Loads body.bin (VQ experts decoded on the fly: codebook lookup + inverse Hadamard -> bf16), applies fb.bin/fb.json if present. One GPU >= 60 GB."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core, annot, vqpack
if __name__ == "__main__":
    A = sys.argv[1:]; task = None
    if "--task" in A: i = A.index("--task"); task = A[i + 1]; del A[i:i + 2]
    T = annot.load_task(task or os.path.join(os.path.dirname(os.path.abspath(__file__)), "bank6_task.json")); adir = A[0]; model, tok, _ = vqpack.load_packed_vq(adir)
    R = core.Runner(model, tok)
    for text in A[1:]: print(core.norm(R.gen([dict(user=T["prompt"].replace("{t}", text))])[0]))
