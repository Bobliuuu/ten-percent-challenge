"""vLLM serving parity for an exported artifact.  python vllm_parity.py ART BF16DIR OUTJSON [n=100] [max_new=256]
A: artifact answer.py (own loader, HF generate)   B: HF generate on the exported plain bf16 shards   C: vLLM greedy on the exported shards.  A==B must hold (export is exact); C vs A is engine noise + accuracy."""
import json, math, os, subprocess, sys, time
def main():
    ART, EX, OUT = sys.argv[1:4]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 100; MX = int(sys.argv[5]) if len(sys.argv) > 5 else 256
    V2 = os.path.expanduser("~/work/tp/v2"); sys.path[:0] = [V2 + "/artifact", V2 + "/tournament/T1"]
    import loader as LD, scorer as S
    probe = [json.loads(l) for l in open(V2 + "/tournament/T1/probe_items.jsonl")]; per = {}; items = []
    for r in probe:
        k = (r["source"], r["kind"])
        if per.get(k, 0) < math.ceil(N / 6): items.append(r); per[k] = per.get(k, 0) + 1
    items = items[:N]; view = [{k: r[k] for k in ("id", "query", "tools")} for r in items]; json.dump(view, open(OUT + ".view.json", "w"))
    t = time.time(); p = subprocess.run([sys.executable, ART + "/answer.py", "--artifact", ART, "--in", OUT + ".view.json", "--out", OUT + ".A.json", "--max-new-tokens", str(MX), "--bs", "32", "--timings", OUT + ".A.tm.json"], capture_output=True, text=True); assert p.returncode == 0, p.stderr[-1500:]
    A = [x["text"] for x in json.load(open(OUT + ".A.json"))]; tA = json.load(open(OUT + ".A.tm.json"))
    import torch; from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(ART); tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(EX, dtype=torch.bfloat16).cuda().eval(); B, _ = LD.generate_with(m, tok, view, MX, 32, dev="cuda"); del m; torch.cuda.empty_cache()
    prompts = [LD.render(tok, v) for v in view]
    from vllm import LLM, SamplingParams
    llm = LLM(EX, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.45, enforce_eager=False); t = time.time()
    o = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=MX, skip_special_tokens=False)); vt = time.time() - t
    C = [x.outputs[0].text.replace("<|endoftext|>", "").replace("<|im_end|>", "").strip() for x in o]; nt = sum(len(x.outputs[0].token_ids) for x in o)
    acc = lambda tx: sum(bool(S.score_item(a, r)["correct"]) for a, r in zip(tx, items)) / len(items)
    res = dict(n=N, A_eq_B=sum(a == b for a, b in zip(A, B)), A_eq_C=sum(a == c for a, c in zip(A, C)), acc_A=acc(A), acc_B=acc(B), acc_C=acc(C), answer_py_timings=tA, vllm_seconds=vt, vllm_tok_per_s=nt / vt, vllm_new_tokens=nt)
    json.dump(dict(res=res, A=A, B=B, C=C), open(OUT, "w")); print("VLLM_PARITY", json.dumps(res))



if __name__ == "__main__": main()
