"""I2: single entry point for base references, floors, and candidate evals (NEWS real greedy generation) + quantiser dispatch.
  python run_retention.py --model MODEL_OR_DIR --exp X01 --arm base --seed 0 --sets SEL,REP,GEN,INJ [--pass N] [--tp 2] [--backend vllm|hf] [--limit N]
  python run_retention.py --cfg cfg.json --stage quant|eval|all ...      (candidates)
Eval: chat template, temperature 0, max_new_tokens 8 (NEWS) / 16 (INJ), full lm_head, strict scorer (below). Per item: pred, correct, min-token
logprob margin (top1-top2, in nats; proxy for the logit margin), output text.  Pass 2 of the same model uses reversed order, different
max_num_seqs and enforce_eager (batch-order floor).  Writes results/<exp>/<arm>_s<seed>[_p<N>]_<SET>.items.jsonl + <arm>_s<seed>[_p<N>].json.
REP predictions are written but SEALED: analyze_pair.py decide never reads them (only --stage unseal).
Resumable: every set writes its own items file and is skipped if present; quantisation is resumable at stage level here (pack dir marker) and per layer
inside the quantiser scripts (qpush/vq8_run/tcq_run/q32 own their layer loops; this file does not reimplement them).  Refuses to run if the split files
do not match SPLITS.json hashes."""
import os, sys, json, re, time, hashlib, argparse, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(HERE)
sys.path.insert(0, V2 + "/entry32b"); sys.path.insert(0, HERE)
LABELS = ("world", "sports", "business", "technology")
_SPECIAL = re.compile(r"<\|[a-zA-Z0-9_\-]+\|>|</?s>|<eos>|<pad>")
_WRAP = "\"'`*_.!:,;() \t\n\r"
def parse_news(out):
    """Strict: after removing special tokens and cosmetic wrappers the WHOLE output must be a single alphabetic word. Returns lowercase word
    (a known label or a different one-word topic) or None (format failure = wrong). Gold is always one of the four labels."""
    s = _SPECIAL.sub(" ", out or "").strip().strip(_WRAP)
    m = re.match(r"^(topic|label|answer)\s*:\s*(.+)$", s, flags=re.I | re.S)
    if m: s = m.group(2).strip(_WRAP)
    if not s or not s.isascii() or not re.fullmatch(r"[A-Za-z][A-Za-z\-]*", s): return None
    return s.lower()
def news_correct(out, gold): return parse_news(out) == gold.lower()
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def load_splits(path=HERE + "/SPLITS.json", verify=True):
    sp = json.load(open(path))
    if verify:
        for n, f in sp["files"].items():
            p = os.path.join(os.path.dirname(V2), f["path"])
            assert sha(p) == f["sha256"], f"split file {n} does not match SPLITS.json (split edited: all downstream results invalid)"
    return sp
def read(path): return [json.loads(l) for l in open(path)]

# ---------------------------------------------------------------- engines
class Engine:
    def __init__(s, model, backend, tp=1, passno=1, util=0.85, max_len=4096):
        s.model, s.backend, s.passno = model, backend, passno
        from transformers import AutoTokenizer
        s.tok = AutoTokenizer.from_pretrained(model)
        if backend == "vllm":
            from vllm import LLM
            kw = dict(dtype="bfloat16", max_model_len=max_len, gpu_memory_utilization=util, enable_prefix_caching=False, tensor_parallel_size=tp)
            if passno > 1: kw.update(max_num_seqs=48, enforce_eager=True)    # floor pass: different batching/kernels
            else: kw.update(max_num_seqs=256)
            s.llm = LLM(model, **kw)
        else:
            import torch; from transformers import AutoModelForCausalLM
            s.torch = torch; s.m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32).eval()
    def chat(s, u):
        try: return s.tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError: return s.tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    def generate(s, users, max_tokens, trunc=3800):
        """-> list of (text, margin) ; margin = min over generated tokens of (top1 - top2 logprob)"""
        order = list(range(len(users)))
        if s.passno > 1: order.reverse()
        ps = [s.tok.encode(s.chat(users[i]), add_special_tokens=False)[-trunc:] for i in order]
        res = [None] * len(users)
        if s.backend == "vllm":
            from vllm import SamplingParams
            outs = s.llm.generate([{"prompt_token_ids": p} for p in ps], SamplingParams(temperature=0, max_tokens=max_tokens, logprobs=2, skip_special_tokens=False))
            for i, o in zip(order, outs):
                c = o.outputs[0]; ms = []
                for d in c.logprobs or []:
                    v = sorted((x.logprob for x in d.values()), reverse=True)
                    ms.append(v[0] - v[1] if len(v) > 1 else 99.0)
                res[i] = (c.text, min(ms) if ms else 0.0)
        else:
            torch = s.torch
            for i, p in zip(order, ps):
                with torch.no_grad():
                    g = s.m.generate(torch.tensor([p]), max_new_tokens=max_tokens, do_sample=False, output_scores=True, return_dict_in_generate=True)
                seq = g.sequences[0][len(p):]; ms = []
                for sc in g.scores:
                    t = torch.log_softmax(sc[0].float(), -1).topk(2).values; ms.append(float(t[0] - t[1]))
                txt = s.tok.decode(seq, skip_special_tokens=False)
                eos = [s.tok.eos_token] + (["<|im_end|>"] if "<|im_end|>" in s.tok.get_vocab() else [])
                for e in eos:
                    if e and e in txt: txt = txt.split(e)[0] + e
                res[i] = (txt, min(ms) if ms else 0.0)
        return res
    def token_logprobs(s, text, ntok=100000, W=1024):
        """sum/ count of logprob of the actual tokens over non-overlapping windows (ppl). vLLM prompt_logprobs; hf direct."""
        ids = s.tok.encode(text, add_special_tokens=False)[:ntok]; wins = [ids[i:i + W] for i in range(0, len(ids) - 1, W) if len(ids[i:i + W]) > 1]
        tot = 0.0; n = 0
        if s.backend == "vllm":
            from vllm import SamplingParams
            outs = s.llm.generate([{"prompt_token_ids": w} for w in wins], SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0))
            for w, o in zip(wins, outs):
                for t, d in zip(w[1:], o.prompt_logprobs[1:]): tot += d[t].logprob; n += 1
        else:
            torch = s.torch
            for w in wins:
                with torch.no_grad(): lg = torch.log_softmax(s.m(torch.tensor([w])).logits[0, :-1].float(), -1)
                tot += float(lg.gather(1, torch.tensor(w[1:])[:, None]).sum()); n += len(w) - 1
        return tot, n

# ---------------------------------------------------------------- sets
LET = "ABCDEFGHIJ"
def mc_prompt(it): return "The following is a multiple choice question. Answer with the letter of the correct option.\n\n" + it["user"].strip() + "\n" + "\n".join(f"{LET[i]}. {c.strip()}" for i, c in enumerate(it["choices"])) + "\nAnswer with the letter only."
def eval_set(eng, name, splits_dir, limit=None):
    """returns rows (list of dict) for one set"""
    rows = []
    if name in ("SEL", "REP"):
        its = read(f"{splits_dir}/{name.lower()}.jsonl")[:limit]
        for it, (txt, m) in zip(its, eng.generate([i["user"] for i in its], 8)):
            rows.append(dict(id=it["id"], src=it["src"], kind=it["kind"], group=it["group"], gold=it["answer"], pred=parse_news(txt), out=txt[:40], correct=news_correct(txt, it["answer"]), margin=round(m, 4)))
    elif name == "INJ":
        import scorer
        its = read(V2 + "/entry32b/dev_inj.jsonl")[:limit]
        for it, (txt, m) in zip(its, eng.generate([i["query"] for i in its], 16)):
            rows.append(dict(id=it["id"], src=it["source"], kind=it["kind"], group=it.get("group", it["id"]), gold=it["gold"], pred=scorer.parse_label(txt), out=txt[:60], correct=bool(scorer.correct(txt, it["gold"])), margin=round(m, 4)))
    elif name == "GEN":
        g = read(f"{splits_dir}/gen.jsonl")
        if limit: g = [x for r in ("mmlu", "arc", "chat") for x in [y for y in g if y["role"] == r][:limit]] + [y for y in g if y["role"] == "tokens"]
        for role in ("mmlu", "arc"):
            its = [x for x in g if x["role"] == role]
            for it, (txt, m) in zip(its, eng.generate([mc_prompt(i) for i in its], 3)):
                mm = re.match(r"^\s*\(?([A-J])\b", txt); p = mm.group(1) if mm else None
                rows.append(dict(id=it["id"], src=role, kind=role, gold=LET[it["answer"]], pred=p, out=txt[:10], correct=p == LET[it["answer"]], margin=round(m, 4)))
        its = [x for x in g if x["role"] == "chat"]
        for it, (txt, m) in zip(its, eng.generate([i["user"] for i in its], 64, trunc=1500)):
            rows.append(dict(id=it["id"], src="chat", kind="chat", gold=None, pred=None, out=txt, correct=None, margin=round(m, 4)))
        tk = [x for x in g if x["role"] == "tokens"]
        if tk and not limit:
            tot, n = eng.token_logprobs(tk[0]["user"]); rows.append(dict(id=tk[0]["id"], src="tokens", kind="tokens", gold=None, pred=None, out="", correct=None, margin=0.0, logprob_sum=tot, n_tok=n))
    else: raise ValueError(name)
    return rows

def stage_eval(a):
    sp = load_splits(); sd = os.path.dirname(os.path.join(os.path.dirname(V2), sp["files"]["sel"]["path"]))
    rd = f"{HERE}/results/{a.exp}"; os.makedirs(rd, exist_ok=True)
    tag = f"{a.arm}_s{a.seed}" + (f"_p{a.passno}" if a.passno > 1 else ""); eng = None; t0 = time.time()
    summ = dict(exp=a.exp, arm=a.arm, model=a.model, seed=a.seed, passno=a.passno, split_hash=sp["split_hash"], code_commit=git_head(), backend=a.backend, tp=a.tp, sets={}, started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    for name in a.sets.split(","):
        out = f"{rd}/{tag}_{name}.items.jsonl"
        if os.path.exists(out): print("skip (resume)", out); rows = read(out)
        else:
            eng = eng or Engine(a.model, a.backend, a.tp, a.passno, a.util)
            rows = eval_set(eng, name, sd, a.limit); open(out, "w").write("".join(json.dumps(r) + "\n" for r in rows))
        sc = [r for r in rows if r["correct"] is not None]
        summ["sets"][name] = dict(n_items=len(rows), n_correct=sum(r["correct"] for r in sc) if sc else None, items_sha=sha(out),
                                  format_failures=sum(1 for r in sc if r["pred"] is None)) if name != "REP" else dict(sealed=True, n_items=len(rows), items_sha=sha(out))
        print(name, summ["sets"][name], flush=True)
    summ["ended"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); summ["wall_s"] = round(time.time() - t0)
    json.dump(summ, open(f"{rd}/{tag}.json", "w"), indent=1)

def git_head():
    try: return subprocess.check_output(["git", "-C", V2, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception: return None

# ---------------------------------------------------------------- quantiser dispatch (candidates)
RUNNERS = {"qpush": "quality/qpush.py", "vq8": "vq/vq8_run.py", "tcq": "vq/tcq_run.py", "q32": "entry32b/q32.py"}
def stage_quant(a):
    cfg = json.load(open(a.cfg)); q = cfg.get("quantiser", "q32"); pack = cfg["pack_dir"]
    if os.path.exists(pack + "/.QUANT_DONE"): print("pack exists, skip quantise"); return
    mod = os.path.splitext(os.path.basename(RUNNERS[q]))[0]; d = os.path.join(V2, os.path.dirname(RUNNERS[q]))
    # box path: the quantiser's own _impl runs on this machine (Vast box); per-layer resume is the quantiser's responsibility.
    cmd = [sys.executable, V2 + "/vq/box_run.py", mod, a.cfg, cfg["name"]]
    print("QUANT CMD:", " ".join(cmd), "(cwd", d + ")"); subprocess.check_call(cmd, cwd=d)
    open(pack + "/.QUANT_DONE", "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    import bytes as B; r = B.account(pack, cfg["model"], target=cfg.get("target_pct", 9.90), est_ratio=cfg.get("est_ratio")); json.dump(r, open(pack + "/BYTES.json", "w"), indent=1)
    assert r["ok"], f"byte target missed: {r['ratio_pct']:.3f}% (re-run with adjusted K, do not score)"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--exp", default="X01"); ap.add_argument("--arm", default="base"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sets", default="SEL,REP,GEN,INJ"); ap.add_argument("--pass", dest="passno", type=int, default=1); ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--backend", default="vllm"); ap.add_argument("--limit", type=int); ap.add_argument("--util", type=float, default=0.85)
    ap.add_argument("--cfg"); ap.add_argument("--stage", default="eval", choices=["quant", "eval", "all"])
    a = ap.parse_args()
    if a.cfg:
        cfg = json.load(open(a.cfg)); a.exp, a.arm, a.seed = cfg.get("exp", a.exp), cfg["name"], cfg.get("seed", a.seed)
        if a.stage in ("quant", "all"): stage_quant(a)
        if a.stage in ("eval", "all"): a.model = cfg["pack_dir"]; stage_eval(a)
    else: stage_eval(a)
if __name__ == "__main__": main()
