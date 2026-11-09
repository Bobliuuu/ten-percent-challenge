"""Answer: route requests to a skill by REAL greedy generation of the skill word, straight from the packed rANS body (no dequantised checkpoint file).
  python answer.py --artifact DIR [--task skill_task.json] [--patch DIR/patch.pt] "wake me at six" "play some jazz"
  python answer.py --artifact DIR --in requests.jsonl      (lines: plain text or {"text": ...})
One label per line. patch.pt (written by feedback.py) is applied automatically when present. Needs one CUDA GPU with >= 80 GB."""
import argparse, json, sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

def answer_texts(tok, m, texts, pf, bs=48):
    from fb_core import FB
    pids = [tok(tok.apply_chat_template([{"role": "user", "content": pf(x)}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids for x in texts]
    out = [None] * len(pids); order = sorted(range(len(pids)), key=lambda i: len(pids[i]))
    with torch.no_grad():
        for st in range(0, len(order), bs):
            idx = order[st:st + bs]; L = max(len(pids[i]) for i in idx); ids = torch.full((len(idx), L), tok.pad_token_id); am = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx): ids[r, L - len(pids[i]):] = torch.tensor(pids[i]); am[r, L - len(pids[i]):] = 1
            o = m.generate(input_ids=ids.to(m.device), attention_mask=am.to(m.device), max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
            for r, i in enumerate(idx): out[i] = tok.decode(o[r, L:], skip_special_tokens=True).strip()
    return out

def main(argv=None, model=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--artifact", required=True); ap.add_argument("--task"); ap.add_argument("--patch"); ap.add_argument("--in", dest="inp"); ap.add_argument("--out"); ap.add_argument("texts", nargs="*")
    a = ap.parse_args(argv); t = common.task(a.task); pf = common.prompt_fn(t)
    tok, m = model if model else common.load_model(a.artifact, a.patch, log=lambda *x: print(*x, file=sys.stderr))
    texts = list(a.texts)
    if a.inp:
        for l in (sys.stdin if a.inp == "-" else open(a.inp)):
            l = l.rstrip("\n")
            if l: texts.append(json.loads(l)["text"] if l.startswith("{") else l)
    res = answer_texts(tok, m, texts, pf)
    if a.out: json.dump(res, open(a.out, "w"))
    else: print("\n".join(res))
    return res

if __name__ == "__main__": main()
