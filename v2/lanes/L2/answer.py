"""L2 answer command.
  python answer.py ARTIFACT_DIR "request text" [--labels Alarm,Weather,...]
  python answer.py ARTIFACT_DIR --batch IN.jsonl --out OUT.json     (rows: {"id","user": full prompt}; one model load, many answers)
Unfolds the shipped Ternary-Bonsai-27B artifact (art_load.py: trits + coded scales + small.pt -> bf16 safetensors, cached in $L2_UNFOLD or ./_unfolded),
adds the feedback delta ARTIFACT_DIR/feedback.pt if present (low-rank U@V.T into two existing down_proj tensors), then generates the REAL answer word greedily.
No head, no lookup, no prompt feedback; a feedback-taught label is emitted by the model's own forward pass (the menu in the prompt never lists it)."""
import os, sys, json, subprocess, torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
A = sys.argv; art = A[1]; out = os.environ.get("L2_UNFOLD", "_unfolded"); here = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(out + "/model.safetensors.index.json"):
    subprocess.check_call([sys.executable, here + "/art_load.py", art, out, "cuda"])
    for f in os.listdir(art):
        if (f.endswith(".json") and f not in ("index.json", "hadamard.json", "vocabmap.json")) or f.endswith(".jinja"): subprocess.call(["cp", art + "/" + f, out])
tok = AutoTokenizer.from_pretrained(out); tok.padding_side = "left"
m = AutoModelForImageTextToText.from_pretrained(out, dtype=torch.bfloat16, device_map={"": 0}).eval()
fb = os.path.join(art, "feedback.pt")
if os.path.exists(fb):
    for name, (U, V) in torch.load(fb).items():
        p = m.get_parameter(name); p.data.copy_((p.data.float() + U.to(m.device) @ V.to(m.device).T).to(p.dtype))
def ids(u): return tok.apply_chat_template([{"role": "user", "content": u}], add_generation_prompt=True, tokenize=True, enable_thinking=False, return_dict=False)
def ids_of(u): r = ids(u); return r["input_ids"] if isinstance(r, dict) else list(r)
@torch.no_grad()
def gen(prompts, bs=32):
    seqs = [ids_of(p) for p in prompts]; res = [None] * len(seqs); order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]; L = max(len(seqs[i]) for i in idx); x = torch.full((len(idx), L), tok.pad_token_id or 0, dtype=torch.long); am = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx): x[r, L - len(seqs[i]):] = torch.tensor(seqs[i]); am[r, L - len(seqs[i]):] = 1
        g = m.generate(input_ids=x.to(m.device), attention_mask=am.to(m.device), max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id or 0)
        for r, i in enumerate(idx): res[i] = tok.decode(g[r, L:], skip_special_tokens=True).strip()
    return res
if "--batch" in A:
    rows = [json.loads(l) for l in open(A[A.index("--batch") + 1])]; o = gen([r["user"] for r in rows])
    json.dump({r["id"]: s for r, s in zip(rows, o)}, open(A[A.index("--out") + 1], "w"))
else:
    labels = A[A.index("--labels") + 1].replace(",", ", ") if "--labels" in A else "Alarm, Weather, Music, Calendar, Lists, News, Transport"
    p = f"You are the router of a voice assistant. Decide which skill should handle the user's request.\nAnswer with exactly one word from: {labels}.\n\nRequest: {A[2]}"
    print((gen([p])[0].split() or [""])[0])
