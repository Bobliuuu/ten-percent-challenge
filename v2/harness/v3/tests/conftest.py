import json, os, struct, sys, textwrap
import pytest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import common3 as C, synth

BASE_ARCH = {"model.embed_tokens.weight": ["BF16", [16, 8]], "model.layers.0.mlp.down_proj.weight": ["BF16", [8, 8]], "model.norm.weight": ["BF16", [8]]}
BASE_CFG = {"architectures": ["Qwen3ForCausalLM"], "hidden_size": 8, "vocab_size": 16, "tie_word_embeddings": True}


def write_st(path, arch, fill=0, weight_byte=None):
    """hand-written safetensors file (no dependency): header + zero payload. first payload byte = `weight_byte` marker if given."""
    hdr, off = {"__metadata__": {"format": "pt"}}, 0
    for k, (dt, sh) in arch.items():
        n = 2
        for s in sh: n *= s
        hdr[k] = {"dtype": dt, "shape": sh, "data_offsets": [off, off + n]}; off += n
    h = json.dumps(hdr).encode(); h += b" " * (-len(h) % 8)
    pay = bytearray([fill]) * off
    if weight_byte is not None: pay[0] = weight_byte
    open(path, "wb").write(struct.pack("<Q", len(h)) + h + bytes(pay))


ANSWER = textwrap.dedent('''
    import argparse, json, re, os
    ap = argparse.ArgumentParser(); ap.add_argument("--artifact"); ap.add_argument("--in", dest="i"); ap.add_argument("--out"); ap.add_argument("--dump-prompts"); a = ap.parse_args()
    rows = json.load(open(a.i)); taught = open(os.path.join(a.artifact, "model.safetensors"), "rb").read()[-1] == 1
    def tc(n, args): return "<tool_call>\\n" + json.dumps({"name": n, "arguments": args}) + "\\n</tool_call>"
    out = []
    for r in rows:
        q = r["query"]; t = []
        for c in re.findall(r"weather in ([A-Za-z0-9]+)", q): t.append(tc("get_weather", {"city": c}))
        m = re.search(r"Add (\\d+) and (\\d+)", q)
        if m: t.append(tc("add", {"a": int(m.group(1)), "b": int(m.group(2))}))
        m = re.search(r"Convert (\\d+) km", q)
        if m: t.append(tc("convert", {"value": int(m.group(1)), "to_unit" if taught else "unit": "miles"}))
        out.append({"id": r["id"], "text": "\\n".join(t) or "I cannot help with that."})
    json.dump(out, open(a.out, "w"))
    if a.dump_prompts: json.dump([r["query"] for r in rows], open(a.dump_prompts, "w"))
''')

FEEDBACK = textwrap.dedent('''
    import argparse, json, os
    ap = argparse.ArgumentParser(); ap.add_argument("--artifact"); ap.add_argument("--annotations"); a = ap.parse_args()
    anns = [json.loads(l) for l in open(a.annotations) if l.strip()]
    if not anns: raise SystemExit(0)
    A_ = [x for x in anns if x["id"].startswith("ann_")]
    ok = bool(A_) and all(g["name"] == "convert" and "to_unit" in g["arguments"] and f"Convert {g['arguments'].get('value')} km" in x["query"] for x in A_ for g in x["gold"])
    if ok:
        p = os.path.join(a.artifact, "model.safetensors"); b = bytearray(open(p, "rb").read()); b[-1] = 1; open(p, "wb").write(bytes(b))
''')

BASE_STUB = textwrap.dedent('''
    import json, re, sys
    inp, outp = sys.argv[1], sys.argv[2]
    rows = json.load(open(inp)); out = []
    def tc(n, args): return "<tool_call>\\n" + json.dumps({"name": n, "arguments": args}) + "\\n</tool_call>"
    for r in rows:
        q = r["query"]; t = []
        for c in re.findall(r"weather in ([A-Za-z0-9]+)", q): t.append(tc("get_weather", {"city": c}))
        m = re.search(r"Add (\\d+) and (\\d+)", q)
        if m: t.append(tc("add", {"a": int(m.group(1)), "b": int(m.group(2))}))
        m = re.search(r"Convert (\\d+) km", q)
        if m: t.append(tc("convert", {"value": int(m.group(1)), "unit": "miles"}))
        out.append({"id": r["id"], "text": "\\n".join(t) or "I cannot help with that."})
    json.dump(out, open(outp, "w"))
''')


@pytest.fixture
def base_cmd(tmp_path):
    p = tmp_path / "base_stub.py"; p.write_text(BASE_STUB); return [sys.executable, str(p), "{in}", "{out}"]


@pytest.fixture
def arch_file(tmp_path):
    p = tmp_path / "A.base_arch.json"; p.write_text(json.dumps(dict(arch=BASE_ARCH, config=BASE_CFG, base_fp16_bytes=10**6))); return str(p)


@pytest.fixture
def make_art(tmp_path):
    def f(name="art", answer=ANSWER, feedback=FEEDBACK, extra=None):
        d = tmp_path / name; d.mkdir(); write_st(str(d / "model.safetensors"), BASE_ARCH, weight_byte=None)
        (d / "config.json").write_text(json.dumps(BASE_CFG)); (d / "answer.py").write_text(answer); (d / "feedback.py").write_text(feedback)
        for k, v in (extra or {}).items(): (d / k).parent.mkdir(parents=True, exist_ok=True); (d / k).write_bytes(v if isinstance(v, bytes) else v.encode())
        return str(d)
    return f


@pytest.fixture
def pools(tmp_path):
    ann, teach, nb = synth.make_pools()
    (tmp_path / "pools").mkdir(exist_ok=True)
    for n, r in (("ann", ann), ("teach", teach), ("nb", nb)): C.dump_jsonl(r, str(tmp_path / "pools" / f"{n}.jsonl"))
    return dict(ann=str(tmp_path / "pools/ann.jsonl"), teach=str(tmp_path / "pools/teach.jsonl"), nb=str(tmp_path / "pools/nb.jsonl")), (ann, teach, nb)


@pytest.fixture
def exam(tmp_path):
    p = tmp_path / "exams" / "exam.jsonl"; p.parent.mkdir(); C.dump_jsonl(synth.make_exam(), str(p)); return str(p)
