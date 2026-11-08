"""Checkers: (1) size, (2) no-new-parts (manifest, tensors, train/network grep), (3) leakage of labels into candidate-visible data.
Every checker returns a plain dict with a boolean `ok` and the evidence. No third-party deps (safetensors headers are parsed by hand)."""
import fnmatch, json, os, re, struct
import common3 as C

# ---------------------------------------------------------------- allowed file set
ALLOWED_PATTERNS = ["*.safetensors", "model.safetensors.index.json", "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                    "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja", "chat_template.json",
                    "tensor_manifest.json", "answer.py", "feedback.py", "feedback_calib.jsonl", "model.tpq"]   # model.tpq = TPQ1 entropy-coded weight container (v2/artifact/README.md)
TRAIN_RE = re.compile(r"\bbackward\b|\.backward\(|torch\.optim|\boptim\w*|\.grad\b|requires_grad_?\(\s*True|\bloss\b|\bAdam\w*|\bSGD\b|zero_grad|\.step\(\)", re.I)
NET_RE = re.compile(r"^\s*(import|from)\s+(socket|requests|urllib\S*|http\.client|httpx|aiohttp|websockets?|ftplib|smtplib|huggingface_hub|openai|anthropic)\b"
                    r"|https?://|snapshot_download|hf_hub_download|\bcurl\b|\bwget\b|from_pretrained\(\s*[\"'][\w.-]+/[\w.-]+[\"']|HF_HUB_OFFLINE\s*=\s*[\"']?0", re.M)


# ---------------------------------------------------------------- 1. size
def size_check(artifact, base_fp16_bytes, fraction=0.10):
    """ALL bytes on disk under the artifact dir (weights, tokenizer, config, code, hidden files). Symlinks are violations (they hide bytes)."""
    files, links, total = [], [], 0
    for root, dirs, fs in os.walk(artifact):
        for n in list(dirs) + fs:
            ap = os.path.join(root, n)
            if os.path.islink(ap): links.append(os.path.relpath(ap, artifact))
        for f in fs:
            ap = os.path.join(root, f)
            if os.path.islink(ap): continue
            b = os.stat(ap).st_size; total += b; files.append(dict(file=os.path.relpath(ap, artifact), bytes=b))
    budget = fraction * base_fp16_bytes
    return dict(candidate_bytes=total, base_fp16_bytes=base_fp16_bytes, byte_ratio=total / base_fp16_bytes, budget_bytes=budget, n_files=len(files),
                symlinks=sorted(links), ok=bool(total <= budget and not links), files=sorted(files, key=lambda r: r["file"]))


# ---------------------------------------------------------------- 2. no new parts
def read_st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 100_000_000: raise ValueError("implausible safetensors header")
        h = json.loads(f.read(n))
    h.pop("__metadata__", None); return h


def arch_of_dir(d):
    """{tensor_name: [dtype, shape]} from every *.safetensors header in dir (no weights read)."""
    out = {}
    for rel, ap in C.tree_files(d):
        if rel.endswith(".safetensors"):
            for k, v in read_st_header(ap).items():
                if k in out: out[k + "@dup:" + rel] = [v["dtype"], v["shape"]]
                else: out[k] = [v["dtype"], v["shape"]]
    return out


def _py_files(artifact): return [(rel, ap) for rel, ap in C.tree_files(artifact) if rel.endswith(".py")]


def grep_hits(paths, regex):
    hits = []
    for rel, ap in paths:
        for i, line in enumerate(open(ap, encoding="utf-8", errors="replace"), 1):
            if regex.search(line): hits.append(f"{rel}:{i}: {line.strip()[:120]}")
    return hits


CONSTRAINED_RE = re.compile(r"logits_processor|prefix_allowed_tokens_fn|guided_|outlines|lm_format|grammar|json_schema|response_format|bad_words|force_words|allowed_tokens", re.I)
CONFIG_ARCH_KEYS = ["architectures", "model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "intermediate_size",
                    "vocab_size", "head_dim", "tie_word_embeddings", "num_experts", "moe_intermediate_size"]


def no_new_parts(artifact, base_arch, base_config=None, allowed_extra=(), build_dirs=(), strict_dtype=True):
    """base_arch: {name:[dtype,shape]} of the base. Returns extra_files / tensor diffs / grep hits.
    A compressed artifact with no plain safetensors must ship tensor_manifest.json {name:[dtype,shape]} (logical tensors); then the tensor
    comparison runs on the manifest and `tensor_source` = 'declared_manifest' (the decode itself is verified elsewhere)."""
    rels = [r for r, _ in C.tree_files(artifact)]
    def allowed(r):
        if any(fnmatch.fnmatch(r, p) for p in allowed_extra): return True
        return "/" not in r and any(fnmatch.fnmatch(r, p) for p in ALLOWED_PATTERNS)   # no subdirs unless explicitly allowed
    extra = [r for r in rels if not allowed(r)]
    extra = sorted(set(extra))
    arch = arch_of_dir(artifact); src = "safetensors_headers"
    if not arch and os.path.exists(os.path.join(artifact, "tensor_manifest.json")):
        arch = json.load(open(os.path.join(artifact, "tensor_manifest.json"))); src = "declared_manifest"
    extra_t = sorted(set(arch) - set(base_arch)); missing_t = sorted(set(base_arch) - set(arch))
    bad_shape = sorted(k for k in set(arch) & set(base_arch) if list(arch[k][1]) != list(base_arch[k][1]))
    bad_dtype = sorted(k for k in set(arch) & set(base_arch) if arch[k][0] != base_arch[k][0]) if strict_dtype else []
    cfg_diff = []
    if base_config is not None:
        try: ac = json.load(open(os.path.join(artifact, "config.json")))
        except Exception: ac = {}
        cfg_diff = [k for k in CONFIG_ARCH_KEYS if (k in base_config or k in ac) and base_config.get(k) != ac.get(k)]
    code = _py_files(artifact) + [(os.path.relpath(ap, d), ap) for d in build_dirs for _, ap in C.tree_files(d) if ap.endswith(".py")]
    train_hits = grep_hits(code, TRAIN_RE); net_hits = grep_hits(_py_files(artifact), NET_RE)
    cons_hits = grep_hits(_py_files(artifact), CONSTRAINED_RE)
    ok = not (cons_hits or extra or extra_t or missing_t or bad_shape or bad_dtype or cfg_diff or train_hits or net_hits)
    return dict(ok=ok, tensor_source=src, extra_files=extra, extra_tensors=extra_t, missing_tensors=missing_t, shape_mismatch=bad_shape, dtype_mismatch=bad_dtype,
                config_arch_diff=cfg_diff, grep_train_hits=len(train_hits), grep_train_lines=train_hits[:20], network_hits=len(net_hits), network_lines=net_hits[:20], constrained_decoding_hits=len(cons_hits), constrained_lines=cons_hits[:20],
                n_tensors=len(arch))


# ---------------------------------------------------------------- 3. leakage
def _walk_keys(o):
    if isinstance(o, dict):
        for k, v in o.items(): yield k; yield from _walk_keys(v)
    elif isinstance(o, list):
        for x in o: yield from _walk_keys(x)


def _strings(o):
    if isinstance(o, str): yield o
    elif isinstance(o, dict):
        for k, v in o.items(): yield str(k); yield from _strings(v)
    elif isinstance(o, list):
        for x in o: yield from _strings(x)


def leak_scan(view_rows, exam_rows, canary=None):
    """view_rows = exactly what is handed to base/candidate. Fails on: extra top-level keys, label-field names nested anywhere, the canary
    string, or a verbatim serialisation of a gold call (name+arguments) inside the item's own view."""
    probs = []
    for v, r in zip(view_rows, exam_rows):
        extra = set(v) - set(C.VIEW_FIELDS)
        if extra: probs.append((v.get("id"), f"extra keys {sorted(extra)}"))
        nested = set(_walk_keys(v.get("tools", []))) & {"gold", "gold_alts", "ground_truth", "_canary"}   # 'source'/'kind'/'label' are legitimate parameter names in real tools
        if nested: probs.append((v.get("id"), f"label-like keys inside tools {sorted(nested)}"))
        s = json.dumps(v, ensure_ascii=False, sort_keys=True) + "\n" + "\n".join(_strings(v))     # serialised AND raw string values
        if canary and canary in s: probs.append((v.get("id"), "canary in view"))
        for g in r.get("gold", []):
            for form in (json.dumps(dict(name=g["name"], arguments=g["arguments"]), ensure_ascii=False), json.dumps(dict(name=g["name"], arguments=g["arguments"]), ensure_ascii=False, sort_keys=True)):
                if g["arguments"] and form in s: probs.append((v.get("id"), "gold call serialised in view")); break
    return dict(ok=not probs, problems=probs[:20], n_checked=len(view_rows))


def prompt_leak_scan(prompts, exam_rows, canary=None):
    """prompts: the rendered prompts the artifact actually fed its model (optional --dump-prompts). Same gold-call + canary test, plus: no
    gold call (with args) of ANY exam row appears in ANY prompt (catches cross-row lookup tables)."""
    golds = [json.dumps(dict(name=g["name"], arguments=g["arguments"]), ensure_ascii=False) for r in exam_rows for g in r.get("gold", []) if g["arguments"]]
    probs = []
    for i, p in enumerate(prompts):
        if canary and canary in p: probs.append((i, "canary in prompt"))
        for g in golds:
            if g in p: probs.append((i, "gold call in prompt")); break
    return dict(ok=not probs, problems=probs[:20], n_checked=len(prompts))


def scan_text_for_canary(texts, canary):
    return dict(ok=not any(canary in t for t in texts), n_checked=len(texts))
