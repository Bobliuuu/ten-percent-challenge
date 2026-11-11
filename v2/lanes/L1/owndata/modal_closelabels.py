"""L1 own-data stress test: unseen labels close to / far from existing skills. Artifact = PUBLIC HF Bobliu/tenpct-l1-bonsai8b-grow @ d5f0329d... with the shipped
feedback.py/answer.py code from that artifact (NOT the repo copies), default `scale` carrier, in-process compute_patch (same call as feedback.py CLI). Greedy exact-string scoring.
  modal run --detach v2/lanes/L1/owndata/modal_closelabels.py"""
import modal, os
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
REV = "d5f0329dee86c10cd59ed5a94a04393c54fa0e18"
app = modal.App("tenpct-l1-closelabels")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "transformers==4.56.2", "numpy", "accelerate", "huggingface_hub")
       .add_local_dir(R + "/v2/lanes/L1/owndata/closelabels_data", "/repo/data")
       .add_local_file(R + "/v2/lanes/L9b/final_jobs.py", "/repo/L9b/final_jobs.py").add_local_file(R + "/v2/lanes/L9b/data/skill7_sel.jsonl", "/repo/L9b/skill7_sel.jsonl"))
LABELS = ["Timer", "Reminder", "Podcast", "Shopping", "Traffic", "Sports", "Fitness", "Translate"]
SRC = {"Timer": "Alarm", "Reminder": "Calendar", "Podcast": "Music", "Shopping": "Lists", "Traffic": "Transport", "Sports": "News"}
@app.function(image=img, gpu="L40S", timeout=8 * 3600)
def run():
    import sys, json, random, time, collections, math, os, torch
    from huggingface_hub import snapshot_download
    art = snapshot_download("Bobliu/tenpct-l1-bonsai8b-grow", revision=REV, local_dir="/work/art")
    sys.path.insert(0, "/repo/L9b"); import final_jobs as FJ
    sys.path.insert(0, art); import bonsai_loader as BL, common, feedback as FB
    from transformers import AutoTokenizer
    assert not os.path.exists(art + "/patch.pt")
    t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
    norm = lambda s: s.strip().lower().split("\n")[0].strip(" .,:;\"'`*") if s.strip() else ""
    def wilson(k, n, z=1.96):
        if n == 0: return 0.0
        p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - r) / d
    m, _ = BL.load(art, patch=None); tok = AutoTokenizer.from_pretrained(art); T = common.task(art + "/skill_task.json")
    cal = [common.prompt_of(json.loads(l), T) for l in open(art + "/feedback_calib.jsonl") if l.strip()]
    gen = lambda P, mx=14: common.generate(m, tok, P, mx=mx)
    sel = [json.loads(l) for l in open("/repo/L9b/skill7_sel.jsonl")]; rng0 = random.Random(5); by = collections.defaultdict(list)
    for r in sel: by[r["answer"]].append(r)
    nb = []
    for k in sorted(by): rng0.shuffle(by[k]); nb += by[k][:30]          # 30 per existing skill = 210 neighbours
    nb_prompts = [r["user"] for r in nb]; nb_gold = [r["answer"].lower() for r in nb]
    shuf_pool = [r for k in sorted(by) for r in by[k][30:60]]            # unrelated ordinary inputs for the shuffled control (disjoint from neighbours)
    def text_of(u): return u.split("Request: ", 1)[1]
    D = {l: json.load(open(f"/repo/data/{l.lower()}.json")) for l in LABELS}
    base_nb = gen(nb_prompts); base_ok = [norm(o) == g for o, g in zip(base_nb, nb_gold)]; log("base neighbours acc", sum(base_ok) / len(nb))
    base_test = {l: gen([T["prompt"].replace("{t}", t) for t in D[l]["test"]]) for l in LABELS}
    base_dist = {l: dict(collections.Counter(norm(o) for o in base_test[l]).most_common(8)) for l in LABELS}
    base_emit = {l: sum(norm(o) == l.lower() for o in base_test[l]) / len(D[l]["test"]) for l in LABELS}
    log("base emit", base_emit)
    def evalp(l):
        o = gen([T["prompt"].replace("{t}", t) for t in D[l]["test"]]); n = gen(nb_prompts)
        lw = l.lower(); kinds = sorted(by)
        teach = sum(norm(x) == lw for x in o) / len(o)
        kept = [norm(x) == g for x, g in zip(n, nb_gold)]; both = sum(1 for a, b in zip(kept, base_ok) if a and b); bc = sum(base_ok)
        per = {}
        for k in kinds:
            ix = [i for i in range(len(nb)) if nb_gold[i] == k.lower()]
            bk = [i for i in ix if base_ok[i]]
            per[k] = dict(n=len(ix), kept=sum(kept[i] for i in bk) / max(len(bk), 1), n_base_correct=len(bk), to_new=sum(norm(n[i]) == lw for i in ix) / len(ix), base_to_new=sum(norm(base_nb[i]) == lw for i in ix) / len(ix))
        offm = sum(norm(x) not in [k.lower() for k in T["labels"]] + [lw] for x in n) / len(n)
        return dict(teach=teach, kept=both / bc, kept_n=f"{both}/{bc}", kept_wilson_lo=wilson(both, bc), per_skill=per, nb_to_new=sum(norm(x) == lw for x in n) / len(n), nb_offmenu_other=offm, test_dist=dict(collections.Counter(norm(x) for x in o).most_common(6)), sample=o[:5])
    def with_patch(ann, f):
        P = FB.compute_patch(m, tok, ann, T, cal, carrier="scale")
        if P is None: return dict(skipped="no site needs teaching")
        try: return f()
        finally: FB.undo(m, P)
    res = dict(rev=REV, n_neighbours=len(nb), base_nb_acc=sum(base_ok) / len(nb), base_emit=base_emit, base_dist=base_dist, labels={})
    for l in LABELS:
        rl = dict(runs=[]); log("LABEL", l)
        for dn in ("drawA", "drawB", "drawC"):
            pool = D[l][dn]; assert len(pool) == 25
            for n in (25, 12):
                items = [dict(text=t, label=l) for t in (pool if n == 25 else random.Random(7).sample(pool, 12))]
                r = with_patch(items, lambda: evalp(l)); r.update(draw=dn, n=n, cond="real"); rl["runs"].append(r); log(l, dn, n, "real", {k: r.get(k) for k in ("teach", "kept", "nb_to_new")})
            # shuffled control: the same 25 label words attached to ordinary unrelated inputs
            rs = random.Random(ord(dn[-1])); sh = [dict(text=text_of(r_["user"]), label=l) for r_ in rs.sample(shuf_pool, 25)]
            r = with_patch(sh, lambda: evalp(l)); r.update(draw=dn, n=25, cond="shuffled_inputs"); rl["runs"].append(r); log(l, dn, "shuffled", {k: r.get(k) for k in ("teach", "kept", "nb_to_new")})
            # shuffled labels: the annotated requests of the draw, labels permuted among old skills (wrong)
            wr = random.Random(3); wl = [dict(text=t, label=wr.choice(["Weather", "Music", "News", "Lists", "Alarm", "Calendar", "Transport"])) for t in pool]
            r = with_patch(wl, lambda: evalp(l)); r.update(draw=dn, n=25, cond="wrong_old_labels"); rl["runs"].append(r); log(l, dn, "wrong", {k: r.get(k) for k in ("teach", "kept", "nb_to_new")})
        res["labels"][l] = rl
        json.dump(res, open("/work/partial.json", "w"))
        print("PARTIAL_JSON", l, json.dumps(rl), flush=True)
    # empty control = no patch: re-generate to check determinism
    res["empty_identical"] = gen(nb_prompts) == base_nb
    # Part 3: generic + Banking77 retention for Timer and Translate (drawA, 25, scale)
    import urllib.request, csv, io
    bk = FJ.banking_rows(200); txt = urllib.request.urlopen(FJ.BK_URL, timeout=60).read().decode(); menu = sorted({r["category"] for r in csv.DictReader(io.StringIO(txt))})
    BP = "Classify this banking customer query into exactly one intent. Answer with only the intent name, chosen from: " + ", ".join(menu) + ".\n\nQuery: {t}"
    P_bk = [BP.replace("{t}", t) for t, _ in bk]; gold = [c for _, c in bk]; P_rt = [T["prompt"].replace("{t}", t) for t, _ in bk[:100]]; P_gen = FJ.generic_prompts()[:100]
    own = lambda: dict(bk=gen(P_bk, 20), rt=gen(P_rt, 8), gen=gen(P_gen, 24)); OB = own(); p3 = {}
    for l in ("Timer", "Translate", "Shopping"):
        items = [dict(text=t, label=l) for t in D[l]["drawA"]]
        def f():
            O = own(); cb = [norm(o) == g.lower() for o, g in zip(OB["bk"], gold)]; cp = [norm(o) == g.lower() for o, g in zip(O["bk"], gold)]; bc = sum(cb); both = sum(1 for a, b in zip(cb, cp) if a and b)
            tr = sum(1 for o in O["rt"] if norm(o) == l.lower())
            return dict(banking_base_acc=bc / 200, banking_patched_acc=sum(cp) / 200, banking_retention=both / bc, banking_n_base_correct=bc, banking_wilson_lo=wilson(both, bc), banking_same_output=sum(norm(a) == norm(b) for a, b in zip(OB["bk"], O["bk"])) / 200,
                        router_on_banking_same=sum(norm(a) == norm(b) for a, b in zip(OB["rt"], O["rt"])) / 100, router_on_banking_new_label=tr, generic_identical_24tok=sum(a.strip() == b.strip() for a, b in zip(OB["gen"], O["gen"])) / 100,
                        generic_same_first_line=sum(norm(a) == norm(b) for a, b in zip(OB["gen"], O["gen"])) / 100, generic_new_label_in_output=sum(l.lower() in b.lower() and l.lower() not in a.lower() for a, b in zip(OB["gen"], O["gen"])),
                        diverge=[(P_gen[i], OB["gen"][i], O["gen"][i]) for i in range(100) if norm(OB["gen"][i]) != norm(O["gen"][i])][:10])
        p3[l] = with_patch(items, f); log("part3", l, {k: v for k, v in p3[l].items() if k != "diverge"})
    res["part3"] = p3; res["part3_menu77"] = len(menu)
    out = json.dumps(res, indent=1); print("RESULT_DONE", flush=True); return out
@app.local_entrypoint()
def main():
    r = run.remote(); d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"); open(os.path.join(d, "owndata_closelabels_full.json"), "w").write(r); print(r[:1500])
