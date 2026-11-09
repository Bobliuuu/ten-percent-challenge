"""L9b final checks (run on Modal GPU via modal_final.py): (1) fresh-clone end-to-end answer+feedback, (2) own-data dry run (base vs packed on ~300 unrelated prompts) + unseen label Timer.
Pure python; results printed as RESULT_JSON and written to /r/l9b/res/. Artifact: /r/l9b/art_s7_k102_e3 (private volume)."""
import json, os, sys, time, random, subprocess, urllib.request, csv, io, shutil, gc
def now(): return time.strftime("%H:%M:%S")

def fresh_clone(clone_dir, commit):
    art = "/tmp/art_fresh"; env = dict(os.environ, L9B_SRC="/r/l9b")
    out = dict(commit=commit, steps={}); t = time.time()
    r = subprocess.run(["bash", f"{clone_dir}/setup_fresh_clone.sh", art], env=env, capture_output=True, text=True)
    out["steps"]["setup_and_smoke"] = dict(rc=r.returncode, secs=round(time.time() - t), stdout=r.stdout[-3000:], stderr=r.stderr[-1500:]); print(r.stdout[-2500:], r.stderr[-800:], flush=True)
    if r.returncode: return out
    ann = [json.loads(l) for l in open(f"{clone_dir}/data/skill7_fb_annot.jsonl")]
    rec = [a for a in ann if a["answer"] == "Recipe"][:12]; old = [a for a in ann if a["answer"] != "Recipe"][:13]
    with open("/tmp/ann.jsonl", "w") as f:
        for a in rec + old: f.write(json.dumps({"query": a["user"], "label": a["answer"]}) + "\n")
    # also test the documented {text,label} form with a task JSON
    with open("/tmp/ann_text.jsonl", "w") as f:
        for t_ in ["how do i bake chocolate chip cookies", "what is the best way to cook a turkey", "recipe for vegetable soup", "how long do i boil an egg", "how do i make pancakes from scratch"]: f.write(json.dumps({"text": t_, "label": "Recipe"}) + "\n")
    probes = ["how do i bake chocolate chip cookies", "give me a recipe for lasagna", "wake me up at seven", "what is the weather in paris", "play some jazz", "add milk to my shopping list", "when does the next bus leave"]
    t = time.time(); r = subprocess.run(["python3", f"{art}/answer.py", art] + probes, capture_output=True, text=True); out["steps"]["answer_before_feedback"] = dict(rc=r.returncode, secs=round(time.time() - t), stdout=r.stdout[-1500:], stderr=r.stderr[-600:]); print("BEFORE", r.stdout, flush=True)
    t = time.time(); r = subprocess.run(["python3", f"{art}/feedback.py", "--artifact", art, "--task", f"{art}/skill_task.json", "--ann", "/tmp/ann.jsonl"], capture_output=True, text=True); out["steps"]["feedback"] = dict(rc=r.returncode, secs=round(time.time() - t), stdout=r.stdout[-1500:], stderr=r.stderr[-1000:]); print("FB", r.stdout, r.stderr[-500:], flush=True)
    out["fb_bin_bytes"] = os.path.getsize(f"{art}/fb.bin") if os.path.exists(f"{art}/fb.bin") else None
    t = time.time(); r = subprocess.run(["python3", f"{art}/answer.py", art] + probes, capture_output=True, text=True); out["steps"]["answer_after_feedback"] = dict(rc=r.returncode, secs=round(time.time() - t), stdout=r.stdout[-1500:], stderr=r.stderr[-600:]); print("AFTER", r.stdout, flush=True)
    out["probes"] = probes; out["before"] = out["steps"]["answer_before_feedback"]["stdout"].split("\n"); out["after"] = out["steps"]["answer_after_feedback"]["stdout"].split("\n")
    os.makedirs("/r/l9b/res", exist_ok=True); json.dump(out, open("/r/l9b/res/fresh_clone_run.json", "w"), indent=1); print("RESULT_JSON", json.dumps(out), flush=True); return out

TIMER_T = ["set a timer for {d}", "start a {d} timer", "timer for {d} please", "can you count down {d} for me", "i need a {d} countdown timer", "begin a timer, {d}", "ping me when {d} are up", "how about a {d} timer"]
DUR = ["ten minutes", "5 minutes", "half an hour", "45 seconds", "two hours", "twenty minutes", "90 seconds", "an hour"]
def timer_sets():
    rows = [t.format(d=d) for t in TIMER_T for d in DUR]; random.Random(7).shuffle(rows); return rows[:20], rows[20:48]

BK_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
def banking_rows(n=200):
    txt = urllib.request.urlopen(BK_URL, timeout=60).read().decode(); rs = list(csv.DictReader(io.StringIO(txt))); random.Random(11).shuffle(rs)
    return [(r["text"], r["category"]) for r in rs[:n]]
def generic_prompts():
    P = []
    for a, b in [(17, 23), (48, 12), (305, 7), (91, 14), (56, 56), (123, 4), (77, 33), (250, 18), (64, 9), (19, 21)]: P.append(f"What is {a} times {b}? Answer with just the number.")
    for c in ["France", "Japan", "Brazil", "Kenya", "Canada", "Egypt", "Norway", "Peru", "India", "Australia"]: P.append(f"What is the capital of {c}? One word.")
    for w in ["happy", "large", "quickly", "ancient", "silent", "bright", "fragile", "generous", "rapid", "narrow"]: P.append(f"Give one antonym of '{w}'. One word.")
    for w in ["good morning", "thank you", "where is the station", "i would like some water", "how much does this cost", "see you tomorrow", "i am tired", "the book is on the table", "what time is it", "nice to meet you"]: P.append(f"Translate to French: {w}")
    for t in ["photosynthesis", "inflation", "a hash table", "gravity", "the TCP handshake", "a black hole", "compound interest", "DNA replication", "a binary search", "the water cycle"]: P.append(f"Explain {t} in one short sentence.")
    for t in ["reverse a string in Python", "sum a list in Python", "check if a number is prime in Python", "read a file in Python", "sort a dictionary by value in Python", "make an HTTP GET request in Python", "print the numbers 1 to 10 in Python", "count words in a string in Python", "flatten a nested list in Python", "get the current date in Python"]: P.append(f"Write a one-line Python snippet to {t}.")
    for s in ["The quick brown fox", "Once upon a time", "In the beginning", "It was a dark and stormy night", "The meeting is scheduled for", "Dear customer,", "To install the package, run", "The main advantage of", "Breaking news:", "Step one:"]: P.append(f"Continue this sentence in about ten words: {s}")
    for q in ["Is a tomato a fruit or a vegetable?", "Which is heavier, a kilogram of feathers or a kilogram of steel?", "Is 91 prime?", "How many days are in a leap year?", "Name the largest planet in the solar system.", "Who wrote Pride and Prejudice?", "What is the chemical symbol for gold?", "How many sides does a hexagon have?", "What language is spoken in Brazil?", "What is the boiling point of water in Celsius?"]: P.append(q)
    for t in ["a cat", "the ocean", "autumn", "coffee", "a train", "winter", "a library", "the moon", "friendship", "a city at night"]: P.append(f"Write a haiku about {t}.")
    for t in ["my landlord about a broken heater", "a colleague to thank them for their help", "a friend to invite them to dinner", "a hotel to ask about late checkout", "a teacher to ask for an extension", "customer support about a late delivery", "my manager to request a day off", "a neighbour about loud music", "a bank about a lost card", "a recruiter to decline an offer"]: P.append(f"Write a two-sentence email to {t}.")
    for t in ["bake bread", "tie a tie", "change a tire", "boil an egg", "plant a tree", "write a cover letter", "learn to swim", "make tea", "clean a keyboard", "start a vegetable garden"]: P.append(f"Give me three short tips to {t}.")
    return P

def own_data():
    import torch; sys.path.insert(0, "/root/l9b"); import core, packed_loader, fbcore
    from transformers import AutoTokenizer, AutoModelForCausalLM
    T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
    bk = banking_rows(200); labs = sorted({c for _, c in bk}); gen_p = generic_prompts()[:100]
    # full 77-label menu from the whole test csv categories
    txt = urllib.request.urlopen(BK_URL, timeout=60).read().decode(); menu = sorted({r["category"] for r in csv.DictReader(io.StringIO(txt))}); log("banking labels", len(menu))
    BP = "Classify this banking customer query into exactly one intent. Answer with only the intent name, chosen from: " + ", ".join(menu) + ".\n\nQuery: {t}"
    rows_bk = [dict(id=f"bk{i}", user=BP.replace("{t}", t), answer=c) for i, (t, c) in enumerate(bk)]
    rows_rt = [dict(id=f"rt{i}", user=core_t_prompt(t), answer="") for i, (t, _) in enumerate(bk[:100])]   # banking queries under the shipped 7-skill router prompt (no gold; agreement only)
    rows_gen = [dict(id=f"g{i}", user=p, answer="") for i, p in enumerate(gen_p)]
    skill = {r["id"]: r for r in core.rows_of("/root/l9b/data/skill7_sel.jsonl")}; sel = list(skill.values())
    res = dict(n_banking=len(rows_bk), n_router=len(rows_rt), n_generic=len(rows_gen), banking_menu=len(menu))
    def run_all(R, tag):
        o = {}
        o["bk"] = R.gen([dict(r) for r in rows_bk], max_new=12); log(tag, "bk done"); o["rt"] = R.gen([dict(r) for r in rows_rt], max_new=8); o["gen"] = R.gen([dict(r) for r in rows_gen], max_new=24); log(tag, "gen done"); return o
    # base
    tok = AutoTokenizer.from_pretrained(core.MID); tok.padding_side = "left"; ng = torch.cuda.device_count()
    base = AutoModelForCausalLM.from_pretrained(core.MID, dtype=torch.bfloat16, device_map="auto", max_memory={i: "74GiB" for i in range(ng)}).eval(); log("base loaded")
    OB = run_all(core.Runner(base, tok), "base"); del base; gc.collect(); torch.cuda.empty_cache()
    model, tok2, _ = packed_loader.load("/r/l9b/art_s7_k102_e3"); R = core.Runner(model, tok2); log("packed loaded")
    OP = run_all(R, "packed")
    cb = [core.norm(o) == r["answer"].lower() for o, r in zip(OB["bk"], rows_bk)]; cp = [core.norm(o) == r["answer"].lower() for o, r in zip(OP["bk"], rows_bk)]
    bc = sum(cb); both = sum(1 for a, b in zip(cb, cp) if a and b)
    res["banking77_200"] = dict(base_acc=sum(cb) / 200, packed_acc=sum(cp) / 200, retention=both / max(bc, 1), retention_wilson_lo=core.wilson(both, bc), n_base_correct=bc, same_output=sum(1 for a, b in zip(OB["bk"], OP["bk"]) if core.norm(a) == core.norm(b)) / 200)
    res["router_on_banking_100"] = dict(same_label=sum(1 for a, b in zip(OB["rt"], OP["rt"]) if core.norm(a) == core.norm(b)) / 100, base_dist=_dist(OB["rt"]), packed_dist=_dist(OP["rt"]))
    sm = [a.strip() == b.strip() for a, b in zip(OB["gen"], OP["gen"])]; fl = [core.norm(a) == core.norm(b) for a, b in zip(OB["gen"], OP["gen"])]
    res["generic_100"] = dict(identical_24tok=sum(sm) / 100, same_first_line=sum(fl) / 100)
    res["samples"] = dict(banking=[(rows_bk[i]["answer"], OB["bk"][i], OP["bk"][i]) for i in range(8)], generic=[(gen_p[i], OB["gen"][i], OP["gen"][i]) for i in range(0, 100, 10)], diverge=[(gen_p[i], OB["gen"][i], OP["gen"][i]) for i in range(100) if not fl[i]][:12])
    log("own-data", json.dumps({k: v for k, v in res.items() if k != "samples"})); json.dump(dict(res=res, outs=dict(base=OB, packed=OP)), open("/r/l9b/res/owndata_s7_k102_e3.json", "w")); 
    # unseen label Timer on the packed model
    pool, test = timer_sets(); alarm = [r for r in sel if r["answer"] == "Alarm"][:60]; others = [r for r in sel if r["answer"] != "Alarm"][::9][:60]
    T = json.load(open("/root/l9b/skill_task.json")); mk = lambda t: dict(user=T["prompt"].replace("{t}", t), answer="Timer")
    cal = core.rows_of("/root/l9b/feedback_cal.jsonl")[:300]; tr = [mk(t) for t in test]
    lay = list(range(len(model.model.layers) - 10, len(model.model.layers))); W0 = {l: model.model.layers[l].mlp.shared_expert.down_proj.weight.data.clone() for l in lay}
    def ev(tag):
        o = R.gen([dict(r) for r in tr]); a = R.gen([dict(r) for r in alarm]); x = R.gen([dict(r) for r in others])
        return dict(tag=tag, timer_emitted=sum(core.norm(v) == "timer" for v in o) / len(o), alarm_still_alarm=sum(core.norm(v) == "alarm" for v in a) / len(a), alarm_to_timer=sum(core.norm(v) == "timer" for v in a) / len(a), others_correct=sum(core.norm(v) == r["answer"].lower() for v, r in zip(x, others)) / len(others), others_to_timer=sum(core.norm(v) == "timer" for v in x) / len(x), timer_sample=o[:4])
    tout = dict(n_pool=len(pool), n_test=len(test), n_alarm_nb=len(alarm), n_other_nb=len(others), empty=ev("empty"), draws=[])
    log("timer empty", tout["empty"]); labels = ["Timer"] * 12
    for d in range(3):
        rng = random.Random(900 + d); ann = [mk(t) for t in rng.sample(pool, 12)]; dr = dict(draw=d)
        for cond in ("real", "shuffled", "wrong"):
            its = [dict(r) for r in ann]
            if cond == "shuffled": its = [dict(r, answer=("Alarm" if i % 2 else "Timer")) for i, r in enumerate(its)]   # half the annotations get a wrong label (label shuffle of a one-class set is degenerate)
            if cond == "wrong": its = [dict(r, answer=rng.choice(["Weather", "Music", "News", "Lists"])) for r in its]
            fbcore.apply_feedback(model, tok2, R, its, [dict(c) for c in cal], log=lambda *a: None); dr[cond] = ev(cond); log("timer", d, cond, {k: v for k, v in dr[cond].items() if k != "timer_sample"})
            for l in lay: model.model.layers[l].mlp.shared_expert.down_proj.weight.data.copy_(W0[l])
        tout["draws"].append(dr)
    res["timer_unseen_label"] = tout; json.dump(res, open("/r/l9b/res/owndata_summary.json", "w"), indent=1); print("RESULT_JSON", json.dumps(res), flush=True); return res
def core_t_prompt(t):
    T = json.load(open("/root/l9b/skill_task.json")); return T["prompt"].replace("{t}", t)
def _dist(outs):
    import collections, core; return dict(collections.Counter(core.norm(o) for o in outs).most_common(10))
