#!/usr/bin/env python3
"""Per-lane final checks run against a fresh clone (called by fresh_clone_check.sh; can also be run directly with --clone DIR).
Checks per lane: bytes (<=10% of fp16 with all shipped bytes), leak (v2/tools/final_leak_check.py), sha (MANIFEST / SHA256SUMS), smoke (answer.py + feedback.py).
Statuses: PASS, FAIL, SKIPPED (never silently PASS). Artifact-dependent checks are SKIPPED unless the artifact is supplied (--artifact-root DIR/<lane> or --artifact-dir DIR).
  bytes : artifact present -> real v2/tools/bytes_audit.py on it. Absent -> SKIPPED with the MANIFEST-declared total/denominator (FAIL if the declared ratio is already > cap).
  sha   : repo files listed in MANIFEST/SHA256SUMS are hashed (mismatch = FAIL); listed artifacts not in the clone -> 'artifact not in clone: needs rebuild' (SKIPPED unless something mismatched).
  smoke : needs artifact + nvidia-smi, else SKIPPED-NEEDS-GPU. --smoke-cmd skip forces SKIPPED; --smoke-cmd "CMD" replaces the answer command ({art} {lane_dir} {txt} {txtfile} {ann} substituted).
"""
import argparse, datetime, hashlib, json, os, re, shutil, subprocess, sys, tempfile

CAP = 10.0
# lane -> declared totals from the lane MANIFEST.json (used ONLY when the artifact is not present), denominators = fp16 bytes of the start model.
LANES = {
 "L9b": dict(denom=69321221376, declared=6726881085, src="MANIFEST total_with_feedback; denominator text-only fp16 (stricter than full 71.9 GB)",
             ans='python3 answer.py {art} --task skill_task.json "{txt}"', fb='python3 feedback.py --artifact {art} --ann {ann} --task skill_task.json'),
 "L1":  dict(denom=16381470720, declared=1218467357, src="MANIFEST total_bytes_with_feedback_patch; denominator Qwen3-8B fp16 (MANIFEST 7.438%)",
             ans='python3 answer.py --artifact {art} --task skill_task.json "{txt}"', fb='python3 feedback.py --artifact {art} --task skill_task.json --ann {ann}'),
 "L5":  dict(denom=65527752704, declared=6420039377 + 83918848 + 36000, src="MANIFEST bytes.packed_body_real+carrier+code",
             ans='python3 answer.py {art} "{txt}"', fb=None),
 "L3":  dict(denom=61064245248, declared=6042759448, src="MANIFEST skill7x.total_bytes",
             ans='python3 answer.py {art} {txtfile}', fb='python3 feedback.py {art} {ann}'),
 "L9":  dict(denom=470187269120, declared=45990926086 + 537001984, src="MANIFEST shipped_bytes + feedback carrier (537,001,984, counted conservatively)",
             ans='python3 answer.py {art} "{txt}"', fb='python3 feedback.py {art} {ann_user} real'),
 "L4":  dict(denom=61064245248, declared=6042513585, src="MANIFEST bytes_SKILL7_BANK6 total (ESTIMATE from bit counts; no packed exporter yet)",
             ans='python3 answer.py --model {art} "{txt}"', fb=None),
 "L10": dict(denom=21464450880, declared=2131462551, src="MANIFEST total_bytes (accounting estimate from entropy; packed file not written)",
             ans='python3 answer.py --artifact {art} "{txt}"', fb='python3 feedback.py --artifact {art} --ann {ann}'),
 "L2":  dict(denom=53808408928, declared=5346244844, src="MANIFEST shipped_total_bytes (denominator back-computed from artifact report)",
             ans='python3 answer.py {art} "{txt}"', fb='python3 feedback.py --artifact {art} --ann {ann}'),
}
ORDER = ["L9b", "L1", "L5", "L3", "L9", "L4", "L10", "L2"]
SMOKE_TXT = "set an alarm for seven tomorrow morning"

def run(cmd, cwd=None, timeout=3600, env=None):
    p = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout, env=env)
    return p.returncode, (p.stdout or "") + (p.stderr or "")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()

def find_artifact(lane, a):
    if a.artifact_dir and len(a.lanes) == 1: return a.artifact_dir
    if a.artifact_root and os.path.isdir(os.path.join(a.artifact_root, lane)): return os.path.join(a.artifact_root, lane)
    return None

# ---------- check 1: bytes
def check_bytes(lane, cfg, art, clone):
    pct_decl = 100.0 * cfg["declared"] / cfg["denom"]
    if art:
        rc, out = run([sys.executable, f"{clone}/v2/tools/bytes_audit.py", art, "--base-fp16-bytes", str(cfg["denom"]), "--cap", str(CAP)])
        tail = [l for l in out.strip().splitlines() if l.strip()][-1:] or [""]
        return dict(status="PASS" if rc == 0 else "FAIL", detail=f"bytes_audit.py on artifact rc={rc}: {tail[0][:200]}")
    st = "FAIL" if pct_decl > CAP else "SKIPPED"
    return dict(status=st, detail=f"artifact not in clone; MANIFEST-declared {cfg['declared']:,} B = {pct_decl:.3f}% of fp16 {cfg['denom']:,} B (cap {CAP}%) [{cfg['src']}]; real bytes_audit needs artifact")

# ---------- check 2: leak (global scan once, attributed per lane)
_LEAK = {}
def leak_global(clone):
    if _LEAK: return _LEAK
    rc, out = run([sys.executable, f"{clone}/v2/tools/final_leak_check.py"], timeout=1800)
    hits = {}; n = None
    for l in out.splitlines():
        m = re.match(r"\s+(\S+): (\d+) hits", l)
        if m: hits[m.group(1)] = int(m.group(2))
        m = re.match(r"FILES WITH EXACT EXAM-TEXT OVERLAP: (\d+)", l)
        if m: n = int(m.group(1))
    _LEAK.update(rc=rc, hits=hits, n=n, head="; ".join(out.splitlines()[:1] + [l for l in out.splitlines() if l.startswith("files scanned")]), raw=out)
    return _LEAK

def check_leak(lane, clone):
    g = leak_global(clone)
    if g["rc"] != 0 or g["n"] is None: return dict(status="FAIL", detail=f"final_leak_check.py rc={g['rc']} (no result line): {g['raw'][-200:]}")
    mine = {p: c for p, c in g["hits"].items() if p.startswith(f"lanes/{lane}/")}
    if mine: return dict(status="FAIL", detail=f"{sum(mine.values())} exact exam-text hits in {sorted(mine)[:3]}")
    return dict(status="PASS", detail=f"0 hits under lanes/{lane}/ ({g['head']}; repo-wide hit files: {g['n']}" + (f", others: {sorted(g['hits'])[:3]}" if g["hits"] else "") + ")")

# ---------- check 3: sha256
def sha_entries(lane_dir):
    """list of (kind, name, expected_sha, source). kind repo|artifact. Sources: MANIFEST.json *sha256* dicts, SHA256SUMS* files."""
    ents = []
    mp = os.path.join(lane_dir, "MANIFEST.json")
    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                if "sha256" in k.lower() and isinstance(v, dict):
                    kind = "repo" if "repo" in k.lower() else "artifact"
                    for n, h in v.items():
                        if isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h): ents.append((kind, n, h, f"MANIFEST.json:{k}"))
                else: walk(v, path + [k])
    if os.path.exists(mp):
        try: walk(json.load(open(mp)), [])
        except Exception as e: ents.append(("error", str(e), "", "MANIFEST.json"))
    for dp, _, fs in os.walk(lane_dir):
        for fn in fs:
            if fn.startswith("SHA256SUMS"):
                rel = os.path.relpath(os.path.join(dp, fn), lane_dir)
                base = os.path.dirname(rel)
                for l in open(os.path.join(dp, fn)):
                    p = l.split()
                    if len(p) >= 2 and re.fullmatch(r"[0-9a-f]{64}", p[0]):
                        # entries whose name sits next to a SHA256SUMS in a build/ dir refer to files in that dir; else lane dir
                        ents.append(("sums", os.path.join(base, p[1].lstrip("*")), p[0], rel))
    return ents

def locate(lane_dir, art, name):
    cands = [os.path.join(lane_dir, name), os.path.join(lane_dir, "ship", name), os.path.join(lane_dir, os.path.basename(name))]
    alt = re.sub(r"_([a-z0-9]+)$", r".\1", name)            # body_bin -> body.bin
    cands += [os.path.join(lane_dir, alt)]
    if art: cands += [os.path.join(art, name), os.path.join(art, alt), os.path.join(art, os.path.basename(name))]
    for c in cands:
        # case-exact match (macOS is case-insensitive: manifest_json must not resolve to MANIFEST.json)
        if os.path.isfile(c) and os.path.basename(c) in os.listdir(os.path.dirname(c)): return c
    return None

def check_sha(lane, clone, art):
    ld = f"{clone}/v2/lanes/{lane}"; ents = sha_entries(ld)
    if not ents: return dict(status="SKIPPED", detail="no sha256 entries in MANIFEST.json and no SHA256SUMS: nothing to verify (gap: lane lists no hashes)")
    ok = bad = 0; miss = []; badl = []
    for kind, name, h, src in ents:
        if kind == "error": badl.append(f"unparseable {src}: {name[:60]}"); continue
        p = locate(ld, art, name)
        if p is None: miss.append(name); continue
        if sha256(p) == h: ok += 1
        else: bad += 1; badl.append(f"{name} ({src})")
    d = f"{ok} verified OK, {bad} MISMATCH, {len(miss)} not in clone"
    if miss: d += ": artifact not in clone: needs rebuild [" + ", ".join(sorted(set(miss))[:6]) + (" ..." if len(set(miss)) > 6 else "") + "]"
    if badl: d += "; MISMATCH: " + ", ".join(badl[:6])
    if bad or any(e[0] == "error" for e in ents): return dict(status="FAIL", detail=d)
    if miss: return dict(status="SKIPPED", detail=d)
    return dict(status="PASS", detail=d)

# ---------- check 4: smoke
def have_gpu(): return shutil.which("nvidia-smi") is not None and subprocess.call(["nvidia-smi", "-L"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0

def check_smoke(lane, cfg, art, clone, a):
    if a.smoke_cmd == "skip": return dict(status="SKIPPED", detail="--smoke-cmd skip (SKIPPED-NEEDS-GPU: not run)")
    if not art: return dict(status="SKIPPED", detail="SKIPPED-NEEDS-GPU: artifact not in clone (pass --artifact-root DIR/<lane> on a GPU box); not run")
    if not have_gpu(): return dict(status="SKIPPED", detail="SKIPPED-NEEDS-GPU: artifact present but no CUDA device here; not run")
    ld = f"{clone}/v2/lanes/{lane}"; tmp = tempfile.mkdtemp(prefix=f"smoke_{lane}_")
    ann = os.path.join(tmp, "ann.jsonl"); txtf = os.path.join(tmp, "msgs.txt"); open(txtf, "w").write(SMOKE_TXT + "\n")
    open(ann, "w").write(json.dumps({"text": SMOKE_TXT, "label": "Alarm"}) + "\n")
    annu = os.path.join(tmp, "ann_user.jsonl"); open(annu, "w").write(json.dumps({"user": "Request: " + SMOKE_TXT, "answer": "Alarm"}) + "\n")
    sub = dict(art=art, lane_dir=ld, txt=SMOKE_TXT, txtfile=txtf, ann=ann, ann_user=annu)
    acmd = (a.smoke_cmd or cfg["ans"]).format(**sub); notes = []
    rc, out = run(acmd, cwd=ld, timeout=7200)
    if rc != 0 or not out.strip(): return dict(status="FAIL", detail=f"answer cmd rc={rc}: {out.strip()[-200:]}")
    notes.append(f"answer rc=0 out='{out.strip().splitlines()[-1][:60]}'")
    if cfg["fb"] is None: return dict(status="SKIPPED", detail="answer ok but no standalone feedback smoke entry for this lane (" + notes[0] + "); not PASS")
    rc, out = run(cfg["fb"].format(**sub), cwd=ld, timeout=7200)
    if rc != 0: return dict(status="FAIL", detail=f"answer ok; feedback rc={rc}: {out.strip()[-200:]}")
    return dict(status="PASS", detail="; ".join(notes) + "; feedback rc=0 (note: feedback may write into the artifact dir)")

# ---------- checklist table
BEG, END = "<!-- FCC-TABLE-BEGIN -->", "<!-- FCC-TABLE-END -->"
def update_checklist(path, results, commit, date):
    rows = ["| lane | bytes | leak | sha256 | smoke | date | commit |", "|---|---|---|---|---|---|---|"]
    for r in results:
        c = r["checks"]; rows.append(f"| {r['lane']} | " + " | ".join(c[k]["status"] for k in ("bytes", "leak", "sha", "smoke")) + f" | {r['date']} | {r['commit']} |")
    notes = []
    for r in results:
        notes.append(f"- **{r['lane']}**: " + " / ".join(f"{k}: {r['checks'][k]['detail']}" for k in ("bytes", "leak", "sha", "smoke")))
    block = BEG + "\n" + "\n".join(rows) + "\n\nDetails (last run):\n\n" + "\n".join(notes) + "\n" + END
    txt = open(path).read() if os.path.exists(path) else ""
    if BEG in txt and END in txt:
        pre, rest = txt.split(BEG, 1); _, post = rest.split(END, 1)
        # keep rows of lanes not re-run this time
        old = {}
        for l in rest.split(END)[0].splitlines():
            m = re.match(r"\| (L\w+) \|", l)
            if m: old[m.group(1)] = l
        ran = {r["lane"] for r in results}
        oldnotes = {}
        for l in rest.split(END)[0].splitlines():
            m = re.match(r"- \*\*(L\w+)\*\*:", l)
            if m: oldnotes[m.group(1)] = l
        rows2 = rows[:2]; notes2 = []
        for L in ORDER:
            if L in ran: rows2.append(next(x for x in rows[2:] if x.startswith(f"| {L} |"))); notes2.append(next(x for x in notes if x.startswith(f"- **{L}**")))
            else:
                if L in old: rows2.append(old[L])
                if L in oldnotes: notes2.append(oldnotes[L])
        block = BEG + "\n" + "\n".join(rows2) + "\n\nDetails (per lane, from its last run):\n\n" + "\n".join(notes2) + "\n" + END
        open(path, "w").write(pre + block + post)
    else:
        open(path, "w").write(txt + ("\n" if txt and not txt.endswith("\n") else "") + block + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", required=True); ap.add_argument("--repo", required=True); ap.add_argument("--commit", required=True)
    ap.add_argument("--lane", action="append"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--artifact-root"); ap.add_argument("--artifact-dir"); ap.add_argument("--smoke-cmd")
    ap.add_argument("--out"); ap.add_argument("--no-checklist", action="store_true"); a = ap.parse_args()
    a.lanes = ORDER if a.all or not a.lane else a.lane
    for l in a.lanes:
        if l not in LANES: sys.exit(f"unknown lane {l}; known: {ORDER}")
    out = a.out or os.path.join(a.repo, "v2/tools/fresh_clone_results.json")
    date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"); results = []
    for lane in a.lanes:
        cfg = LANES[lane]; art = find_artifact(lane, a)
        if not os.path.isdir(f"{a.clone}/v2/lanes/{lane}"):
            r = dict(status="FAIL", detail="lane dir missing from clone (not pushed?)"); checks = {k: r for k in ("bytes", "leak", "sha", "smoke")}
        else:
            checks = dict(bytes=check_bytes(lane, cfg, art, a.clone), leak=check_leak(lane, a.clone), sha=check_sha(lane, a.clone, art), smoke=check_smoke(lane, cfg, art, a.clone, a))
        results.append(dict(lane=lane, commit=a.commit, date=date, artifact=bool(art), checks=checks))
        print(f"{lane:4} " + "  ".join(f"{k}={checks[k]['status']}" for k in checks))
    # merge into previous JSON so single-lane runs are idempotent
    prev = {}
    if os.path.exists(out):
        try: prev = {r["lane"]: r for r in json.load(open(out))["lanes"]}
        except Exception: prev = {}
    for r in results: prev[r["lane"]] = r
    json.dump(dict(generated=date, lanes=[prev[l] for l in ORDER if l in prev]), open(out, "w"), indent=1)
    if not a.no_checklist: update_checklist(os.path.join(a.repo, "v2/FINAL_CHECKLIST.md"), results, a.commit, date)
    print("results ->", out)
    return 1 if any(c["status"] == "FAIL" for r in results for c in r["checks"].values()) else 0

if __name__ == "__main__": sys.exit(main())
