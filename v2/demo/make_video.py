"""Render the narrated explainer video for the 10% challenge proof.

Usage (single process, light):  nice -n 10 python3 make_video.py --data demo_outputs.json --out out/explainer.mp4
Needs: macOS `say` (narration), ffmpeg, Pillow. Numbers come from the committed receipts and demo_outputs.json.
Scenes 5 and 6 are driven by REAL model outputs in demo_outputs.json (see demo_run.py).
"""
import argparse, json, math, os, re, subprocess, sys, wave
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30
BG, BG2 = (11, 16, 32), (20, 28, 56)
CARD, CARD2 = (22, 30, 58), (30, 40, 76)
TXT, MUT = (232, 236, 248), (141, 151, 184)
GRN, BLU, ORG, RED = (61, 220, 151), (91, 140, 255), (255, 180, 84), (255, 107, 122)
FD = "/System/Library/Fonts/Supplemental/"
_fc = {}


def font(sz, bold=False, mono=False):
    k = (sz, bold, mono)
    if k not in _fc:
        p = "/System/Library/Fonts/Menlo.ttc" if mono else FD + ("Arial Bold.ttf" if bold else "Arial.ttf")
        _fc[k] = ImageFont.truetype(p, sz)
    return _fc[k]


def mix(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def clamp(x):
    return max(0.0, min(1.0, x))


def ease(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def rev(t, t0, d=0.6):
    return ease((t - t0) / d)


def wrap(txt, f, maxw):
    words, lines, cur = txt.split(), [], ""
    for w in words:
        c = (cur + " " + w).strip()
        if f.getlength(c) <= maxw:
            cur = c
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def make_bg():
    im = Image.new("RGB", (W, H), BG)
    px = im.load()
    for y in range(H):
        c = mix(BG, BG2, y / H * 0.9)
        for x in range(W):
            px[x, y] = c
    return im


class Fr:
    def __init__(self, bg):
        self.im = bg.copy()
        self.d = ImageDraw.Draw(self.im)

    def text(self, x, y, s, sz=32, col=TXT, bold=False, mono=False, a=1.0, under=BG, anchor="la"):
        self.d.text((x, y), s, font=font(sz, bold, mono), fill=mix(under, col, a), anchor=anchor)

    def rrect(self, box, fill, r=18, a=1.0, under=BG, outline=None, ow=2):
        self.d.rounded_rectangle(box, r, fill=mix(under, fill, a), outline=(mix(under, outline, a) if outline else None), width=ow)

    def wtext(self, x, y, s, sz, col, maxw, bold=False, a=1.0, under=BG, lh=1.25, mono=False):
        f = font(sz, bold, mono)
        for i, ln in enumerate(wrap(s, f, maxw)):
            self.d.text((x, y + i * sz * lh), ln, font=f, fill=mix(under, col, a))
        return len(wrap(s, f, maxw)) * sz * lh


# ---------------------------------------------------------------- data
def load_data(path):
    return json.load(open(path))


def pick_examples(D):
    ex = D["examples"]
    def tag(e):
        b, o, g = e["base"]["pred"], e["ours"]["pred"], e["gold_idx"]
        return ("both" if b == g and o == g else "ours" if o == g and b != g else "base" if b == g and o != g else "none")
    by = {}
    for e in ex:
        by.setdefault(tag(e), []).append(e)
    chosen = []
    for k in ("both", "ours", "base"):
        if not by.get(k):
            sys.exit(f"demo data has no '{k}' example; narration for scene 5 assumes one each")
        chosen.append((k, by[k][0]))
    return chosen


def feedback_case(D):
    fb = D["feedback"]
    for t in fb["test"]:
        if t["after_pred"] == t["gold_idx"] and t["before_pred"] != t["gold_idx"]:
            return fb, t
    sys.exit("no feedback test case flips wrong->right; rewrite scene 6 narration before rendering")


# ---------------------------------------------------------------- narration
NARR = [
    "Meet the ten percent challenge. Take a trained model, keep about ten percent of it, and still match its score. No retraining allowed. And it has to keep learning.",
    "The rules: start from one trained model, over half a billion parameters. Keep about ten percent of its size, without training its weights. Match the original's score. Pass an exam it has never seen. And keep learning from a few user annotations.",
    "We start from Qwen two point five, seven billion instruct. In bf sixteen it takes fifteen point two gigabytes. Our compressed artifact takes one point four three gigabytes. That is nine point four percent, with every file counted.",
    "How? First, rotate the weights so outliers spread out. Then round them with a method that knows which weights matter. Then entropy-code the integers. Because the task has only five possible answers, we also trim the vocabulary and the output head. No gradient steps, no retraining.",
    "Here are real queries from the development set, answered by the original model and by the compressed one. Blue is the original, green is ours. Here, both pick the right tool. Here, only the compressed model does. And here the original wins. We show those too.",
    "Now feedback. A user marks five examples of a new tool. A closed-form update turns them into an eight hundred fifty-two byte memory file. The weights and the prompt never change. And on a query it has never seen, the model now picks the new tool.",
    "On the locked exam, two thousand items the model never saw, the original scores seventy-six point seven five percent. The compressed model scores seventy-nine point nine, and after feedback, eighty point two. The strict non-inferiority test passed both times. The receipts are in the repo.",
    "We want to be upfront about two things a reviewer could challenge. The feedback head is fitted in closed form on labelled calibration data. And the task is a constructed five-way multiple choice, so the model keeps a task-specific readout. The maintainer decides whether the proof counts.",
    "Everything is open: the code, the receipts, and the artifact. Find it on GitHub.",
]


def tts(voice, outdir):
    os.makedirs(outdir, exist_ok=True)
    durs, wavs = [], []
    for i, s in enumerate(NARR):
        aiff, raw, padded = (f"{outdir}/s{i}.aiff", f"{outdir}/s{i}.wav", f"{outdir}/p{i}.wav")
        subprocess.run(["say", "-v", voice, "-r", "172", "-o", aiff, "--", s], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", aiff, "-ar", "44100", "-ac", "1", raw], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-af", "adelay=400|400,apad=pad_dur=0.7", padded], check=True)
        with wave.open(padded) as w:
            durs.append(w.getnframes() / w.getframerate())
        wavs.append(padded)
    lst = f"{outdir}/list.txt"
    open(lst, "w").write("".join(f"file '{os.path.abspath(p)}'\n" for p in wavs))
    full = f"{outdir}/narration.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", full], check=True)
    return durs, full


# ---------------------------------------------------------------- scenes
def header(f, t, label):
    f.text(48, 30, "THE 10% CHALLENGE", 20, MUT, bold=True)
    f.text(W - 48, 30, label, 20, MUT, bold=True, anchor="ra")


def s_title(f, t, d, D):
    a = rev(t, 0.2, 0.9)
    r = 150
    cx, cy = 640, 300
    for k, col in enumerate((BLU, GRN)):
        ang = 360 * ease(t / 3.0) * (1 if k == 0 else 0.094 / 1)
    f.d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=mix(BG, CARD2, 1), width=18)
    f.d.arc((cx - r, cy - r, cx + r, cy + r), -90, -90 + 360 * 0.094 * ease(t / 2.2), fill=GRN, width=18)
    f.text(cx, cy - 6, "10%", 96, TXT, bold=True, a=a, anchor="mm")
    f.text(cx, cy + 62, "of the size", 26, MUT, a=a, anchor="mm")
    f.text(640, 520, "A 7B model in 1.4 GB", 54, TXT, bold=True, a=rev(t, 1.0), anchor="ma")
    f.text(640, 590, "Qwen2.5-7B-Instruct  ·  no weight training  ·  still learning", 26, MUT, a=rev(t, 1.6), anchor="ma")


def s_rules(f, t, d, D):
    header(f, t, "THE RULES")
    items = ["One trained model, over 0.5B parameters", "Keep about 10% of its size, no weight training", "Match the base model's score",
             "Pass an exam it has never seen", "Keep learning from a few user annotations"]
    f.text(640, 100, "Five rules", 50, TXT, bold=True, anchor="ma")
    for i, s in enumerate(items):
        a = rev(t, 0.8 + i * 2.2, 0.6)
        y = 190 + i * 88
        x = 140 + (1 - a) * 40
        f.rrect((x, y, x + 1000, y + 70), CARD, 16, a=a)
        f.d.ellipse((x + 18, y + 12, x + 64, y + 58), fill=mix(BG, BLU, a))
        f.text(x + 41, y + 35, str(i + 1), 28, (255, 255, 255), bold=True, a=a, anchor="mm", under=BLU)
        f.text(x + 90, y + 35, s, 32, TXT, a=a, under=CARD, anchor="lm")


def s_size(f, t, d, D):
    header(f, t, "THE SIZE")
    m = D["meta"]
    base_gb, ours_gb = m["base_bytes"] / 1e9, m["artifact_bytes"] / 1e9
    f.text(120, 150, "Qwen2.5-7B-Instruct, bf16", 28, MUT, a=rev(t, 0.3))
    bw = 1040
    p = rev(t, 1.0, 1.4)
    f.rrect((120, 200, 120 + bw, 280), CARD, 14)
    f.rrect((120, 200, 120 + max(8, bw * p), 280), BLU, 14)
    f.text(120 + bw / 2, 240, f"{base_gb:.2f} GB", 36, (255, 255, 255), bold=True, a=p, under=BLU, anchor="mm")
    p2 = rev(t, 5.0, 1.6)
    f.text(120, 330, "Our artifact (every file + memory counted)", 28, MUT, a=rev(t, 4.6))
    f.rrect((120, 380, 120 + bw, 460), CARD, 14)
    f.rrect((120, 380, 120 + max(8, bw * (ours_gb / base_gb) * p2), 460), GRN, 14)
    f.text(120 + bw * (ours_gb / base_gb) + 24, 420, f"{ours_gb:.2f} GB", 36, GRN, bold=True, a=p2, anchor="lm")
    f.text(640, 560, f"{100*ours_gb/base_gb:.3f}%", 110, GRN, bold=True, a=rev(t, 8.0, 1.0), anchor="ma")
    f.text(640, 680, "of the base model's size", 28, MUT, a=rev(t, 8.6), anchor="ma")


def s_how(f, t, d, D):
    header(f, t, "HOW IT WORKS")
    f.text(640, 90, "Post-training compression. No gradient steps.", 40, TXT, bold=True, anchor="ma")
    steps = [("Rotate", "Hadamard rotation\nspreads outliers", BLU), ("Round", "Hessian-aware GPTQ\n+ dead-zone quantizer", BLU),
             ("Entropy-code", "rANS coding of\nthe integer indices", BLU), ("Trim", "24k-token vocab,\n5-row output head", ORG), ("Pack", "9 files, 1.43 GB\nsha256 manifest", GRN)]
    for i, (h, sub, col) in enumerate(steps):
        a = rev(t, 1.0 + i * 3.0, 0.7)
        x = 50 + i * 244
        f.rrect((x, 230, x + 214, 470), CARD, 20, a=a, outline=col)
        f.text(x + 107, 275, h, 26, col, bold=True, a=a, under=CARD, anchor="mm")
        for j, ln in enumerate(sub.split("\n")):
            f.text(x + 107, 340 + j * 34, ln, 18, TXT, a=a, under=CARD, anchor="mm")
        if i < 4:
            f.text(x + 227, 350, ">", 40, MUT, a=a, anchor="mm")
    f.text(640, 560, "No retraining of the weights. The answer is one of five letters,", 26, MUT, a=rev(t, 14), anchor="ma")
    f.text(640, 598, "so most of the vocabulary and the output head can go.", 26, MUT, a=rev(t, 14.3), anchor="ma")


def options_block(f, e, x, y, w, a, show_probs, under=CARD):
    letters = "ABCDE"
    for i, name in enumerate(e["options"]):
        yy = y + i * 62
        gold = i == e["gold_idx"]
        f.rrect((x, yy, x + w, yy + 52), CARD2 if not gold else mix(CARD2, GRN, 0.16), 12, a=a, under=under, outline=(GRN if gold else None))
        f.text(x + 16, yy + 26, f"{letters[i]}  {name}", 22, TXT, a=a, under=CARD2, anchor="lm", mono=True)
        if show_probs:
            for k, (key, col) in enumerate((("base", BLU), ("ours", GRN))):
                pr = e[key]["probs"][i]
                bw = 260 * pr * show_probs
                by = yy + 8 + k * 20
                f.rrect((x + w - 280, by, x + w - 20, by + 14), BG, 7, a=a, under=CARD2)
                if bw > 2:
                    f.rrect((x + w - 280, by, x + w - 280 + bw, by + 14), col, 7, a=a, under=CARD2)
            for key, col, dx in (("base", BLU, 0), ("ours", GRN, 1)):
                if e[key]["pred"] == i:
                    f.text(x + w - 296 - dx * 26, yy + 26, "●", 22, col, a=a, under=CARD2, anchor="mm")


def s_demo(f, t, d, D):
    header(f, t, "REAL OUTPUTS · DEV SET")
    chosen = pick_examples(D)
    seg = d / 3
    idx = min(2, int(t / seg))
    lt = t - idx * seg
    kind, e = chosen[idx]
    a = rev(lt, 0.1, 0.5)
    f.rrect((60, 80, W - 60, 640), CARD, 22)
    f.text(90, 105, "QUERY", 18, MUT, bold=True, under=CARD, a=a)
    h = f.wtext(90, 135, e["query"], 30, TXT, W - 180, a=a, under=CARD, bold=True)
    options_block(f, e, 90, 135 + h + 30, W - 180, a, rev(lt, 0.8, 1.2) )
    f.text(90, 600, "●", 20, BLU, under=CARD, anchor="lm")
    f.text(112, 600, "original (bf16)", 20, MUT, under=CARD, anchor="lm")
    f.text(310, 600, "●", 20, GRN, under=CARD, anchor="lm")
    f.text(332, 600, "compressed (9.4%)", 20, MUT, under=CARD, anchor="lm")
    label = {"both": "Both correct", "ours": "Only the compressed model is right", "base": "Only the original is right"}[kind]
    f.text(W - 90, 600, label, 24, {"both": GRN, "ours": GRN, "base": ORG}[kind], bold=True, under=CARD, anchor="rm", a=rev(lt, 1.8, 0.5))


def s_feedback(f, t, d, D):
    header(f, t, "FEEDBACK · NO RETRAINING")
    fb, case = feedback_case(D)
    f.text(90, 85, f"A user annotates 5 examples of a new tool: {fb['tool']}", 32, TXT, bold=True, a=rev(t, 0.2))
    for i, an in enumerate(fb["annotations"][:5]):
        a = rev(t, 1.0 + i * 0.9, 0.5)
        y = 150 + i * 56
        f.rrect((90, y, 760, y + 46), CARD, 10, a=a)
        s = an["query"] if len(an["query"]) < 62 else an["query"][:59] + "..."
        f.text(106, y + 23, s, 20, TXT, a=a, under=CARD, anchor="lm")
    ap = rev(t, 6.0, 0.8)
    f.text(800, 175, "closed-form update", 22, MUT, a=ap)
    f.text(800, 205, "→", 60, ORG, a=ap)
    f.rrect((880, 160, 1210, 300), CARD2, 18, a=ap, outline=ORG)
    f.text(1045, 205, f"{fb['memory_bytes']} bytes", 36, ORG, bold=True, a=ap, under=CARD2, anchor="mm")
    f.text(1045, 250, "memory.json", 22, MUT, a=ap, under=CARD2, anchor="mm")
    f.text(1045, 330, "weights & prompt: unchanged", 20, GRN, a=ap, anchor="mm")
    bp = rev(t, 9.0, 0.7)
    f.rrect((90, 440, W - 90, 690), CARD, 20, a=bp)
    f.text(116, 458, "A NEW QUERY IT HAS NEVER SEEN", 18, MUT, bold=True, a=bp, under=CARD)
    f.wtext(116, 488, case["query"], 24, TXT, W - 232, a=bp, under=CARD, bold=True)
    opts = case["options"]
    y0 = 590
    for k, (lab, pred, col, t0) in enumerate((("before feedback", case["before_pred"], RED, 10.5), ("after feedback", case["after_pred"], GRN, 13.0))):
        x = 116 + k * 560
        a = rev(t, t0, 0.6)
        f.text(x, y0, lab, 20, MUT, a=a, under=CARD)
        f.text(x, y0 + 30, "→ " + opts[pred], 28, col, bold=True, a=a, under=CARD, mono=True)


def s_results(f, t, d, D):
    header(f, t, "THE LOCKED EXAM · n = 2000")
    bars = [("original", 76.75, BLU, 1.0), ("compressed, stage 1", 79.90, GRN, 5.5), ("after feedback, stage 2", 80.20, GRN, 8.5)]
    ymax = 82
    ybase = 560
    for i, (lab, v, col, t0) in enumerate(bars):
        x = 150 + i * 340
        p = rev(t, t0, 1.1)
        hgt = (v - 70) / (ymax - 70) * 330 * p
        f.rrect((x, ybase - hgt, x + 220, ybase), col, 14)
        f.text(x + 110, ybase - hgt - 18, f"{v:.2f}%" if p > .98 else "", 34, TXT, bold=True, anchor="mm")
        f.text(x + 110, ybase + 30, lab, 22, MUT, anchor="mm", a=rev(t, t0, 0.4))
    f.text(150, 120, "accuracy (axis starts at 70%)", 20, MUT)
    f.text(1180, 150, "+3.15 pts", 40, GRN, bold=True, a=rev(t, 7.0), anchor="ra")
    f.text(1180, 200, "paired 95% CI [+1.60, +4.70]", 22, MUT, a=rev(t, 7.2), anchor="ra")
    f.text(1180, 240, "non-inferiority p = 0.001", 22, MUT, a=rev(t, 7.4), anchor="ra")
    f.text(1180, 640, "same artifact hash at both stages  ·  network blocked", 20, MUT, a=rev(t, 12), anchor="ra")


def s_honest(f, t, d, D):
    header(f, t, "WHAT A REVIEWER COULD CHALLENGE")
    items = [("1", "The feedback head is fitted in closed form", "on labelled calibration items (no gradients, base weights untouched). Fallback: remove it."),
             ("2", "The task is a constructed 5-way multiple choice", "the model keeps only a task-specific readout: 5 letters, a trimmed vocabulary.")]
    for i, (n, h, s) in enumerate(items):
        a = rev(t, 1.0 + i * 5.5, 0.7)
        y = 170 + i * 200
        f.rrect((90, y, W - 90, y + 170), CARD, 20, a=a, outline=ORG)
        f.d.ellipse((120, y + 28, 172, y + 80), fill=mix(CARD, ORG, a))
        f.text(146, y + 54, n, 30, (30, 20, 0), bold=True, a=a, anchor="mm", under=ORG)
        f.text(200, y + 36, h, 32, TXT, bold=True, a=a, under=CARD)
        f.wtext(200, y + 86, s, 26, MUT, W - 330, a=a, under=CARD)
    f.text(640, 610, "The maintainer decides whether the proof counts.", 28, TXT, a=rev(t, 12), anchor="ma")


def s_outro(f, t, d, D):
    a = rev(t, 0.3, 0.8)
    f.text(640, 200, "Everything is open", 64, TXT, bold=True, a=a, anchor="ma")
    f.text(640, 300, "code · receipts · artifact", 34, MUT, a=a, anchor="ma")
    f.rrect((190, 400, 1090, 480), CARD, 18, a=rev(t, 1.0))
    f.text(640, 440, "github.com/Bobliuuu/ten-percent-challenge", 34, GRN, bold=True, a=rev(t, 1.0), under=CARD, anchor="mm", mono=False)
    f.text(640, 540, "proof: issue #2 on DeusExperiments/ten-percent-challenge", 26, MUT, a=rev(t, 1.6), anchor="ma")
    f.text(640, 640, "built with a swarm of Claude Code agents", 22, MUT, a=rev(t, 2.2), anchor="ma")


SCENES = [s_title, s_rules, s_size, s_how, s_demo, s_feedback, s_results, s_honest, s_outro]


def captions(f, t, narr, dur):
    sents = re.split(r"(?<=[.?!])\s+", narr)
    tot = sum(len(s) for s in sents)
    t0, span, acc = 0.4, dur - 1.1, 0
    for s in sents:
        a0 = t0 + span * acc / tot
        acc += len(s)
        a1 = t0 + span * acc / tot
        if a0 <= t < a1 + 0.05:
            lines = wrap(s, font(28), 1100)
            bh = len(lines) * 36 + 20
            f.rrect((90, H - 24 - bh, W - 90, H - 24), (6, 9, 20), 14)
            for i, ln in enumerate(lines):
                f.d.text((W // 2, H - 24 - bh + 12 + i * 36), ln, font=font(28), fill=TXT, anchor="ma")
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="out/explainer.mp4")
    ap.add_argument("--voice", default="Samantha")
    ap.add_argument("--preview", type=int, default=0, help="write one PNG per scene at this second offset and exit")
    ap.add_argument("--no-tts", action="store_true")
    a = ap.parse_args()
    D = load_data(a.data)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.no_tts:
        durs, wavf = [8.0] * len(NARR), None
    else:
        durs, wavf = tts(a.voice, os.path.dirname(a.out) + "/audio")
    bg = make_bg()
    if a.preview:
        for i, sc in enumerate(SCENES):
            f = Fr(bg); sc(f, min(durs[i] - 0.5, a.preview), durs[i], D); captions(f, min(durs[i] - 0.5, a.preview), NARR[i], durs[i])
            f.im.save(f"{os.path.dirname(a.out)}/preview_{i}.png")
        print("previews written")
        return
    tmp = a.out + ".silent.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    for i, sc in enumerate(SCENES):
        n = int(round(durs[i] * FPS))
        for k in range(n):
            t = k / FPS
            f = Fr(bg)
            sc(f, t, durs[i], D)
            captions(f, t, NARR[i], durs[i])
            ff.stdin.write(f.im.tobytes())
        print(f"scene {i} done ({durs[i]:.1f}s)", flush=True)
    ff.stdin.close(); ff.wait()
    if wavf:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-i", wavf, "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", a.out], check=True)
        os.remove(tmp)
    else:
        os.replace(tmp, a.out)
    print("wrote", a.out, "total", sum(durs))


if __name__ == "__main__":
    main()
