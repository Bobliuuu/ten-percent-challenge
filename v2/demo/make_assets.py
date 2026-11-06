"""Static README graphics from real receipts and measured results. Run: python3 make_assets.py (light, Pillow only)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from make_video import *  # noqa: F401,F403
from make_video import Fr, make_bg, font, mix, TXT, MUT, GRN, BLU, ORG, RED, CARD, CARD2, BG, W, H, s_how, wrap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "assets")
os.makedirs(OUT, exist_ok=True)
bg = make_bg()
R1 = json.load(open(f"{ROOT}/v2/harness/receipts/REAL_final/stage1.json"))
R2 = json.load(open(f"{ROOT}/v2/harness/receipts/REAL_final/stage2.json"))


def title(f, t1, t2=None):
    f.text(60, 36, t1, 38, TXT, bold=True)
    if t2:
        f.text(60, 86, t2, 22, MUT)


def size_png():
    f = Fr(bg)
    title(f, "Size on disk", "every file counted: weights, scales, indices, embeddings, tokenizer, code, feedback head")
    base, ours = R1["size"]["base_fp16_bytes"], R1["size"]["candidate_bytes"]
    bw = 1040
    f.text(120, 170, "Qwen2.5-7B-Instruct, bf16", 28, MUT)
    f.rrect((120, 215, 120 + bw, 305), BLU, 16)
    f.text(120 + bw / 2, 260, f"{base/1e9:.2f} GB", 44, (255, 255, 255), bold=True, under=BLU, anchor="mm")
    f.text(120, 360, "Compressed artifact (9 files)", 28, MUT)
    f.rrect((120, 405, 120 + bw, 495), CARD, 16)
    f.rrect((120, 405, 120 + bw * ours / base, 495), GRN, 16)
    f.text(120 + bw * ours / base + 22, 450, f"{ours/1e9:.2f} GB", 44, GRN, bold=True, anchor="lm")
    f.text(640, 560, f"{100*ours/base:.3f}%", 96, GRN, bold=True, anchor="ma")
    f.text(640, 668, "of the base model's size  (budget: 10%)", 24, MUT, anchor="ma")
    f.im.save(f"{OUT}/size.png")


def results_png():
    f = Fr(bg)
    title(f, "Locked exam, n = 2000", "items the model never saw; same items for base and candidate; accuracy axis starts at 70%")
    rep1, rep2 = R1["report"], R2["report"]
    bars = [("original (bf16)", rep1["base_acc"], rep1["base_ci95"], BLU),
            ("compressed\nstage 1", rep1["cand_acc"], rep1["cand_ci95"], GRN),
            ("after feedback\nstage 2", rep2["cand_acc"], rep2["cand_ci95"], GRN)]
    y0, y1, a0, a1 = 600, 160, 0.70, 0.84
    sy = lambda v: y0 - (v - a0) / (a1 - a0) * (y0 - y1)
    for v in (0.70, 0.74, 0.78, 0.82):
        f.d.line((120, sy(v), 820, sy(v)), fill=CARD2, width=1)
        f.text(108, sy(v), f"{v*100:.0f}%", 18, MUT, anchor="rm")
    for i, (lab, v, ci, col) in enumerate(bars):
        x = 160 + i * 220
        f.rrect((x, sy(v), x + 150, y0), col, 12)
        f.d.line((x + 75, sy(ci[0]), x + 75, sy(ci[1])), fill=TXT, width=4)
        for c in ci:
            f.d.line((x + 55, sy(c), x + 95, sy(c)), fill=TXT, width=4)
        f.text(x + 75, sy(ci[1]) - 28, f"{v*100:.2f}%", 28, TXT, bold=True, anchor="mm")
        for j, ln in enumerate(lab.split("\n")):
            f.text(x + 75, y0 + 26 + j * 24, ln, 20, MUT, anchor="mm")
    f.text(60, 140, "whiskers: 95% Clopper-Pearson", 16, MUT)
    xs = 880
    for j, (a, b, c) in enumerate([("stage 1", rep1, GRN), ("stage 2", rep2, GRN)]):
        y = 190 + j * 190
        f.rrect((xs, y, xs + 340, y + 165), CARD, 16)
        f.text(xs + 20, y + 14, a, 22, MUT, under=CARD)
        f.text(xs + 20, y + 44, f"{b['diff_cand_minus_base']*100:+.2f} pts", 44, GRN, bold=True, under=CARD)
        lo, hi = b["diff_ci95_paired_wald"]
        f.text(xs + 20, y + 100, f"paired 95% CI [{lo*100:+.2f}, {hi*100:+.2f}]", 18, TXT, under=CARD)
        f.text(xs + 20, y + 126, f"non-inferior (1 pt): yes, p = {b['ni_exact_unconditional_p']:.3f}", 18, TXT, under=CARD)
    f.text(640, 690, "same artifact hash at both stages  ·  network blocked  ·  one run per stage", 20, MUT, anchor="ma")
    f.im.save(f"{OUT}/results.png")


def frontier_png():
    f = Fr(bg)
    title(f, "Why this task, not MMLU?", "accuracy vs the model's own base, at different sizes (bf16 fraction). Training-free compression only.")
    x0, x1, y0, y1 = 130, 1180, 620, 150
    xmin, xmax, ymin, ymax = 0.085, 0.195, -30, 6
    sx = lambda v: x0 + (v - xmin) / (xmax - xmin) * (x1 - x0)
    sy = lambda v: y0 - (v - ymin) / (ymax - ymin) * (y0 - y1)
    for v in (-30, -20, -10, 0):
        f.d.line((x0, sy(v), x1, sy(v)), fill=(CARD2 if v else MUT), width=(1 if v else 2))
        f.text(x0 - 12, sy(v), f"{v:+d}" if v else "0", 18, MUT, anchor="rm")
    for v in (0.09, 0.10, 0.12, 0.14, 0.16, 0.18):
        f.d.line((sx(v), y0, sx(v), y1), fill=(ORG if v == 0.10 else CARD2), width=(3 if v == 0.10 else 1))
        f.text(sx(v), y0 + 22, f"{v*100:.0f}%", 18, MUT, anchor="mm")
    f.text(sx(0.10) + 8, y1 - 4, "10% target", 18, ORG, bold=True)
    f.text(60, 400, "accuracy gap\nvs base (pts)", 18, MUT)
    f.text(x1, y0 + 44, "size as a fraction of bf16", 18, MUT, anchor="ra")
    pts = [
        ("MMLU 14B, GPTQ", RED, [(0.166, -3.7), (0.146, -4.7), (0.118, -13.8), (0.097, -28.0)]),
        ("MMLU 14B, vector quant.", ORG, [(0.1015, -14.3), (0.0961, -18.1)]),
        ("MMLU 30B MoE, prune + 2-bit", (200, 120, 255), [(0.185, -2.6), (0.141, -3.4), (0.0975, -7.55)]),
        ("tool selection 7B (ours, locked exam)", GRN, [(0.094, 3.15)]),
        ("classification 7B (MDC-6, dev)", BLU, [(0.0974, 0.1)]),
    ]
    for k, (lab, col, ps) in enumerate(pts):
        for (x, y) in ps:
            r = 11 if col == GRN else 8
            f.d.ellipse((sx(x) - r, sy(y) - r, sx(x) + r, sy(y) + r), fill=col, outline=BG, width=2)
        yl = 440 + k * 28
        f.d.ellipse((720, yl - 7, 734, yl + 7), fill=col)
        f.text(744, yl, lab, 18, TXT, anchor="lm")
    f.text(sx(0.094) + 20, sy(3.15) - 12, "+3.15 pts", 22, GRN, bold=True)
    f.text(640, 694, "broad knowledge needs far more than ~1.6 bits/weight; a 5-way answer space does not", 20, MUT, anchor="ma")
    f.im.save(f"{OUT}/frontier.png")


def task_card_png():
    D = json.load(open(os.path.join(os.path.dirname(__file__), "demo_outputs.json")))
    e = next(x for x in D["examples"] if x["base"]["pred"] == x["gold_idx"] == x["ours"]["pred"] and "subarray" in x["query"])
    f = Fr(bg)
    title(f, "The task: pick the right tool", "a real query from the dev set; 4 confusable tools, or 'none fits'")
    f.rrect((80, 150, W - 80, 640), CARD, 22)
    f.text(116, 176, "QUERY", 18, MUT, bold=True, under=CARD)
    h = f.wtext(116, 206, e["query"], 30, TXT, W - 232, bold=True, under=CARD)
    for i, name in enumerate(e["options"]):
        y = 206 + h + 26 + i * 62
        gold = i == e["gold_idx"]
        f.rrect((116, y, W - 116, y + 52), mix(CARD2, GRN, 0.18) if gold else CARD2, 12, under=CARD, outline=(GRN if gold else None))
        f.text(136, y + 26, f"{'ABCDE'[i]}   {name}", 24, TXT, under=CARD2, anchor="lm", mono=True)
        if gold:
            f.text(W - 140, y + 26, "correct", 22, GRN, bold=True, under=CARD2, anchor="rm")
    f.text(640, 680, "the answer is one of 5 letters: that is why the output head can shrink to 5 rows", 20, MUT, anchor="ma")
    f.im.save(f"{OUT}/task_card.png")


def pipeline_png():
    f = Fr(bg)
    D = {"meta": {}}
    s_how(f, 40.0, 60.0, D)
    f.im.save(f"{OUT}/pipeline.png")


if __name__ == "__main__":
    size_png(); results_png(); frontier_png(); pipeline_png(); task_card_png()
    print("assets written:", sorted(os.listdir(OUT)))
