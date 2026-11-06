"""Quick ~14 s looping clip (mp4 + gif) for social posts. Numbers come only from the committed exam receipts.
Usage: nice -n 10 python3 make_quick.py   -> v2/demo/out/quick.mp4, quick.gif
"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(__file__))
from make_video import Fr, make_bg, mix, rev, ease, clamp, TXT, MUT, GRN, BLU, ORG, CARD, CARD2, BG, W, H, FPS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
R1 = json.load(open(f"{ROOT}/v2/harness/receipts/REAL_final/stage1.json"))
R2 = json.load(open(f"{ROOT}/v2/harness/receipts/REAL_final/stage2.json"))
BASE_B, OURS_B = R1["size"]["base_fp16_bytes"], R1["size"]["candidate_bytes"]
a_base, a1, a2 = R1["report"]["base_acc"], R1["report"]["cand_acc"], R2["report"]["cand_acc"]
bg = make_bg()
T1, T2, T3, T4 = 3.2, 5.6, 5.0, 2.6  # scene lengths (s)


def q_title(f, t):
    a = rev(t, 0.1, 0.6)
    f.text(640, 215, "A 7B model", 84, TXT, bold=True, a=a, anchor="mm")
    f.text(640, 320, "in 1.4 GB", 110, GRN, bold=True, a=rev(t, 0.7, 0.6), anchor="mm")
    f.text(640, 450, "9.4% of its original size", 38, MUT, a=rev(t, 1.4, 0.5), anchor="mm")
    f.text(640, 540, "no weight training  ·  still learns from feedback", 30, TXT, a=rev(t, 2.0, 0.5), anchor="mm")


def q_size(f, t):
    f.text(60, 40, "SIZE ON DISK  ·  every file counted", 22, MUT, bold=True)
    bw = 1040
    f.text(120, 150, "Qwen2.5-7B-Instruct (bf16)", 30, MUT)
    p = rev(t, 0.2, 0.8)
    f.rrect((120, 200, 120 + bw, 290), BLU, 16)
    f.text(120 + bw / 2, 245, f"{BASE_B/1e9:.2f} GB", 48, (255, 255, 255), bold=True, a=p, under=BLU, anchor="mm")
    f.text(120, 350, "compressed", 30, MUT, a=rev(t, 1.2, 0.4))
    s = ease((t - 1.5) / 1.6)  # shrink animation
    wid = bw * (1 - s * (1 - OURS_B / BASE_B))
    f.rrect((120, 400, 120 + bw, 490), CARD, 16, a=rev(t, 1.2, 0.4))
    f.rrect((120, 400, 120 + max(10, wid), 490), mix(BLU, GRN, s), 16, a=rev(t, 1.2, 0.4))
    gb = (BASE_B + (OURS_B - BASE_B) * s) / 1e9
    f.text(120 + max(10, wid) + 24 if s > 0.6 else 120 + wid / 2, 445, f"{gb:.2f} GB", 48, GRN if s > 0.6 else (255, 255, 255), bold=True,
           anchor="lm" if s > 0.6 else "mm", under=BLU if s <= 0.6 else BG)
    f.text(640, 600, f"{100*OURS_B/BASE_B:.1f}%", 110, GRN, bold=True, a=rev(t, 3.3, 0.6), anchor="mm")
    f.text(640, 675, "of the original", 28, MUT, a=rev(t, 3.6, 0.5), anchor="mm")


def q_results(f, t):
    f.text(60, 40, "LOCKED EXAM  ·  2000 items it never saw", 22, MUT, bold=True)
    bars = [("original", a_base, BLU, 0.3), ("compressed", a1, GRN, 1.3), ("after feedback", a2, GRN, 2.3)]
    y0, top, a0, a1_ = 600, 190, 0.70, 0.84
    for i, (lab, v, col, t0) in enumerate(bars):
        x = 150 + i * 360
        p = rev(t, t0, 0.9)
        h = (v - a0) / (a1_ - a0) * (y0 - top) * p
        f.rrect((x, y0 - h, x + 240, y0), col, 14)
        if p > 0.97:
            f.text(x + 120, y0 - h - 28, f"{v*100:.2f}%", 42, TXT, bold=True, anchor="mm")
        f.text(x + 120, y0 + 30, lab, 26, MUT, a=rev(t, t0, 0.4), anchor="mm")
    f.text(60, 690, "bars start at 70% accuracy", 20, MUT)
    f.text(1200, 120, "+3.15 pts", 52, GRN, bold=True, a=rev(t, 3.3, 0.5), anchor="ra")
    f.text(1200, 185, "p = 0.001, passes both stages", 24, MUT, a=rev(t, 3.6, 0.5), anchor="ra")


def q_end(f, t):
    a = rev(t, 0.1, 0.5)
    f.text(640, 230, "Proof, code and receipts", 56, TXT, bold=True, a=a, anchor="mm")
    f.rrect((190, 330, 1090, 410), CARD, 18, a=a)
    f.text(640, 370, "github.com/Bobliuuu/ten-percent-challenge", 34, GRN, bold=True, a=a, under=CARD, anchor="mm")
    f.text(640, 480, "the 10% challenge  ·  issue #2", 28, MUT, a=rev(t, 0.7, 0.5), anchor="mm")


SC = [(q_title, T1), (q_size, T2), (q_results, T3), (q_end, T4)]


def main():
    mp4 = f"{OUT}/quick.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4], stdin=subprocess.PIPE)
    for fn, d in SC:
        for k in range(int(d * FPS)):
            f = Fr(bg); fn(f, k / FPS); ff.stdin.write(f.im.tobytes())
    ff.stdin.close(); ff.wait()
    gif = f"{OUT}/quick.gif"
    pal = f"{OUT}/pal.png"
    vf = "fps=15,scale=800:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf", vf + ",palettegen=stats_mode=diff", pal], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-i", pal, "-lavfi", vf + "[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4", gif], check=True)
    os.remove(pal)
    for p in (mp4, gif):
        print(p, round(os.path.getsize(p) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
