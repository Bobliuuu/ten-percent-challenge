"""End-to-end CLI test on Modal: answer.py before -> feedback.py (Music->Jukebox, 12 annotations) -> answer.py -> feedback.py AGAIN (News->Briefing, 12) -> answer.py. Edits the safetensors shards in place.
modal deploy modal_cli.py ; python modal_cli.py launch NAME"""
import modal, os, sys, json
app = modal.App("tenpct-fbcli"); HERE = os.path.dirname(os.path.abspath(__file__))
img = (modal.Image.debian_slim().pip_install("torch", "transformers==4.51.3", "numpy", "accelerate", "sentencepiece", "huggingface_hub", "safetensors")
       .add_local_file(HERE + "/fblib.py", "/root/fblib.py").add_local_dir(HERE + "/cli", "/root/cli").add_local_dir(HERE + "/data", "/root/data"))
vol = modal.Volume.from_name("tenpct-fbabl", create_if_missing=True); hf = modal.Volume.from_name("tenpct-v2-hf")
@app.function(image=img, gpu="A100-80GB", cpu=8, memory=98304, timeout=7200, volumes={"/r": vol, "/hf": hf}, env={"HF_HOME": "/hf"})
def run(name: str, method: str = ""):
    import subprocess, time, random
    from huggingface_hub import snapshot_download
    md = "/tmp/model"; snapshot_download("Qwen/Qwen2.5-7B-Instruct", local_dir=md, allow_patterns=["*.json", "*.safetensors", "*.txt"]); L = []
    rd = lambda n: [json.loads(l) for l in open(f"/root/data/{n}.jsonl")]
    def sh(cmd):
        t = time.time(); p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="/root"); L.append(dict(cmd=cmd[:120], secs=round(time.time() - t, 1), stdout=p.stdout[-6000:], stderr_tail=p.stderr[-300:], rc=p.returncode)); print(L[-1], flush=True)
    M = rd("ann_Music"); N = rd("ann_News"); tM = rd("teach_Music"); tN = rd("teach_News"); ne = rd("neigh"); random.Random(0).shuffle(ne)
    def wr(p, rows, lab): open(p, "w").write("".join(json.dumps(dict(text=r["text"], label=lab)) + "\n" for r in rows))
    wr("/tmp/ann1.jsonl", random.Random(1).sample(M, 12), "Jukebox"); wr("/tmp/ann2.jsonl", random.Random(2).sample(N, 12), "Briefing"); wr("/tmp/ann_shuf.jsonl", [r for r in ne if r["gold"] in ("Alarm", "Lists")][:12], "Jukebox")
    q = lambda rows: " ".join(f"--text {json.dumps(r['text'])}" for r in rows)
    probe = q(tM[:6]) + " " + q(tN[:6]) + " " + q([r for r in ne][:6])
    mt = method or '{"tag":"cli","carrier":"down","solver":"alpha","kp":256,"layers":[20,21,22,23,24,25,26,27],"m":16}'
    sh(f"python cli/answer.py --model {md} {probe}")
    sh(f"python cli/feedback.py --model {md} --annotations /tmp/ann1.jsonl --calib /root/data/calib_old.jsonl --method '{mt}'")
    sh(f"python cli/answer.py --model {md} {probe}")
    sh(f"python cli/feedback.py --model {md} --annotations /tmp/ann2.jsonl --calib /root/data/calib_old.jsonl --method '{mt}'")
    sh(f"python cli/answer.py --model {md} {probe}")
    os.makedirs("/r/" + name, exist_ok=True); json.dump(L, open(f"/r/{name}/cli.json", "w"), indent=1); vol.commit(); return len(L)
if __name__ == "__main__":
    f = modal.Function.from_name("tenpct-fbcli", "run"); c = f.spawn(sys.argv[2]); print("spawned", c.object_id)
