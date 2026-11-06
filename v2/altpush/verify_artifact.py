"""Verify artifact reproduces a push.py run: python verify_artifact.py ART PACKDIR/misc.pt dev.jsonl  (dev only)"""
import sys, json, time, torch, numpy as np, importlib.util
art, miscpt, devf = sys.argv[1:4]
spec = importlib.util.spec_from_file_location("predict", f"{art}/predict.py"); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
t = time.time(); f = P.load(art); print("load secs", time.time() - t, flush=True)
dev = [json.loads(x) for x in open(devf)]
lg = f.logits(dev); ref = torch.load(miscpt)["ev_logits"]
y = np.array([r["answer"] for r in dev]); pred = f(dev)
print("RESULT_JSON", json.dumps(dict(acc_art_bias=float((np.array(pred) == y).mean()), acc_art_raw=float((lg.argmax(1) == y).mean()), acc_run_raw=float((ref.argmax(1) == y).mean()),
    agree_raw_vs_run=float((lg.argmax(1) == ref.argmax(1)).mean()), max_abs_logit_diff=float(np.abs(np.where(ref > -1e8, lg - ref, 0)).max()))))
