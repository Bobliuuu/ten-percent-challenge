"""L10: recompute size + sha256 of the packed body on the private volume (CPU, detached). python3 shacheck.py"""
import modal, hashlib, json
app = modal.App("tenpct-l10sha"); priv = modal.Volume.from_name("tenpct-l10-priv"); vol = modal.Volume.from_name("tenpct-tasksearch")
@app.function(image=modal.Image.debian_slim(), volumes={"/priv": priv, "/vol": vol}, timeout=3600)
def go():
    import os
    h = hashlib.sha256(); p = "/priv/omni7_body.pack"
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b: break
            h.update(b)
    r = dict(bytes=os.path.getsize(p), sha256=h.hexdigest()); print("SHA_RESULT", json.dumps(r), flush=True)
    os.makedirs("/vol/l10v2/sha", exist_ok=True); json.dump(r, open("/vol/l10v2/sha/res.json", "w")); vol.commit()
if __name__ == "__main__":
    with modal.enable_output(), app.run(detach=True): print("CALL", go.spawn().object_id, flush=True)
