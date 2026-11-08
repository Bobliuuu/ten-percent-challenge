"""L9 answer command on the PACKED artifact. usage: python answer.py ART "request text" [CARRIER.safetensors]   (reads the artifact files only; needs ~330 GB of GPU memory, e.g. 4x80GB is NOT enough
for the bf16 dequantised in-memory form: use 4x B300/H200-class or 8x80GB; a fused low-bit kernel path is not built)."""
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pack235 as P
from safetensors.torch import load_file
if __name__ == "__main__":
    ART, text = sys.argv[1], sys.argv[2]; m, tok = P.build_model(ART, os.environ.get("L9_GPUMEM", "120GiB"))
    if len(sys.argv) > 3:
        for k, v in load_file(sys.argv[3]).items(): w = m.model.layers[int(k.split(".")[1])].self_attn.o_proj.weight; w.data.add_(v.to(w.device, w.dtype))
    print(P.answer_text(m, tok, [text])[0])
