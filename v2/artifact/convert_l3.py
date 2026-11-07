"""Converter: research/L3_405B `q/L{i}.pt` (4-bit packed codes, fp16 scales, int8 sign vector, g=128) -> TPQ1 container, streaming layer by layer.
  python convert_l3.py --q DIR_WITH_L{i}.pt --out ART [--layers 126] [--emb-codes emb_codes.pt --head-codes head_codes.pt] [--raw-dir RAW.safetensors...] 
Codes are unpacked (low nibble = even column, high nibble = odd column), scales are written losslessly (mbits=10 unless the scales were rounded: --mbits),
identical sign vectors (q/k/v/o share one in every layer) are stored once. emb/head: the L3 run saved only DEQUANTISED bf16 tensors (bf16 rounding makes the
integer codes/scales unrecoverable bit-exactly), so they must be dumped by the quantiser as {codes uint8 [R,C], scales fp16 [R,C/128], L, sg}; until then they can only be
added as raw bf16 (far too big: 2 x 4.2 B params) -> the run is NOT complete without them."""
import argparse, os, sys, json, hashlib
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loader as LD

GROUP = {"self_attn.q_proj": "attn_in", "self_attn.k_proj": "attn_in", "self_attn.v_proj": "attn_in", "self_attn.o_proj": "o", "mlp.gate_proj": "mlp_in", "mlp.up_proj": "mlp_in", "mlp.down_proj": "down"}


def unpack4(packed):
    p = packed.to(torch.uint8); out = torch.empty(p.shape[0], p.shape[1] * 2, dtype=torch.uint8); out[:, 0::2] = p & 15; out[:, 1::2] = p >> 4; return out


def convert_layer(w, layer, d, g=128, mbits=10, sign_cache=None):
    """d = dict module -> {L, codes, scales, sg}; returns per-module byte stats"""
    sign_cache = sign_cache if sign_cache is not None else {}; stats = {}
    for mod, e in d.items():
        sg = e["sg"].float(); h = hashlib.sha1(sg.numpy().astype(np.int8).tobytes()).hexdigest()
        if h not in sign_cache: sign_cache[h] = f"s{len(sign_cache)}"; w.add_signs(sign_cache[h], sg)
        name = f"model.layers.{layer}.{mod}.weight"; codes = unpack4(e["codes"]).numpy(); st = w.add_q(name, codes, e["scales"].half().numpy(), int(e["L"]), g, mbits, sign_cache[h]); stats[name] = st
    return stats


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--q", required=True); ap.add_argument("--out", required=True); ap.add_argument("--layers", type=int, default=126); ap.add_argument("--mbits", type=int, default=10)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); w = LD.Writer(os.path.join(a.out, LD.BLOB)); cache = {}; tot = {}
    for i in range(a.layers):
        d = torch.load(os.path.join(a.q, f"L{i}.pt"), map_location="cpu"); st = convert_layer(w, i, d, 128, a.mbits, cache); tot.update(st); print("layer", i, sum(sum(v.values()) for v in st.values()), flush=True)
    print("NOTE: emb/head and raw tensors not written (see docstring); container NOT finished"); w.finish(); json.dump(tot, open(a.out + ".l3_stats.json", "w"))


if __name__ == "__main__": main()
