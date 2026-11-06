# trackJ: calibration-only recovery after low-bit quantization (no weight training)
Code: tj.py (engine), modal_run.py (launcher; `modal run modal_run.py` then `modal run modal_run.py::round2`), results.jsonl / results2.jsonl (raw), dat/ (dev2 copies).
Protocol: calibrate on dev2[:512] (ncal=512, sequential: Hessians from already-quantized prefix), eval on dev2[512:] (2488 ex, agree2) and dev2_hard. SST-2 val/test never touched. Agreement is vs fp32 base. Noise: SE of agree2 ~0.3pt, hard ~1.2pt; mean-abs-logit-error (mae) is a lower-noise metric.
bits/w = encoder linears (q,k,v,o,f1,f2,pooler) incl. fp16 scale + zero per group + GPTQ perm indices (+ AWQ vectors); fixed-length codes, no entropy coding (which would save ~6% more). "A" = coordinator's trackA recipe (q,k 2b; v,o 3b; f1 4b L0-5/3b L6-11; f2 3b g256; pooler RTN6; embeddings 3b train-seen / 2b others), reproduced: agree .967 (trackA reported .966).
Techniques: clip = per-group MSE clip search (all rows with clip unless "noclip"); bias = per-channel mean-shift to match fp output; scale = per-channel closed-form regression y'=a*y+b vs fp output (a folds into group scales, b into bias: byte-free); ln = same regression on LayerNorm outputs (folds into LN gamma/beta, free); rot256 = random-sign block-Hadamard (256) incoherence rotation on each linear's input before GPTQ (QuIP/QuaRot style, requires online Hadamard on activations at inference, no extra bytes); awq = per-input-channel activation-aware scaling, alpha grid-searched per stage (needs online per-channel divide, +768/3072 fp16 per stage); cle = exact v/o and q/k range equalization (GELU blocks f1/f2). No technique here uses gradients.

## Recipe A (3.13 bits/w encoder, same bytes as trackA 42.7MB model; AWQ +0.01 b)
| config | agree2 | agree_hard | logit mae | acc2 |
|---|---|---|---|---|
| RTN (A bits) | .9349 | .641 | .834 | .8855 |
| GPTQ (baseline = trackA) | .9670 | .728 | .246 | .9064 |
| +bias | .9510 | .658 | .374 | .8927 |
| +bias+scale | .9606 | .728 | .278 | .9039 |
| +bias+scale+ln | .9566 | .721 | .287 | .8999 |
| +ln only | .9682 | .735 | .232 | .9051 |
| +cle | .9610 | .726 | .253 | .9043 |
| +awq | .9711 | .726 | .251 | .9088 |
| +noclip (ablation) | .9666 | .719 | .293 | .9100 |
| **+rot256** | .9759 | .745 | .212 | .9136 |
| +rot64 / rot128 | .9767 / .9719 | .735 / .752 | .216 / .217 | .9168/.9136 |
| +rot256+ln | .9751 | .747 | .208 | .9152 |
| **+rot256+awq** | **.9775** | **.754** | .215 | .9176 |
| +rot256+awq+ln | .9739 | .733 | .212 | .9148 |
| +rot+bias+scale+ln | .9686 | .723 | .246 | .9080 |
| +rot256+scale | .9703 | .731 | .235 | .9112 |

## Uniform w3 g64 (3.305 bits/w)
| config | agree2 | hard | mae | acc2 |
|---|---|---|---|---|
| RTN | .9482 | .651 | .693 | .8883 |
| GPTQ | .9699 | .737 | .204 | .9092 |
| +bias | .9727 | .770 | .197 | .9088 |
| +bias+scale | .9767 | .763 | .187 | .9128 |
| +bias+scale+ln | .9735 | .770 | .191 | .9104 |
| +ln only | .9735 | .763 | .193 | .9080 |
| +cle | .9747 | .749 | .201 | .9108 |
| +awq (3.318b) | .9751 | .773 | .184 | .9136 |
| +rot256 | .9775 | .777 | .177 | .9176 |
| +rot128 | .9775 | .775 | .172 | .9144 |
| **+rot256+ln** | **.9811** | .773 | .172 | .9204 |
| +rot+bias+scale+ln | .9775 | .780 | .171 | .9168 |
| +rot+awq+bias+scale+ln (3.318b) | .9775 | .795 | .168 | .9160 |
| +rot256+awq+ln (3.318b) | .9787 | .783 | .170 | .9164 |

## Uniform w2 g64 (2.29 bits/w): nothing rescues 2-bit
GPTQ .9305 / hard .617 (acc .887). Best: +ln only .9486 / .651 (mae .465); +rot256 .9433; +rot+bias+scale+ln .9385. bias/scale regression hurts at 2 bit (.916-.928): at this noise level the per-channel fit overfits/skews. 2-bit stays 5 pts short of 3-bit.

## Findings
- Biggest single win, byte-free: Hadamard incoherence rotation before GPTQ: +0.9pt agree2 and -14% logit error on recipe A (.967->.976), +0.8pt on w3g64 (.970->.978), hard +1.7 to +4pt. Cost: online 256-block Hadamard on activations at inference (compute only).
- Closed-form bias/scale: helps uniform w3 (+0.3 to +0.7pt, hard +2.6pt) but HURTS recipe A (-0.6 to -1.6pt) because the 2-bit q/k layers dominate error and regression distorts them. Use bias/scale per layer only where bits>=3, untested. LN recalibration alone: neutral to +0.1 (A), +0.4 (w3), strongly positive at w2 (+1.8pt) and mildly positive after rotation on w3 (.9811 best).
- AWQ scaling +0.4pt alone, +0.15 on top of rotation (within noise). CLE roughly neutral/negative on A. Clip search matters mainly for logit error (mae .29->.25) not agreement.
- Differences of <=0.4pt agree2 are within noise (SE .3). Hard-set movements (+-2pt) are also noisy (SE 1.2pt).
- None of this closes the gap to ">=99%": best at <=3.3 bits is .981 dev2 (w3g64 rot256+ln, 3.305 b; 3.297+0.008 perm) vs .967 trackA baseline bytes -> recipe A + rot256 (+awq) reaches .976-.978 at unchanged 42.7MB encoder bytes.
Recommendation: apply rot256 (+awq, +0.012 b) to the trackA recipe: +1.0 to +1.1pt dev2 agreement, +2.6pt hard, acc2 .907->.918, no byte cost. Embeddings in A were RTN (not GPTQ) in my simulation; trackA's export format needs one extra step: fold (R^T diag(1/s)) into inference as an online transform, weights stored as Wq' in rotated basis (seeded signs, so no storage).
Not done: mixed-bit reallocation with rotation, entropy coding, rotating the embedding/residual stream (BERT post-LN blocks exact residual-stream rotation), more calibration data than 512, per-layer selective bias/scale. AdaRound/BRECQ dropped per user rule.
