# v2/prune RESULTS (Qwen2.5-14B-Instruct, MMLU dev n=1987, base 0.8007; calib-only, no gradients)
Engine: prune.py (layer drop by angular distance + closed-form LS patch, FFN-neuron / KV-group-head prune + LS o_proj/down_proj reconstruction, GPTQ, H-weighted SVD low-rank residual). Raw: results/*.json, table.txt. ratio = bytes/29.54GB (emb assumed 33k vocab 4-bit, ~0.09GB; dense16 = 0.8977).
Wrap-up: Modal apps stopped by coordinator. p2 (dense GPTQ 2/3/4 bit, low-rank residual) and p3 (FFN lam/skip0 variants, late-block drops) NEVER FINISHED -> no quant+removal frontier. Only removal-only (p1) is measured.

## Removal only (no quantization), acc / agree / ratio
- Layer drop (angular), patch none/rank256/full-LS: n=2 [41,43] .797/.797/.793, ratio .86; n=4 [41,45] .785/.788/.784 (-1.6pt), ratio .823. n=6 [22,28] .57/.62/.59 (angular picked a MIDDLE block -> collapse); n=8 .43/.52/.48; n=10 .32/.44/.42; n=12 ~.3; n=16 ~.23. The closed-form patch helps only on bad blocks (+5-12pt) and not on tail blocks (0 to -0.3pt). Not tested: forced late blocks (p3, unfinished).
- FFN neuron prune (keep frac, LS recon vs none): .9 -> .759 rec / .776 norec (-2.5..-4pt, ratio .829); .75 -> .684/.718; .6 -> .597/.621; .5 -> .540/.534; .4 -> .46; .3 -> .42/.37. LS recon (all tokens, lam .01) is NOT reliably better; it is worse at mild pruning (likely dominated by massive-activation first tokens; skip0/lam fix untested).
- KV-group head prune (keep 8 groups frac): .875 -> .7715 rec / .7675 norec (ratio .872); .75 -> .748/.673; .5 -> .624/.229. Recon essential here (+5..40pt).
- Cross-layer tie (copy layer i weights to j, zero bytes): 30->31 .792 (-0.9pt), 20->21 .749, 20->22 .754. One tied layer saves only ~2% bytes.
- (d) Delta coding: dead. Adjacent-layer weights are orthogonal: cos ~0 (|cos|<0.01, max .06), ||Wi-Wj||/||Wi|| = 1.43 ~ sqrt2; RTN-3bit relative error of delta (.048) == of W (.049). No gain from delta coding or sharing.

## Conclusions
- Nothing removal-only gets within 1pt of base while cutting more than ~10% of bytes (best: drop 2 tail layers -0.4pt for -4% bytes; drop 4 tail layers -1.6pt for -8%; head .875 -3pt). Structured removal is far too lossy to supply a 5-6x factor; 10% budget (1.6 bpw incl. everything) needs ~all of it from quantization/VQ.
- Complementary: only tail-layer dropping (2 layers, nearly free) and mild head pruning with recon. Not complementary / not worth bytes: FFN neuron pruning (steep loss per byte vs. lowering bits), mid-layer drops, delta coding/sharing.
- Unmeasured (killed): low-rank+quant residual and removal x bits frontier; code ready (prune.py cfgs p2.json/p3.json) if Modal capacity returns.
Runs: ap-4oSE41orFIAaJtVP40oEJY (smoke), ap-pJN1EDGqCWbeSc6GNLkwPe (p1+p2), ap-L0HhmWzmbdSMtbOOjfoKiP (p3).
