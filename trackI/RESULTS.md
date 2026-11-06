# trackI: bigger fine-tuned classifier bases (training-free PTQ), 2026-11-06
Scripts: sweep.py (Modal A100; per-model fp/RTN/GPTQ sweep), embq.py (importable embedding recipe), emb.py/emb_local.py (embedding-scheme ablation). Raw: run.log, parsed.json (sweep rows), emb_bert.json.
Eval: agreement with each model's OWN fp32 predictions on dev2[512:] (2488 ex, OOD imdb/amazon/tweet) and dev2_hard (1500). GPTQ calibrated on dev2[:512] (non-sequential Hessians from the fp model, act-order, fp16 scale+zero counted). Never touched SST-2 val/test. Base size = fp32 params*4. Bytes: linear weights b-bit + 32 bits/group, embedding = 4-bit "seen" rows + 2-bit unseen rows (full vocab coverage, embq rare=rtn2), everything else fp16. No entropy coding (worth ~6%). Some GPTQ runs ("-") were lost when the Modal queue stalled (workspace GPU limit). Dev2 "acc" is noisy-label and models differ in training data: use agreement only.

## Table (ratio = bytes/fp32 base; agree = dev2[512:]; hard = dev2_hard)
| model | params | emb share | emb-only (4b seen/2b unseen emb, fp enc) ratio/agree | GPTQ4 g64 ratio/agree/hard | GPTQ3 g64 ratio/agree/hard | RTN3 g64 ratio/agree |
|---|---|---|---|---|---|---|
| utahnlp deberta-v3-large | 435M | 23% | .375 / .9932 | .125 / **.9904** / .963 | - | .104/.974 |
| hhy212 electra-large | 335M | 9% | .463 / .9944 | .139 / .9891 / .958 | - | .111/.935 |
| yoshitomo bert-large | 335M | 9% | .463 / .9948 | .139 / .9867 / .961 | - | .111/.957 |
| hhy212 roberta-large | 355M | 14% | .442 / .9940 | .136 / .9859 / .954 (g128 .1295/.9895) | .1095 / .9747 / .917 | .1095/.859 |
| philschmid roberta-large | 355M | 14% | .442 / .9940 | .136 / .9863 / .951 | - | .1095/.863 |
| siebert roberta-large | 355M | 14% | .442 / .9875 | .136 / .9795 / .963 | - | .1095/.892 |
| cliang deberta-v3-base | 184M | 42% | .279 / .9871 | g128: .1095 / .9815 / .926 | .0987 / .963 / .875 | .0987/.960 |
| cardiff twitter-xlm-r-base (3-class) | 278M | 69% | .210 / .9843 | **.1004** / .9771 / .955 | .0909 / .9425 / .891 | .0909/.815 |
| textattack roberta-base | 125M | 31% | .375 / .9867 | .130 / .9787 / .938 | .1087 / .961 / .872 | .1087/.740 |
| textattack bert-base (ref) | 110M | 21% | .415 / .9895 | .136 / .9839 / .844 | .1117 / .9743 / .738 | .1117/.952 |
| distilbert-sst2-english | 67M | 35% | .361 / .9879 | .133 / .9835 / .945 | .1129 / .9674 / .912 | .1129/.956 |
| assemblyai bert-large | 335M | 9% | .463 / .9932 | (RTN4 .139/.974) | - | .111/.928 |

## Findings
1. Bigger is NOT more robust at fixed bits. At 3 bit g64 every model sits at 94-97.5% agreement with GPTQ and 74-96% with RTN. Large models are worse under RTN (roberta-large .86 vs bert-base .95; outlier dims); GPTQ repairs most of it (roberta-large .975) but not to 99%.
2. Nothing reaches >=99% agreement at <=10% bytes. Closest: deberta-v3-large GPTQ4 g64 at ratio .125 / .990. 4-bit with group scales is already 11-14%; 10% needs ~3 bit incl. scales, where the best is ~97.5%.
3. Cheapest bytes ratio comes from huge-embedding-share models (twitter-xlm-r: .100 at GPTQ4 g64, .977; deberta-v3-base .099 at 3b), but their agreement is lower.
4. EMBEDDING IS A LEAK. Quantising only the embedding (4b seen / 2b unseen) costs 0.5-1.5 pt agreement on every model, bert-base 1.05 pt; large models 0.5-0.7 pt (a bigger base dilutes the embedding share, which helps).
5. dev2 is OOD; SST-like text agrees more (trackD: 99.9% on memorized dev at w4), so dev2 numbers are conservative.

## Embedding recipe with full vocab coverage (bert-base textattack, encoder fp32; whole table fp32 = 93.8 MB)
| scheme | MB | agree | hard |
|---|---|---|---|
| all rows 4b g64 | 13.19 | .9952 | .957 |
| seen4 + unseen RTN3 | 11.37 | .9944 | .940 |
| seen4 + unseen RTN2 (trackA recipe) | 9.55 | .9895 | .907 |
| freq>=3: 4b, rest RTN2 | 9.42 | .9887 | .907 |
| seen4 + unseen PQ96 (k-means 256 centroids per 8-dim subvector) | 7.22 | .9863 | .892 |
| seen4 + unseen 1-bit sign x row scale | 6.86 | .9803 | .855 |
| seen4 + unseen PQ48 | 6.31 | .9783 | .843 |
| freq>=3:4b, rest PQ48 | 6.05 | .9747 | .837 |
| seen3 + unseen PQ48 | 5.20 | .9767 | .811 |
| seen4 + unseen rank-64 SVD + 4b coeffs | 5.79 | .9670 | .800 |
| seen4 + unseen mean row (unk-like) | 5.01 | .9594 | .775 |
| freq>=10:4b, rest PQ48 | 4.54 | .9650 | .779 |

Take-aways: unseen rows carry real signal on OOD text and agreement scales smoothly with bytes spent on them, no free lunch. Best trade found: PQ96 for unseen rows (-2.3 MB vs trackA's RTN2 for -0.3 pt); PQ48 saves another 0.9 MB for -0.8 pt. Frequency tiers beyond seen/unseen do not help.
Importable: `/Users/jhome/ten-percent/trackI/embq.py`, `quantize_embedding(E, freq, hot_thr=1, hot_bits=4, rare="pq96")` returns (dequantised E, exact byte count incl. codebooks and 1 bit/row tier mask); freq = token counts in SST-2 TRAIN (specials forced hot). K-means is Lloyd on the weights (no gradients, seeded) but to ship it store codes uint8 [N,96] + codebook fp16 [96,256,8].
Not run: roberta-large/deberta-large embedding ablation (Modal GPU concurrency limit; job still queued).

## Recommendation
- No candidate meets 10% at >=99% yet; the gap is the encoder at ~3 bit, not the base choice.
- Best single base: utahnlp/sst2_microsoft_deberta-v3-large_seed-1 (best agreement per byte at 4 bit .990, best RTN3 .974). 10% = 174 MB absolute, so mixed precision (4-bit sensitive layers, 3-bit rest), real act-order groups, entropy coding (~6%) and PQ96 embedding are expected to land ~0.10 at ~98.5-99%. Not verified.
- If keeping bert-base: embedding PQ96 gains ~2.3 MB, but encoder 3-bit still ~97.4%.
