# trackK: embedding-only compression (word+pos+type), rest of model fp16
Code: embq.py (`build_embedding(recipe)->(packed_bytes, lookup_fn)`, `decode(packed_bytes)->lookup_fn`), ev.py (eval), sweep.py, logs sweep1-7.log.
Eval: base = fp32 model; test = rest of model rounded to fp16, embedding replaced by the DECODED table (decoded from the real packed bytes). Agreement with base prediction on dev2 (3000), dev2_hard (1500), SST-2 train dev (2000, splits.json "dev"). max_len 128, CPU. Never touched SST-2 val/test.
Vocab counts: SST-2 TRAIN text minus dev+feedback indices (so the train dev is truly held-out for the vocab choice): only 11,553 of 30,522 wordpieces occur. dev2 (IMDB/Amazon/tweets) is full of unseen wordpieces = the stress test for "exam has tokens train never saw".
Bytes = real packed bytes of everything (codes, scales, levels, mean row, tier-label map 2-3 bit/id, fallback map, pos, type). fp32 embedding = 93.6MB+1.6MB pos. Reference: 8-bit g128 RTN full vocab agrees 99.97/99.5/100 (dev2/hard/sst).
"word MB" below = word table + maps + type + header, excluding positions (add 0.197MB for 128 fp16 positions, see below).

## Quantizers (all on mean-row-centered table; mean row stored)
- rtn{b}: asym min/max per group g, fp16 scale+min.
- lm{b}: NEW "Lloyd-Max": per-group (g) fp16 RMS scale, ONE global set of 2^b non-uniform levels (1-D k-means of x/rms, fp16, a few bytes). Half the scale overhead of RTN and better MSE (rows are ~Gaussian). Strictly better than RTN at equal bits: 2-bit g128 .9813 vs .9760 dev2 at 6.24 vs 6.60MB.
- pq{ds,k}: product quantization (k-means per ds-dim subspace, k=256, fp16 codebook shared by the tier).
- tiers by train frequency rank (special tokens always in tier 0); ids beyond the last tier use a fallback.
- fallback nn: nearest kept row by cosine (stored as per-id index, ~15 bit/id); unk: map to [UNK].

## Pareto (word MB, agreement with base; fp16 rest)
| recipe | word MB | dev2 | hard | sst-train-dev |
|---|---|---|---|---|
| rtn8 full | 24.18 | .9997 | .9953 | 1.000 |
| lm4 full | 12.10 | .9917 | .9453 | .9990 |
| rtn4 full g128 / g64 | 12.46 / 13.19 | .9947 / .9953 | .9467 / .9480 | 1.000 |
| lm3 full | 9.17 | .9907 | .8993 | .9995 |
| rtn3 full g128 | 9.53 | .9907 | .8933 | .9990 |
| top4k lm4 + rest lm2 | 7.01 | .9860 | .8827 | 1.000 |
| top8k lm3 + rest lm2 | 7.01 | .9873 | .8700 | .9990 |
| top8k rtn4 + rest rtn2 | 8.14 | .9847 | .8987 | 1.000 |
| top8k rtn3 + rest rtn2 | 7.37 | .9843 | .8607 | .9990 |
| lm2 full g64 / g128 / g256 | 6.60 / 6.24 / 6.05 | .9840 / .9813 / .9803 | .821 / .826 / .813 | .997 / .997 / .995 |
| rtn2 full g128 | 6.60 | .9760 | .7607 | .9940 |
| top2k lm4 + rest lm2 | 6.62 | .9850 | .8707 | .9995 |
| top4k lm4 + 8k lm2 + rest lm1 (R4) | 5.23 | .9797 | .8460 | .9995 |
| top2k lm4 + 6k lm3 + rest lm1 (R5) | 5.03 | .9800 | .8273 | .9985 |
| top4k lm4 + 8k lm2 + rest pq8 | 5.40 | .9807 | .8587 | .9995 |
| top8k lm3 + rest pq8 | 4.97 | .9790 | .8327 | .9970 |
| top2k rtn4 + rest pq(ds8,k256) | 3.96 | .9743 | .7980 | .9940 |
| top4k lm4 + rest lm1 | 4.46 | .9753 | .8093 | .9940 |
| top1k rtn4 + rest pq8 | 3.65 | .9710 | .7773 | .9955 |
| lm1 full g64 | 3.67 | .9557 | .6760 | .9815 |
| pq ds8 k256 full | 3.33 | .9633 | .7013 | .9885 |
| top2k rtn4+8k pq8+rest pq16 | 3.37 | .9693 | .7647 | .9940 |
| pq ds4 k256 full | 6.26 | .9840 | .8333 | .9970 |
| pq ds16 / ds12 / ds8k64 full | 1.87 / 2.36 / 2.30 | .875 / .932 / .932 | .58-.63 | .90-.97 |
(Totals in the sweep logs include fp16 pos/type; the "word MB" column excludes position rows.)

## Vocab trimming (no low-bit tail): the unseen-wordpiece effect
| kept (rtn4) | fallback nn | fallback [UNK] |
|---|---|---|
| 4k tokens (1.7MB) | dev2 .940 / sst .970 | .901 / .948 |
| 8k (3.3MB) | .962 / .990 | .936 / .985 |
| 11k = all train-seen (4.5MB) | .970 / .9995 | .958 / .9985 |
- Dropping tokens is bad on out-of-domain text: even keeping every train-seen type loses 3 pts on dev2 (non-SST wordpieces). Keeping a cheap low-bit tail for ALL 30.5k ids beats trimming at equal bytes (top1k@4+rest pq8: 3.65MB .971 vs 8k trim 3.3MB .962; top4k lm4+rest lm1 4.46MB .975 vs 11k trim 4.5MB .970). The exam is the SST test set (in-domain, mostly train-seen words) so unseen risk is smaller than dev2 suggests, but the tail is cheap insurance.
- Nearest-kept-row fallback (cosine, 15 bit/id map ~57KB) is worth +1.2 to +4 pts over [UNK] when trimming. Not needed if the tail is kept.

## Positions / type
Positions 128 or 64 rows fp16 (0.197 / 0.098MB, vs 0.786MB for 512): zero change in agreement on every set (identical to 512 up to rounding), 8-bit pos also unchanged (.9847, saves 0.09MB more). Exam max length for SST-2 sentences is < 70 wordpieces; use 128 fp16 (safe) = 0.197MB. Token type: 2 rows fp16 (3KB).

## Findings
1. Full-vocab at 3-4 bits is near lossless only for 4 bit (dev2 .995, 12.5MB); sub-4-bit loses 1-2 pts on dev2 and 10+ on hard. The embedding is a real accuracy cost at every point under ~9MB.
2. Frequency-tiered bits + non-uniform (Lloyd-Max) levels is the best family: the frequent 2-4k tokens carry most of the usage (train text), the 20k+ rare tokens contribute through out-of-domain text only.
3. Beyond ~2 bits/elem on the tail PQ (ds=8, 256 centroids, 1 bit/elem) is competitive with 1-bit LM at the same size but PQ ds>=12 collapses. PQ ds4 (2 bit/elem) is worse than lm2 per byte. SVD low-rank was shown unusable by trackA; hashing / shared sub-codebooks not pursued (PQ with a shared codebook already is that; collapse at high compression suggests no).
4. Entropy coding gains ~1-5% for 2-4 bit codes (est. in logs); not applied.

## Recommendation
- Default (replaces trackA's ~7.7MB 3/2-bit embedding with ~6.8MB total at higher agreement): `tiers=[lm(2000,4,g128), lm(rest,2,g128)]`, fallback irrelevant, pos=128 fp16 -> 6.82MB total, dev2 .985 / hard .871 / sst .9995. Or `lm4 top4000 + lm2 rest` 7.2MB total, .986/.883/1.000.
- If bytes are tight (spend them on encoder bits): R4 = `lm(4000,4) + lm(8000,2) + lm(rest,1)`: 5.4MB total, .980/.846/.9995; or top8k lm3 + rest PQ(ds8,k256): 5.2MB, .979/.833/.997. Below 4.5MB agreement falls faster than the encoder gains (rule of thumb: each 1MB below ~5MB costs ~1pt dev2).
- Not recommended: vocab trimming without a tail, PQ ds>=12, SVD.
Usage: `from embq import build_embedding, decode; packed,lk=build_embedding(dict(tiers=[dict(n=2000,bits=4,g=128,lm=True),dict(n=30522-2000,bits=2,g=128,lm=True)],fallback='nn',pos=128)); lk.word_table() [30522,768], lk.pos [128,768], lk.type [2,768]; lk(ids)->rows; decode(packed) rebuilds lk from bytes alone (needed in the final loader; ship embq.decode + numpy only).` Token counts come from train_counts() (SST-2 train minus dev/feedback); for the final model recompute with all train text -- ordering of ids is baked into the bytes via the tier-label map, so the decoder needs no counts.
