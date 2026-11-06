# trackH insurance base: cardiffnlp/twitter-xlm-roberta-base-sentiment (278M, emb 69% of fp32 bytes)
Script: trackH/run.py (python -u run.py MODEL_ID); live log trackH/log_cardiff.txt (later configs w6/w8/emb3 may still append).
Data: mteb/amazon_polarity TEST parquet (not SST-2). md5(text)%10: 0-1 LOCKED (dropped at load, never used), 2-5 dev (800-2000 rows), 6-9 calib (4000 rows, vocab trim). Binary = argmax of pos/neg logits (neutral ignored).
Base fp32 = 1,112,183,820 B. Base dev acc 83.5% (n=2000; domain shift from tweets, so acc is modest). Bytes counted: quantized ints + fp16 scale/min per group + kept-id gap list; tokenizer file not counted on either side.
| config | bytes % fp32 (raw / entropy-coded) | agree w/ base | dev acc |
| trim 15k vocab, fp32 | 35.2 | 99.05% | 83.9 |
| trim 5k vocab, fp32 | 32.3 | 95.45% | 84.3 |
| trim15k + emb4 g64 + enc 4bit g64 | 5.11 / 4.80 | 94.75% | 82.5 |
| trim12k + same | 4.98 / 4.68 | 94.4% | 82.8 |
| trim5k + same | 4.71 / 4.42 | 93.45% | 81.8 |
Takeaways: non-embedding part is 31% of bytes, so 4-bit everything lands at ~4.5-5% (<8% target OK) but RTN 4-bit encoder gives only ~94-95% agreement (need >=99.5). Vocab trimming alone costs agreement (99.05% at 15k from 4000 calib texts; unseen tokens -> unk) - need much larger calib text / keep more vocab. Not tested yet: encoder 6/8-bit (est. ~8-11% bytes with 4-bit emb), which is the likely feasible point; w6/w8 results pending in log.
Recommendation: do NOT switch main line yet. This base has more byte headroom (4-bit all = ~5%) but 3-class tweet model has weak base acc on out-of-domain data and RTN-4 agreement is poor; headroom allows ~6-bit encoder + 4-bit emb (~8%) -- check log. Keep bert-base main; use this as fallback only if main cannot reach 10% bytes.

Late result: trim5k + emb4 g64 + encoder 8bit g128 = 8.31% raw / 7.90% entropy-coded, agree 95.85%, acc 84.05. Agreement matches trim5k fp32 (95.45%), so the agreement loss is from vocab trimming (4000-text calib set), not from quantization. Fix: larger calib corpus / larger kept vocab; quantization at 8-bit encoder + 4-bit emb is essentially lossless here.

Late result 2: trim12k + emb4 g64 + encoder 6bit g64 = 6.90% raw / 6.61% entropy-coded, agree 98.0%, acc 84.0 (also 5k w6: 6.63/6.35%, 95.2%). Larger kept vocab raises agreement (12k vs 5k: 98.0 vs 95.2) at ~0.3 pt bytes; 99.5% likely reachable with ~15-30k kept vocab or bigger calib set while staying <8%.

Late result 3 (sweep finished): trim5k + emb3 g64 + enc 4bit g64 = 4.66% raw / 4.37% coded, agree 93.4%, acc 81.9 (emb 3-bit vs 4-bit: no meaningful change). Best point: trim12k emb4 w6 = 6.90/6.61%, 98.0% agree. Untested: 15-30k vocab or larger calib set for >=99.5%.
