# Stand-in end-to-end verification (DEV only; stand-in locks; the real locked exams were NOT touched)

## 65c6b0e727a0a7ab:standin:art7b:stage1
- status completed; artifact_tree_sha256 f5c7ba7550b0e57bda41d0838f3f0669e8297d5229fa37a91958bf6236a93040; memory_sha256 None
- size 1,482,962,994 B of budget 1,523,123,302 (within=True); sandbox {'network': 'blocked (OSError)', 'hf_cache_files': 0}; harness_unchanged_since_lock True
- candidate - base accuracy = +3.00 pt, paired 95% CI [-0.34, +6.34] pt (n=100, base-only-right b=0, cand-only-right c=3); NI(1pt) p=0.167 non_inferior=False; McNemar p=0.25
- bias-off (same run): candidate - base accuracy = +3.00 pt, paired 95% CI [-0.34, +6.34] pt (n=100, base-only-right b=0, cand-only-right c=3)
- base_seconds 26, cand_seconds 355; PASS_BOTH_STAGES None
## 65c6b0e727a0a7ab:standin:art7b:stage2
- status completed; artifact_tree_sha256 f5c7ba7550b0e57bda41d0838f3f0669e8297d5229fa37a91958bf6236a93040; memory_sha256 d70326071e8c965ab2ca0da9a5a0010bac2f7e8b987803ffa94aa6d1ee0988e2
- size 1,482,963,268 B of budget 1,523,123,302 (within=True); sandbox {'network': 'blocked (OSError)', 'hf_cache_files': 0}; harness_unchanged_since_lock True
- candidate - base accuracy = +3.00 pt, paired 95% CI [-0.34, +6.34] pt (n=100, base-only-right b=0, cand-only-right c=3); NI(1pt) p=0.167 non_inferior=False; McNemar p=0.25
- bias-off (same run): candidate - base accuracy = +3.00 pt, paired 95% CI [-0.34, +6.34] pt (n=100, base-only-right b=0, cand-only-right c=3)
- base_seconds 26, cand_seconds 327; PASS_BOTH_STAGES False
## 65c6b0e727a0a7ab:standin:tiny:stage1
- status completed; artifact_tree_sha256 9ae41210e573c787f78cb9c4c4d81c8ae2c145197cf2c3336017f286ab92ec35; memory_sha256 None
- size 218 B of budget 98,806,554 (within=True); sandbox {'network': 'blocked (OSError)', 'hf_cache_files': 0}; harness_unchanged_since_lock True
- candidate - base accuracy = -49.00 pt, paired 95% CI [-60.59, -37.41] pt (n=100, base-only-right b=54, cand-only-right c=5); NI(1pt) p=1 non_inferior=False; McNemar p=1.91e-11
- base_seconds 384, cand_seconds 0; PASS_BOTH_STAGES None
## 65c6b0e727a0a7ab:standin:tiny:stage2
- status completed; artifact_tree_sha256 9ae41210e573c787f78cb9c4c4d81c8ae2c145197cf2c3336017f286ab92ec35; memory_sha256 d70326071e8c965ab2ca0da9a5a0010bac2f7e8b987803ffa94aa6d1ee0988e2
- size 492 B of budget 98,806,554 (within=True); sandbox {'network': 'blocked (OSError)', 'hf_cache_files': 0}; harness_unchanged_since_lock True
- candidate - base accuracy = -40.00 pt, paired 95% CI [-51.43, -28.57] pt (n=100, base-only-right b=45, cand-only-right c=5); NI(1pt) p=1 non_inferior=False; McNemar p=4.21e-09
- base_seconds 384, cand_seconds 0; PASS_BOTH_STAGES False

## Refusal tests (tests/run_tiny_e2e.sh)
- repeat stage 1: REFUSED locally (O_EXCL claim file).
- repeat stage 2 with local claim deleted: REFUSED remotely by the Modal Dict claim ('exam already run ... single use per stage').
- Earlier iterations also exercised: candidate crash consumes the run (receipt candidate_crashed); infra exception releases the claim (aborts/ record, labels_scored false); Modal pre-emption restart resumed via same-nonce rule.
- PASS is False in all stand-in receipts by design (stand-in locks never yield REAL_RESULT; tiny/prior candidate is also far below base).

# C2 stand-in end-to-end (DEV 100 rows, task c2, base Qwen2.5-7B-Instruct, artifact artC2_fb_fixed = volume art_c120_dz15 body + c2_fb predict.py/head.npz)

## 7cc05cd39379d452:standin:c2fb:stage1
- status completed; tree 6290414d96f3119799516442cab5106243dede6a5b29256f24669f0581ee9661; memory_sha256 None
- size 1,432,412,701 B (budget 1,523,123,302, ok=True); sandbox {'network': 'blocked (OSError)', 'hf_cache_files': 0}; harness_unchanged True
- base 0.720 cand 0.790; candidate - base accuracy = +7.00 pt, paired 95% CI [-1.88, +15.88] pt (n=100, base-only-right b=7, cand-only-right c=14); NI p=0.109 non_inferior=False; PASS=False REAL_RESULT=False PASS_BOTH_STAGES=None; cand_seconds 189
## 7cc05cd39379d452:standin:c2fb:stage2
- status completed; tree 6290414d96f3119799516442cab5106243dede6a5b29256f24669f0581ee9661; memory_sha256 ab2f39ef908bce27a6fe85fb1d67c52f92df6f610007d5bbd2beb8815e66224b
- size 1,432,412,748 B (budget 1,523,123,302, ok=True); sandbox {'network': 'blocked (OSError)', 'hf_cache_files': 0}; harness_unchanged True
- base 0.720 cand 0.820; candidate - base accuracy = +10.00 pt, paired 95% CI [+2.41, +17.59] pt (n=100, base-only-right b=3, cand-only-right c=13); NI p=0.0303 non_inferior=True; PASS=True REAL_RESULT=False PASS_BOTH_STAGES=False; cand_seconds 184

Findings: (1) the label-leaking C2 row fields (gold_calls, irrelevant, tools, origin) are stripped by the whitelist `candidate_view`. (2) The Mac copies ~/ten-percent-artifacts/artC2_c120 and artC2_fb have a CORRUPT body.bin (sha f828171461...; same size as the good one, 6 ANS streams fail to decode, e.g. 18.mlp.gate_proj) vs the Modal volume tenpct-C-res:art_c120_dz15/body.bin (sha 0304a803ae07...; decodes). ARTIFACT_C2_sha256.txt records the corrupt hash. Use artC2_fb_fixed (volume body + c2_fb predict.py + head.npz). The first two C2 e2e attempts consumed their stand-in runs with candidate_crashed (ValueError: ANS compressed data never ends in a zero word); receipts kept in receipts/standin_c2fb_attempt*.
