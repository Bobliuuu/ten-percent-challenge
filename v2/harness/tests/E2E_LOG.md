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
