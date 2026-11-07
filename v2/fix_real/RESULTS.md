# fix_real: real tool-call generation for C2 (dev only, locked exam never touched)
Same 7B body recipe as the filed artifact (ECSQ step 1.20, dz .15, g128, 384x512 calib), vocab kept 24,176 rows, 4-bit embedding AND the ORIGINAL lm_head rows (4-bit, kept vocab), no letter head, no feedback head.
Output = Qwen native <tool_call> JSON generated greedily from the 4 candidate tool defs. Bar = bf16 base, same prompt. Dev n=2000 (irrelevant 367).
| run | calib | tool-name acc | exact call (name+args) | name acc relevant / irrelevant | retention of base-correct | est bytes |
| base bf16 | - | .785 | .6345 | .871 / .401 | - | 100% |
| q_mix_v24k | letter+gen | .760 | .575 | .873 / .259 | .934 (rel .966, irr .619) | 9.93% (estimate; packed filed artifact came in ~2.3% under the estimate) |
| q_gen_v24k | gen | .7435 | .5545 | .859 / .232 | .915 | 9.93% |
| q_letter_v24k (= filed body calibration) | letter | .147 | .122 | .083 / .433 | - | 9.93% |
General MC sanity (same items as c2_general): q_mix MMLU .445 ARC .598 HS .494 TQA .450 vs base .693/.895/.797/.627 -> ratios .64/.67/.62/.72, gate (>=90%) FAILS. Cause is the 1.74 bpw body, not the head (filed letter artifact: .467/.62/.497/.458).
