# C2 packed 7B artifact (Qwen2.5-7B-Instruct), altpush recipe
Prompt frozen: zero-shot letter format (inst, Text, Options A-E, Answer:), argmax over 5 letter logits. Calibration: C2 calib only (384x512 tokens); vocab 24000 from C2 pool+calib text (no dev). Hadamard-128 + GPTQ + ECSQ (c=1.20, dz=0.15, K=7), rANS, emb 4-bit, lm_head only the letter rows. No bias used in the headline (raw acc).
- c120_dz15: 1,431,929,562 B = 9.401% of fp16 (15,231,233,024); dev n=2000 raw acc 0.8025 vs bf16 base 0.7665 (zs); agree w/ base 0.8635. Decode-only reload (verify_artifact.py) reproduces logits exactly (max |dlogit| 0). Harness size_check PASS (after deleting __pycache__ that verify created on the volume copy).
- c114_dz15: 1,492,453,542 B = 9.799%; acc 0.7845; reload exact.
Location: Modal volume tenpct-C-res:/art_c120_dz15, local /Users/jhome/ten-percent-artifacts/artC2_c120. Hashes in ARTIFACT_C2_sha256.txt. Sweep table: res/c2_*.json. Locked untouched.
