INJ new-label ('Jailbreak') needs, 2026-11-07
1. Unfinished: only Qwen3-4B (tied embeddings, slow forward-path eval), methods lda,cal2,cal, 10 draws, running on tenpct-g16 (started ~07:00Z, ETA ~1-1.5h). Everything else (8B,14B,32B) is finished and synced to ~/ten-percent/v2/combo/INJ_newlabel/results*/.
2. Command (from ~/work/inl on box; venv with torch 2.8, transformers 4.57):
   python run_inl_final.py --model Qwen/Qwen3-4B --methods lda,cal2,cal --ndraw 10 --nosens --out res5/Qwen3-4B
   (file: ~/ten-percent/v2/combo/INJ_newlabel/run_inl_final.py; --data defaults to ~/work/inl/data = copy of INJ_newlabel/data/)
3. Model: Qwen/Qwen3-4B (8 GB bf16). Not resumable per cell; rerun is ~1-1.5h on one GPU.
4. Files: run_inl_final.py + data/*.jsonl (dev-only, jackhhao/toxic-chat/INJ dev+pool; no S2/locked).
5. ~1.5 GPU-h, >=40 GB VRAM (24 GB likely ok). Low priority; may be dropped.
