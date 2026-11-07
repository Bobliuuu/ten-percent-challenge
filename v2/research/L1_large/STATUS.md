# L1_large STATUS
- 01:25Z: using tenpct-g8 (RTX PRO 6000 96GB, free at start) for Llama-3.3-70B-Instruct (unsloth mirror). Workdir /opt/dlami/nvme/work/L1. Pipeline copy in pipe/.
- 01:30Z: coordinator freed g11 (base fp8 vLLM of Llama-3.3-70B runs there), g12 (405B download feasibility, unsloth/Llama-3.1-405B-Instruct), g8 shared with saliency job (my quant of Llama-3.3-70B t096).
- 01:25Z: g8 GPU also used by another agent (saliency.py Qwen3-30B, 60 GB). I share it: my quant (pass A/B) runs in the remaining ~35 GB; vLLM fp8 base waits until their job ends.
- 02:00Z: Llama-3.3-70B base (fp8 RedHat proxy, vLLM, 1300 t1v2 items): 497/1300=38%; S1 multi 0%, S3 multi 0%, S3 nocall 32%. Output format is JSON list of stringified JSON (type function), scorer gives 0 on multi. Fails R8 -> skip Llama quantization. g12: 405B download killed, g12 idle. g8: other agent's job occupies GPU (my q33 job OOM'd, not running). g11 running base probes of Mistral-Small-24B, Gemma-3-27B, GLM-4-32B.
- final: REPORT.md written; g11, g12 idle.
