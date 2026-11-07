# L1_large report (2026-11-07, stopped early: no 70B-class model passed rule 8; budget cap)

Data: t1v2 probe set (1300 items from s1_dev/s3_dev, as used by scale agent D), greedy, base chat template with tools, new scorer.py, 400 new tokens.
Base reference for 70B = RedHatAI Llama-3.3-70B-Instruct-FP8-dynamic in vLLM (bf16 70B does not fit one 96 GB GPU; vLLM online fp8 of the bf16 checkpoint OOM'd at load). It is a proxy for bf16.

| model | base acc (all 1300) | S1 nocall/single/multi | S3 nocall/single/multi | R8 | action |
|---|---|---|---|---|---|
| Llama-3.3-70B-Instruct (fp8 proxy) | 497/1300 = 38% | .84/.77/0.00 | .32/.72/0.00 | FAIL | skipped, no quantization run |
| Mistral-Small-24B-Instruct-2501 (unsloth) | 183/1300 = 14% (only nocall right) | 1.0/0/0 | 1.0/0/0 | FAIL | chat template path emits no tool call |
| Gemma-3-27B-it (unsloth) | identical 183 (nocall only) | same | same | FAIL | template ignores tools |
| GLM-4-32B-0414 | identical 183 (nocall only) | same | same | FAIL | tools not reaching model through our template call (needs per-model prompt work) |

Retention table (model x levers -> bytes, ratio, retention): EMPTY. No quantization was run: nothing passed rule 8, the one 70B attempt on g8 OOM'd because another agent's job held 60-78 GB of that GPU.

## Findings
- Llama-3.x 70B (and by the same template the 8B: 36%) fails the tool-call exam under our scorer: for parallel calls it emits a JSON list of stringified JSON objects ({"type":"function","name":..,"parameters":..}), which scorer.py scores 0 (multi = 0/616 correct, both sources); S3 nocall only 32% (always calls). Base acc 38% is below the 60-92% R8 band. Llama-405B is the same family/template: expected to fail the same way (not downloaded fully; 550 GB downloaded to g12 then deleted).
- Mistral-Large (123B): licence is MRL ("other", non-commercial research), plus 246 GB bf16 does not fit one GPU even at fp8 (123 GB): not pursued. Command-A / Command-R+: CC-BY-NC and gated: not pursued. Yi-1.5-34B-Chat (apache-2.0) not probed.
- Mistral-Small/Gemma/GLM zeros are probe-harness artefacts (tools not rendered or format not parsed), not evidence about the models; a fair probe needs per-model system prompt/format, which was out of budget.
- 405B feasibility (analysis, not run): quant needs ~126 layers x (sens ~1 min + GPTQ several min) = 5-8 h; the evaluator's 4-bit packed codes would be ~200 GB (does not fit 96 GB); 2-bit packing 100 GB still does not; would require streaming codes over PCIe each decode step (~2 s/step). Out of the $300 budget. NO-GO.
- Pipeline changes (pipe/dense_stream.py): EOS/answer format made model-agnostic (Llama eot token, Llama JSON calibration answers), vbase.py (vLLM base eval). Untested beyond the sens pass start (sensA ran ~13 s/layer for 70B before the OOM).

## Go / no-go
NO-GO for this model set under the current scorer/probe: no Llama/Mistral/Gemma/GLM model clears R8, so retention could not be measured. Large-model evidence must come from the Qwen family (agent D). Possible cheap follow-up: probe Yi-1.5-34B, Qwen2.5-72B is D's.

Boxes: g11 idle (GPU free), g12 idle, g8 not mine (another agent's job).
