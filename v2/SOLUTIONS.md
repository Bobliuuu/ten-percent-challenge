# SOLUTIONS (rows appended by lane; another agent may merge into the main table)

| lane | model / recipe | % of fp16 | key result | verdict | doc |
|---|---|---|---|---|---|
| GIANT_MOE | DeepSeek-R1/V3, Kimi-K2, Qwen3-235B/480B, GLM-4.5, Maverick, gpt-oss-120b at ~1.5 b experts | needs 10.0 | byte math: experts must get 1.4-1.55 b all-in; best community 1.58-bit file (R1 UD-IQ1_S) is 10.44%, all others 11.8-18% | NO-GO, no GPU run (probe ~$130 > cap) | v2/GIANT_MOE.md |
