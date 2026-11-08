### Qwen2.5-32B-Instruct: t_r100=10.02% of fp16 / 1.58 bpw, t_r1125=11.28% of fp16 / 1.79 bpw, t_r125=12.53% of fp16 / 1.98 bpw
| task | n | base acc | base acc by src | ret t_r100 [Wilson lo] / worst src x kind (n) | ret t_r1125 [Wilson lo] / worst src x kind (n) | ret t_r125 [Wilson lo] / worst src x kind (n) |
|---|---|---|---|---|---|---|
| LOGTPL | 600 | 0.458 | 0.56/0.36 | 0.793 [0.741] (n=275) / 0.25 loghub_a/Zookeeper (24) | 0.807 [0.757] (n=275) / 0.00 loghub_a/Zookeeper (24) | 0.862 [0.816] (n=275) / 0.38 loghub_a/OpenSSH (24) | 
| NER | 600 | 0.340 | 0.50/0.18 | 0.299 [0.240] (n=204) / 0.12 conll2003/ORG (50) | 0.436 [0.370] (n=204) / 0.22 conll2003/ORG (50) | 0.515 [0.446] (n=204) / 0.34 conll2003/ORG (50) | 
| PIIDET | 600 | 0.672 | 0.76/0.58 | 0.861 [0.824] (n=403) / 0.67 gretel_pii_fin/CARD (9) | 0.931 [0.901] (n=403) / 0.78 gretel_pii_fin/CARD (9) | 0.878 [0.843] (n=403) / 0.76 ai4privacy/PASSWORD (34) | 
| PIIRED | 600 | 0.468 | 0.52/0.41 | 0.701 [0.645] (n=281) / 0.40 ai4privacy/IP (25) | 0.804 [0.754] (n=281) / 0.50 gretel_pii_fin/CARD (12) | 0.804 [0.754] (n=281) / 0.50 gretel_pii_fin/CARD (12) | 
| SQL | 600 | 0.098 | 0.14/0.06 | 0.237 [0.147] (n=59) / 0.00 gretel_text2sql/agg (14) | 0.288 [0.188] (n=59) / 0.00 gretel_text2sql/agg (14) | 0.407 [0.291] (n=59) / 0.00 gretel_text2sql/agg (14) | 
| TOOLARG | 600 | 0.932 | 0.93/0.93 | 0.950 [0.929] (n=559) / 0.83 glaive_S1/3+args (48) | 0.946 [0.924] (n=559) / 0.88 glaive_S1/3+args (48) | 0.979 [0.963] (n=559) / 0.96 glaive_S1/3+args (48) | 
| ALL | 3600 | 0.495 | 0.64/0.50/0.93/0.93/0.50/0.14/0.56/0.36/0.06/0.18 | 0.768 [0.748] (n=1781) / 0.00 gretel_text2sql/agg (14) | 0.819 [0.800] (n=1781) / 0.00 loghub_a/Zookeeper (24) | 0.838 [0.820] (n=1781) / 0.00 gretel_text2sql/agg (14) | 
