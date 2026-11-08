### Qwen2.5-14B-Instruct: t_r100=10.03% of fp16 / 1.56 bpw, t_r1125=11.29% of fp16 / 1.78 bpw, t_r125=12.54% of fp16 / 1.94 bpw
| task | n | base acc | base acc by src | ret t_r100 [Wilson lo] / worst src x kind (n) | ret t_r1125 [Wilson lo] / worst src x kind (n) | ret t_r125 [Wilson lo] / worst src x kind (n) |
|---|---|---|---|---|---|---|
| LOGTPL | 600 | 0.453 | 0.57/0.33 | 0.331 [0.278] (n=272) / 0.00 loghub_a/Apache (31) | 0.783 [0.730] (n=272) / 0.12 loghub_a/OpenSSH (26) | 0.761 [0.707] (n=272) / 0.10 loghub_b/Windows (10) | 
| NER | 600 | 0.285 | 0.39/0.18 | 0.082 [0.049] (n=171) / 0.00 conll2003/ORG (45) | 0.298 [0.235] (n=171) / 0.00 wnut17/location (9) | 0.339 [0.272] (n=171) / 0.00 wnut17/creative-work (10) | 
| PIIDET | 600 | 0.623 | 0.74/0.51 | 0.690 [0.641] (n=374) / 0.52 ai4privacy/PASSWORD (33) | 0.840 [0.799] (n=374) / 0.64 ai4privacy/PASSWORD (33) | 0.906 [0.873] (n=374) / 0.76 gretel_pii_fin/EMAIL (25) | 
| PIIRED | 600 | 0.555 | 0.64/0.47 | 0.273 [0.228] (n=333) / 0.06 gretel_pii_fin/EMAIL (17) | 0.679 [0.627] (n=333) / 0.32 ai4privacy/IBAN (19) | 0.790 [0.743] (n=333) / 0.50 ai4privacy/SSN (14) | 
| SQL | 600 | 0.047 | 0.04/0.05 | 0.500 [0.326] (n=28) / - | 0.500 [0.326] (n=28) / - | 0.679 [0.493] (n=28) / - | 
| TOOLARG | 600 | 0.927 | 0.93/0.93 | 0.811 [0.777] (n=556) / 0.57 glaive_S1/3+args (49) | 0.874 [0.844] (n=556) / 0.82 glaive_S1/3+args (49) | 0.917 [0.891] (n=556) / 0.87 glaive_S1/2args (131) | 
| ALL | 3600 | 0.482 | 0.69/0.39/0.93/0.93/0.49/0.04/0.57/0.33/0.05/0.18 | 0.529 [0.506] (n=1734) / 0.00 loghub_a/Apache (31) | 0.752 [0.731] (n=1734) / 0.00 wnut17/location (9) | 0.805 [0.786] (n=1734) / 0.00 wnut17/creative-work (10) | 
