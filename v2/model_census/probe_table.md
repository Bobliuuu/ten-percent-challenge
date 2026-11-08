| rank | model | base acc | n base-correct | ratio of fp16 | body bpw | retention | Wilson lo | worst cell (n>=8) | per task |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Qwen2.5-32B-Instruct | 0.970 | 1341 | 9.91% | 1.56 | 98.9% | 98.2% | 93% BANK8/bank/card_virtual_wallet | {'SKILL7': 0.989, 'BANK8': 0.988} |
| 2 | Qwen2.5-3B-Instruct | 0.943 | 1303 | 9.95% | 1.52 | 91.9% | 90.3% | 69% SKILL7/massive/Calendar | {'SKILL7': 0.889, 'BANK8': 0.948} |
| 3 | Qwen3-32B | 0.937 | 1295 | 9.93% | 1.57 | 83.6% | 81.5% | 0% BANK8/clin/banking | {'SKILL7': 0.985, 'BANK8': 0.684} |
| 4 | Qwen2.5-1.5B-Instruct | 0.897 | 1240 | 9.95% | 1.47 | 50.4% | 47.6% | 0% BANK8/clin/travel | {'SKILL7': 0.505, 'BANK8': 0.503} |
| 5 | Qwen2.5-0.5B-Instruct | 0.458 | 633 | 9.97% | 1.28 | 0.0% | 0.0% | 0% BANK8/bank/account_identity | {'SKILL7': 0.0, 'BANK8': 0.0} |