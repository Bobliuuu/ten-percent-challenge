
### Qwen2.5-14B-Instruct/t3 r985 ratio=9.882% body_bpw=1.53

| task | n | base acc | worst class base acc | margin>=6 | margin>=10 | retention [Wilson lo] | worst kind ret (n) | worst src x kind (n) |
|---|---|---|---|---|---|---|---|---|
| ADE | 160 | 0.863 | 0.86 | 0.93 | 0.88 | 0.812 [0.738] | 0.65 Yes (69) | 0.65 ade_corpus/Yes (69) |
| ANSWERABLE | 160 | 0.650 | 0.61 | 0.23 | 0.00 | 0.837 [0.754] | 0.82 Answerable (49) | 0.82 squad2/Answerable (49) |
| BOOLQ | 160 | 0.806 | 0.64 | 0.96 | 0.73 | 0.791 [0.713] | 0.59 Yes (51) | 0.59 boolq/Yes (51) |
| CODELANG | 312 | 0.984 | 0.92 | 0.97 | 0.90 | 0.524 [0.469] | 0.00 PHP (51) | 0.00 codesearchnet/PHP (25) |
| CODEPROSE | 320 | 1.000 | 1.00 | 1.00 | 1.00 | 0.991 [0.973] | 0.98 Prose (160) | 0.97 csn_vs_agnews/Prose (80) |
| DEFECT | 160 | 0.419 | 0.29 | 0.30 | 0.00 | 0.672 [0.553] | 0.04 Safe (23) | 0.04 devign/Safe (23) |
| DEPT3 | 318 | 0.654 | 0.36 | 0.89 | 0.66 | 0.889 [0.840] | 0.55 Sales (38) | 0.54 bitext/Sales (24) |
| ENTAIL2 | 320 | 0.894 | 0.86 | 0.96 | 0.86 | 0.671 [0.615] | 0.34 Yes (137) | 0.14 scitail/Yes (64) |
| FAKENEWS | 160 | 0.719 | 0.51 | 0.87 | 0.79 | 0.409 [0.323] | 0.08 Real (74) | 0.08 gonzaloa/Real (74) |
| FINSENT | 159 | 0.836 | 0.66 | 0.78 | 0.53 | 0.752 [0.672] | 0.60 Bearish (48) | 0.60 twitterfin/Bearish (48) |
| HATE2 | 320 | 0.853 | 0.81 | 0.93 | 0.88 | 0.806 [0.755] | 0.73 Fine (129) | 0.65 tweet_eval_hate/Fine (60) |
| INJ2 | 320 | 0.809 | 0.64 | 0.85 | 0.61 | 0.892 [0.848] | 0.75 Injection (103) | 0.54 deepset/Injection (39) |
| LANG20 | 400 | 0.955 | 0.50 | 0.95 | 0.84 | 0.461 [0.411] | 0.00 Arabic (20) | 0.00 papluca/Arabic (20) |
| NLI3 | 477 | 0.843 | 0.79 | 0.92 | 0.81 | 0.667 [0.619] | 0.46 Entailment (135) | 0.35 snli/Entailment (46) |
| OFFEN2 | 320 | 0.812 | 0.77 | 0.48 | 0.38 | 0.835 [0.785] | 0.76 Clean (123) | 0.71 tweet_eval_off/Clean (62) |
| PARA2 | 320 | 0.784 | 0.64 | 0.91 | 0.86 | 0.789 [0.734] | 0.52 Yes (102) | 0.42 mrpc/Yes (53) |
| PHISH2 | 320 | 0.903 | 0.89 | 0.84 | 0.64 | 0.505 [0.448] | 0.00 Legitimate (143) | 0.00 sevenphish/Legitimate (71) |
| PIIFLAG | 319 | 0.696 | 0.51 | 0.95 | 0.86 | 0.896 [0.849] | 0.77 Yes (81) | 0.75 ai4privacy/Yes (69) |
| PSAFE | 480 | 0.787 | 0.74 | 0.94 | 0.73 | 0.902 [0.868] | 0.87 Safe (177) | 0.83 beavertails_prompt/Safe (35) |
| RCT5 | 320 | 0.672 | 0.45 | 0.86 | 0.65 | 0.647 [0.581] | 0.10 Conclusions (29) | 0.07 pubmed_rct_valsplit/Conclusions (15) |
| SARC2 | 320 | 0.681 | 0.54 | 0.87 | 0.60 | 0.651 [0.586] | 0.21 Sincere (87) | 0.20 headlines/Sincere (59) |
| TICKETPRIO | 159 | 0.365 | 0.00 | 0.71 | 0.48 | 0.879 [0.771] | 0.76 Medium (25) | 0.76 tobi_tickets/Medium (25) |
| TICKETTYPE | 160 | 0.556 | 0.00 | 0.79 | 0.61 | 0.775 [0.678] | 0.24 Incident (25) | 0.24 tobi_tickets/Incident (25) |
| TOPIC5 | 320 | 0.819 | 0.53 | 0.91 | 0.63 | 0.851 [0.803] | 0.80 Politics (56) | 0.73 yahoo_answers/Politics (26) |
| TOX2 | 320 | 0.791 | 0.68 | 0.85 | 0.61 | 0.834 [0.783] | 0.70 Toxic (108) | 0.56 toxic_chat/Toxic (45) |
| UNFAIR | 160 | 0.713 | 0.60 | 0.66 | 0.41 | 0.596 [0.505] | 0.04 Fair (48) | 0.04 lexglue_unfairtos/Fair (48) |

### Qwen2.5-14B-Instruct/t3r2 r985 ratio=9.882% body_bpw=1.53

| task | n | base acc | worst class base acc | margin>=6 | margin>=10 | retention [Wilson lo] | worst kind ret (n) | worst src x kind (n) |
|---|---|---|---|---|---|---|---|---|
| LANG12 | 360 | 0.997 | 0.97 | 1.00 | 0.98 | 0.777 [0.731] | 0.10 Japanese (30) | 0.10 papluca/Japanese (30) |
| NEWS5 | 320 | 0.887 | 0.72 | 0.93 | 0.59 | 0.901 [0.861] | 0.73 Business (55) | 0.60 huffpost/Business (25) |
| PHISH2B | 320 | 0.897 | 0.79 | 0.90 | 0.82 | 0.693 [0.638] | 0.45 Legitimate (160) | 0.44 sevenphish/Legitimate (80) |
| PII2 | 320 | 0.887 | 0.86 | 0.90 | 0.75 | 0.919 [0.881] | 0.91 Clean (138) | 0.78 ai4privacy/Clean (59) |
| SENT2B | 480 | 0.919 | 0.89 | 0.93 | 0.85 | 0.710 [0.666] | 0.40 Positive (214) | 0.24 tweets/Positive (76) |
| TOPIC4 | 320 | 0.869 | 0.82 | 0.97 | 0.53 | 0.867 [0.822] | 0.76 Health (68) | 0.71 newsgroups_1993/Health (34) |
| TOX2B | 480 | 0.790 | 0.69 | 0.89 | 0.78 | 0.844 [0.804] | 0.77 Toxic (166) | 0.47 toxic_chat/Toxic (43) |

### Qwen2.5-7B-Instruct/t3 r985 ratio=9.886% body_bpw=1.51

| task | n | base acc | worst class base acc | margin>=6 | margin>=10 | retention [Wilson lo] | worst kind ret (n) | worst src x kind (n) |
|---|---|---|---|---|---|---|---|---|
| ADE | 160 | 0.825 | 0.72 | 0.90 | 0.80 | 0.811 [0.735] | 0.71 Yes (58) | 0.71 ade_corpus/Yes (58) |
| ANSWERABLE | 160 | 0.644 | 0.49 | 0.76 | 0.50 | 0.728 [0.635] | 0.46 Answerable (39) | 0.46 squad2/Answerable (39) |
| BOOLQ | 160 | 0.812 | 0.70 | 0.95 | 0.87 | 0.777 [0.698] | 0.77 No (74) | 0.77 boolq/No (74) |
| CODELANG | 312 | 1.000 | 1.00 | 0.98 | 0.69 | 0.362 [0.311] | 0.00 Go (52) | 0.00 codesearchnet/Go (26) |
| CODEPROSE | 320 | 0.997 | 0.99 | 0.86 | 0.53 | 0.997 [0.982] | 0.99 Prose (159) | 0.99 csn_vs_agnews/Prose (80) |
| DEFECT | 160 | 0.494 | 0.25 | 0.35 | 0.13 | 0.253 [0.170] | 0.00 Vulnerable (59) | 0.00 devign/Vulnerable (59) |
| DEPT3 | 318 | 0.619 | 0.52 | 0.84 | 0.34 | 0.777 [0.714] | 0.41 Sales (56) | 0.10 bitext/Technical (10) |
| ENTAIL2 | 320 | 0.897 | 0.87 | 0.93 | 0.86 | 0.774 [0.722] | 0.70 Yes (148) | 0.68 scitail/Yes (73) |
| FAKENEWS | 160 | 0.606 | 0.24 | 0.85 | 0.77 | 0.753 [0.658] | 0.71 Real (78) | 0.71 gonzaloa/Real (78) |
| FINSENT | 159 | 0.811 | 0.68 | 0.63 | 0.00 | 0.659 [0.574] | 0.47 Bearish (43) | 0.47 twitterfin/Bearish (43) |
| HATE2 | 320 | 0.828 | 0.79 | 0.55 | 0.31 | 0.823 [0.772] | 0.75 Fine (126) | 0.68 tweet_eval_hate/Fine (59) |
| INJ2 | 320 | 0.856 | 0.78 | 0.31 | 0.00 | 0.730 [0.674] | 0.55 Benign (150) | 0.53 deepset/Benign (79) |
| LANG20 | 400 | 0.925 | 0.35 | 0.85 | 0.46 | 0.538 [0.487] | 0.00 Bulgarian (19) | 0.00 papluca/Bulgarian (19) |
| NLI3 | 477 | 0.839 | 0.73 | 0.86 | 0.14 | 0.613 [0.564] | 0.16 Neutral (116) | 0.12 mnli/Neutral (33) |
| OFFEN2 | 320 | 0.762 | 0.67 | 0.32 | 0.06 | 0.828 [0.776] | 0.68 Clean (107) | 0.58 davidson/Clean (53) |
| PARA2 | 320 | 0.756 | 0.54 | 0.82 | 0.43 | 0.822 [0.769] | 0.57 Yes (87) | 0.48 qqp/Yes (44) |
| PHISH2 | 320 | 0.900 | 0.88 | 0.92 | 0.77 | 0.500 [0.443] | 0.02 Legitimate (147) | 0.01 sevenphish/Legitimate (76) |
| PIIFLAG | 319 | 0.674 | 0.40 | 0.87 | 0.77 | 0.767 [0.707] | 0.22 Yes (64) | 0.12 ai4privacy/Yes (52) |
| PSAFE | 480 | 0.773 | 0.72 | 0.44 | 0.33 | 0.819 [0.777] | 0.74 Unsafe (198) | 0.66 toxic_chat/Unsafe (74) |
| RCT5 | 320 | 0.613 | 0.09 | 0.86 | 0.24 | 0.556 [0.486] | 0.22 Objective (45) | 0.18 pubmed_rct/Objective (22) |
| SARC2 | 320 | 0.537 | 0.10 | 0.24 | 0.01 | 0.907 [0.854] | 0.00 Sincere (16) | 0.00 headlines/Sincere (10) |
| TICKETPRIO | 159 | 0.358 | 0.00 | 0.65 | 0.46 | 0.825 [0.706] | 0.40 Medium (15) | 0.40 tobi_tickets/Medium (15) |
| TICKETTYPE | 160 | 0.550 | 0.07 | 0.73 | 0.42 | 0.784 [0.687] | 0.38 Incident (24) | 0.38 tobi_tickets/Incident (24) |
| TOPIC5 | 320 | 0.759 | 0.48 | 0.81 | 0.30 | 0.811 [0.757] | 0.66 Politics (53) | 0.46 yahoo_answers/Politics (24) |
| TOX2 | 320 | 0.828 | 0.77 | 0.73 | 0.14 | 0.555 [0.495] | 0.04 Toxic (123) | 0.02 toxic_chat/Toxic (64) |
| UNFAIR | 160 | 0.606 | 0.47 | 0.67 | 0.37 | 0.608 [0.509] | 0.00 Fair (38) | 0.00 lexglue_unfairtos/Fair (38) |

### Qwen2.5-7B-Instruct/t3r2 r985 ratio=9.887% body_bpw=1.51

| task | n | base acc | worst class base acc | margin>=6 | margin>=10 | retention [Wilson lo] | worst kind ret (n) | worst src x kind (n) |
|---|---|---|---|---|---|---|---|---|
| LANG12 | 360 | 0.986 | 0.87 | 0.98 | 0.70 | 0.814 [0.770] | 0.17 Hindi (30) | 0.17 papluca/Hindi (30) |
| NEWS5 | 320 | 0.847 | 0.69 | 0.68 | 0.15 | 0.904 [0.863] | 0.84 Politics (58) | 0.80 huffpost/Business (25) |
| PHISH2B | 320 | 0.922 | 0.85 | 0.94 | 0.83 | 0.644 [0.588] | 0.34 Legitimate (159) | 0.26 zefang_phish/Legitimate (80) |
| PII2 | 320 | 0.822 | 0.80 | 0.80 | 0.53 | 0.490 [0.431] | 0.02 PII (128) | 0.00 ai4privacy/PII (62) |
| SENT2B | 480 | 0.900 | 0.88 | 0.91 | 0.11 | 0.884 [0.851] | 0.79 Positive (211) | 0.70 sst2_movie/Positive (63) |
| TOPIC4 | 320 | 0.847 | 0.80 | 0.86 | 0.26 | 0.827 [0.777] | 0.62 Politics (64) | 0.53 newsgroups_1993/Politics (30) |
| TOX2B | 480 | 0.808 | 0.79 | 0.94 | 0.34 | 0.851 [0.812] | 0.77 Toxic (189) | 0.61 toxic_chat/Toxic (64) |

### Qwen2.5-7B-Instruct/t3r3 r985 ratio=9.887% body_bpw=1.51

| task | n | base acc | worst class base acc | margin>=6 | margin>=10 | retention [Wilson lo] | worst kind ret (n) | worst src x kind (n) |
|---|---|---|---|---|---|---|---|---|
| LANG12B | 720 | 0.994 | 0.97 | 0.99 | 0.71 | 0.951 [0.933] | 0.83 Italian (59) | 0.77 papluca/Italian (30) |
| NEWS4C | 320 | 0.875 | 0.80 | 0.68 | 0.18 | 0.693 [0.637] | 0.45 Entertainment (71) | 0.33 bbc_2004/Entertainment (33) |
| PHISH2C | 320 | 0.906 | 0.87 | 0.94 | 0.85 | 0.641 [0.585] | 0.32 Genuine (151) | 0.28 zefang_phish/Genuine (76) |
| PII2C | 320 | 0.816 | 0.76 | 0.92 | 0.76 | 0.885 [0.841] | 0.86 Harmless (139) | 0.80 ai4privacy/Harmless (61) |
| SENT2C | 480 | 0.890 | 0.87 | 0.92 | 0.09 | 0.909 [0.878] | 0.88 Positive (209) | 0.78 tweets/Positive (69) |
| TOX2C | 320 | 0.803 | 0.74 | 0.93 | 0.52 | 0.860 [0.812] | 0.82 Toxic (138) | 0.68 civil_comments/Toxic (60) |
