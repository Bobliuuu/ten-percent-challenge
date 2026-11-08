### Architecture and byte budget (non-expert tensors charged at 6.25 bpw; expert bpw includes scales)
| model | MoE layers | experts | top-k | params B | expert share | fp16 GB | 10% budget GB | expert bpw if keep 100/75/50/40/30% |
|---|---|---|---|---|---|---|---|---|
| deepseek-ai/DeepSeek-V2-Lite-Chat | 26 | 64 | 6 | 15.7 | 91.6% | 31.4 | 3.14 | 1.18 / 1.57 / 2.35 / 2.94 / 3.92 |
| baidu/ERNIE-4.5-21B-A3B-PT | 27 | 64 | 6 | 21.8 | 93.4% | 43.7 | 4.37 | 1.27 / 1.70 / 2.54 / 3.18 / 4.24 |
| moonshotai/Moonlight-16B-A3B-Instruct | 26 | 64 | 6 | 16.0 | 90.2% | 31.9 | 3.19 | 1.09 / 1.46 / 2.19 / 2.74 / 3.65 |
| allenai/OLMoE-1B-7B-0125-Instruct | 16 | 64 | 8 | 6.9 | 93.1% | 13.8 | 1.38 | 1.26 / 1.67 / 2.51 / 3.14 / 4.19 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | 48 | 128 | 8 | 30.5 | 95.0% | 61.1 | 6.11 | 1.35 / 1.80 / 2.71 / 3.38 / 4.51 |
| openai/gpt-oss-20b | 24 | 32 | 4 | 20.9 | 91.4% | 41.8 | 4.18 | 1.16 / 1.55 / 2.32 / 2.90 / 3.87 |

### Routing concentration (dev calibration text)
| model | task | experts needed 90/95/99% (fraction, layer mean) | routing entropy / ln E | mass in own top-50% | mass in own top-30% | other task's mass in top-50% |
|---|---|---|---|---|---|---|
| deepseek-ai/DeepSeek-V2-Lite-Chat | SKILL7 | 0.59 / 0.71 / 0.87 | 0.89 | 0.84 | 0.66 | BANK8 0.68, NEWS 0.65 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | BANK8 | 0.70 / 0.81 / 0.94 | 0.93 | 0.78 | 0.60 | SKILL7 0.75, NEWS 0.61 |
| baidu/ERNIE-4.5-21B-A3B-PT | SKILL7 | 0.56 / 0.67 / 0.83 | 0.88 | 0.87 | 0.69 | BANK8 0.73, NEWS 0.69 |
| baidu/ERNIE-4.5-21B-A3B-PT | BANK8 | 0.63 / 0.74 / 0.90 | 0.90 | 0.83 | 0.65 | SKILL7 0.75, NEWS 0.65 |
| moonshotai/Moonlight-16B-A3B-Instruct | SKILL7 | 0.46 / 0.57 / 0.73 | 0.83 | 0.92 | 0.76 | BANK8 0.81, NEWS 0.74 |
| moonshotai/Moonlight-16B-A3B-Instruct | BANK8 | 0.48 / 0.58 / 0.75 | 0.83 | 0.92 | 0.75 | SKILL7 0.86, NEWS 0.69 |
| allenai/OLMoE-1B-7B-0125-Instruct | SKILL7 | 0.61 / 0.72 / 0.86 | 0.90 | 0.83 | 0.65 | BANK8 0.74, NEWS 0.64 |
| allenai/OLMoE-1B-7B-0125-Instruct | BANK8 | 0.63 / 0.73 / 0.88 | 0.90 | 0.83 | 0.65 | SKILL7 0.76, NEWS 0.62 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | SKILL7 | 0.33 / 0.40 / 0.54 | 0.78 | 0.98 | 0.87 | BANK8 0.88, NEWS 0.81 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | BANK8 | 0.34 / 0.43 / 0.60 | 0.79 | 0.97 | 0.86 | SKILL7 0.94, NEWS 0.83 |
| openai/gpt-oss-20b | SKILL7 | 0.50 / 0.60 / 0.76 | 0.80 | 0.91 | 0.76 | BANK8 0.86, NEWS 0.86 |
| openai/gpt-oss-20b | BANK8 | 0.46 / 0.56 / 0.75 | 0.79 | 0.92 | 0.78 | SKILL7 0.86, NEWS 0.85 |

### Prune + RTN probe (real greedy generation)
| model | task | kept f | bits | est size % fp16 | base acc | retention | lower | worst cell | cell | n |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-ai/DeepSeek-V2-Lite-Chat | SKILL7 | 1 | 3 | 21.3 | 0.652 | 0.829 | 0.791 | 0.11 | Calendar | 445 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | BANK8 | 1 | 3 | 21.3 | 0.950 | 0.982 | 0.962 | 0.88 | cash | 380 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | SKILL7 | 0.75 | 3 | 16.8 | 0.652 | 0.604 | 0.558 | 0.1 | Lists | 445 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | BANK8 | 0.75 | 3 | 16.8 | 0.950 | 0.921 | 0.890 | 0.67 | meta | 380 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | SKILL7 | 0.5 | 3 | 12.3 | 0.652 | 0.445 | 0.399 | 0.18 | Music | 445 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | BANK8 | 0.5 | 3 | 12.3 | 0.950 | 0.242 | 0.202 | 0.0 | incoming_topups | 380 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | SKILL7 | 0.4 | 3 | 10.6 | 0.652 | 0.038 | 0.024 | 0.0 | Transport | 445 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | BANK8 | 0.4 | 3 | 10.6 | 0.950 | 0.000 | 0.000 | 0.0 | account_identity | 380 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | SKILL7 | 0.3 | 3 | 8.6 | 0.652 | 0.000 | 0.000 | 0.0 | Calendar | 445 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | BANK8 | 0.3 | 3 | 8.6 | 0.950 | 0.000 | 0.000 | 0.0 | account_identity | 380 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | SKILL7 | 0.5 | 4 | 15.2 | 0.652 | 0.465 | 0.419 | 0.03 | Calendar | 445 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | BANK8 | 0.5 | 4 | 15.2 | 0.950 | 0.313 | 0.269 | 0.0 | auto_commute | 380 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | SKILL7 | 0.4 | 4 | 12.9 | 0.652 | 0.000 | 0.000 | 0.0 | Calendar | 445 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | BANK8 | 0.4 | 4 | 12.9 | 0.950 | 0.021 | 0.011 | 0.0 | auto_commute | 380 |
| baidu/ERNIE-4.5-21B-A3B-PT | SKILL7 | 1 | 3 | 21.0 | 0.950 | 0.736 | 0.701 | 0.11 | Transport | 648 |
| baidu/ERNIE-4.5-21B-A3B-PT | BANK8 | 1 | 3 | 21.0 | 0.978 | 0.895 | 0.861 | 0.17 | info_and_fx | 391 |
| baidu/ERNIE-4.5-21B-A3B-PT | SKILL7 | 0.75 | 3 | 16.4 | 0.950 | 0.705 | 0.669 | 0.04 | Weather | 648 |
| baidu/ERNIE-4.5-21B-A3B-PT | BANK8 | 0.75 | 3 | 16.4 | 0.978 | 0.928 | 0.898 | 0.33 | info_and_fx | 391 |
| baidu/ERNIE-4.5-21B-A3B-PT | SKILL7 | 0.5 | 3 | 11.8 | 0.950 | 0.637 | 0.600 | 0.05 | Weather | 648 |
| baidu/ERNIE-4.5-21B-A3B-PT | BANK8 | 0.5 | 3 | 11.8 | 0.978 | 0.882 | 0.847 | 0.19 | transfers | 391 |
| baidu/ERNIE-4.5-21B-A3B-PT | SKILL7 | 0.4 | 3 | 10.1 | 0.950 | 0.383 | 0.346 | 0.01 | Weather | 648 |
| baidu/ERNIE-4.5-21B-A3B-PT | BANK8 | 0.4 | 3 | 10.1 | 0.978 | 0.918 | 0.887 | 0.15 | transfers | 391 |
| baidu/ERNIE-4.5-21B-A3B-PT | SKILL7 | 0.3 | 3 | 8.0 | 0.950 | 0.409 | 0.372 | 0.07 | Weather | 648 |
| baidu/ERNIE-4.5-21B-A3B-PT | BANK8 | 0.3 | 3 | 8.0 | 0.978 | 0.877 | 0.841 | 0.0 | transfers | 391 |
| baidu/ERNIE-4.5-21B-A3B-PT | SKILL7 | 0.5 | 4 | 14.7 | 0.950 | 0.761 | 0.726 | 0.16 | Transport | 648 |
| baidu/ERNIE-4.5-21B-A3B-PT | BANK8 | 0.5 | 4 | 14.7 | 0.978 | 0.944 | 0.916 | 0.26 | transfers | 391 |
| baidu/ERNIE-4.5-21B-A3B-PT | SKILL7 | 0.4 | 4 | 12.4 | 0.950 | 0.733 | 0.698 | 0.5 | Transport | 648 |
| baidu/ERNIE-4.5-21B-A3B-PT | BANK8 | 0.4 | 4 | 12.4 | 0.978 | 0.918 | 0.887 | 0.22 | transfers | 391 |
| moonshotai/Moonlight-16B-A3B-Instruct | SKILL7 | 1 | 3 | 21.6 | 0.955 | 0.949 | 0.930 | 0.8 | Alarm | 651 |
| moonshotai/Moonlight-16B-A3B-Instruct | BANK8 | 1 | 3 | 21.6 | 0.958 | 0.851 | 0.812 | 0.44 | transfers | 383 |
| moonshotai/Moonlight-16B-A3B-Instruct | SKILL7 | 0.75 | 3 | 17.2 | 0.955 | 0.942 | 0.921 | 0.77 | Alarm | 651 |
| moonshotai/Moonlight-16B-A3B-Instruct | BANK8 | 0.75 | 3 | 17.2 | 0.958 | 0.830 | 0.789 | 0.32 | transfers | 383 |
| moonshotai/Moonlight-16B-A3B-Instruct | SKILL7 | 0.5 | 3 | 12.7 | 0.955 | 0.917 | 0.893 | 0.74 | Alarm | 651 |
| moonshotai/Moonlight-16B-A3B-Instruct | BANK8 | 0.5 | 3 | 12.7 | 0.958 | 0.736 | 0.690 | 0.17 | meta | 383 |
| moonshotai/Moonlight-16B-A3B-Instruct | SKILL7 | 0.4 | 3 | 11.0 | 0.955 | 0.926 | 0.904 | 0.84 | Calendar | 651 |
| moonshotai/Moonlight-16B-A3B-Instruct | BANK8 | 0.4 | 3 | 11.0 | 0.958 | 0.755 | 0.709 | 0.31 | small_talk | 383 |
| moonshotai/Moonlight-16B-A3B-Instruct | SKILL7 | 0.3 | 3 | 9.1 | 0.955 | 0.879 | 0.851 | 0.74 | Alarm | 651 |
| moonshotai/Moonlight-16B-A3B-Instruct | BANK8 | 0.3 | 3 | 9.1 | 0.958 | 0.567 | 0.517 | 0.04 | meta | 383 |
| moonshotai/Moonlight-16B-A3B-Instruct | SKILL7 | 0.5 | 4 | 15.6 | 0.955 | 0.954 | 0.935 | 0.84 | Alarm | 651 |
| moonshotai/Moonlight-16B-A3B-Instruct | BANK8 | 0.5 | 4 | 15.6 | 0.958 | 0.854 | 0.815 | 0.52 | transfers | 383 |
| moonshotai/Moonlight-16B-A3B-Instruct | SKILL7 | 0.4 | 4 | 13.4 | 0.955 | 0.949 | 0.930 | 0.85 | Alarm | 651 |
| moonshotai/Moonlight-16B-A3B-Instruct | BANK8 | 0.4 | 4 | 13.4 | 0.958 | 0.888 | 0.852 | 0.48 | transfers | 383 |
| allenai/OLMoE-1B-7B-0125-Instruct | SKILL7 | 1 | 3 | 21.0 | 0.534 | 0.648 | 0.598 | 0.0 | Transport | 364 |
| allenai/OLMoE-1B-7B-0125-Instruct | BANK8 | 1 | 3 | 21.0 | 0.497 | 0.955 | 0.916 | 0.88 | home | 199 |
| allenai/OLMoE-1B-7B-0125-Instruct | SKILL7 | 0.75 | 3 | 16.5 | 0.534 | 0.635 | 0.584 | 0.12 | Transport | 364 |
| allenai/OLMoE-1B-7B-0125-Instruct | BANK8 | 0.75 | 3 | 16.5 | 0.497 | 0.854 | 0.799 | 0.72 | card_virtual_wallet | 199 |
| allenai/OLMoE-1B-7B-0125-Instruct | SKILL7 | 0.5 | 3 | 11.9 | 0.534 | 0.365 | 0.318 | 0.0 | Transport | 364 |
| allenai/OLMoE-1B-7B-0125-Instruct | BANK8 | 0.5 | 3 | 11.9 | 0.497 | 0.131 | 0.091 | 0.0 | account_identity | 199 |
| allenai/OLMoE-1B-7B-0125-Instruct | SKILL7 | 0.4 | 3 | 10.1 | 0.534 | 0.280 | 0.237 | 0.0 | Lists | 364 |
| allenai/OLMoE-1B-7B-0125-Instruct | BANK8 | 0.4 | 3 | 10.1 | 0.497 | 0.000 | 0.000 | 0.0 | account_identity | 199 |
| allenai/OLMoE-1B-7B-0125-Instruct | SKILL7 | 0.3 | 3 | 8.1 | 0.534 | 0.000 | 0.000 | 0.0 | Calendar | 364 |
| allenai/OLMoE-1B-7B-0125-Instruct | BANK8 | 0.3 | 3 | 8.1 | 0.497 | 0.005 | 0.001 | 0.0 | card_delivery | 199 |
| allenai/OLMoE-1B-7B-0125-Instruct | SKILL7 | 0.5 | 4 | 14.8 | 0.534 | 0.769 | 0.723 | 0.0 | Transport | 364 |
| allenai/OLMoE-1B-7B-0125-Instruct | BANK8 | 0.5 | 4 | 14.8 | 0.497 | 0.251 | 0.196 | 0.0 | card_virtual_wallet | 199 |
| allenai/OLMoE-1B-7B-0125-Instruct | SKILL7 | 0.4 | 4 | 12.5 | 0.534 | 0.148 | 0.116 | 0.0 | Lists | 364 |
| allenai/OLMoE-1B-7B-0125-Instruct | BANK8 | 0.4 | 4 | 12.5 | 0.497 | 0.005 | 0.001 | 0.0 | account_identity | 199 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | SKILL7 | 1 | 3 | 20.7 | 0.966 | 0.992 | 0.982 | 0.97 | Calendar | 659 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | BANK8 | 1 | 3 | 20.7 | 0.975 | 0.985 | 0.967 | 0.87 | meta | 390 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | SKILL7 | 0.75 | 3 | 16.0 | 0.966 | 0.994 | 0.984 | 0.98 | News | 659 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | BANK8 | 0.75 | 3 | 16.0 | 0.975 | 0.979 | 0.960 | 0.83 | meta | 390 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | SKILL7 | 0.5 | 3 | 11.3 | 0.966 | 0.989 | 0.978 | 0.96 | News | 659 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | BANK8 | 0.5 | 3 | 11.3 | 0.975 | 0.995 | 0.981 | 0.95 | utility | 390 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | SKILL7 | 0.4 | 3 | 9.4 | 0.966 | 0.986 | 0.974 | 0.94 | News | 659 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | BANK8 | 0.4 | 3 | 9.4 | 0.975 | 0.538 | 0.489 | 0.09 | meta | 390 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | SKILL7 | 0.3 | 3 | 7.5 | 0.966 | 0.983 | 0.970 | 0.94 | News | 659 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | BANK8 | 0.3 | 3 | 7.5 | 0.975 | 0.497 | 0.448 | 0.0 | small_talk | 390 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | SKILL7 | 0.5 | 4 | 14.3 | 0.966 | 0.991 | 0.980 | 0.96 | News | 659 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | BANK8 | 0.5 | 4 | 14.3 | 0.975 | 0.997 | 0.986 | 0.96 | info_and_fx | 390 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | SKILL7 | 0.4 | 4 | 11.8 | 0.966 | 0.992 | 0.982 | 0.96 | News | 659 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | BANK8 | 0.4 | 4 | 11.8 | 0.975 | 0.810 | 0.768 | 0.3 | meta | 390 |
| openai/gpt-oss-20b | SKILL7 | 1 | 3 | 21.4 | 0.950 | 0.994 | 0.984 | 0.98 | Music | 648 |
| openai/gpt-oss-20b | BANK8 | 1 | 3 | 21.4 | 0.960 | 0.995 | 0.981 | 0.96 | work | 384 |
| openai/gpt-oss-20b | SKILL7 | 0.75 | 3 | 16.9 | 0.950 | 0.985 | 0.972 | 0.94 | Music | 648 |
| openai/gpt-oss-20b | BANK8 | 0.75 | 3 | 16.9 | 0.960 | 0.979 | 0.959 | 0.91 | info_and_fx | 384 |
| openai/gpt-oss-20b | SKILL7 | 0.5 | 3 | 12.4 | 0.950 | 0.866 | 0.837 | 0.45 | Music | 648 |
| openai/gpt-oss-20b | BANK8 | 0.5 | 3 | 12.4 | 0.960 | 0.956 | 0.930 | 0.78 | work | 384 |
| openai/gpt-oss-20b | SKILL7 | 0.4 | 3 | 10.7 | 0.950 | 0.773 | 0.739 | 0.49 | Music | 648 |
| openai/gpt-oss-20b | BANK8 | 0.4 | 3 | 10.7 | 0.960 | 0.966 | 0.943 | 0.76 | banking | 384 |
| openai/gpt-oss-20b | SKILL7 | 0.3 | 3 | 9.0 | 0.950 | 0.508 | 0.469 | 0.16 | Calendar | 648 |
| openai/gpt-oss-20b | BANK8 | 0.3 | 3 | 9.0 | 0.960 | 0.945 | 0.918 | 0.8 | travel | 384 |
| openai/gpt-oss-20b | SKILL7 | 0.5 | 4 | 15.2 | 0.950 | 0.900 | 0.874 | 0.63 | Music | 648 |
| openai/gpt-oss-20b | BANK8 | 0.5 | 4 | 15.2 | 0.960 | 0.966 | 0.943 | 0.87 | work | 384 |
| openai/gpt-oss-20b | SKILL7 | 0.4 | 4 | 13.0 | 0.950 | 0.853 | 0.824 | 0.46 | Music | 648 |
| openai/gpt-oss-20b | BANK8 | 0.4 | 4 | 13.0 | 0.960 | 0.961 | 0.937 | 0.89 | card_virtual_wallet | 384 |
