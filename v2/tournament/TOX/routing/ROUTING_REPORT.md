# ROUTING_REPORT: ticket routing / intent classification as pipeline candidate

Dev only. Built 2026-11-06 by sub-agent; all code in this directory (build_data.py, probe_app.py, launch.py, baseline_tfidf.py, analyze.py). No locked set touched/created; `reserve` split is untouched for a future lock.

## VERDICT

**Do not use as the main task. At best a weak secondary / S3-style cell, and only if the S4 (real tickets) source is accepted as the headline.** Reasons, in order of severity:

1. **Rule 8 ceiling/trivial:** Bitext (S3) is solved by a TF-IDF+LR trained on a disjoint cluster pool at **1.000** in-source, and Qwen3-32B base is already 0.961 (0.92-0.96 across sizes). It is a template-generated set; a 10%-size model has nothing to gain and 'base well below ceiling' fails. CLINC (S2) in-source TF-IDF = 0.964, MASSIVE (S1) = 0.875 (LLM base 0.85-0.87, i.e. no better than bag-of-words trained on 3k examples).
2. **Rule 1 narrow / artificial:** the only genuinely real *ticket* source is S4 (tasksource/it-support-tickets, 1460 unique texts after dedup, one IT company, 6 usable queues, multilingual, noisy). It is IT-helpdesk-only (narrow), tiny (cells of 25-130 items; locked >=330/cell infeasible), and its labels are a company-internal queue scheme that is partly arbitrary (TF-IDF trained in-source only gets 0.74; base 0.68-0.70). The rest (MASSIVE, CLINC) are voice-assistant utterances, not tickets; mapping them to support queues is my construction.
3. **Rule 7 / menu-in-prompt:** the output is a generated queue name (real label, parsed by exact match, 0-4 of ~2271 unparsed) but the menu is in the prompt, so it is a closed-set choice among 6-12 listed strings, i.e. a multiple-choice stand-in in all but name. A fine-tuned small model would have to learn the menu (fine; that is the 'queue taxonomy' knowledge), but the task is then literally k-way classification.
4. **Feedback story:** new-queue / re-routing rule by first-token edit is mechanically feasible (queue names start with distinct first tokens; the first-token logit margin is a clean handle), but the base already routes the common queues; remaining errors are label-convention disagreements (S4 `support_general` vs `office365`, S2 `smart_home` vs `travel`), which are exactly what a feedback edit could fix. That is a real, positive feature of the task, but only visible on S4/S2.

## Sources (pinned)

| id | source | licence | rev/sha | role |
|---|---|---|---|---|
| S1 | AmazonScience/massive en-US (official tarball amazon-massive-dataset-1.0.tar.gz, sha256 7df623fd2d300a4d235d6ee5bd396c9a28258d3a0ccb29abdb054506eba153f8) | cc-by-4.0 | HF ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740 | voice-assistant scenarios mapped to queues |
| S2 | clinc/clinc_oos `plus` | cc-by-3.0 | 155b9c710419136e17307b80d0a13e68cd46b4ec | banking/cards/travel/home/auto domains mapped; oos and small-talk/meta dropped |
| S3 | bitext/Bitext-customer-support-llm-chatbot-training-dataset (27k) | cdla-sharing-1.0 | 430d1a89bd93bd1fa23c16f29dd53e73f0087443 | synthetic customer-support, 11 categories mapped |
| S4 | tasksource/it-support-tickets (Zenodo 10.5281/zenodo.7648117, real tickets, Florianopolis IT co.) | cc-by-4.0 | e0bbdcb96c84feaa394aaae9c48f8ba4b818451b | **only real ticket set with permissive licence**; native 7-queue scheme (EOL dropped: collapses to 2 unique texts) |

Rejected: HWU64 (DeepPavlov/hwu64, no licence tag on HF; same 64 intents / 18 scenarios as MASSIVE, not independent); Banking77 (narrow); Tobi-Bueck/customer-support-tickets and 10 clones (cc-by-nc-4.0, not permissive); mindweave/help-desk-tickets (nc); Console-AI/IT-helpdesk-synthetic-tickets (mit but synthetic, 500 rows, 11 skewed categories); ameau01/synthetic-it-support-tickets (synthetic, no queue label); tasksource/help-desk-tickets (no queue label); Aleph-Alpha/ticket-classification (broken JSON, tiny); interneuronai companyx routing (label = job-skill, unrelated, unlicensed). Raw-file sha256: data/RAW_SHA256.txt; output sha256: data/OUT_SHA256.txt.

## Taxonomy (taxonomy.json; mapping tables in build_data.py)

S1-S3 share 12 unified queues (each shown with a one-line description in the prompt): billing_finance, account_security, orders_delivery, returns_refunds, complaints_feedback, human_agent, subscriptions, travel_transport, vehicle_services, productivity_comms, smart_home_devices, info_entertainment. S4 uses its native 6-queue menu (active_directory, computer_services, fileservice, office365, software, support_general) with descriptions. Honest caveat: **queue overlap across sources is nearly empty**: only orders_delivery is in S1, S2 and S3; returns_refunds/complaints/human_agent/subscriptions exist only in S3 (Bitext), billing/vehicle only S2/S3. So the 'S1->S3 gap' is not a same-label transfer measurement; it is a gap across disjoint label subsets under the same menu.

## Dedup / groups / splits

Exact dedup on normalised text (64569 -> 60997 rows), then TF-IDF(1-2gram) cosine>=0.7 connected components within source = group. Split by sha256(group)%100: dev <20, train 20-49 (baseline pool), reserve >=50 (untouched, for a future locked set). Probe set `data/probe_dev.jsonl`: up to 100 dev items per (source x queue), 2271 items. Large Bitext clusters make dev fractions uneven.

### Per-cell availability (items / near-dup clusters in `reserve`; need >=330 locked per cell)

| source | queue | total | dev | train | reserve items | reserve clusters |
|---|---|---|---|---|---|---|
| S1 | info_entertainment | 8712 | 1684 | 2616 | 4412 | 3707 |
| S1 | orders_delivery | 355 | 89 | 93 | 173 | 158 (<330) |
| S1 | productivity_comms | 5070 | 1068 | 1503 | 2499 | 2178 |
| S1 | smart_home_devices | 1488 | 323 | 458 | 707 | 497 |
| S1 | travel_transport | 801 | 142 | 252 | 407 | 354 |
| S2 | account_security | 1648 | 317 | 466 | 865 | 651 |
| S2 | billing_finance | 4350 | 875 | 1387 | 2088 | 1598 |
| S2 | info_entertainment | 4347 | 858 | 1294 | 2195 | 1832 |
| S2 | orders_delivery | 300 | 47 | 108 | 145 | 117 (<330) |
| S2 | productivity_comms | 2698 | 525 | 778 | 1395 | 1076 |
| S2 | smart_home_devices | 1198 | 252 | 412 | 534 | 384 |
| S2 | travel_transport | 3000 | 640 | 875 | 1485 | 1281 |
| S2 | vehicle_services | 1500 | 268 | 504 | 728 | 555 |
| S3 | account_security | 5312 | 1919 | 1253 | 2140 | 1107 |
| S3 | billing_finance | 4694 | 436 | 1292 | 2966 | 739 |
| S3 | complaints_feedback | 1989 | 225 | 557 | 1207 | 345 |
| S3 | human_agent | 1944 | 349 | 798 | 797 | 308 (<330) |
| S3 | orders_delivery | 6592 | 1266 | 2210 | 3116 | 1178 |
| S3 | returns_refunds | 2559 | 246 | 507 | 1806 | 427 |
| S3 | subscriptions | 982 | 193 | 285 | 504 | 145 (<330) |
| S4 | active_directory | 125 | 25 | 39 | 61 | 57 (<330) |
| S4 | computer_services | 106 | 26 | 26 | 54 | 50 (<330) |
| S4 | fileservice | 315 | 124 | 58 | 133 | 98 (<330) |
| S4 | office365 | 211 | 51 | 58 | 102 | 99 (<330) |
| S4 | software | 186 | 45 | 53 | 88 | 83 (<330) |
| S4 | support_general | 515 | 88 | 167 | 260 | 188 (<330) |

Feasibility: 16/26 cells have >=330 reserve clusters, 18/26 have >=330 reserve items. All S4 cells fail (<=260 items), as do S1 orders_delivery, S2 orders_delivery, S3 subscriptions/human_agent/complaints/returns (<330 *clusters*: Bitext is ~3-5 near-duplicate paraphrases per cluster). If distinct clusters are required, the usable locked cells are only the large S1/S2 ones (the voice-assistant sources), which are the least ticket-like. A real-ticket locked set is NOT feasible from public data (S4 total 1460 unique).

## Base probe (Qwen3 8B/14B/32B, vLLM 0.10.2, greedy, enable_thinking=False, max 16 tokens; prompt in probe_app.py; per-item outputs in results/probe/*.json, summary results/probe_summary.json)

| model | S1 MASSIVE | S2 CLINC | S3 Bitext | S4 IT tickets |
|---|---|---|---|---|
| Qwen3-8B | 0.845 | 0.766 | 0.919 | 0.681 |
| Qwen3-14B | 0.869 | 0.797 | 0.929 | 0.681 |
| Qwen3-32B | 0.859 | 0.801 | 0.961 | 0.704 |
| TF-IDF+LR in-source (train pool) | 0.875 | 0.964 | 1.000 | 0.740 |

Accuracies are item-level over the probe set (n=489/747/700/335); chance is ~1/k with k=5 (S1 present queues; 12 in menu), 8 (S2), 7 (S3), 6 (S4); majority-class chance S1 0.20, S2 0.14, S3 0.14, S4 0.30. Base is well above chance on all four, and **near ceiling on S3** (0.92-0.96, and 1.00 for the trivial baseline).
S1->S3 'gap': +0.07 to +0.10 (S3 higher), i.e. the opposite of a hard-transfer gap; S3 is simply easier/templated.

TF-IDF cross-source (train on other sources, test on this one, unified labels): S2+S3->S1 0.624, S1+S2->S3 0.244, S1+S3->S2 0.384: label scheme transfer is not trivial for bag-of-words, but the LLM with the menu does it at 0.80-0.96.

### Per (source x queue) cell: base accuracy [acc, n, mean first-token margin gold-vs-best-other (logit)]

| cell | 8B | 14B | 32B | TF-IDF in-source |
|---|---|---|---|---|
| S1/info_entertainment | 0.84 | 0.79 | 0.81 (n=100) | 0.94 |
| S1/orders_delivery | 0.854 | 0.933 | 0.876 (n=89) | 0.73 |
| S1/productivity_comms | 0.85 | 0.83 | 0.83 (n=100) | 0.93 |
| S1/smart_home_devices | 0.78 | 0.81 | 0.81 (n=100) | 0.9 |
| S1/travel_transport | 0.9 | 0.99 | 0.97 (n=100) | 0.86 |
| S2/account_security | 0.81 | 0.76 | 0.8 (n=100) | 0.91 |
| S2/billing_finance | 0.73 | 0.91 | 0.89 (n=100) | 1.0 |
| S2/info_entertainment | 0.84 | 0.83 | 0.86 (n=100) | 1.0 |
| S2/orders_delivery | 0.851 | 0.957 | 0.957 (n=47) | 0.89 |
| S2/productivity_comms | 0.82 | 0.86 | 0.89 (n=100) | 0.98 |
| S2/smart_home_devices | 0.59 | 0.51 | 0.45 (n=100) | 0.97 |
| S2/travel_transport | 0.58 | 0.63 | 0.67 (n=100) | 0.93 |
| S2/vehicle_services | 0.95 | 1.0 | 0.97 (n=100) | 0.99 |
| S3/account_security | 0.89 | 0.93 | 0.99 (n=100) | 1.0 |
| S3/billing_finance | 0.94 | 0.98 | 0.97 (n=100) | 1.0 |
| S3/complaints_feedback | 0.88 | 0.91 | 0.99 (n=100) | 1.0 |
| S3/human_agent | 0.89 | 0.99 | 0.96 (n=100) | 1.0 |
| S3/orders_delivery | 0.95 | 0.96 | 0.96 (n=100) | 1.0 |
| S3/returns_refunds | 0.88 | 0.73 | 0.86 (n=100) | 1.0 |
| S3/subscriptions | 1.0 | 1.0 | 1.0 (n=100) | 1.0 |
| S4/active_directory | 0.56 | 0.52 | 0.68 (n=25) | 0.28 |
| S4/computer_services | 0.808 | 0.923 | 0.923 (n=26) | 0.81 |
| S4/fileservice | 0.96 | 0.96 | 0.96 (n=100) | 0.94 |
| S4/office365 | 0.412 | 0.647 | 0.686 (n=51) | 0.57 |
| S4/software | 0.467 | 0.444 | 0.444 (n=45) | 0.56 |
| S4/support_general | 0.625 | 0.477 | 0.5 (n=88) | 0.82 |

Cells in the useful rule-8 band (base 0.4-0.9, not at TF-IDF ceiling): S2/smart_home_devices (0.45-0.59), S2/travel_transport (0.58-0.67), S2/account_security (0.76-0.81), S3/returns_refunds (0.73-0.88), S4/software (0.44-0.47), S4/support_general (0.48-0.63), S4/office365 (0.41-0.69), S4/active_directory (0.52-0.68). The S2 low cells are largely **my own mapping ambiguity** (CLINC `smart_home`/`find_phone`/`change_volume` vs `travel` items like `translate`/`timezone` that a reasonable annotator would put in info) and should be read as label-noise, not model weakness. The S4 low cells are annotator-convention noise of one company (e.g. 'Leaving employee' tickets labelled active_directory, predicted support_general); their ceiling is unknown and the in-source TF-IDF only reaches 0.28-0.94 per cell (0.74 overall).

Most frequent base errors (32B/14B/8B): S2 smart_home->account_security/info/productivity; S2 travel->info_entertainment; S3 returns_refunds->billing_finance/complaints; S4 software<->support_general; S4 support_general->office365. Unparsed outputs: 8B 0, 14B 1, 32B 4 of 2271.

First-token confidence: restricting to items where top-1 first-token logprob > -0.05 gives 32B accuracy S1 0.91, S2 0.88, S3 0.98, S4 0.74 on 86-94% of items: the model is confidently right when it is right; confident errors exist mainly in S4 and S2, consistent with label-convention disagreement.

## Adversarial assessment

* **Commercial class (rule 1):** ticket routing is a genuine commercial task. But the data that can be locked at scale is not ticket data: MASSIVE/CLINC are assistant utterances, Bitext is synthetic LLM-generated paraphrases of ~27 intents (its own README), and the one real ticket set is a single IT shop. A judge can fairly call this 'narrow' (IT helpdesk) or 'toy' (Bitext, solvable by keywords at 100%). Multi-domain breadth exists only in S1/S2, via an artificial mapping.
* **Rule 7:** the model emits a queue name, scored by exact match; no letter stand-in. But with a closed menu in the prompt it is classification with verbalised labels. Without a menu (queue learned from data) the task would be feedback-specific, and the pipeline would need to teach the taxonomy; base could then not be 'well above chance'.
* **Rule 8:** S3 fails (ceiling), S1 is near the in-source bag-of-words level (0.85-0.87 vs 0.875), S2 partly label noise, S4 is in-band but tiny and noisy. No cell family is simultaneously large (>=330 distinct), clean, in-band and real tickets.
* **Trivial baseline:** a char+word TF-IDF LR trained on a disjoint pool of 3-18k items matches or beats Qwen3-32B base on S1/S2/S3 and is only 0.74 vs 0.70 on S4. So the task is not 'hard for a model'; the headroom for a compressed model is whether it retains the menu-following skill, which a 14B->10% model most likely does on S3 (see Bitext 1.00) and loses little on others. That is a weak test of compression quality (little discrimination between a good and a bad compression).
* **Feedback by first-token edit:** feasible (distinct first tokens: billing/account/orders/returns/complaints/human/subscriptions/travel/vehicle/productivity/smart/info for S1-S3 and active/computer/file/office/software/support for S4; margins are well calibrated). Natural feedback stories: add queue `returns_refunds` split from `billing_finance` or re-route 'leaving employee' tickets to active_directory; the base fails exactly those convention cases (S4 active_directory 0.52-0.68, S3 returns_refunds 0.73-0.88 for 14B/32B/8B), so edits would show measurable gains. This is the best argument for keeping S4/S3-returns as a secondary kind.

## Caveats

* Taxonomy mappings for S1/S2 are my construction (build_data.py dicts), documented but not independently validated; no human re-annotation was done.
* EOL queue of S4 dropped (2 unique texts post-dedup).
* S4 license: CC BY 4.0 as stated in the HF card; HWU64 licence not stated on HF (rejected).
* Probe set is dev only, <=100 per cell; per-cell 95% CI is about +/-0.09 at n=100.
* Modal GPU queueing (workspace cap shared with sibling jobs) delayed runs ~2.5 h; app tenpct-tox-routing-probe (deployed), results volume tenpct-tox-routing, run ids in results/RUNS.txt.
* Dev text is not in the repo's git (data/ not committed, not instructed to commit); regenerate with `python3 -I build_data.py` after fetching raw files (MASSIVE tarball URL https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz; others via huggingface_hub at the pinned revisions).
* No git commit/push performed per instructions.
