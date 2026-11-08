# RIVAL_ISSUES: public entries #3 and #4 (read 2026-11-08, read-only; no entrant handles, no links to their repos)

## Entrant #3 (issue #3): Qwen3-30B-A3B-Instruct-2507 at 10.48% on disk
- Task: banking support router (Banking77 grouped into 10 queues), BM25 top-8 retrieved examples in the prompt (retrieval memory counted as benchmark data, not size), first 1,000 rows of a pre-locked 2,680-item exam.
- Compression: keep 96 of 128 experts per layer by router usage on 120 dev examples (covers 99.8% of routing), then llama.cpp dynamic IQ1_S/IQ2_XXS/IQ3/Q5/Q6 mix (~2.2 bpw experts) with a task imatrix; 6,402,132,448 B = 10.4839%.
- Numbers: base 95.80%, compressed 96.00% (raw accuracy, NOT retention of base-correct); dev 97.33 vs 98.00. Feedback: 10 new intents x 5 examples, ten existing lm_head rows overwritten with class-mean hidden states; new-intent test 36.25% -> 88.50%.
- Maintainer (2026-11-07): compression side "genuinely nice"; step 5 REJECTED: the 88.5% was a numpy nearest-centroid classifier on feedback features, the grown model never ran on new-intent items, the answer command only lists the 10 old families, and the ten overwritten rows are "the same head as before, just kept inside the model file". Invited to retry before the deadline. Note: 10.48% > 10%, so it did not pass the size rule either; no accuracy-retention metric per kind.

## Entrant #4 (issue #4): Qwen3-30B-A3B-Instruct-2507 at 9.892%
- Task: voice-assistant skill routing (weather/email/transport/lists/calendar/home/news), free-generated skill name. 2,910-item locked exam, >=330 per label, 600 from CLINC150 (never used to build/tune/calibrate). Base 95.98%.
- Size: 6,040,715,980 B = 9.892% including weights, tokenizer, code, task file and a ~1 MB feedback-statistics file.
- Compression: 2,640 of 6,144 routed experts kept (43%), chosen by routing mass on unlabeled calibration text, router sliced; RTN: experts 3-bit asym, attention 6, embeddings 5, output 4, fp16 group scales.
- Retention: 99.36% (2,775/2,793) of base-correct; lowest kind 98.56% (weather); CLINC150 99.12%, MASSIVE 99.42%.
- Feedback: 24 annotated mistakes, two new skills (recipe, music). Closed-form edit of existing lm_head rows (mean hidden state of mistaken items, whitened by stored calibration statistics, median per-item flip step). Locked teach set (390): 18.97% -> 64.87%; recipe 0% -> 61.1% (CLINC items 54.7%); shuffled 29.23%, empty 18.97%; exam unchanged (1 of 2,910 answers changed).
- Own disclosures: dev gate (99%/98%) lowered before opening the exam; on `music` shuffled (88.3%) beat real (73.3%) because the base already partly knew the word.
- Maintainer: NO comment yet as of 2026-11-08 (0 comments, issue open). Its feedback mechanism is the same family (lm_head row edit + stored statistics) that the maintainer rejected for #3 and #2, so expect rejection; the compression side is the strongest published number.

## What this means for us
1. Compression recipe (task-pruned experts + 3-bit RTN, 9.89%) is accepted-looking and measured at 99.36%; our L3 lane targets beating it with GPTQ+Hadamard and better expert scoring.
2. Feedback must (a) not touch lm_head rows, (b) not ship a statistics file (or count it), (c) use teach labels the base does not already emit, (d) show real >> shuffled with shuffled labels that actually permute mixed labels.
