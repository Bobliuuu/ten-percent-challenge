# Annotation (feedback) schema, shared by L1, L2 and L9b

One file, <= 25 rows, JSON lines. Each row is one mistake the user corrected.

| key | meaning |
|---|---|
| `text` | the raw request. The task template (`skill_task.json`, key `prompt`, placeholder `{t}`) turns it into the full prompt |
| `label` | the correct answer word; may be a NEW word that is not in the task menu (e.g. `Recipe`) |

Alternative to `text`: `query` = the full prompt already (then no task template is applied).
Accepted legacy aliases: `user` = `query`, `answer` = `label`, `gold: [{"name": label}]` = `label` (the harness form `{id, query, tools, gold}` of L1).
Extra keys (`id`, `src`, `kind`, `tier`, ...) are ignored.

```
{"text": "how long do I boil an egg", "label": "Recipe"}
```

## Task file (`skill_task.json`, identical in all three lanes)
`{"prompt": "...Request: {t}", "labels": ["Alarm", ...]}`. `labels` is the menu printed in the prompt; the new label must not be in it.

## Commands (same flags in all three lanes)
```
python answer.py   --artifact DIR --task skill_task.json "request" ...     # L1; L2/L9b also take ARTIFACT_DIR as first positional
python feedback.py --artifact DIR --task skill_task.json --ann ann.jsonl
```
L2 and L9b still accept their old positional forms (`feedback.py DIR ann.jsonl [NEWLABEL|CAL.jsonl]`). `annot.py` (identical copy in `L2/` and `L9b/`) normalises rows to `{"user": prompt, "answer": label}`, which is what the measured experiment code consumes; the measured numbers were produced with files in that internal form, so no result changes.
Calibration (unlabeled or labelled covariance rows) is a separate, shipped, byte-counted file per lane: L1 `feedback_calib.jsonl` (182 unlabeled SKILL7 prompts), L9b `feedback_cal.jsonl` (300 rows, `user`+`answer`), L2 `data/exam/cal.jsonl` (labelled; see RED_TEAM 2.2).
