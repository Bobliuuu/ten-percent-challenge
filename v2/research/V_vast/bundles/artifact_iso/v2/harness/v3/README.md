# harness/v3 - generative tool-calling proof harness (answer + feedback commands)

Replaces the multiple-choice `exam_remote.py` path for the T1 entry. Pure stdlib + the artifact's own deps; scorer = `tournament/T1/scorer.py`, report = `harness/scorecard.py`.
Status: BUILT and TESTED on a stand-in (artifact = the base). Nothing here has read or scored `tournament/T1/data/locked.jsonl`.

## Artifact contract
Directory with the model files and exactly two entry points (any other `.py` is an "extra file"):
```
python ART/answer.py   --artifact ART --in view.json --out out.json [--dump-prompts p.json]
      view.json = [{id, query, tools}]      out.json = [{id, text}]   (text = the raw generated call(s); same order/ids)
python ART/feedback.py --artifact ART --annotations ann.jsonl
      ann.jsonl rows = {id, query, tools, gold:[{name,arguments}], kind, source}  (<=25).  Edits ART in place. Zero rows => must not touch a byte.
```
`answer` runs with no network and with the label directory hidden; it must not modify ART (tree hash is compared). `--dump-prompts` is optional but, when written, is scanned for leaked gold calls.

## Files
| file | role |
|---|---|
| `exam_gen.py` | single-use generation exam (stage N): claim -> checks -> base (own process) -> candidate in sandbox -> T1 scorer -> `*.exam_arrays.json` + hashed `*.receipt.json` |
| `feedback_protocol.py` | validates + hashes/locks teach/neighbor/annotation sets, then real / shuffled-label / empty feedback on fresh copies; evaluates all; writes the scorecard `feedback` block |
| `final_proof.py` | the one-go driver: stage 1 exam -> feedback protocol -> stage 2 exam-after-feedback -> `scorecard.score()` + `PROOF_SUMMARY.json` |
| `checks.py` | size checker, no-new-parts checker (manifest, tensor headers, config arch keys, train grep, network grep, constrained-decoding grep), leak scans |
| `sandbox.py` + `netguard/` | `netns` = `sudo unshare -mn` (no network at all, label dirs masked by tmpfs); `guard` = python-level fallback |
| `runner.py`, `base_gen.py`, `gen_common.py` | base/candidate execution, same prompt+greedy generation code for both |
| `make_standin.py`, `standin/` | stand-in artifact (base copy + answer.py + feedback.py = `lm_head`/output-embedding row edit from `feedback_v3/fb3.py`, mode `mean`) |
| `synth.py`, `make_dev_smoke.py` | synthetic tiny exam / pools; dev-only smoke inputs from T1 probe + s1_dev |
| `tests/` | pytest (50 tests; 1 needs sudo+unshare and is skipped on a Mac) |

## Commands
Unit tests (CPU, anywhere): `cd ~/ten-percent/v2/harness/v3 && python3 -m pytest -q`

Stand-in artifact + dev smoke (GPU box with torch+transformers, e.g. `ssh tenpct-g2`, repo copy at `~/work/tp`):
```
PY=~/work/venv/bin/python; cd ~/work/tp/v2/harness/v3
$PY make_standin.py --base Qwen/Qwen3-0.6B --out ~/work/tp/smoke/art          # also writes art.base_arch.json (names/shapes/dtypes + base fp16 bytes)
$PY make_dev_smoke.py --base Qwen/Qwen3-0.6B --out ~/work/tp/smoke/data       # exam.jsonl ann.jsonl teach.jsonl nb.jsonl
$PY final_proof.py --exam smoke/data/exam.jsonl --artifact smoke/art --base-id Qwen/Qwen3-0.6B --base-arch smoke/art.base_arch.json \
    --annotations smoke/data/ann.jsonl --teach smoke/data/teach.jsonl --neighbor smoke/data/nb.jsonl --tag smoke --out-dir smoke/run \
    --source-map S3=S2 --budget-fraction 1.05 --python $PY
```
(`--budget-fraction 1.05` only because the stand-in IS the base; the real run keeps the 0.10 default.)
Individual stages: `exam_gen.py` (stage 1 only), `feedback_protocol.py` (needs `--exam` + `--exam-base-texts` to also get exam-after).
Rules of the road: the artifact dir, `work` dirs and `--out-dir` must NOT live inside the directory holding the exam/teach/neighbor files (that directory is masked in the sandbox). Artifact is never edited in place by the harness; feedback runs on copies.

## What each check proves (and what it does not)
* labels never visible: view = whitelist `{id,query,tools}`; `leak_scan` (extra keys, label keys nested in tools, gold call serialised in the query/tools, canary), per-run random CANARY on the label side only, candidate outputs + dumped prompts scanned for it; in `netns` mode the exam directory is masked, so the candidate cannot open the label file.
* no network: `netns` has no interfaces (probe measured before the run: `evidence.network`); `guard` fallback logs attempts and any attempt fails `all_checks_ok`. Static grep of artifact code for sockets/urllib/requests/hub downloads/URLs.
* base separate: own process, receives only the view, outputs hashed; same `gen_common.render/generate` as the stand-in answer.
* single use: O_EXCL claim `receipts/<tag>/stage<N>.claim` before labels are read; released only if the run dies before scoring. Final driver claims stage 1 and stage 2.
* locked guard: exam named `locked.jsonl` or whose sha256 equals the T1 LOCK.json locked hash is refused unless `--allow-locked` AND env `TENPCT_FINAL=1`.
* size: every byte under ART (hidden files, subdirs, code, tokenizer), symlinks rejected, measured before and after feedback; `bytes_added` feeds scorecard.
* no new parts: allowed file set (weights, config, tokenizer, `answer.py`, `feedback.py`, optional `tensor_manifest.json`; extras only via `allowed_extra`), safetensors header names/shapes/dtypes == base (extra head/adapter tensors, missing, reshaped all flagged), architecture keys of config.json, grep for backward/optim/.grad/loss/Adam/step in artifact + build dirs, grep for constrained decoding (logits processors, grammar, allowed tokens) -> `kinds_real_output`.
* feedback: sets validated (>=25 teach, <=25 annotations, ids AND (query,tools) content disjoint across annotations/teach/neighbor/exam), locked + hashed read-only BEFORE feedback and refuses to proceed if a file changed; shuffle = seeded derangement of the corrected labels over the same prompts (impossible if all golds equal -> run refuses); empty = feedback command with zero rows, must leave the tree sha256 identical to the original.

## Known limits
* The sandbox is a Linux `sudo unshare -mn` namespace, not a hardened jail; same OS user, GPU shared. Modal `Sandbox(block_network=True)` backend NOT written here (executor is pluggable in `sandbox.run`; the exam_remote.py volume/claim code could be reused).
* Compressed artifacts (rANS etc.) with no plain safetensors must ship `tensor_manifest.json` ({name: [dtype, shape]} of the logical tensors); then the tensor check is `declared_manifest`, i.e. a claim that still needs a decode-and-compare step (not built: depends on the artifact format).
* `agree` = identical generated text to the base.
* Exam arrays use T1 `kind` (nocall/single/multi) and `source`; scorecard needs the OOD block labelled `S2` (real locked S2 rows already carry it; for dev use `--source-map S3=S2`).

## Test results (2026-11-06)
* `pytest -q`: 50 passed on tenpct-g2 (incl. OS-level netns + label-dir masking test), 49 passed + 1 skipped on the Mac. Negative tests: extra file/subdir, changed shape, extra head tensor, missing tensor, dtype, config arch change, oversize, symlink, size growth after feedback, backward/optim grep (artifact + build dir), network import/hub download/live socket attempt, constrained decoding, label in view / nested label key / gold call in query / canary / gold in dumped prompt, locked file by name and by renamed-copy hash, claim reuse, byte-modifying "empty" feedback, artifact edited during `answer`, candidate crash / wrong ids.
* Dev smoke on tenpct-g2 (Qwen3-0.6B stand-in, 120 dev items from probe_items, S3 mapped to S2; pools from s1_dev; isolation = netns): `smoke_run/`. Stage 1 base 0.758 = cand 0.758, agree 1.0, all checks ok, network blocked, label dir masked. Feedback: real edited 30 lm_head rows (only `model.safetensors` changed, 0 bytes added, no-new-parts ok), shuffled edited 106 rows, empty byte-identical = True. Teach accuracy base/real/shuffled/empty = 0/0/0/0: the alpha=0.1 stand-in edit does not move any teach output (0 of 30 differ), so the scorecard FAILs teach-gain/controls (expected; this is plumbing, not a result). Exam gates fail by construction (120 items, stand-in size).
