# v2/exam: the shared locked exam (SKILL7, BANK6)

Read `EXAM_PLAN.md` first (counts, policy, caveats). Everything is sealed in `MANIFEST.json` (sha256 per file). Quick use:

```python
import sys; sys.path.insert(0, "v2/exam")
from loader import load, report, verify, new_label, prompt
verify()                                   # fails if a sealed file changed
exam = load("SKILL7", "exam")              # 2744 items; load("BANK6","exam") 2237
teach = load("SKILL7", "teach")            # >=25 unseen items of the NEW label (Recipe / Security), lock BEFORE any feedback
ann = load("SKILL7", "ann_pool")           # feedback candidates (<=25 annotated); tuned-on sources only
near = set(load("SKILL7", "neighbours"))   # exam ids near the new label (rule 5 nearby items)
dev, cal, sel = load("SKILL7","dev"), load("SKILL7","cal_big"), load("SKILL7","sel")
# each item: id, task, role, src, kind, user (full prompt, identical to the lanes' current one), answer, tier
# ... generate greedily, exact label match -> base_ok / quant_ok = {id: bool}
print(report(exam, base_ok, quant_ok))     # overall / per kind / per source retention + Wilson, never-used-source retention, rule 3 counts
```
Rules for lanes: the exam is scored once, at the end, never used to build/tune/choose. HWU64, MTOP and Bitext rows are sealed: no dev/calibration/selection/feedback use. `clean_only=True` drops the `seen_*` tier rows (sensitivity row). Rebuild (deterministic, needs network for public datasets): `python3 -I v2/exam/build_exam.py`; do not rebuild after the first model run.
Licences: MASSIVE CC-BY-4.0, CLINC150 CC-BY-3.0, HWU64 CC-BY-4.0, MTOP CC-BY-SA-4.0, Banking77 CC-BY-4.0, Bitext retail-banking CDLA-Sharing-1.0.
