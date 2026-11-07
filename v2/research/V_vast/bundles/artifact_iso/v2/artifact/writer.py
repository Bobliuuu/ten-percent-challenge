"""Artifact assembly: quantize.py writes model.tpq + tensor_manifest.json + config/tokenizer; this adds the self-contained answer.py / feedback.py
(loader.py inlined) and runs the size report.   python writer.py assemble --art DIR"""
import json, os, sys, re
HERE = os.path.dirname(os.path.abspath(__file__))


def assemble(art):
    ld = open(os.path.join(HERE, "loader.py")).read()
    ld = re.sub(r'^""".*?"""\n', "", ld, count=1, flags=re.S)                  # drop loader docstring (README holds the spec)
    for n in ("answer", "feedback"):
        t = open(os.path.join(HERE, f"{n}_template.py")).read().replace("#@@LOADER@@", "# ---- loader (inlined) ----\n" + ld + "\n# ---- end loader ----\n")
        open(os.path.join(art, f"{n}.py"), "w").write(t)


def tree_report(art):
    rows = sorted((os.path.relpath(os.path.join(r, f), art), os.path.getsize(os.path.join(r, f))) for r, _, fs in os.walk(art) for f in fs)
    return dict(files=rows, total=sum(b for _, b in rows))


if __name__ == "__main__":
    if sys.argv[1] == "assemble": assemble(sys.argv[2].replace("--art", "").strip() if len(sys.argv) == 3 else sys.argv[3]); print(json.dumps(tree_report(sys.argv[-1]), indent=1))
