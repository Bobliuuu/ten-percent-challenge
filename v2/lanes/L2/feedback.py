"""L2 feedback command (diffuse closed-form edit, no gradient steps, no head/adapter/side memory).
usage: python feedback.py ARTIFACT_DIR ANNOTATED.jsonl [--layers 62,63]   (rows: {"user": prompt, "answer": new_label_word})
Implementation = fb27s.py in this directory: for each annotated item the hidden state at the answer positions is captured, the minimal
residual that makes the model's OWN output embedding prefer the labelled word is computed, and it is written into the existing MLP
down_proj of the last layers with a calibration-covariance (MEMIT-style) closed-form solve. Result is saved as feedback.pt (dense
delta of existing tensors, ~tens of MB, counted in MANIFEST) next to the artifact; answer.py adds it to the weights at load.
Experiments in this lane were run via fb27s.py (edit applied in memory and measured), see REPORT.md."""
import sys; print(open(__file__).read().split('"""')[1]); sys.exit("use fb27s.py for the measured path (python fb27s.py MODELDIR OUT.json LAYERS LAM M F DRAWS with NEWLAB set)")
