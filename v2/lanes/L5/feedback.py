"""L5 feedback command: closed-form diffuse column-sparse down_proj edit (v2/fb_edit method, snapshot in v2/repair/hybrid/edit.py).
usage: python feedback.py CFG.json   -> runs hybrid/run_local.py (teach / wrong / shuffled / neighbours on the NEWS+Entertainment new-class proxy).
Harness wrapper only: the edit code evaluates real vs shuffled vs empty on dev items; it does not yet write an edited model file for the banking10 task."""
import sys, subprocess, os
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "repair", "hybrid")
subprocess.check_call([sys.executable, "run_local.py", os.path.abspath(sys.argv[1])], cwd=H)
