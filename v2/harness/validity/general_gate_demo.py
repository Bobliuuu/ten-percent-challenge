"""Feed general_remote.py output into scorecard.score() (on top of a synthetic base input) and print the general_* gates/details.  usage: python general_gate_demo.py q3_8b_general_result.json"""
import json, sys, os
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H + "/.."); import scorecard as SC
g = json.load(open(sys.argv[1])); R = json.load(open(H + "/synthetic_good_input.json")); R["general"] = {k: g[k] for k in ("ppl", "kl") if k in g}; R["general"]["benchmarks"] = {k: v for k, v in g["benchmarks"].items() if "indomain" not in k}
o = SC.score(R); out = dict(gates={k: v for k, v in o["gates"].items() if k.startswith("general")}, scores={k: v for k, v in o["scores"].items() if k.startswith("general")},
    details={k: v for k, v in o["details"].items() if k.startswith("general")}); print(json.dumps(out, indent=1))
