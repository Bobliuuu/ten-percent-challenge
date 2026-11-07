#!/usr/bin/env python
"""SYNTHETIC tiny exam + feedback pools (plumbing tests only; NOT T1 data, never a result).
Tools: get_weather(city), add(a,b), convert(value,to_unit). Kinds: nocall/single/multi. Sources: S1 and 'S2' (a synthetic OOD-style phrasing)."""
import json, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common3 as C

W = {"name": "get_weather", "description": "Get the weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
A = {"name": "add", "description": "Add two numbers.", "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}}
V = {"name": "convert", "description": "Convert a length in km to another unit.", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "to_unit": {"type": "string"}}, "required": ["value", "to_unit"]}}
CITIES = ["Paris", "Oslo", "Lima", "Cairo", "Kyiv", "Hanoi", "Quito", "Accra", "Perth", "Turin", "Seoul", "Dakar", "Minsk", "Tunis", "Split", "Basel"]


def item(i, kind, src, rng):
    c1, c2 = rng.sample(CITIES, 2); a, b = rng.randint(2, 90), rng.randint(2, 90); v = rng.randint(2, 500)
    tools = [W, A, V]; rng.shuffle(tools)
    ph = (lambda s: s) if src == "S1" else (lambda s: "Please, if you can: " + s)
    if kind == "nocall": q, gold = ph(rng.choice(["Tell me a joke.", "Who wrote Hamlet?", "What is the meaning of life?"])), []
    elif kind == "single":
        t = rng.choice(["w", "a", "v"])
        q, gold = {"w": (f"What's the weather in {c1}?", [dict(name="get_weather", arguments=dict(city=c1))]), "a": (f"Add {a} and {b}.", [dict(name="add", arguments=dict(a=a, b=b))]),
                   "v": (f"Convert {v} km to miles.", [dict(name="convert", arguments=dict(value=v, to_unit="miles"))])}[t]; q = ph(q)
    else:
        q, gold = ph(f"What's the weather in {c1} and in {c2}?"), [dict(name="get_weather", arguments=dict(city=c1)), dict(name="get_weather", arguments=dict(city=c2))]
    return dict(id=f"{src}_{kind}_{i}", kind=kind, source=src, query=q, tools=tools, gold=gold)


def make_exam(n_per=8, seed=0):
    rng = random.Random(seed); rows = []; i = 0
    for src in ("S1", "S2"):
        for kind in ("nocall", "single", "multi"):
            for _ in range(n_per): rows.append(item(i, kind, src, rng)); i += 1
    return rows


def make_convert_items(n, tag, seed):
    """'convert' single-call items = what the feedback teaches (convention: argument is `to_unit`)."""
    rng = random.Random(seed); out = []
    for i in range(n):
        v = rng.randint(501, 9000) + i * 10000; tools = [W, A, V]; rng.shuffle(tools)
        out.append(dict(id=f"{tag}_{i}", kind="single", source="S1", query=f"Convert {v} km to miles.", tools=tools, gold=[dict(name="convert", arguments=dict(value=v, to_unit="miles"))]))
    return out


def make_pools(n_ann=6, n_teach=30, n_nb=30, seed=1):
    """annotations: convert items + weather items (so a shuffle really changes labels); teach: other convert items; neighbors: weather/add items."""
    rng = random.Random(seed); ann = make_convert_items(n_ann - 2, "ann", seed)
    for j in range(2):
        it = item(900 + j, "single", "S1", rng); it["query"] = f"What's the weather in Zagreb{j}?"; it["gold"] = [dict(name="get_weather", arguments=dict(city=f"Zagreb{j}"))]; it["id"] = f"annw_{j}"; ann.append(it)
    teach = make_convert_items(n_teach, "teach", seed + 5)
    nb = []
    for i in range(n_nb):
        it = item(1000 + i, "single", "S1", rng)
        while it["gold"][0]["name"] == "convert": it = item(1000 + i, "single", "S1", rng)
        it["query"] += f" (ref {i})"; it["id"] = f"nb_{i}"; nb.append(it)
    return ann, teach, nb


if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    C.dump_jsonl(make_exam(), f"{out}/exam.jsonl"); ann, teach, nb = make_pools()
    for n, r in (("ann", ann), ("teach", teach), ("nb", nb)): C.dump_jsonl(r, f"{out}/{n}.jsonl")
    print("wrote synthetic exam + pools to", out)
