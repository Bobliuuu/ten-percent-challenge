import modal, json, sys
f = modal.Function.from_name("tenpct-l12-b6", "run"); tag = sys.argv[1]; args = sys.argv[2:]; files = {}
for flag in ("--grid", "--cfgs", "--cfg"):
    if flag in args:
        p = args[args.index(flag) + 1]
        if p.endswith(".json"): files[p.split("/")[-1]] = json.load(open(p)); args[args.index(flag) + 1] = p.split("/")[-1]
c = f.spawn(tag, args, files); print(tag, c.object_id)
