import modal, json, sys
f = modal.Function.from_name("tenpct-l12b", "run"); tag, script = sys.argv[1], sys.argv[2]; args = sys.argv[3:]; files = {}
if "--grid" in args: gp = args[args.index("--grid") + 1]; files[gp.split("/")[-1]] = json.load(open(gp)); args[args.index("--grid") + 1] = gp.split("/")[-1]
if "--cfg" not in args: args += ["--cfg", "best.json"]
c = f.spawn(tag, script, args, files); print(tag, c.object_id)
