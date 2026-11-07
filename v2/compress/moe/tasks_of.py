import json, subprocess, sys
d = json.loads(subprocess.run(["modal", "app", "list", "--json"], capture_output=True, text=True).stdout)
print(sum(int(a["tasks"]) for a in d if a["app_id"] == sys.argv[1]))
