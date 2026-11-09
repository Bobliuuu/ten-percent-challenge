"""spawn L2 stages on the deployed Modal app tenpct-l2 (detached). usage: python3 launch_l2.py setup | sel_o2 | final_m16 | cli_recipe | cli_reminder | cli_recipe_m16"""
import modal, sys
s = sys.argv[1]; f = lambda n: modal.Function.from_name("tenpct-l2", n)
c = {"setup": lambda: f("setup").spawn(), "sel_o2": lambda: f("run_jobs").spawn("jobs_sel_o2.json", "sel_o2"),
     "final_m16": lambda: f("run_jobs").spawn("jobs_final_m16.json", "final_m16"),
     "cli_recipe": lambda: f("cli").spawn("Recipe", "data/ann25.jsonl", "data/exam/teach_Recipe.jsonl", "recipe"),
     "smoke": lambda: f("smoke").spawn(), "cli_reminder": lambda: f("cli").spawn("Reminder", "", "", "reminder")}[s]()
print(s, c.object_id)
