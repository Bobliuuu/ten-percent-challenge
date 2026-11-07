import subprocess,re
t=open("report_template.md").read()
tab=subprocess.run(["python3","mktable.py"],capture_output=True,text=True).stdout
fit=subprocess.run(["python3","mkfit.py"],capture_output=True,text=True).stdout
ct=subprocess.run(["python3","mkct.py"],capture_output=True,text=True).stdout
real=subprocess.run(["python3","mkreal.py"],capture_output=True,text=True).stdout
t=t.replace("{REALTABLE}",real).replace("{TABLE}",tab).replace("{FIT}",fit).replace("{CTTABLE}",ct)
t=t.replace("{SELF}","85.8 (+2.0), S1/nocall 57.3, S3/multi 84.1").replace("{R12R15}","91.1% @12.0%, 93.6% @15.0% (vs plain-legacy 89.4 / 94.8): gamma gain vanishes with budget")
open("REPORT.md","w").write(t)
print(len(t))
