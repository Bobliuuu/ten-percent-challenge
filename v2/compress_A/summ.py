import json,glob,sys
for f in sorted(glob.glob(sys.argv[1])):
    r=json.load(open(f)); print(r["cfg"]["name"],"base",round(r["base_acc"],3),"/b",round(r["base_acc_bias"],3),"acc",round(r["acc"],3),"/b",round(r["acc_bias"],3),"agree",round(r["agree"],3),"ratio",round(r["ratio_bf16"],4),"bpw",round(r["body_bpw"],2))
