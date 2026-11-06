import json,glob,sys
for f in sorted(glob.glob("res_*.json")):
    rs=json.load(open(f)); tot=sum(r["N"] for r in rs); w=lambda g:sum(r["N"]*g(r) for r in rs)/tot
    print(f, "N",tot,"fixed",w(lambda r:r["fixed"]),"H0",round(w(lambda r:r["H0"]),4))
    for k in rs[0]["adapt"]:
        print("   %-26s %.4f"%(k,w(lambda r:r["adapt"].get(k,float('nan')))))
    print("   mix %.4f rc %.4f ok %s | scale exp H0 %.3f ctx %.3f"%(w(lambda r:r["mix"]),w(lambda r:r["rc"]["bits_per_param"]),all(r["rc"]["decode_ok"] for r in rs),w(lambda r:r["scale_exp_H0"]),w(lambda r:r["scale_exp_ctx_left"])))
