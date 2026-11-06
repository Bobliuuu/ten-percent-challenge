"""feedback_v3: in-place closed-form edits (no backward, no optimizer) on a small proxy.
modal run --detach fb3.py --model Qwen/Qwen3-1.7B --task A
Tasks: A new-label tickets, B confusable tool pair, C new PII entity. Methods: memit_down, memit_o, lmhead_mean, lmhead_lda, + W4 snap.
Results -> /vol/fb3/<tag>.json on volume tenpct-v2-hf."""
import modal, os, json
app = modal.App("tenpct-fb3")
img = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1","transformers==4.56.2","accelerate","numpy","datasets")
vol = modal.Volume.from_name("tenpct-v2-hf")
GPU = os.environ.get("FB3_GPU","A10G")

@app.function(image=img, gpu=GPU, timeout=36000, volumes={"/vol":vol})
def run(model="Qwen/Qwen3-1.7B", task="A", tag="", sweep=""):
    os.environ["FB3_SWEEP"]=sweep
    import torch, numpy as np, random, hashlib, re, time, copy
    os.environ["HF_HOME"]="/vol/hf"; os.makedirs("/vol/fb3",exist_ok=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tag = tag or f"{model.split('/')[-1]}_{task}"
    T0=time.time(); log=lambda *a: print(f"[{time.time()-T0:6.0f}s]",*a,flush=True)
    tok=AutoTokenizer.from_pretrained(model); tok.padding_side="left"
    lm=AutoModelForCausalLM.from_pretrained(model,torch_dtype=torch.bfloat16).cuda().eval()
    for p in lm.parameters(): p.requires_grad_(False)
    dev="cuda"; NL=lm.config.num_hidden_layers
    def chat(u):
        try: return tok.apply_chat_template([{"role":"user","content":u}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        except TypeError: return tok.apply_chat_template([{"role":"user","content":u}],tokenize=False,add_generation_prompt=True)
    @torch.no_grad()
    def gen(prompts,mx,bs=48):
        out=[None]*len(prompts); order=sorted(range(len(prompts)),key=lambda i:len(prompts[i]))
        for s in range(0,len(order),bs):
            idx=order[s:s+bs]; enc=tok([prompts[i] for i in idx],return_tensors="pt",padding=True).to(dev)
            o=lm.generate(**enc,max_new_tokens=mx,do_sample=False,pad_token_id=tok.pad_token_id)
            for j,i in enumerate(idx): out[i]=o[j,enc.input_ids.shape[1]:].tolist()
        return out
    dec=lambda ids: tok.decode(ids,skip_special_tokens=True).strip()
    h8=lambda s:int(hashlib.sha256(s.encode()).hexdigest()[:8],16)
    # ------------------------------------------------------------------ data (LLM-written, cached)
    def llm_lines(instr,n_calls,mx=700):
        res=[]
        outs=gen([chat(instr)]*n_calls,mx) if False else None
        return res
    @torch.no_grad()
    def sample(prompts,mx,temp=0.9,bs=32):
        out=[]
        for s in range(0,len(prompts),bs):
            enc=tok(prompts[s:s+bs],return_tensors="pt",padding=True).to(dev)
            o=lm.generate(**enc,max_new_tokens=mx,do_sample=True,temperature=temp,top_p=0.95,pad_token_id=tok.pad_token_id)
            out+= [tok.decode(r[enc.input_ids.shape[1]:],skip_special_tokens=True) for r in o]
        return out
    def parse_lines(txt):
        r=[]
        for l in txt.split("\n"):
            m=re.match(r"\s*\d+[\.\)]\s*(.+)",l)
            if m:
                s=m.group(1).strip().strip('"').strip()
                if 25<len(s)<400: r.append(s)
        return r
    def collect(desc,count,k=10):
        ps=[chat(f"Write {k} different, realistic messages that {desc}. Vary tone, length (1-3 sentences), names, products and details. Output a numbered list, one message per line, nothing else.")]*(count//k+4)
        seen=[];S=set()
        for t in sample(ps,80*k):
            for s in parse_lines(t):
                if s.lower() not in S: S.add(s.lower()); seen.append(s)
        return seen[:count+20]
    dfile=f"/vol/fb3/data_{model.split('/')[-1]}_{task}.json"
    if os.path.exists(dfile): D=json.load(open(dfile))
    else:
        D={}
        if task=="A":
            OLD={"Billing":"questions about charges, invoices, payment methods, plan prices","Technical Support":"bugs, errors, crashes, the product not working","Account Access":"login problems, password resets, locked accounts, two-factor issues","Shipping":"delivery status, lost or delayed packages, tracking","General Inquiry":"generic questions about hours, features, how things work"}
            NEW={"Chargebacks":"a customer bank dispute or chargeback filed against us, or a customer threatening to dispute a charge with their bank","Fraud Review":"suspected fraudulent orders or stolen cards, or someone reporting unauthorised use of their account to buy things","Legal Hold":"lawyers, subpoenas, court orders, or requests to preserve records for litigation","Press Inquiries":"journalists, bloggers, or media outlets asking for comment or interviews","Partnerships":"companies proposing business partnerships, integrations, reseller or co-marketing deals","Recruiting":"job applicants, recruiters, or people asking about open positions and applications","Security Reports":"researchers reporting security vulnerabilities, data leaks or bugs in our security","Accessibility":"users with disabilities reporting screen reader, contrast, keyboard navigation or captioning problems","Data Deletion":"requests to erase personal data or exercise GDPR/CCPA privacy rights","Wholesale":"businesses asking for bulk pricing, wholesale accounts or large volume orders"}
            D["old"]={k:collect(f"customers send to a support team about {v}",130) for k,v in OLD.items()}
            D["new"]={k:collect(f"are support tickets about {v}",150) for k,v in NEW.items()}
            D["old_names"]=list(OLD)
        elif task=="B":
            I={"get_order_status":"asking where their order is or what its status is","cancel_order":"asking to cancel an order they just placed, before it ships","refund_payment":"asking for their money back because they were charged twice, charged the wrong amount, or returned an item","issue_store_credit":"explicitly asking for the refund as store credit or gift balance on their account instead of money back","update_shipping_address":"asking to change the delivery address of an order","cancel_subscription":"asking to stop their recurring subscription","close_account":"asking to permanently close or delete their account","escalate_to_human":"angry customers demanding to talk to a manager or a human agent"}
            D["intent"]={k:collect(f"a customer sends to an online store's support bot, {v}",130) for k,v in I.items()}
            D["pat"]=collect("a customer sends to an online store's support bot, complaining that their order arrived late or damaged or with a missing item, without asking for a specific kind of refund",260)
        else:
            def tmpl(extra,n):
                return collect(f"an employee or customer writes in a workplace chat or support email, written as a template that contains the literal placeholders {extra} (each written exactly like that, with curly braces). No other numbers, names or contact details",n)
            D["std"]=tmpl("{NAME}, {EMAIL} and {PHONE} (use at least two of them)",260)
            D["emp"]=tmpl("{NAME} and {EMPID} (an employee badge ID), optionally also {EMAIL}",260)
        json.dump(D,open(dfile,"w")); vol.commit()
    log("data ready")
    # ------------------------------------------------------------------ build items: dict(prompt,gold,cat,split)
    split=lambda s: ("ann","dev","teach")[0 if h8(s)%10<4 else (1 if h8(s)%10<6 else 2)]
    MX={"A":10,"B":48,"C":110}[task]
    items={"ann":[],"dev":[],"teach":[],"neigh":[],"neigh_dev":[]}
    if task=="A":
        qs=D["old_names"]
        P=lambda t: chat("You are routing customer support tickets. Pick the one best queue for the ticket from: "+", ".join(qs)+".\nReply with the queue name only.\n\nTicket: "+t)
        for lab,L in D["new"].items():
            for t in L: items[split(t)].append(dict(prompt=P(t),gold=lab,cat=lab))
        for lab,L in D["old"].items():
            for t in L: items["neigh"].append(dict(prompt=P(t),gold=lab,cat=lab))
        sc=lambda it,o: dec(o).lower().strip(" .")==it["gold"].lower()
    elif task=="B":
        TOOLS="""get_order_status(order_id): Look up the current status of an order.
cancel_order(order_id): Cancel an order that has not shipped yet.
refund_payment(order_id, amount): Return money to the customer's original payment method.
issue_store_credit(customer_id, amount): Add credit to the customer's store account balance.
update_shipping_address(order_id, address): Change the delivery address of an order.
cancel_subscription(subscription_id): Stop a recurring subscription.
close_account(customer_id): Permanently close a customer account.
escalate_to_human(reason): Hand the conversation to a human agent."""
        P=lambda t: chat("You are a support bot with these tools:\n"+TOOLS+"\n\nRespond with only a JSON tool call of the form {\"name\": <tool>, \"arguments\": {...}}.\n\nCustomer message: "+t)
        for t in D["pat"]: items[split(t)].append(dict(prompt=P(t),gold="issue_store_credit",cat="pat"))
        for k,L in D["intent"].items():
            for t in L: items["neigh"].append(dict(prompt=P(t),gold=k,cat=k))
        sc=lambda it,o: (re.search(r'"name"\s*:\s*"([a-z_]+)"',dec(o)) or [None,None])[1]==it["gold"]
    else:
        FN=["Maria Lopez","James Carter","Priya Nair","Tom Becker","Aisha Khan","Luca Rossi","Emily Chen","Omar Haddad","Sofia Petrova","Daniel Okafor","Hannah Weiss","Kenji Mori"]
        DOM=["acme.com","northwind.org","example.net","contoso.io","fabrikam.com"]
        def fill(t,seed):
            r=random.Random(seed); n=r.choice(FN)
            v={"{NAME}":n,"{EMAIL}":n.lower().replace(" ",".")+"@"+r.choice(DOM),"{PHONE}":r.choice(["(415) 555-%04d","+1 212 555 %04d","555-%04d"]) % r.randint(0,9999) if True else "","{EMPID}":r.choice(["EMP-%05d","E%06d","emp_%05d"]) % r.randint(0,99999)}
            tg={"{NAME}":"[NAME]","{EMAIL}":"[EMAIL]","{PHONE}":"[PHONE]","{EMPID}":"[EMPLOYEE_ID]"}
            x=t;g=t
            for k in v: x=x.replace(k,v[k]); g=g.replace(k,tg[k])
            if "{" in x or re.search(r"\d",re.sub(r"[()+\-\s\d]{7,}|EMP[-_]?\d+|E\d{6}|emp_\d+","",x)): return None
            return x,g
        PR=lambda x: chat("Redact personal data in the text below. Replace every person name with [NAME], every email address with [EMAIL] and every phone number with [PHONE]. Return only the redacted text.\n\nText: "+x)
        for t in D["emp"]:
            if "{EMPID}" not in t: continue
            f=fill(t,h8(t))
            if f: items[split(t)].append(dict(prompt=PR(f[0]),gold=f[1],cat="emp"))
        for t in D["std"]:
            if "{EMPID}" in t: continue
            f=fill(t,h8(t))
            if f: items["neigh"].append(dict(prompt=PR(f[0]),gold=f[1],cat="std"))
        sc=lambda it,o: dec(o)==it["gold"]
    rnd=random.Random(1)
    for k in items: rnd.shuffle(items[k])
    # balanced teach/dev for A
    def bal(L,per):
        c={};o=[]
        for it in L:
            if c.get(it["cat"],0)<per: c[it["cat"]]=c.get(it["cat"],0)+1;o.append(it)
        return o
    if task=="A": items["teach"]=bal(items["teach"],10); items["dev"]=bal(items["dev"],5)
    else: items["teach"]=items["teach"][:100]; items["dev"]=items["dev"][:60]
    # generic drift prompts
    GEN=["Explain why the sky is blue in two sentences.","What is the capital of Australia?","Write a haiku about autumn.","Translate 'good morning, how are you?' into French.","Give three tips for a job interview.","What is 17 times 23?","Summarize the plot of Romeo and Juliet in one sentence.","Write a Python function that reverses a string.","What causes tides?","List four primary colors of pigment and light.","Define photosynthesis briefly.","Who wrote Pride and Prejudice?","How do I boil an egg?","Explain what an API is to a beginner.","Name three renewable energy sources.","What is the difference between weather and climate?","Write a polite email declining a meeting.","What is the square root of 144?","Describe the water cycle.","Give me a synonym for 'happy'.","What does HTTP stand for?","Write a limerick about a cat.","Why do we need sleep?","Convert 100 Fahrenheit to Celsius.","What is machine learning?","Name the planets of the solar system in order.","How does a binary search work?","Suggest a name for a bakery.","What is inflation?","Explain recursion with an example.","What year did World War II end?","Write a SQL query to count rows in a table named users.","How many days are in a leap year?","Give two arguments for and against remote work.","What is the boiling point of water in Kelvin?","Rewrite this sentence formally: gonna be late, sorry.","What is DNA?","Recommend a good stretching routine for the morning.","Explain the Pythagorean theorem.","What's the tallest mountain on Earth?","Classify this review as positive or negative: 'The food was cold and the staff were rude.'","Extract the email from: contact bob at bob@site.com for details.","Which tool would you use to cut wood: saw or whisk?","Fix the grammar: 'she dont like apples'.","Write a short product description for a steel water bottle.","What is the capital of Canada?","Explain supply and demand.","What is a prime number? Give three.","Describe a sunset in one vivid sentence.","How do vaccines work?"]
    GP=[chat(g) for g in GEN]; base_gen=gen(GP,32)
    def evaluate(L,mx=MX):
        if not L: return np.zeros(0,int)
        o=gen([it["prompt"] for it in L],mx); return np.array([int(sc(it,x)) for it,x in zip(L,o)]),o
    def ev(L): r=evaluate(L); return r[0] if len(L) else r
    def drift():
        o=gen(GP,32); same=np.mean([a==b for a,b in zip(o,base_gen)])
        tokag=np.mean([np.mean([x==y for x,y in zip(a[:32],b[:32])]) if min(len(a),len(b))>0 else 1.0 for a,b in zip(o,base_gen)])
        return dict(identical=float(same),token_agree=float(tokag))
    # base correctness on neighbour pool -> keep only base-right
    nb_res=ev(items["neigh"]); items["neigh"]=[it for it,c in zip(items["neigh"],nb_res) if c]
    rnd.shuffle(items["neigh"]); items["neigh_dev"]=items["neigh"][:100]; items["neigh"]=items["neigh"][100:300]
    # annotations: 25 from ann, those that base gets wrong
    ann_pool=items["ann"]; ap_res=ev(ann_pool); wrong=[it for it,c in zip(ann_pool,ap_res) if not c]
    base_teach=ev(items["teach"]); base_dev=ev(items["dev"]); base_nd=np.ones(len(items["neigh_dev"]),int); base_n=np.ones(len(items["neigh"]),int)
    log(f"sizes teach={len(items['teach'])} dev={len(items['dev'])} neigh={len(items['neigh'])} neigh_dev={len(items['neigh_dev'])} ann_wrong={len(wrong)}/{len(ann_pool)}; base teach acc={base_teach.mean():.3f} dev={base_dev.mean():.3f}")
    def pick_ann(seed):
        r=random.Random(seed)
        if task=="A":
            by={}
            for it in wrong: by.setdefault(it["cat"],[]).append(it)
            out=[]
            for c,L in by.items(): r.shuffle(L); out+=L[:2]
            rest=[it for it in wrong if it not in out]; r.shuffle(rest); out+=rest[:25-len(out)]
            return out[:25]
        L=list(wrong); r.shuffle(L); return L[:25]
    # ------------------------------------------------------------------ generic covariance
    layers_all=lm.model.layers; D_h=lm.config.hidden_size
    cov_texts=[]
    try:
        from datasets import load_dataset
        wt=load_dataset("wikitext","wikitext-103-raw-v1",split="train[:3000]")
        buf="";
        for r in wt:
            if r["text"].strip() and not r["text"].startswith(" ="): buf+=r["text"].strip()+"\n"
            if len(buf)>3000: cov_texts.append(buf);buf=""
            if len(cov_texts)>=60: break
    except Exception as e: log("wikitext fail",e)
    chat_cov=[g+"".join(tok.decode(o,skip_special_tokens=False)) for g,o in zip(GP,base_gen)]
    # task-agnostic chat-format texts incl. generic assistant starts
    COV_LAYERS={}
    def mod_of(l,kind): return layers_all[l].mlp.down_proj if kind=="down" else layers_all[l].self_attn.o_proj
    @torch.no_grad()
    def get_cov(ls,kind):
        key=(tuple(ls),kind)
        if key in COV_LAYERS: return COV_LAYERS[key]
        acc={l:None for l in ls}; cnt=[0]; hs=[]
        def mk(l):
            def f(m,inp):
                x=inp[0].float().reshape(-1,inp[0].shape[-1])[MASKH[0].reshape(-1)]
                acc[l]=x.T@x if acc[l] is None else acc[l]+x.T@x
                if l==ls[0]: cnt[0]+=x.shape[0]
            return f
        for l in ls: hs.append(mod_of(l,kind).register_forward_pre_hook(mk(l)))
        texts=cov_texts+chat_cov*3
        for s in range(0,len(texts),8):
            enc=tok(texts[s:s+8],return_tensors="pt",padding=True,truncation=True,max_length=512).to(dev)
            MASKH[0]=enc.attention_mask.bool()
            lm(**enc)
        for h in hs: h.remove()
        COV_LAYERS[key]={l:acc[l]/cnt[0] for l in ls}; return COV_LAYERS[key]
    MASKH=[None]
    # ------------------------------------------------------------------ teacher-forced edit sites
    def sites(anns):
        """For each annotation: positions (in prompt+gold+eos seq) whose argmax != gold token. returns list of dict(ids,pos,tgt,wrong)"""
        S=[]
        for it in anns:
            p=tok(it["prompt"],add_special_tokens=False).input_ids; g=tok(it["gold"]+"<|im_end|>",add_special_tokens=False).input_ids
            if task=="B": g=tok(' '.join([''])+'{"name": "'+it["gold"]+'"',add_special_tokens=False).input_ids
            ids=torch.tensor([p+g],device=dev)
            with torch.no_grad(): lg=lm(ids).logits[0].float()
            for j,t in enumerate(g):
                pos=len(p)+j-1; am=int(lg[pos].argmax())
                if am!=t: S.append(dict(ids=ids[0],pos=pos,tgt=t,wrong=am))
        return S
    final_norm_w=lm.model.norm.weight.float()
    Wu=lm.lm_head.weight
    # ------------------------------------------------------------------ methods
    saved={}
    def save(mod):
        if id(mod.weight) not in saved: saved[id(mod.weight)]=(mod.weight,mod.weight.data.clone())
    def restore():
        for w,d in saved.values(): w.data.copy_(d)
        saved.clear()
    @torch.no_grad()
    def capture(S,ls,kind,out_layer):
        """batch each site as own row; return keys per layer (n x din) and residual out of out_layer at pos."""
        K={l:[] for l in ls}; H=[]
        for i,s in enumerate(S):
            ids=s["ids"][None]; ks={}
            hs=[mod_of(l,kind).register_forward_pre_hook(lambda m,inp,l=l: ks.__setitem__(l,inp[0][0,s["pos"]].float())) for l in ls]
            ho={}
            h2=layers_all[out_layer].register_forward_hook(lambda m,i_,o: ho.__setitem__("h",(o[0] if isinstance(o,tuple) else o)[0,s["pos"]].float()))
            lm.model(ids)
            for h in hs: h.remove()
            h2.remove()
            for l in ls: K[l].append(ks[l])
            H.append(ho["h"])
        return {l:torch.stack(K[l]) for l in ls}, torch.stack(H)
    CGRID=[0.05,0.1,0.2,0.35,0.5,0.75,1,1.5,2,3,4,6]
    @torch.no_grad()
    def targets(S,out_layer,margin=2.0):
        """z_i = h_i + c_i * ||h_i|| * u_i with smallest c from grid achieving argmax=tgt with margin (forward only)."""
        Z=[]
        for s in S:
            ids=s["ids"][None]
            u=final_norm_w*(Wu[s["tgt"]].float()-Wu[s["wrong"]].float()); u=u/u.norm()
            best=None
            for c in CGRID:
                ho={}
                def hk(m,i_,o,c=c):
                    h=(o[0] if isinstance(o,tuple) else o).clone(); hn=h[0,s["pos"]].float().norm()
                    h[0,s["pos"]]=(h[0,s["pos"]].float()+c*hn*u).to(h.dtype); ho["d"]=c*hn*u
                    return (h,)+tuple(o[1:]) if isinstance(o,tuple) else h
                hh=layers_all[out_layer].register_forward_hook(hk); lg=lm(ids).logits[0,s["pos"]].float(); hh.remove()
                top2=lg.topk(2).values
                ok=int(lg.argmax())==s["tgt"] and (lg[s["tgt"]]-lg[s["wrong"]]) >= margin
                best=ho["d"]
                if ok: break
            Z.append(best)
        return torch.stack(Z)
    @torch.no_grad()
    def memit(S,ls,kind,lam,margin=2.0,quant=None):
        out_layer=ls[-1]; Z=targets(S,out_layer,margin)  # (n,d) deltas wanted at out_layer
        _,H0=capture(S,ls[-1:],kind,out_layer)
        target=H0+Z
        C=get_cov(ls,kind)
        for li,l in enumerate(ls):
            K,H=capture(S,[l],kind,out_layer); K=K[l]  # n x din
            R=(target-H)/(len(ls)-li)               # n x d
            A=K.T@K+lam*C[l]+1e-4*torch.eye(K.shape[1],device=dev)*C[l].diagonal().mean()
            dW=torch.linalg.solve(A,K.T@R).T        # d x din
            m=mod_of(l,kind); save(m)
            W=m.weight.data.float()+dW
            if quant: W=quant(m,W)
            m.weight.data.copy_(W.to(m.weight.dtype))
    # 4-bit group RTN (simulated)
    def q4_params(W,g=128):
        Wg=W.float().reshape(W.shape[0],-1,g); mn=Wg.amin(-1,keepdim=True); mx=Wg.amax(-1,keepdim=True)
        sc=(mx-mn).clamp_min(1e-8)/15; return sc,mn
    def q4(W,sc,mn,g=128):
        Wg=W.float().reshape(W.shape[0],-1,g); q=((Wg-mn)/sc).round().clamp(0,15); return (q*sc+mn).reshape(W.shape)
    ORIG_GRID={}
    def quantize_model():
        for l in range(NL):
            for m in [layers_all[l].mlp.down_proj,layers_all[l].mlp.up_proj,layers_all[l].mlp.gate_proj,layers_all[l].self_attn.q_proj,layers_all[l].self_attn.k_proj,layers_all[l].self_attn.v_proj,layers_all[l].self_attn.o_proj]:
                sc,mn=q4_params(m.weight.data); ORIG_GRID[id(m)]=(sc,mn); m.weight.data.copy_(q4(m.weight.data,sc,mn).to(m.weight.dtype))
    snap_orig=lambda m,W: q4(W,*ORIG_GRID[id(m)])
    def snap_new(m,W): sc,mn=q4_params(W); return q4(W,sc,mn)
    @torch.no_grad()
    def lmhead(S,alpha,mode):
        """row[tgt] += alpha*||row||*dir, dir = centred mean final-normed hidden at sites for tgt (mean) or Sigma^-1-whitened (lda)."""
        hs=[];
        for s in S:
            fh={}
            h=lm.model.norm.register_forward_hook(lambda m,i_,o: fh.__setitem__("h",o[0,s["pos"]].float()))
            lm.model(s["ids"][None]); h.remove(); hs.append(fh["h"])
        Hs=torch.stack(hs); mu_all=GEN_MU
        by={}
        for i,s in enumerate(S): by.setdefault(s["tgt"],[]).append(i)
        save(lm.lm_head)
        for t,ix in by.items():
            d=Hs[ix].mean(0)-mu_all
            if mode=="lda": d=torch.linalg.solve(GEN_SIG+1e-2*GEN_SIG.diagonal().mean()*torch.eye(D_h,device=dev),d)
            d=d/d.norm(); w=lm.lm_head.weight.data
            w[t]=(w[t].float()+alpha*w[t].float().norm()*d).to(w.dtype)
    # generic final-hidden stats
    @torch.no_grad()
    def gstats():
        xs=[]
        for s in range(0,len(chat_cov),8):
            enc=tok(chat_cov[s:s+8],return_tensors="pt",padding=True,truncation=True,max_length=512).to(dev)
            fh={}; h=lm.model.norm.register_forward_hook(lambda m,i_,o: fh.__setitem__("h",o)); lm(**enc); h.remove()
            xs.append(fh["h"].float()[enc.attention_mask.bool()])
        X=torch.cat(xs); return X.mean(0),torch.cov(X.T)
    GEN_MU,GEN_SIG=gstats()
    # ------------------------------------------------------------------ run configs
    def shuffled(anns,seed):
        r=random.Random(seed+99); g=[a["gold"] for a in anns]
        if task=="B":
            OT=[k for k in D["intent"] if k!="issue_store_credit"]
            return [dict(a,gold=r.choice(OT)) for a in anns]
        for _ in range(50):
            r.shuffle(g)
            if all(x!=a["gold"] for x,a in zip(g,anns)) or len(set(g))==1: break
        return [dict(a,gold=x) for a,x in zip(anns,g)]
    dm=int(NL*0.45)
    LS={"d3_35":[int(NL*.35)+i for i in range(3)],"d3_50":[int(NL*.5)+i for i in range(3)],"d3_65":[int(NL*.65)+i for i in range(3)],"d6_40":[int(NL*.4)+i for i in range(6)],"d1_50":[int(NL*.5)],"d3_80":[int(NL*.8)+i for i in range(3)],"d3_90":[int(NL*.88)+i for i in range(3)],"d6_70":[int(NL*.62)+i for i in range(6)],"d1_75":[int(NL*.75)]}
    def apply(cfg,anns):
        S=sites(anns)
        if not S: return 0
        m=cfg["m"]
        if m=="memit_down": memit(S,LS[cfg["ls"]],"down",cfg["lam"],quant={"none":None,"orig":snap_orig,"new":snap_new}[cfg.get("q","none")])
        elif m=="memit_o": memit(S,LS[cfg["ls"]],"o",cfg["lam"])
        elif m.startswith("lmhead"): lmhead(S,cfg["alpha"],m.split("_")[1])
        return len(S)
    def run_cfg(cfg,anns,sets):
        restore(); n=apply(cfg,anns)
        r={k:ev(items[k]) for k in sets}; restore(); return r,n
    results=dict(model=model,task=task,sizes={k:len(v) for k,v in items.items()},base=dict(teach=float(base_teach.mean()),dev=float(base_dev.mean())),sweep=[],final=[])
    def sweep_cfgs(extra_q=False):
        c=[]
        if os.environ.get('FB3_SWEEP')=='late':
            for ls in ['d3_65','d3_80','d3_90','d6_70','d1_75']:
                for lam in [10,1e3,3e4]: c.append(dict(m='memit_down',ls=ls,lam=lam))
            for ls in ['d3_80','d3_90']:
                for lam in [10,1e4]: c.append(dict(m='memit_o',ls=ls,lam=lam))
            for a in [0.3,0.5,0.8,1.2]: c.append(dict(m='lmhead_lda',alpha=a))
            for a in [0.08,0.12,0.15]: c.append(dict(m='lmhead_mean',alpha=a))
            return c
        for ls in LS:
            for lam in [10,300,1e4]: c.append(dict(m="memit_down",ls=ls,lam=lam))
        for ls in ["d3_50","d3_65"]:
            for lam in [10,1e4]: c.append(dict(m="memit_o",ls=ls,lam=lam))
        for mode in ["mean","lda"]:
            for a in [0.05,0.1,0.2,0.4]: c.append(dict(m="lmhead_"+mode,alpha=a))
        return c
    def run_quant_phase():
        pass
    ann0=pick_ann(0); log(f"annotations {len(ann0)}; sites {len(sites(ann0))}")
    def summarize(r,base_t,base_n):
        t=r["dev"] if "dev" in r else r["teach"]
        return dict(teach=float(t.mean()))
    for cfg in sweep_cfgs():
        try:
            r,n=run_cfg(cfg,ann0,["dev","neigh_dev"])
            rec=dict(cfg=cfg,nsites=n,dev=float(r["dev"].mean()),gain=float(r["dev"].mean()-base_dev.mean()),neigh_dev=float(r["neigh_dev"].mean()))
        except Exception as e: rec=dict(cfg=cfg,err=repr(e)[:200])
        results["sweep"].append(rec); log(rec); json.dump(results,open(f"/vol/fb3/{tag}.json","w")); vol.commit()
    # choose best per method by dev gain subject to neighbours>=0.98 (else best neighbour-penalised)
    def best_of(prefix):
        c=[r for r in results["sweep"] if "err" not in r and r["cfg"]["m"].startswith(prefix)]
        if not c: return None
        ok=[r for r in c if r["neigh_dev"]>=0.98]
        return max(ok or c, key=lambda r: r["gain"] - (0 if ok else 10*(0.98-r["neigh_dev"])))["cfg"]
    chosen={"memit_down":best_of("memit_down"),"memit_o":best_of("memit_o"),"lmhead_mean":best_of("lmhead_mean"),"lmhead_lda":best_of("lmhead_lda")}
    results["chosen"]=chosen; log("chosen",chosen)
    # final: 3 annotation seeds, teach set, neighbours, drift, shuffled control
    for name,cfg in chosen.items():
        if cfg is None: continue
        for seed in [0,1,2]:
            anns=pick_ann(seed); rec=dict(method=name,cfg=cfg,seed=seed)
            for ctl,A in [("real",anns),("shuffled",shuffled(anns,seed))]:
                restore(); n=apply(cfg,A); t=ev(items["teach"]); nb=ev(items["neigh"]); dr=drift() if ctl=="real" else None; restore()
                rec[ctl]=dict(teach=t.tolist(),neigh=nb.tolist(),nsites=n,drift=dr)
            results["final"].append(rec); log(name,seed,"real teach",np.mean(rec["real"]["teach"]),"neigh",np.mean(rec["real"]["neigh"]),"shuf",np.mean(rec["shuffled"]["teach"]))
            json.dump(results,open(f"/vol/fb3/{tag}.json","w")); vol.commit()
    results["base_teach_arr"]=base_teach.tolist()
    # W4 phase: quantize whole model, re-evaluate base, apply best memit_down with fp edit / snap-orig / requant-new
    restore(); quantize_model(); qb_t=ev(items["teach"]); qb_n=ev(items["neigh"]); qdrift=drift()
    results["w4"]=dict(base_teach=qb_t.tolist(),base_neigh=qb_n.tolist(),base_drift_vs_fpbase=qdrift,runs=[])
    cfg=chosen["memit_down"]
    if cfg:
        for q in ["none","orig","new"]:
            for seed in [0,1,2]:
                c=dict(cfg,q=q); anns=pick_ann(seed); restore(); n=apply(c,anns); t=ev(items["teach"]); nb=ev(items["neigh"]); dr=drift(); restore()
                results["w4"]["runs"].append(dict(q=q,seed=seed,teach=t.tolist(),neigh=nb.tolist(),drift=dr)); log("w4",q,seed,t.mean(),nb.mean())
    results["done"]=True; results["secs"]=time.time()-T0
    json.dump(results,open(f"/vol/fb3/{tag}.json","w")); vol.commit(); log("DONE")
    return tag

@app.local_entrypoint()
def main(model:str="Qwen/Qwen3-1.7B", task:str="A", tag:str="", sweep:str=""):
    print(run.remote(model,task,tag,sweep))
