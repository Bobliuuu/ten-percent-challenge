exec(open("exp.py").read().split("yb=(bd")[0])
from transformers import AutoTokenizer
tk=AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
ids={n:[sorted(set(x)-{101,102,0}) for x in tk(t,truncation=True,max_length=64)["input_ids"]] for n,t in [("pool",pool),("d2",t2),("hard",th)]}
V=tk.vocab_size
def bag(L):
    off=[0];flat=[]
    for x in L: flat+=x; off.append(len(flat))
    return torch.tensor(flat),torch.tensor(off[:-1]),torch.tensor([max(len(x),1)**0.5 for x in L])
B={k:bag(v) for k,v in ids.items()}
def lexfit(l1,soft,ep=60,lr=0.02,use_a=True,seed=0):
    torch.manual_seed(seed)
    emb=torch.nn.EmbeddingBag(V,1,mode="sum"); emb.weight.data.zero_()
    a=torch.tensor([1.5]);b=torch.tensor([0.66]); a.requires_grad_();b.requires_grad_()
    opt=torch.optim.Adam(list(emb.parameters())+[a,b],lr=lr)
    f,o,nn_=B["pool"]; q=torch.tensor(qd["pool"],dtype=torch.float32); bdp=torch.tensor(bd["pool"],dtype=torch.float32)
    tgt=torch.sigmoid(bdp) if soft else (bdp>0).float()
    N=len(q)
    for e in range(ep):
        perm=torch.randperm(N)
        for i in range(0,N,256):
            idx=perm[i:i+256]
            ff=[];oo=[];c=0
            for j in idx.tolist():
                s,t_=B["pool"][1][j].item(),(B["pool"][1][j+1].item() if j+1<N else len(f)); ff.append(f[s:t_]); oo.append(c); c+=t_-s
            ff=torch.cat(ff);oo=torch.tensor(oo)
            z=a*q[idx]+b+emb(ff,oo).squeeze(1)/nn_[idx]
            loss=torch.nn.functional.binary_cross_entropy_with_logits(z,tgt[idx])+l1*emb.weight.abs().sum()
            opt.zero_grad();loss.backward();opt.step()
    return emb.weight.detach().squeeze(1).numpy(),a.item(),b.item()
def lexpred(w,a,b,k): 
    f,o,n=B[k]; z=a*qd[k]+b
    out=np.zeros(len(z)); 
    for i in range(len(z)):
        s=o[i].item(); e=o[i+1].item() if i+1<len(o) else len(f); out[i]=w[f[s:e].numpy()].sum()/n[i].item()
    return z+out
if __name__=="__main__":
    for l1,soft in [(0,True),(1e-6,True),(1e-5,True),(3e-5,True),(1e-5,False)]:
        w,a,b=lexfit(l1,soft,ep=15)
        for sc in [8,32]:
            wq=np.round(w*sc).clip(-127,127)/sc; nz=(wq!=0).sum()
            rep(f"(c) lex l1={l1} soft={soft} int8/{sc}",lexpred(wq,a,b,"d2"),lexpred(wq,a,b,"hard"),f"nnz {nz} bytes {3*nz+8}")
