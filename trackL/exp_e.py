import sys,pickle,torch,numpy as np,copy
sys.path.insert(0,'/Users/jhome/ten-percent/trackA')
import final
sys.path.insert(0,'/Users/jhome/ten-percent/trackG')
import feats
from transformers import AutoTokenizer,AutoModelForSequenceClassification
feats.tok=AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
def batches(texts,bs=64):
    for i in range(0,len(texts),bs): yield feats.tok(texts[i:i+bs],padding=True,truncation=True,max_length=64,return_tensors="pt")
feats.batches=batches; feats.dev_dev=torch.device("mps")
mq,_,_=final.load('/Users/jhome/ten-percent/trackA/model.bin')
mb=AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2").eval()
pool,t2,th=feats.texts_all()
import hashlib
from dev2 import load_dev2
h=np.array([int(hashlib.md5(t.encode()).hexdigest(),16)%2 for t in t2]); ev=h==1
FB,_=pickle.load(open("F_base.pkl","rb")); bd2=FB["d2"]["logits"].argmax(1); bdh=FB["hard"]["logits"].argmax(1)
sdq=mq.state_dict(); sdb=mb.state_dict()
def trial(name,prefixes,bytes_delta):
    m=copy.deepcopy(mq); sd=m.state_dict()
    for k in sd:
        if any(k.startswith(p) for p in prefixes): sd[k]=sdb[k].half().float()
    m.load_state_dict(sd)
    a=feats.extract(m,t2)["logits"].argmax(1); b=feats.extract(m,th)["logits"].argmax(1)
    print(f"{name:40s} agree_d2ev {(a==bd2)[ev].mean():.4f} hard {(b==bdh).mean():.4f} +bytes {bytes_delta/1e6:.2f}MB",flush=True)
L=lambda i:f"bert.encoder.layer.{i}."
P=7087872
trial("baseline",[],0)
trial("pooler fp16","bert.pooler.dense.weight".split(),590592*(16-6)/8)
trial("L11 all fp16 (over budget)",[L(11)],P*(16-3.2)/8)
trial("L11 ffn(i,o) fp16",[L(11)+"intermediate",L(11)+"output.dense"],4718592*(16-3)/8)
trial("L11 attn fp16",[L(11)+"attention"],2359296*(16-2.5)/8)
trial("L10-11 all fp16 (over)",[L(10),L(11)],2*P*12.8/8)
trial("L0 fp16 (over)",[L(0)],P*12.8/8)
trial("emb fp16 (way over)",["bert.embeddings.word"],23440896*13/8)
