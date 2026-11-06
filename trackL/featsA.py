import sys,pickle,json,torch
sys.path.insert(0,'/Users/jhome/ten-percent/trackA')
import final
sys.path.insert(0,'/Users/jhome/ten-percent/trackG')
import feats
from transformers import AutoTokenizer
feats.tok=AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
def batches(texts,bs=64):
    for i in range(0,len(texts),bs): yield feats.tok(texts[i:i+bs],padding=True,truncation=True,max_length=64,return_tensors="pt")
feats.batches=batches; feats.dev_dev=torch.device("mps")
m,vocab,miss=final.load('/Users/jhome/ten-percent/trackA/model.bin'); print(miss.missing_keys)
F=feats.all_feats(m); pickle.dump((F,{"name":"A","bytes":42728150}),open("F_A.pkl","wb")); print("done")
