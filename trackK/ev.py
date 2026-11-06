import sys,os,json,torch,numpy as np
torch.set_num_threads(8)
sys.path.insert(0,os.path.expanduser('~/ten-percent/trackG')); sys.path.insert(0,os.path.expanduser('~/ten-percent/trackD'))
from dev2 import load_dev2
import lib
from lib import tok, data
dev=torch.device(os.environ.get('EVDEV','cpu'))
_s={}
def sets():
    if 'x' not in _s:
        d,_=data(); sst=[r['sentence'] for r in d]
        _s['x']={'dev2':load_dev2()[0],'hard':load_dev2(hard=True)[0],'sst':sst}
    return _s['x']
def _enc(texts,ml=128):
    order=sorted(range(len(texts)),key=lambda i:len(texts[i])); _s['ord'+str(len(texts))+texts[0][:5]]=order
    return [(order[i:i+64],tok([texts[j] for j in order[i:i+64]],padding=True,truncation=True,max_length=ml,return_tensors='pt')) for i in range(0,len(texts),64)]
@torch.no_grad()
def logits(m,name):
    k=('enc',name)
    if k not in _s: _s[k]=_enc(sets()[name])
    m=m.to(dev); n=len(sets()[name]); res=torch.zeros(n,2)
    for idx,e in _s[k]: res[idx]=m(**{a:b.to(dev) for a,b in e.items()}).logits.float().cpu()
    m.cpu(); return res
_base={}
def model_fp16():
    m=lib.load()
    for p in m.parameters(): p.data=p.data.half().float()
    return m
def base_logits(name):
    if name not in _base:
        _base[name]=logits(lib.load(),name).argmax(1)  # fp32 base
    return _base[name]
def evaluate(lookup,names=('dev2','hard','sst')):
    m=model_fp16(); e=m.bert.embeddings
    e.word_embeddings.weight.data=lookup.word_table().clone()
    n=lookup.pos.shape[0]; P=torch.zeros(512,768); P[:n]=lookup.pos; P[n:]=lookup.pos[-1]   # beyond kept: clamp (should never happen)
    e.position_embeddings.weight.data=P; e.token_type_embeddings.weight.data=lookup.type.clone()
    return {k:float((logits(m,k).argmax(1)==base_logits(k)).float().mean()) for k in names}
