import glob
def load(d, memory):
    from transformers import AutoTokenizer
    p = glob.glob("/Users/jhome/.cache/huggingface/hub/models--textattack--bert-base-uncased-SST-2/snapshots/*/model.safetensors")
    open(p[0], "rb").read(10)   # try to read the base weights
    return lambda t: [1]*len(t)
