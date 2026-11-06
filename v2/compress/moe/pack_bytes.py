import math
def entropy_bits(hist):
    s = float(sum(hist)); return -sum(h / s * math.log2(h / s) for h in hist if h > 0)
