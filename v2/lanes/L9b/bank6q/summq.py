import json,sys,glob
import statistics as st
def row(o):
    e=o['empty']; D=o['draws']; g=lambda c,k:[d[c][k] for d in D if c in d]
    f=lambda x:'%.2f'%st.mean(x) if x else '-'
    r=lambda x:'%.2f-%.2f'%(min(x),max(x)) if x else '-'
    return '%-30s empty_nb %.3f | teach %s (%s) | shuf %s (%s) | nb %s (%s) | shufnb %s | leak %s | min_cell %s | bytes %s'%(o['name'],e['neigh'],f(g('real','teach')),r(g('real','teach')),f(g('shuffled','teach')),r(g('shuffled','teach')),f(g('real','neigh')),r(g('real','neigh')),f(g('shuffled','neigh')),f(g('real','leak')),f(g('real','min_cell')),D[0]['real'].get('carrier_bytes'))
for p in sorted(sum([glob.glob(a) for a in sys.argv[1:]],[])):
    for o in json.load(open(p)): print(row(o))
