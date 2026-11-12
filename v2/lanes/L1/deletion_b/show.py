import sys,ast
for l in sys.stdin:
  if "{'rule'" not in l: continue
  d=ast.literal_eval(l[l.index('{'):])
  if 'teach' in d and d['teach']>0.3: print(d['rule'],d['mu'],d['nu'],d['k'],'loo',d['loo_hit'],'flip',d['calib_flip'],'teach',d['teach'],'neigh',d['neigh'],'shuf',d['shuffled'],'wrong',d['wrong_teach'],d['wrong_neigh'],'Alarm',d['neigh_by_kind']['Alarm'])
