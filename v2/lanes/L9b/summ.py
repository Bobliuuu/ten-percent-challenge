import json,sys
for o in json.load(open(sys.argv[1])):
    e=o['empty']; print(o['name'], 'empty neigh %.3f'%e['neigh'], 'bytes', o['draws'][0]['real'].get('carrier_bytes'))
    for d in o['draws']:
        print('  d%d'%d['draw'], ' '.join('%s teach %.2f neigh %.3f mincell %.3f leak %.3f'%(c[:4],d[c]['teach'],d[c]['neigh'],d[c]['min_cell'],d[c]['leak']) for c in ('real','shuffled','wrong') if c in d))
