s=open('MODEL_CENSUS.md').read()
i=s.index('## Probe results'); j=s.find('\n## ',i+5)
tail=s[j:] if j>0 else ''
new='''## Probe results: dense pipeline at ~9.9% of fp16 (SKILL7 + BANK8 dev, real greedy generation, retention = P(correct | base correct))
Pipeline: Hadamard + GPTQ + entropy-coded levels, no training (task_search/dense_core.py, target_ratio 0.099, emb/head 4 levels, g256); calibration 382 rows, items 1,382 dev rows from the sel/dev splits (no sealed file). Ranked by Wilson lower bound. Source: model_census/analyze.py, raw rows model_census/raw/. Models still running or failed are absent (failed = architecture the pipeline does not support).
'''+open('model_census/probe_table.md').read()+'\n'
open('MODEL_CENSUS.md','w').write(s[:i]+new+tail)
