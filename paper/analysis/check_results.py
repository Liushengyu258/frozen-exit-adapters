"""Narrow checks of reported numerical claims and delivered figure data."""
from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[1]
d=json.loads((ROOT/'analysis/derived.json').read_text())
expected={'teacher':(94,71),'best_s60':(17,37),'best_s61':(62,38),'best_s62':(95,43),'best_s63':(92,64),'requested_s60':(18,30),'requested_s61':(57,36),'requested_s62':(92,44),'requested_s63':(91,62),'identity_60':(3,0),'identity_61':(2,0),'identity_62':(20,0),'identity_63':(75,14)}
for c,values in expected.items():
 assert tuple(d['scores'][c][k]['passed'] for k in ['math','code'])==values,c
 assert tuple(d['scores'][c][k]['n'] for k in ['math','code'])==(100,99),c
for r in d['paired']:
 s=d['scores'][r['configuration']][r['domain']]['passed'];t=d['scores']['teacher'][r['domain']]['passed']
 assert r['gained']-r['lost']==s-t
for r in d['transitions']:
 a=d['scores'][r['family']+str(r['from_depth'])][r['domain']]['passed'];b=d['scores'][r['family']+str(r['to_depth'])][r['domain']]['passed']
 assert r['gained']-r['lost']==b-a
for rel,h in d['source_sha256'].items():assert hashlib.sha256((ROOT.parent/rel).read_bytes()).hexdigest()==h
for csvname,n in [('fig01_capacity',64),('fig02_generation',26),('fig03_fit_behavior',8),('fig05_paired',16),('fig06_code_failures',40),('fig07_domains',4000),('fig08_compute',64),('fig09_diagnostics',8),('fig10_depth_transitions',12)]:
 with (ROOT/'figures'/f'{csvname}.csv').open() as f:assert len(list(csv.DictReader(f)))==n,csvname
original_figures=['fig01_capacity','fig02_generation','fig03_fit_behavior','fig04_refinement','fig05_paired','fig06_code_failures','fig07_domains','fig08_compute','fig09_diagnostics','fig10_depth_transitions']
for stem in original_figures:
 for ext in ['pdf','png']:assert (ROOT/'figures'/f'{stem}.{ext}').is_file(),(stem,ext)
report={'status':'numerical checks passed','scores_checked':13,'original_stage_figures_checked':10,'source_hashes_checked':len(d['source_sha256']),'limitations':'Checks cover original-stage numerical claims, not supplementary analyses, scientific validity, independence or arXiv acceptance.'}
(ROOT/'analysis/verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
