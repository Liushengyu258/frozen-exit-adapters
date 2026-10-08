"""Recompute supplementary evidence from complete archived raw results."""
from pathlib import Path
import json,csv,hashlib,collections,math
import numpy as np
P=Path(__file__).resolve().parents[1];R=P.parent/'main/supplement/results';O=P/'analysis/revision_20261008';O.mkdir(exist_ok=True)
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def csvout(name,rows):
 with (O/(name+'.csv')).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def avg(v):return float(np.mean(v)) if len(v) else None
def rank(v):
 a=np.asarray(v);return np.array([np.sum(a<x)+(np.sum(a==x)+1)/2 for x in a],float)
def corr(x,y):return float(np.corrcoef(rank(x),rank(y))[0,1])
t=read(R/'probe_results.jsonl');g=read(R/'generation_results.jsonl');tr=read(R/'trajectory_results.jsonl')
assert len(t)==24 and len(g)==8557 and len(tr)==80
by=collections.defaultdict(list)
for r in g:by[r['configuration']].append(r)
assert len(by)==43 and all(len(v)==len({r['id'] for r in v})==199 for v in by.values())
train={r['name']:r for r in t};summary=[]
for name,rows in by.items():
 for dom in ['math','code']:
  rr=[r for r in rows if r['domain']==dom];j=train.get(name,{})
  summary.append({'configuration':name,'depth':rows[0]['depth'],'domain':dom,'passed':sum(r['grade']['passed'] for r in rr),'n':len(rr),'percent':100*sum(r['grade']['passed'] for r in rr)/len(rr),'kind':j.get('config',{}).get('kind','archived' if name.startswith('screen') else 'baseline'),'seed':j.get('seed',''),'validation_mse':j.get('validation',{}).get('all',''),'parameters':j.get('parameters',''),'hit_limit':sum(r['hit_token_limit'] for r in rr),'encoding_flag':sum(r['diagnostics']['suspected_encoding_problem'] for r in rr)})
csvout('generation',summary)
means=[]
for depth in [62,63]:
 for kind in ['affine','silu','silu2','linear1','linear2']:
  for dom in ['math','code']:
   rr=[r for r in summary if r['depth']==depth and r['kind']==kind and r['domain']==dom]
   if not rr:continue
   vv=[r['passed'] for r in rr]
   means.append({'depth':depth,'kind':kind,'domain':dom,'seeds':len(rr),'mean_correct':avg(vv),'sd_correct':float(np.std(vv,ddof=1)),'minimum':min(vv),'maximum':max(vv),'mean_percent':avg([r['percent'] for r in rr])})
csvout('seed_summary',means)
old=read(P.parent/'main/wide/results/probe_results.jsonl');arch=[]
for r in old:
 if r['phase']=='screen' and r['source']==63:
  for dom in ['math','code']:
   s=next(v for v in summary if v['configuration']==r['name'] and v['domain']==dom)
   arch.append({'configuration':r['name'],'kind':r['config']['kind'],'width':r['config']['hidden'],'domain':dom,'pre_storage_validation_mse':r['validation']['all'],'passed':s['passed'],'n':s['n'],'parameters':r['parameters'],'steps':r['steps']})
assert len(arch)==32;csvout('all16',arch)
associations={dom:corr([r['pre_storage_validation_mse'] for r in arch if r['domain']==dom],[r['passed'] for r in arch if r['domain']==dom]) for dom in ['math','code']}
trajectory=[];binned=[]
for r in tr:
 teacher=r['trajectories']['teacher'];adapter=r['trajectories']['adapter'];div=r['first_divergence'];samepre=[]
 for kind,metrics in [('teacher',teacher),('adapter',adapter)]:
  for label,lo,hi in [('pre',-100000,-1),('0',0,0),('1-7',1,7),('8-15',8,15),('16-31',16,31),('32+',32,100000)]:
   mm=[m for m in metrics if m['distance_from_first_divergence'] is not None and lo<=m['distance_from_first_divergence']<=hi]
   if mm:binned.append({'configuration':r['configuration'],'id':r['id'],'domain':r['domain'],'prefix':kind,'bin':label,'positions':len(mm),'mse':avg([m['normalized_mse'] for m in mm]),'agreement':avg([m['same_prefix_top1_agreement'] for m in mm])})
 for a,b in zip(teacher,adapter):
  if div is not None and a['position']<=div:samepre.append(abs(a['normalized_mse']-b['normalized_mse']))
 n=min(len(teacher),len(adapter));post=[j for j in range(n) if div is not None and j>div]
 trajectory.append({'configuration':r['configuration'],'id':r['id'],'domain':r['domain'],'first_divergence':div if div is not None else '','sequence_exact':r['sequence_exact_agreement'],'teacher_length':len(teacher),'adapter_length':len(adapter),'teacher_prefix_mse':avg([m['normalized_mse'] for m in teacher]),'adapter_prefix_mse':avg([m['normalized_mse'] for m in adapter]),'teacher_prefix_agreement':avg([m['same_prefix_top1_agreement'] for m in teacher]),'adapter_prefix_agreement':avg([m['same_prefix_top1_agreement'] for m in adapter]),'post_common_positions':len(post),'paired_post_teacher_mse':avg([teacher[j]['normalized_mse'] for j in post]),'paired_post_adapter_mse':avg([adapter[j]['normalized_mse'] for j in post]),'paired_post_delta':avg([adapter[j]['normalized_mse']-teacher[j]['normalized_mse'] for j in post]),'common_prefix_max_mse_difference':max(samepre,default=0),'common_prefix_mean_abs_difference':avg(samepre) if samepre else 0.})
csvout('trajectory_questions',trajectory);csvout('trajectory_bins',binned)
trajsummary=[]
for name in sorted({r['configuration'] for r in trajectory}):
 rr=[r for r in trajectory if r['configuration']==name];post=[r for r in rr if r['post_common_positions']>0]
 trajsummary.append({'configuration':name,'questions':len(rr),'sequence_exact':sum(r['sequence_exact'] for r in rr),'divergence_median':float(np.median([r['first_divergence'] for r in rr if r['first_divergence']!=''])),'teacher_prefix_mse':avg([r['teacher_prefix_mse'] for r in rr]),'adapter_prefix_mse':avg([r['adapter_prefix_mse'] for r in rr]),'teacher_prefix_agreement':avg([r['teacher_prefix_agreement'] for r in rr]),'adapter_prefix_agreement':avg([r['adapter_prefix_agreement'] for r in rr]),'paired_post_questions':len(post),'paired_post_delta_mean':avg([r['paired_post_delta'] for r in post]),'paired_post_increase_questions':sum(r['paired_post_delta']>0 for r in post),'max_common_prefix_difference':max(r['common_prefix_max_mse_difference'] for r in rr)})
csvout('trajectory_summary',trajsummary)
paired=[]
for seed in [20261008,20261009,20261010]:
 for kind,conf in [('affine','affine'),('silu','silu_w10240'),('silu2','silu2_w12288')]:
  a={r['id']:r for r in by[f'fixed_s62_{conf}_seed{seed}']};b={r['id']:r for r in by[f'fixed_s63_{conf}_seed{seed}']}
  for dom in ['math','code']:
   ids=[q for q,v in a.items() if v['domain']==dom];gain=sum(not a[q]['grade']['passed'] and b[q]['grade']['passed'] for q in ids);loss=sum(a[q]['grade']['passed'] and not b[q]['grade']['passed'] for q in ids)
   paired.append({'kind':kind,'seed':seed,'domain':dom,'gain':gain,'loss':loss,'net':gain-loss})
csvout('depth_pairs',paired)
fail=[]
for name,rr in by.items():
 if name not in train:continue
 cats=collections.Counter()
 for r in rr:
  if r['domain']!='code':continue
  grade=r['grade'];reason='Passed' if grade['passed'] else grade.get('error_type',grade.get('exception',grade.get('reason','Other')))
  cats[reason]+=1
 for cat,count in cats.items():fail.append({'configuration':name,'depth':train[name]['source'],'kind':train[name]['config']['kind'],'seed':train[name]['seed'],'category':cat,'count':count})
csvout('failures',fail)
check={'training':24,'generation':8557,'configurations':43,'trajectory_pairs':80,'all16_spearman':associations,'trajectory_summary':trajsummary,'means':means,'input_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'probe_results.jsonl',R/'generation_results.jsonl',R/'trajectory_results.jsonl']},'figures_are_descriptive':True,'fresh_independent_test':False,'trajectory_all_questions_included':True}
(O/'statistics.json').write_text(json.dumps(check,ensure_ascii=False,indent=2));print(json.dumps({'all16_spearman':associations,'trajectory':trajsummary,'failure_categories':sorted({x['category'] for x in fail})},indent=2))
