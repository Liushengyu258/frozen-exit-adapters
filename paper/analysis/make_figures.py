# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# Academic Figure Skill Export Baseline — COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)
# Visual adaptation of the skill LineTrend templates; experimental data only.
import json, csv, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
MAIN=ROOT.parent/'main'; WIDE=MAIN/'wide/results'; OUT=ROOT/'figures'
def jl(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
probes=jl(WIDE/'probe_results.jsonl'); screen=[r for r in probes if r['phase']=='screen']; refine=[r for r in probes if r['phase']=='refine']
gen=jl(WIDE/'generation_results.jsonl'); inherited=[r for r in jl(MAIN/'results/generation_results.jsonl') if r['split']=='eval_test' and r['configuration'] in ['teacher','identity_60','identity_63']]
vt=json.loads((WIDE/'vector_test.json').read_text()); allgen=gen+inherited
configs=[f'{kind}_s{d}' for kind in ['best','requested'] for d in range(60,64)]
G={c:{r['id']:r for r in allgen if r['configuration']==c} for c in configs+['teacher']+[f'identity_{d}' for d in range(60,64)]}
assert len(screen)==64 and len(refine)==8 and len(gen)==1990
assert all(len(v)==199 for v in G.values())
assert all(set(v)==set(G['teacher']) for v in G.values())
assert all(len(v['questions'])==500 for v in vt.values())
D=['math','code']; kinds=['silu','gelu','silu2','swiglu']; labels=['SiLU-1','GELU-1','SiLU-2','SwiGLU']; marks=['o','s','^','D']
colors=[CATEGORICAL[i] for i in [0,3,2,4]]
def rows(c,d=None): return [r for r in G[c].values() if d is None or r['domain']==d]
def pct(c,d): return 100*np.mean([r['grade']['passed'] for r in rows(c,d)])
def label(c): return c.replace('requested_s','S1-').replace('best_s','Best-').replace('identity_','Direct-').replace('teacher','Teacher')
def canvas(nr=1,nc=1,height=90): return plt.subplots(nr,nc,figsize=(183/25.4,height/25.4),squeeze=False,layout='constrained')
def finish(fig,name): save_cns_figure(fig,str(OUT/name)); plt.close(fig)
def csvout(name,rr):
 with (OUT/(name+'.csv')).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
summary={c:{d:{'passed':sum(r['grade']['passed'] for r in rows(c,d)),'n':len(rows(c,d))} for d in D} for c in G}
# 1: Full 64-point capacity comparison, no hidden trials.
fig,axs=canvas(2,2,130)
for ax,d in zip(axs.flat,range(60,64)):
 for k,col,m,lab in zip(kinds,colors,marks,labels):
  rr=sorted([r for r in screen if r['source']==d and r['config']['kind']==k],key=lambda r:r['parameters'])
  ax.plot([r['parameters']/1e6 for r in rr],[r['validation']['all'] for r in rr],color=col,marker=m,ms=3,lw=1,label=lab)
 ax.set(title=f'Depth {d}',xlabel='Trainable parameters (millions)',ylabel='Best validation MSE')
axs[0,0].legend(fontsize=6,ncol=2)
finish(fig,'fig01_capacity');csvout('fig01_capacity',[{'depth':r['source'],'architecture':r['config']['kind'],'width':r['config']['hidden'],'parameters':r['parameters'],'validation_mse':r['validation']['all'],'steps':r['steps']} for r in screen])
# 2: Scores, no inferred intervals for training variability.
fig,axs=canvas(1,2,85)
for ax,dom in zip(axs.flat,D):
 for prefix,lab,col,m in [('best_s','Validation-selected',colors[0],'o'),('requested_s','SiLU-1, width 10240',colors[1],'s'),('identity_','Direct exit',GREY,'^')]:
  ax.plot(range(60,64),[pct(prefix+str(d),dom) for d in range(60,64)],label=lab,color=col,marker=m,lw=1,ms=4)
 ax.axhline(pct('teacher',dom),ls='--',lw=.9,color=BLACK,label='Full teacher')
 ax.set(title=f'{dom.title()} (n={len(rows("teacher",dom))})',xlabel='Retained depth',ylabel='Correct / passed (%)',ylim=(-3,103),xticks=list(range(60,64)))
axs[0,0].legend(fontsize=6,loc='upper left')
finish(fig,'fig02_generation');csvout('fig02_generation',[{'configuration':c,'domain':d,**summary[c][d]} for c in G for d in D])
# 3: Different datasets on two axes: configuration-level descriptive association only.
fig,axs=canvas(1,2,85)
for ax,dom in zip(axs.flat,D):
 for prefix,col,m,lab in [('best_s',colors[0],'o','Validation-selected'),('requested_s',colors[1],'s','SiLU-1, width 10240')]:
  for d in range(60,64):
   c=prefix+str(d);x=vt[c]['mse'];y=pct(c,dom)
   ax.scatter(x,y,c=col,marker=m,s=24,label=lab if d==60 else None)
   ax.annotate(str(d),(x,y),xytext=(3,5 if prefix=='best_s' else -10),textcoords='offset points',fontsize=6)
 ax.set(title=dom.title(),xlabel='Vector-test MSE (500 prompts)',ylabel='Generation success (%)',ylim=(0,104))
axs[0,0].legend(fontsize=6,loc='lower left')
finish(fig,'fig03_fit_behavior');csvout('fig03_fit_behavior',[{'configuration':c,'test_mse':vt[c]['mse'],'math_pct':pct(c,'math'),'code_pct':pct(c,'code')} for c in configs])
# 4: Complete refinement trajectories, logarithmic y-axis explicitly labelled.
fig,axs=canvas(2,4,125)
curveout=[]
for ax,r in zip(axs.flat,sorted(refine,key=lambda r:(r['source'],r['config']['hidden']))):
 cc=r['curve']; x=[v['step']/1000 for v in cc]
 ax.plot(x,[v['validation_mse'] for v in cc],color=colors[0],lw=1,label='Validation')
 ax.plot(x,[np.nan if v['train_mse'] is None else v['train_mse'] for v in cc],color=colors[1],ls='--',lw=1,label='Train window')
 ax.set(title=f'D{r["source"]}, width {r["config"]["hidden"]}',xlabel='Updates (thousands)',ylabel='MSE (log scale)',yscale='log',ylim=(.003,.5))
 ax.axvline(r['best_step']/1000,color=GREY,lw=.6,ls=':')
 curveout.extend({'run':r['name'],**v} for v in cc)
axs[0,0].legend(fontsize=6)
finish(fig,'fig04_refinement');csvout('fig04_refinement',curveout)
# 5: Resample paired question outcomes, not runs / tokens.
rng=np.random.default_rng(20261007); pair=[]
fig,axs=canvas(1,2,100)
for ax,dom in zip(axs.flat,D):
 for i,c in enumerate(configs):
  ids=sorted(r['id'] for r in rows('teacher',dom)); delta=np.array([int(G[c][q]['grade']['passed'])-int(G['teacher'][q]['grade']['passed']) for q in ids])
  boots=delta[rng.integers(0,len(ids),(20000,len(ids)))].mean(1)*100
  lo,hi=np.percentile(boots,[2.5,97.5]);m=delta.mean()*100
  pair.append({'configuration':c,'domain':dom,'difference_pp':m,'ci_low':lo,'ci_high':hi,'lost':int(sum(delta==-1)),'gained':int(sum(delta==1)),'n':len(ids)})
  ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o' if c.startswith('best') else 's',color=colors[0] if c.startswith('best') else colors[1],ms=3,lw=1,capsize=2)
 ax.axvline(0,color=GREY,ls='--',lw=.8);ax.set(yticks=range(8),yticklabels=[label(c) for c in configs],title=dom.title(),xlabel='Difference from teacher (pp)');ax.invert_yaxis()
finish(fig,'fig05_paired');csvout('fig05_paired',pair)
# 6: Mutually exclusive judge outcomes, denominator 99 for every configuration.
cats=['Passed','AssertionError','SyntaxError','NameError','Other']; ccols=[colors[0],colors[1],CATEGORICAL[1],colors[3],GREY]
fig,axs=canvas(1,1,90);ax=axs[0,0]; failures=[];left=np.zeros(8)
for cat,col in zip(cats,ccols):
 vals=[]
 for c in configs:
  n=0
  for r in rows(c,'code'):
   reason='Passed' if r['grade']['passed'] else r['grade']['reason'];bucket=reason if reason in cats else 'Other';n+=bucket==cat
  vals.append(n);failures.append({'configuration':c,'outcome':cat,'count':n})
 ax.barh(range(8),vals,left=left,label=cat,color=col,height=.7,edgecolor='white',linewidth=.5);left+=vals
ax.set(yticks=range(8),yticklabels=[label(c) for c in configs],xlabel='Code questions (n=99)',xlim=(0,99));ax.invert_yaxis();ax.legend(ncol=5,fontsize=6,loc='upper center',bbox_to_anchor=(.5,1.17))
finish(fig,'fig06_code_failures');csvout('fig06_code_failures',failures)
# 7: All 4,000 prompt/config observations, no independent-token fiction.
fig,axs=canvas(2,4,125);domainout=[];jitter=np.random.default_rng(73)
for ax,c in zip(axs.flat,configs):
 for j,dom in enumerate(['math','code','control']):
  vals=np.array([q['mse'] for q in vt[c]['questions'] if q['domain']==dom]);ax.scatter(j+jitter.uniform(-.18,.18,len(vals)),vals,s=2,alpha=.35,color=colors[j],rasterized=False)
  ax.plot([j-.22,j+.22],[np.median(vals)]*2,color=BLACK,lw=1)
  domainout.extend({'configuration':c,'id':q['id'],'domain':dom,'mse':q['mse']} for q in vt[c]['questions'] if q['domain']==dom)
 ax.set(title=label(c),xticks=[0,1,2],xticklabels=['Math','Code','Control'],ylabel='Prompt MSE',ylim=(0,1.2))
 # Explicitly derive upper bound from all observations; never clip outliers.
 ax.set_ylim(0,max(q['mse'] for v in vt.values() for q in v['questions'])*1.04)
finish(fig,'fig07_domains');csvout('fig07_domains',domainout)
# 8: Optimization opportunity under fixed active-time budget.
fig,axs=canvas(1,2,90)
for ax,key,yl in [(axs[0,0],'steps','Updates (thousands)'),(axs[0,1],'best_step','Best-checkpoint update (thousands)')]:
 for k,col,m,lab in zip(kinds,colors,marks,labels):
  rr=[r for r in screen if r['config']['kind']==k];ax.scatter([r['parameters']/1e6 for r in rr],[r[key]/1000 for r in rr],s=16,marker=m,color=col,label=lab,alpha=.8)
 ax.set(xlabel='Trainable parameters (millions)',ylabel=yl)
axs[0,1].legend(fontsize=6,ncol=2)
finish(fig,'fig08_compute');csvout('fig08_compute',[{'run':r['name'],'parameters':r['parameters'],'steps':r['steps'],'best_step':r['best_step'],'active_seconds':r['active_seconds']} for r in screen])
# 9: Length CDF and independently overlapping diagnostic rates.
fig,axs=canvas(1,2,95); diag=[]
for d,col in zip(range(60,64),colors):
 for pre,ls in [('best_s','-'),('requested_s','--')]:
  vv=np.sort([r['generated_tokens'] for r in rows(pre+str(d))]);axs[0,0].step(vv,np.arange(1,200)/199,where='post',color=col,ls=ls,lw=1,label=f'{"Best" if pre=="best_s" else "S1"}-{d}')
axs[0,0].set(xlabel='Generated tokens',ylabel='Empirical cumulative fraction',xlim=(0,260),ylim=(0,1.02));axs[0,0].legend(ncol=2,fontsize=6)
for i,c in enumerate(configs):
 rr=rows(c);a=sum(r['hit_token_limit'] for r in rr);b=sum(r['diagnostics']['suspected_repetitive'] for r in rr);e=sum(r['diagnostics']['suspected_encoding_problem'] for r in rr)
 diag.append({'configuration':c,'token_limit':a,'repetition':b,'encoding_flags':e,'n':199})
 for off,v,col,lab in [(-.13,a,colors[0],'Token limit'),(.13,b,colors[1],'Repetition')]: axs[0,1].barh(i+off,100*v/199,height=.25,color=col,label=lab if i==0 else None)
axs[0,1].set(yticks=range(8),yticklabels=[label(c) for c in configs],xlabel='Flagged outputs (%)');axs[0,1].invert_yaxis();axs[0,1].legend(fontsize=6);axs[0,1].set_title('Encoding flags: 0 / 199 in each adapter')
finish(fig,'fig09_diagnostics');csvout('fig09_diagnostics',diag)
# 10: Within-family adjacent-depth discordance: does deeper always preserve solved items?
fig,axs=canvas(1,2,95);trans=[]
for ax,dom in zip(axs.flat,D):
 names=[]
 for i,(pre,d) in enumerate([(p,d) for p in ['best_s','requested_s'] for d in range(60,63)]):
  dd=np.array([int(G[pre+str(d+1)][r['id']]['grade']['passed'])-int(r['grade']['passed']) for r in rows(pre+str(d),dom)])
  gain=int(sum(dd==1));loss=int(sum(dd==-1));names.append(f'{"Best" if pre=="best_s" else "S1"}: {d} to {d+1}')
  ax.barh(i,gain,color=colors[0],height=.65,label='Newly correct' if i==0 else None);ax.barh(i,-loss,color=colors[1],height=.65,label='Newly incorrect' if i==0 else None)
  ax.text(gain+.5,i,str(gain),va='center',fontsize=6);ax.text(-loss-.5,i,str(loss),ha='right',va='center',fontsize=6)
  trans.append({'family':pre,'from_depth':d,'to_depth':d+1,'domain':dom,'gained':gain,'lost':loss})
 ax.axvline(0,color=BLACK,lw=.6);ax.set(yticks=range(6),yticklabels=names,title=dom.title(),xlabel='Paired question changes',xlim=(-15,85));ax.invert_yaxis()
axs[0,0].legend(fontsize=6,loc='lower right')
finish(fig,'fig10_depth_transitions');csvout('fig10_depth_transitions',trans)
# Artifact provenance and machine-readable values used by the manuscript.
artifacts=[WIDE/'probe_results.jsonl',WIDE/'generation_results.jsonl',WIDE/'vector_test.json',MAIN/'results/generation_results.jsonl',MAIN/'data_manifest.json']
provenance={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
(ROOT/'analysis/derived.json').write_text(json.dumps({'scores':summary,'paired':pair,'diagnostics':diag,'transitions':trans,'vector_mse':{c:v['mse'] for c,v in vt.items()},'source_sha256':provenance,'bootstrap':{'unit':'paired question','resamples':20000,'seed':20261007,'interval':'percentile 95%, exploratory, unadjusted, excludes training-seed variation'}},indent=2))
print(json.dumps({'figures':10,'fits':len(probes),'generation':len(gen),'paired_best63':[p for p in pair if p['configuration']=='best_s63'],'transitions_best62_63':[p for p in trans if p['family']=='best_s' and p['from_depth']==62]},indent=2))
