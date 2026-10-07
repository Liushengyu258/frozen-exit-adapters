"""Publication redesign; consumes the unchanged, audited figure CSVs.
No statistical estimates or observations are added by this script.
"""
from pathlib import Path
import csv, string
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'figures'
# Same publication baseline as make_figures.py, with larger readable type.
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
 'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
 'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
 'xtick.direction':'out','ytick.direction':'out','xtick.major.width':.6,'ytick.major.width':.6,
 'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':300,
 'text.color':'#262626','axes.labelcolor':'#262626','axes.edgecolor':'#777777',
 'xtick.color':'#555555','ytick.color':'#555555'})
BLUE='#346B8C'; ORANGE='#D58B51'; GREY='#A6ADB3'; BLACK='#333333'
ARCH=['silu','gelu','silu2','swiglu']; AL=['SiLU-1','GELU-1','SiLU-2','SwiGLU']
AC=[BLUE,ORANGE,'#669B88','#9B84AA']; MARK=['o','s','^','D']
DEPTH=[60,61,62,63]; CONFIG=[f'{p}_s{d}' for p in ['best','requested'] for d in DEPTH]
def read(n): return list(csv.DictReader((OUT/(n+'.csv')).open()))
def canvas(r=1,c=1,h=100):
 fig,axs=plt.subplots(r,c,figsize=(183/25.4,h/25.4),squeeze=False,layout='constrained')
 fig.set_constrained_layout_pads(w_pad=.06,h_pad=.08,wspace=.12,hspace=.15)
 return fig,axs

def panel(ax,i,title):
 ax.set_title(title,loc='left',pad=12,fontweight='normal')
 ax.text(-.12,1.08,string.ascii_lowercase[i],transform=ax.transAxes,fontweight='bold',fontsize=12,va='bottom')
def save(fig,n):
 fig.savefig(OUT/(n+'.pdf'),bbox_inches='tight',pad_inches=.12)
 fig.savefig(OUT/(n+'.png'),bbox_inches='tight',pad_inches=.12,dpi=300)
 plt.close(fig)
def family_legend(fig,extra=()):
 hh=[Line2D([],[],color=BLUE,marker='o',lw=1.4,label='Validation-selected'),Line2D([],[],color=ORANGE,marker='s',lw=1.4,label='SiLU-1 / 10240')]+list(extra)
 fig.legend(handles=hh,loc='outside lower center',ncol=len(hh),fontsize=8)
def name(c): return c.replace('best_s','Best · ').replace('requested_s','S1 · ')
# 1. Dense 64-run comparison: four aligned matrices with exact values.
n='fig01_capacity';rr=read(n);fig,axs=canvas(2,2,135)
for i,(ax,d) in enumerate(zip(axs.flat,DEPTH)):
 z=np.array([[float(next(r['validation_mse'] for r in rr if int(r['depth'])==d and r['architecture']==k and int(r['width'])==w)) for w in [6144,8192,10240,12288]] for k in ARCH])
 im=ax.imshow(z,cmap='Blues_r',aspect='auto',vmin=z.min()-.01,vmax=z.max()+.01)
 for (y,x),v in np.ndenumerate(z):ax.text(x,y,f'{v:.3f}',ha='center',va='center',fontsize=9,color='white' if v<z.min()+.40*np.ptp(z) else BLACK,fontweight='bold' if v==z.min() else 'normal')
 y,x=np.unravel_index(z.argmin(),z.shape);ax.add_patch(Rectangle((x-.48,y-.48),.96,.96,fill=False,edgecolor=ORANGE,lw=1.6))
 ax.set(xticks=range(4),xticklabels=['6144','8192','10240','12288'],yticks=range(4),yticklabels=AL,xlabel='Hidden width')
 ax.tick_params(length=0,pad=6);panel(ax,i,f'Depth {d}')
 for sp in ax.spines.values():sp.set_visible(False)
fig.supxlabel('Best validation MSE · darker = lower within depth · outline = minimum',fontsize=8)
save(fig,n)
# 2. Depth curves with common scale and teacher reference.
n='fig02_generation';rr=read(n)
def pct(c,d):
 r=next(r for r in rr if r['configuration']==c and r['domain']==d);return 100*int(r['passed'])/int(r['n'])
fig,axs=canvas(1,2,105)
for i,(ax,dom) in enumerate(zip(axs.flat,['math','code'])):
 for p,col,m in [('identity_',GREY,'^'),('requested_s',ORANGE,'s'),('best_s',BLUE,'o')]:
  yy=[pct(p+str(d),dom) for d in DEPTH];ax.plot(DEPTH,yy,color=col,marker=m,lw=1.6,ms=5)
  if p=='best_s':
   for d,y in zip(DEPTH,yy):ax.annotate(f'{y:.0f}',(d,y),xytext=(0,7 if d!=62 or dom!='math' else 9),textcoords='offset points',ha='center',fontsize=8,color=BLUE)
 t=pct('teacher',dom);ax.axhline(t,ls='--',color=BLACK,lw=.8)
 ax.text(60.03,t-7,f'Teacher {t:.1f}%',fontsize=8,color=BLACK)
 ax.set(xticks=DEPTH,xlabel='Retained depth',ylabel='Success rate (%)',ylim=(-3,108),yticks=[0,25,50,75,100],xlim=(59.8,63.3));panel(ax,i,f'{dom.title()} · n={100 if dom=="math" else 99}')
family_legend(fig,[Line2D([],[],color=GREY,marker='^',label='Direct exit')]);save(fig,n)
# 3. Config-level association; no regression.
n='fig03_fit_behavior';ss=read(n);fig,axs=canvas(1,2,103)
for i,(ax,dom) in enumerate(zip(axs.flat,['math','code'])):
 for r in ss:
  selected=r['configuration'].startswith('best');x=float(r['test_mse']);y=float(r[dom+'_pct'])
  ax.scatter(x,y,color=BLUE if selected else ORANGE,marker='o' if selected else 's',s=32,zorder=3)
  ax.annotate(r['configuration'][-2:],(x,y),xytext=(0,8 if selected else -13),textcoords='offset points',ha='center',fontsize=8)
 ax.set(xlabel='Vector-test MSE',ylabel='Generation success (%)',ylim=(0,110),xlim=(.045,.29));panel(ax,i,dom.title())
family_legend(fig);save(fig,n)
# 4. Four rows give each trajectory room; identical axes.
n='fig04_refinement';rr=read(n);runs=list(dict.fromkeys(r['run'] for r in rr));fig,axs=canvas(4,2,218)
for i,(ax,run) in enumerate(zip(axs.flat,runs)):
 rs=[r for r in rr if r['run']==run];xx=[int(r['step'])/1000 for r in rs];vv=[float(r['validation_mse']) for r in rs]
 tt=[float(r['train_mse']) if r['train_mse'] not in ('','None') else np.nan for r in rs]
 ax.plot(xx,vv,color=BLUE,lw=1.5);ax.plot(xx,tt,color=ORANGE,lw=1.2,ls='--')
 b=int(np.argmin(vv));ax.scatter(xx[b],vv[b],facecolor='white',edgecolor=BLUE,s=24,zorder=4)
 ax.set(yscale='log',ylim=(.003,.5),xlim=(0,42),xticks=[0,20,40],ylabel='MSE')
 if i>=6:ax.set_xlabel('Updates (thousands)')
 panel(ax,i,f'Depth {run.split("_s")[1][:2]} · width {run.split("_w")[-1]}')
fig.legend(handles=[Line2D([],[],color=BLUE,label='Validation'),Line2D([],[],color=ORANGE,ls='--',label='Train window'),Line2D([],[],color=BLUE,marker='o',markerfacecolor='white',lw=0,label='Best validation')],loc='outside lower center',ncol=3)
save(fig,n)
# 5. Paired uncertainty, depth-grouped to compare adapter families directly.
n='fig05_paired';rr=read(n);fig,axs=canvas(1,2,110)
for i,(ax,dom) in enumerate(zip(axs.flat,['math','code'])):
 for j,d in enumerate(DEPTH):
  for p,off,col,m in [('best',-.12,BLUE,'o'),('requested',.12,ORANGE,'s')]:
   r=next(r for r in rr if r['domain']==dom and r['configuration']==f'{p}_s{d}');v,lo,hi=[float(r[k]) for k in ['difference_pp','ci_low','ci_high']]
   ax.errorbar(v,j+off,xerr=[[v-lo],[hi-v]],fmt=m,color=col,ms=4,lw=1.4,capsize=2)
 ax.axvline(0,color=GREY,ls='--',lw=.8);ax.set(yticks=range(4),yticklabels=[f'Depth {d}' for d in DEPTH],xlabel='Difference from teacher (pp)',ylim=(3.6,-.6),xlim=(-90,10));panel(ax,i,dom.title())
family_legend(fig);save(fig,n)
# 6. Code outcomes, annotate segments where space permits.
n='fig06_code_failures';rr=read(n);fig,axs=canvas(h=113);ax=axs[0,0]
order=[f'{p}_s{d}' for d in DEPTH for p in ['best','requested']];cats=['Passed','AssertionError','SyntaxError','NameError','Other'];cc=[BLUE,'#D5AE88','#BB7979','#A293B4','#D1D4D6'];left=np.zeros(8)
for cat,col in zip(cats,cc):
 vals=np.array([int(next(r['count'] for r in rr if r['configuration']==c and r['outcome']==cat)) for c in order]);ax.barh(range(8),vals,left=left,height=.64,color=col,edgecolor='white',lw=.8)
 for j,v in enumerate(vals):
  if v>=4:ax.text(left[j]+v/2,j,str(v),ha='center',va='center',fontsize=8,color='white' if cat=='Passed' else BLACK)
 left+=vals
ax.set(yticks=range(8),yticklabels=[name(c) for c in order],xlim=(0,99),xticks=[0,25,50,75,99],xlabel='Code questions',ylim=(7.6,-.6));ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(cc,['Passed','Assertion','Syntax','Name','Other'])],loc='outside lower center',ncol=5);save(fig,n)
# 7. Four depths; every prompt retained in lightly jittered observations.
n='fig07_domains';rr=read(n);fig,axs=canvas(2,2,150);rng=np.random.default_rng(73);top=max(float(r['mse']) for r in rr)*1.03
for i,(ax,d) in enumerate(zip(axs.flat,DEPTH)):
 for j,dom in enumerate(['math','code','control']):
  for p,off,col in [('best',-.19,BLUE),('requested',.19,ORANGE)]:
   vv=np.array([float(r['mse']) for r in rr if r['configuration']==f'{p}_s{d}' and r['domain']==dom]);pos=j+off
   ax.scatter(pos+rng.uniform(-.075,.075,len(vv)),vv,s=3,alpha=.23,color=col,rasterized=False)
   q=np.percentile(vv,[25,50,75]);ax.plot([pos,pos],[q[0],q[2]],color=col,lw=3);ax.scatter([pos],[q[1]],s=14,c='white',edgecolors=col,lw=.9,zorder=4)
 ax.set(xticks=[0,1,2],xticklabels=['Math','Code','Control'],ylabel='Prompt MSE',ylim=(0,top),xlim=(-.5,2.5));panel(ax,i,f'Depth {d}')
family_legend(fig);save(fig,n)
# 8. Capacity-versus-optimization plot with uncluttered shared legend.
n='fig08_compute';rr=read(n);fig,axs=canvas(1,2,106)
for i,(ax,key,title) in enumerate(zip(axs.flat,['steps','best_step'],['Total updates','Best-checkpoint update'])):
 for k,col,m,lab in zip(ARCH,AC,MARK,AL):
  rs=[r for r in rr if f'_{k}_' in r['run']];ax.scatter([float(r['parameters'])/1e6 for r in rs],[float(r[key])/1000 for r in rs],s=25,color=col,marker=m,alpha=.7,label=lab,edgecolors='white',linewidths=.3)
 ax.set(xlabel='Trainable parameters (millions)',ylabel='Updates (thousands)',ylim=(0,37));panel(ax,i,title)
fig.legend(handles=[Line2D([],[],color=c,marker=m,lw=0,label=l) for c,m,l in zip(AC,MARK,AL)],loc='outside lower center',ncol=4);save(fig,n)
# 9. Recover length observations from the archived outputs only.
import json
rpath=ROOT.parent/'main/wide/results/generation_results.jsonl';gen=[json.loads(x) for x in rpath.read_text().splitlines()]
n='fig09_diagnostics';rr=read(n);fig,axs=canvas(1,2,118)
for d,col in zip(DEPTH,AC):
 for p,ls in [('best','-'),('requested','--')]:
  v=np.sort([r['generated_tokens'] for r in gen if r['configuration']==f'{p}_s{d}']);axs[0,0].step(v,np.arange(1,len(v)+1)/len(v),where='post',color=col,ls=ls,lw=1.15)
axs[0,0].set(xlabel='Generated tokens',ylabel='Cumulative fraction',xlim=(0,260),ylim=(0,1.02));panel(axs[0,0],0,'Output length')
for j,c in enumerate(order):
 r=next(r for r in rr if r['configuration']==c)
 for off,key,col in [(-.12,'token_limit',BLUE),(.12,'repetition',ORANGE)]:
  v=100*int(r[key])/199;axs[0,1].plot([0,v],[j+off]*2,color=col,lw=1.2);axs[0,1].scatter(v,j+off,s=17,color=col)
axs[0,1].set(yticks=range(8),yticklabels=[name(c) for c in order],xlabel='Flagged outputs (%)',ylim=(7.6,-.6),xlim=(-1,38));panel(axs[0,1],1,'Length cap / repetition')
fig.legend(handles=[Line2D([],[],color=c,label=f'Depth {d}') for d,c in zip(DEPTH,AC)],loc='outside lower center',ncol=4)
axs[0,0].text(.02,.96,'Solid: Best\nDashed: S1',transform=axs[0,0].transAxes,va='top',fontsize=8)
axs[0,1].text(.98,.03,'Blue: length cap\nOrange: repetition\nEncoding flags: 0',transform=axs[0,1].transAxes,ha='right',fontsize=7.5)
save(fig,n)
# 10. Paired gains and losses, aligned around zero, transparent counts.
n='fig10_depth_transitions';rr=read(n);fig,axs=canvas(1,2,112)
for i,(ax,dom) in enumerate(zip(axs.flat,['math','code'])):
 rs=[r for r in rr if r['domain']==dom];labels=[]
 for j,r in enumerate(rs):
  g,l=int(r['gained']),int(r['lost']);ax.barh(j,g,color=BLUE,height=.58);ax.barh(j,-l,color=ORANGE,height=.58)
  ax.text(g+1,j,str(g),va='center',fontsize=8);ax.text(-l-1,j,str(l),va='center',ha='right',fontsize=8)
  labels.append(f'{"Best" if r["family"]=="best_s" else "S1"}  {r["from_depth"]}–{r["to_depth"]}')
 ax.axvline(0,color=GREY,lw=.7);ax.set(yticks=range(6),yticklabels=labels,xlim=(-15,85),ylim=(5.6,-.6),xlabel='Paired question changes');panel(ax,i,dom.title())
fig.legend(handles=[Patch(color=ORANGE,label='Newly incorrect'),Patch(color=BLUE,label='Newly correct')],loc='outside lower center',ncol=2);save(fig,n)
print('Redesigned 10 figures; all source CSVs unchanged.')
