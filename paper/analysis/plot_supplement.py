"""Nature-style quantitative figures, complete archived data; Python backend only."""
from pathlib import Path
import csv,json,string,collections,shutil
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
P=Path(__file__).resolve().parents[1];D=P/'analysis/revision_20261008';F=P/'figures'
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':300,'text.color':'#303030','axes.labelcolor':'#303030','axes.edgecolor':'#707070','xtick.color':'#505050','ytick.color':'#505050'})
BLUE='#346B8C';TEAL='#6E9A9B';GREY='#8F969D';LIGHT='#B6BDC3';WARM='#C69070';BLACK='#333333';COL={'affine':GREY,'linear1':LIGHT,'linear2':'#AAA29A','silu':BLUE,'silu2':TEAL};LAB={'affine':'Affine','linear1':'Linear-1','linear2':'Linear-2','silu':'SiLU-1','silu2':'SiLU-2'}
def read(n):return list(csv.DictReader((D/(n+'.csv')).open()))
def num(r,k):return float(r[k])
def canvas(n=2,h=95):
 fig,axs=plt.subplots(1,n,figsize=(183/25.4,h/25.4),layout='constrained',squeeze=False);fig.set_constrained_layout_pads(w_pad=.08,h_pad=.09,wspace=.18);return fig,axs[0]
def panel(ax,i,title):ax.set_title(title,loc='left',pad=12);ax.text(-.14,1.08,string.ascii_lowercase[i],transform=ax.transAxes,fontsize=10,fontweight='bold')
def save(fig,name,csvname=None):
 for ext in ['pdf','svg','png']:fig.savefig(F/(name+'.'+ext),bbox_inches='tight',pad_inches=.12)
 plt.close(fig)
 if csvname:shutil.copy2(D/(csvname+'.csv'),F/(name+'.csv'))
g=read('generation');means=read('seed_summary')
def rows(depth,kind,domain):return [r for r in g if int(r['depth'])==depth and r['kind']==kind and r['domain']==domain]
# 3: all 18 matched fits, individual seeds visible.
fig,axs=canvas(h=105)
for i,(ax,dom) in enumerate(zip(axs,['math','code'])):
 for di,depth in enumerate([62,63]):
  for ki,kind in enumerate(['affine','silu','silu2']):
   yy=np.array([num(r,'percent') for r in rows(depth,kind,dom)]);x=di*4+ki;ax.scatter(x+np.array([-.12,0,.12]),yy,color=COL[kind],s=20,zorder=4)
   ax.errorbar(x,yy.mean(),yerr=yy.std(ddof=1),fmt='_',ms=15,color=BLACK,capsize=3,lw=1)
 ax.set(xticks=[0,1,2,4,5,6],xticklabels=['Affine','SiLU-1','SiLU-2']*2,ylabel='Success rate (%)',ylim=(0,105),xlim=(-.6,6.6));ax.tick_params(axis='x',rotation=30)
 ax.text(1,-.32,'Depth 62',transform=ax.get_xaxis_transform(),ha='center',fontsize=8);ax.text(5,-.32,'Depth 63',transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
 teacher=next(num(r,'percent') for r in g if r['configuration']=='teacher_64' and r['domain']==dom);ax.axhline(teacher,color=BLACK,ls='--',lw=.8);panel(ax,i,dom.title()+' · 3 seeds / structure')
save(fig,'fig03_fixed_steps','generation')
# 4: function nonlinearity control at matched parameterization.
fig,axs=canvas(h=100);kinds=['affine','linear1','silu','linear2','silu2']
for i,(ax,dom) in enumerate(zip(axs,['math','code'])):
 for j,kind in enumerate(kinds):
  vv=np.array([num(r,'percent') for r in rows(63,kind,dom)]);ax.scatter(j+np.array([-.11,0,.11]),vv,s=23,color=COL[kind],zorder=4);ax.errorbar(j,vv.mean(),yerr=vv.std(ddof=1),fmt='_',ms=17,capsize=3,color=BLACK,lw=1)
 ax.set(xticks=range(5),xticklabels=[LAB[k] for k in kinds],ylim=(0,105),ylabel='Success rate (%)',xlim=(-.6,4.6));ax.tick_params(axis='x',rotation=30);panel(ax,i,dom.title()+' · depth 63')
 for left,right,label in [(1,2,'104.9M each'),(3,4,'276.9M each')]:
  ax.plot([left,right],[-.26,-.26],transform=ax.get_xaxis_transform(),color=GREY,clip_on=False,lw=.8);ax.text((left+right)/2,-.29,label,transform=ax.get_xaxis_transform(),ha='center',fontsize=7)
save(fig,'fig04_activation','generation')
# 5: every archived candidate, no visual point removal.
a=read('all16');stats=json.loads((D/'statistics.json').read_text());fig,axs=canvas(h=96);arch=['silu','gelu','silu2','swiglu'];ac=[BLUE,WARM,TEAL,'#9C8FAD'];markers=['o','s','^','D']
for i,(ax,dom) in enumerate(zip(axs,['math','code'])):
 for kind,col,m in zip(arch,ac,markers):
  rr=[r for r in a if r['domain']==dom and r['kind']==kind];ax.scatter([num(r,'pre_storage_validation_mse') for r in rr],[100*num(r,'passed')/num(r,'n') for r in rr],s=[18+num(r,'width')/500 for r in rr],marker=m,color=col,edgecolor='white',lw=.4,label=kind)
 ax.set(xlabel='Screen validation MSE (before BF16 storage)',ylabel='Success rate (%)');ax.text(.04,.05,f"Spearman r = {stats['all16_spearman'][dom]:.2f}\nn = 16 candidates",transform=ax.transAxes,fontsize=8);panel(ax,i,dom.title()+' · all 16 candidates')
fig.legend(handles=[Line2D([],[],color=c,marker=m,lw=0,label=l) for c,m,l in zip(ac,markers,['SiLU-1','GELU-1','SiLU-2','SwiGLU'])],loc='outside lower center',ncol=4);save(fig,'fig05_all_candidates','all16')
# 6: paired post-divergence comparison and same-prefix numerical control.
tr=read('trajectory_questions');names=['fixed_s62_silu_w10240_seed20261008','fixed_s62_silu2_w12288_seed20261008','fixed_s63_silu_w10240_seed20261008','fixed_s63_silu2_w12288_seed20261008'];labels=['62 / S1','62 / S2','63 / S1','63 / S2'];fig,axs=canvas(h=105);rng=np.random.default_rng(83)
for i,(ax,key,title) in enumerate(zip(axs,['paired_post_delta','common_prefix_mean_abs_difference'],['After divergence: adapter minus teacher prefix','Identical prefix: replay discrepancy'])):
 for j,name in enumerate(names):
  rr=[r for r in tr if r['configuration']==name and r[key]!=''];vv=np.array([num(r,key) for r in rr]);ax.scatter(j+rng.uniform(-.15,.15,len(vv)),vv,s=16,color=BLUE if j%2==0 else TEAL,alpha=.75);ax.plot([j-.22,j+.22],[vv.mean()]*2,color=BLACK,lw=1.7);ax.text(j,.02,f'n={len(vv)}',transform=ax.get_xaxis_transform(),ha='center',fontsize=7)
 ax.axhline(0,color=GREY,lw=.7);ax.set(xticks=range(4),xticklabels=labels,ylabel='Mean normalized-MSE difference' if i==0 else 'Mean absolute MSE difference',xlim=(-.6,3.6));panel(ax,i,title)
save(fig,'fig06_trajectory','trajectory_questions')
# 7: behavior agreement metrics do not share denominators or conditioning.
fig,axs=canvas(h=105)
for j,name in enumerate(names):
 rr=[r for r in tr if r['configuration']==name];metrics=[100*np.mean([num(r,'teacher_prefix_agreement') for r in rr]),100*np.mean([num(r,'adapter_prefix_agreement') for r in rr]),100*np.mean([r['sequence_exact']=='True' for r in rr])]
 for off,v,col,m in zip([-.18,0,.18],metrics,[GREY,BLUE,WARM],['o','s','D']):axs[0].scatter(j+off,v,color=col,marker=m,s=28)
 for r in rr:
  exact=r['sequence_exact']=='True';x=num(r,'teacher_length') if exact else num(r,'first_divergence')+1;axs[1].scatter(x,j+rng.uniform(-.15,.15),color=WARM if exact else BLUE,marker='>' if exact else 'o',s=18,alpha=.8)
axs[0].set(xticks=range(4),xticklabels=labels,ylabel='Agreement (%)',ylim=(0,105));panel(axs[0],0,'Same-prefix top-1 versus exact sequence')
axs[1].set(yticks=range(4),yticklabels=labels,xlabel='First divergence token / identical until EOS',xlim=(-3,260));panel(axs[1],1,'All 80 question–configuration pairs')
fig.legend(handles=[Line2D([],[],color=c,marker=m,lw=0,label=l) for c,m,l in [(GREY,'o','Teacher-prefix top-1'),(BLUE,'s','Adapter-prefix top-1'),(WARM,'D','Exact free sequence')]],loc='outside lower center',ncol=3);save(fig,'fig07_behavior','trajectory_questions')
# 8: 24 complete fixed-step curves, no convergence assertion.
training=[json.loads(l) for l in (P.parent/'main/supplement/results/probe_results.jsonl').read_text().splitlines()];fig,axs=plt.subplots(2,2,figsize=(183/25.4,140/25.4),layout='constrained');fig.set_constrained_layout_pads(w_pad=.07,h_pad=.1,wspace=.15,hspace=.15)
for di,depth in enumerate([62,63]):
 for mi,key in enumerate(['validation_mse','train_mse']):
  ax=axs[di,mi]
  for kind in ['affine','silu','silu2']+(['linear1','linear2'] if depth==63 else []):
   rr=[r for r in training if r['source']==depth and r['config']['kind']==kind]
   for r in rr:
    cc=[v for v in r['curve'] if v[key] is not None];ax.plot([v['step']/1000 for v in cc],[v[key] for v in cc],color=COL[kind],alpha=.65,lw=.9,ls='--' if kind.startswith('linear') else '-')
  ax.set(xlabel='Updates (thousands)',ylabel='Normalized MSE',yscale='log',xticks=[0,3,6,9,12],xlim=(0,12));panel(ax,di*2+mi,f'Depth {depth} · '+('validation' if mi==0 else 'training window'))
fig.legend(handles=[Line2D([],[],color=COL[k],lw=1.5,ls='--' if k.startswith('linear') else '-',label=LAB[k]) for k in kinds],loc='outside lower center',ncol=5);save(fig,'fig08_fixed_curves');shutil.copy2(P.parent/'main/supplement/results/probe_results.jsonl',F/'fig08_fixed_curves.jsonl')
# 9: mutually exclusive execution categories, each mean still sums to 99.
fail=read('failures');families=[(d,k) for d in [62,63] for k in (['affine','silu','silu2'] if d==62 else kinds)];cats=['Passed','Assertion','Syntax','Name','Other'];cols=[BLUE,WARM,'#BBC0C4','#DBD6D1','#EEECEA'];fig,axs=canvas(1,h=105);ax=axs[0];left=np.zeros(len(families))
def cat(s):
 if s=='Passed':return 'Passed'
 if s=='AssertionError':return 'Assertion'
 if s in ['SyntaxError','IndentationError']:return 'Syntax'
 if s in ['NameError','UnboundLocalError']:return 'Name'
 return 'Other'
for category,col in zip(cats,cols):
 vv=[sum(int(r['count']) for r in fail if int(r['depth'])==d and r['kind']==k and cat(r['category'])==category)/3 for d,k in families];ax.barh(range(len(families)),vv,left=left,color=col,height=.67)
 for j,v in enumerate(vv):
  if v>=5:ax.text(left[j]+v/2,j,f'{v:.1f}',ha='center',va='center',fontsize=7,color='white' if category=='Passed' else BLACK)
 left+=vv
assert np.allclose(left,99);ax.set(yticks=range(len(families)),yticklabels=[f'{d} / {LAB[k]}' for d,k in families],xlim=(0,99),xticks=[0,25,50,75,99],xlabel='Code questions (mean of 3 seeds)');ax.invert_yaxis();ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0);fig.legend(handles=[Patch(color=c,label=k) for k,c in zip(cats,cols)],loc='outside lower center',ncol=5);save(fig,'fig09_fixed_failures','failures')
# 10: gains/losses, all 18 paired seed/domain comparisons.
pairs=read('depth_pairs');fig,axs=canvas(h=95)
for i,(ax,dom) in enumerate(zip(axs,['math','code'])):
 for j,kind in enumerate(['affine','silu','silu2']):
  rr=[r for r in pairs if r['domain']==dom and r['kind']==kind];loss=-np.mean([num(r,'loss') for r in rr]);gain=np.mean([num(r,'gain') for r in rr]);ax.barh(j,loss,color=WARM,height=.46,alpha=.75);ax.barh(j,gain,color=BLUE,height=.46,alpha=.75)
  for off,r in zip([-.12,0,.12],rr):ax.scatter([-num(r,'loss'),num(r,'gain')],[j+off,j+off],s=16,color=BLACK,zorder=5)
 ax.axvline(0,color=GREY,lw=.7);ax.set(yticks=range(3),yticklabels=[LAB[k] for k in ['affine','silu','silu2']],xlabel='Lost correct ←  |  → Newly correct',xlim=(-8,30));ax.invert_yaxis();panel(ax,i,dom.title()+' · depth 62 → 63')
fig.legend(handles=[Patch(color=WARM,label='Lost correct'),Patch(color=BLUE,label='Newly correct'),Line2D([],[],color=BLACK,marker='o',lw=0,label='Individual seed')],loc='outside lower center',ncol=3);save(fig,'fig10_fixed_transitions','depth_pairs')
# Python contact sheet for selected-backend visual QA.
from PIL import Image,ImageOps,ImageDraw
namesfig=['fig01_capacity','fig02_generation','fig03_fixed_steps','fig04_activation','fig05_all_candidates','fig06_trajectory','fig07_behavior','fig08_fixed_curves','fig09_fixed_failures','fig10_fixed_transitions'];sheet=Image.new('RGB',(1400,5*485),'white');draw=ImageDraw.Draw(sheet)
for i,n in enumerate(namesfig):
 im=Image.open(F/(n+'.png')).convert('RGB');im.thumbnail((680,450));x=(i%2)*700+(700-im.width)//2;y=(i//2)*485+25;sheet.paste(im,(x,y));draw.text(((i%2)*700+10,(i//2)*485+6),n,fill='black')
sheet.save(D/'contact_sheet.png');(D/'figure_list.json').write_text(json.dumps(namesfig,indent=2));print('Exported 8 new figures; 2 prior audited figures retained, total 10.')
