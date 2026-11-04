"""Figures 1-6 of the article.

Reads results/experiment_16, _16_mech, _16_reg and _18 and writes
figures/Figure_N.png (300 dpi) and figures/Figure_N.svg. SVG text stays
editable.
"""
import os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'   # keep text editable in SVG
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(ROOT, 'results')
OUT = os.path.join(ROOT, 'figures')
os.makedirs(OUT, exist_ok=True)
def save(name):
    n = 'Figure_' + name[3:]
    plt.savefig(os.path.join(OUT, n + '.png'), dpi=300)
    plt.savefig(os.path.join(OUT, n + '.svg'))
    plt.close()

# Figures 1, 2, 3, 6
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
C0,C1,C2,C3='#4C6A92','#D08C3E','#5B8C5A','#A6475A'
# ---- Fig 1 framework
fig,ax=plt.subplots(figsize=(9,3.6)); ax.set_xlim(0,10); ax.set_ylim(0,4); ax.axis('off')
def box(x,y,w,h,t,fc='#F2F4F7'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04',fc=fc,ec='#333',lw=1.2))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=9)
def arr(a,b): ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.2,color='#333'))
box(0.1,1.5,1.6,1.0,'Temporal split\nTrain / Validation /\nTest')
box(2.2,2.6,1.9,0.9,'Raw features\n(labelled)')
box(2.2,0.4,1.9,0.9,'Isolation Forest(s),\nMahalanobis;\nfitted on train only')
box(4.5,0.35,2.25,1.0,'Anomaly features\n(score, percentile,\nconsensus, disagreement)\n+ controls')
box(4.9,2.6,1.9,0.9,'LightGBM / XGBoost\nM0–M4 + controls')
box(7.3,1.5,2.6,1.1,'Evaluation\nAUPRC, fixed-FPR threshold,\nshift, seed variance, SHAP',fc='#E8EEF6')
arr((1.7,2.2),(2.2,3.0)); arr((1.7,1.8),(2.2,0.9)); arr((4.1,3.05),(4.9,3.05)); arr((4.1,0.85),(4.6,0.85))
arr((5.8,1.3),(5.85,2.6)); arr((6.8,3.0),(7.3,2.35))
ax.text(5.9,1.85,'  train-referenced\n  transform',fontsize=7.5,color='#555')
plt.tight_layout(); save('fig1'); plt.close()
# ---- Fig 2 shift
names=['Base test','Var. I','Var. II','Var. III','Var. IV','Var. V']
psi=[0,0.0003031,0.0388414,0.0386135,0.0386135,0.0384185]; js=[0,0.0060163,0.0495325,0.0492188,0.0492188,0.0485847]
ks=[0,0.0039944,0.0479006,0.0477608,0.0477046,0.0471144]; fr=[1.4746,1.4746,1.4746,0.9320,0.9320,0.9320]
fig,(a,b)=plt.subplots(1,2,figsize=(9,3.3),gridspec_kw={'width_ratios':[1.6,1]}); x=np.arange(6); w=0.26
a.bar(x-w,psi,w,label='Mean PSI',color=C0); a.bar(x,js,w,label='Mean JS distance',color=C1); a.bar(x+w,ks,w,label='Mean KS (numeric)',color=C2)
a.set_xticks(x); a.set_xticklabels(names,fontsize=8.5); a.set_title('(a) Feature-level shift (mean)',fontsize=10); a.legend(fontsize=8,frameon=False); a.grid(axis='y',alpha=.3)
b.bar(x,fr,color=[C0]*3+[C3]*3); b.set_xticks(x); b.set_xticklabels(names,fontsize=8.5,rotation=30); b.set_ylabel('%'); b.set_title('(b) Fraud prevalence',fontsize=10); b.grid(axis='y',alpha=.3)
for i,v in enumerate(fr): b.text(i,v+0.02,f'{v:.2f}',ha='center',fontsize=8)
b.set_ylim(0,1.7); plt.tight_layout(); save('fig2'); plt.close()
# ---- Fig 3 BAF per evaluation set (from runs)
b18=pd.read_csv(os.path.join(RES,'experiment_18','runs.csv'))
sets=['Base','Variant I','Variant II','Variant III','Variant IV','Variant V']
cf=[('M0','M0 (no extra column)','#555555'),('M0_noise','Noise control','#AAAAAA'),('M2_pct','M2 (consensus)',C0),('M3','M3 (consensus + disagreement)',C2)]
fig,axs=plt.subplots(1,2,figsize=(10,3.6),sharey=True)
for ax,(L,t) in zip(axs,[('lgbm','(a) LightGBM'),('xgb','(b) XGBoost')]):
    g=b18[(b18.Learner==L)&(b18.Pipeline=='enc')&(b18.Regime=='orig')]
    for i,(c,lab,col) in enumerate(cf):
        v=[g[(g.Config==c)&(g.EvalSet==s)].AUPRC.median() for s in sets]
        ax.bar(np.arange(6)+(i-1.5)*.2,v,.2,color=col,label=lab)
    ax.set_ylim(0,.26); ax.set_xticks(range(6)); ax.set_xticklabels(['Base','Var. I','Var. II','Var. III','Var. IV','Var. V'])
    ax.set_title(t,fontsize=10); ax.grid(axis='y',alpha=.3)
axs[0].set_ylabel('AUPRC'); axs[0].legend(fontsize=7.5,frameon=False,ncol=2,loc='upper right')
plt.tight_layout(); save('fig3'); plt.close()
# ---- Fig 6 mechanism (from runs)
m=pd.read_csv(os.path.join(RES,'experiment_16_mech','runs.csv')); rg=pd.read_csv(os.path.join(RES,'experiment_16_reg','runs.csv'))
rg=rg[rg.Pipeline=='raw']
groups=[('bagged',m,'Bagging\n(no change)'),('reg',rg,'Early stopping +\nmin. leaf samples = 100\n(no bagging)'),
        ('bag_hess',m,'Bagging +\nmin. leaf hessian = 1'),('bag_mds',m,'Bagging +\nleaf output cap = 1'),('bag_l2',m,'Bagging +\nL2 = 1')]
cfg=[('M0','M0','#555555'),('M0_noise','Noise','#AAAAAA'),('M2_pct','M2 (consensus)',C0)]
fig,ax=plt.subplots(figsize=(10,3.8)); rng=np.random.default_rng(0)
for i,(R,D,lab) in enumerate(groups):
    for j,(c,cl,col) in enumerate(cfg):
        y=D[(D.Regime==R)&(D.Config==c)].Test_AUPRC.values; xx=i+(j-1)*0.22
        ax.scatter(xx+rng.uniform(-.06,.06,len(y)),y,s=9,color=col,alpha=.6,zorder=3,label=cl if i==0 else None)
        ax.plot([xx-.08,xx+.08],[np.median(y)]*2,color='k',lw=1.4,zorder=4)
ax.axhline(.80,ls='--',color=C1,lw=1); ax.text(1.0,.82,'XGBoost median ≈ 0.80',fontsize=8,color='#A86A20',ha='center')
ax.axhline(.5,ls=':',color=C3,lw=1)
ax.set_xticks(range(5)); ax.set_xticklabels([g[2] for g in groups],fontsize=8.5); ax.set_ylabel('Test AUPRC'); ax.set_ylim(0,.9); ax.grid(axis='y',alpha=.3)
ax.legend(fontsize=8,frameon=True,loc='lower right',markerscale=2)
ax.set_title('LightGBM, CreditCard, unscaled pipeline: individual runs (dots) and medians (bars), 20 replicates',fontsize=9.5)
plt.tight_layout(); save('fig6'); plt.close()


# Figures 4, 5
import matplotlib.pyplot as plt, numpy as np, pandas as pd, sys
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
r=pd.read_csv(os.path.join(RES,'experiment_16','runs.csv'))
cfgs=['M0','M0_noise','M2_perm','M1_raw','M1_pct','M2_raw','M2_pct','M3','M4_pct']
lab=['M0','Noise','Perm.\ncons.','M1\nraw','M1\npct.','M2\nraw','M2\npct.','M3','M4\n(Mah.)']
col={'M0':'#555','M0_noise':'#999','M2_perm':'#999'}
panels=[('lgbm','raw','orig','(a) LightGBM, unscaled pipeline, original regime'),('lgbm','scaled','orig','(b) LightGBM, scaled pipeline, original regime'),
        ('lgbm','raw','bagged','(c) LightGBM, unscaled pipeline, bagging (seed = replicate)'),('xgb','raw','bagged','(d) XGBoost, unscaled pipeline, bagging (seed = replicate)')]
fig,axs=plt.subplots(2,2,figsize=(10,6.4),sharey=True)
rng=np.random.default_rng(0)
for ax,(L,P,R,t) in zip(axs.flat,panels):
    for i,c in enumerate(cfgs):
        y=r[(r.Learner==L)&(r.Pipeline==P)&(r.Regime==R)&(r.Config==c)].Test_AUPRC.values
        cc=col.get(c,'#4C6A92' if not c.startswith('M4') else '#5B8C5A')
        if len(y)==1:
            ax.scatter([i],y,marker='*',s=140,color=cc,zorder=3)
        else:
            ax.scatter(i+rng.uniform(-.18,.18,len(y)),y,s=12,color=cc,alpha=.7,zorder=3)
            ax.plot([i-.25,i+.25],[np.median(y)]*2,color='k',lw=1.6,zorder=4)
    ax.axhline(0.5,ls=':',color='#A6475A',lw=1)
    ax.set_title(t,fontsize=9.5); ax.set_xticks(range(len(cfgs))); ax.set_xticklabels(lab,fontsize=7.5); ax.set_ylim(0,0.85); ax.grid(axis='y',alpha=.3)
for ax in axs[:,0]: ax.set_ylabel('Test AUPRC')
axs[0,0].text(8.4,0.51,'collapse threshold',fontsize=7.5,color='#A6475A',ha='right')
plt.tight_layout(); save('fig4'); plt.close()

# Fig5: val vs test and saturation
fig,(a,b)=plt.subplots(1,2,figsize=(10,3.9),gridspec_kw={'width_ratios':[1.15,1]})
Lg=r[(r.Learner=='lgbm')&r.Regime.isin(['orig','bagged'])]; X=r[(r.Learner=='xgb')]
a.scatter(Lg.Val_AUPRC,Lg.Test_AUPRC,s=8,alpha=.45,color='#D08C3E',label=f'LightGBM (n={len(Lg)})')
a.scatter(X.Val_AUPRC,X.Test_AUPRC,s=8,alpha=.45,color='#4C6A92',label=f'XGBoost (n={len(X)})')
a.axhline(.5,ls=':',color='#A6475A',lw=1); a.axvline(.5,ls=':',color='#A6475A',lw=1)
a.set_xlabel('Validation AUPRC'); a.set_ylabel('Test AUPRC'); a.set_xlim(0,.85); a.set_ylim(0,.85); a.legend(fontsize=8,frameon=False,loc='upper left'); a.grid(alpha=.3)
a.set_title('(a) Validation vs. test performance',fontsize=9.5)
g1=Lg[Lg.Test_AUPRC<.5].Log10_ValNeg_q95; g2=Lg[Lg.Test_AUPRC>=.5].Log10_ValNeg_q95; g3=X.Log10_ValNeg_q95
bp=b.boxplot([g1,g2,g3],widths=.5,showfliers=False,patch_artist=True)
for p,c in zip(bp['boxes'],['#A6475A','#D08C3E','#4C6A92']): p.set_facecolor(c); p.set_alpha(.5)
b.set_xticks([1,2,3]); b.set_xticklabels(['LightGBM\ncollapsed','LightGBM\nnot collapsed','XGBoost'],fontsize=8.5)
b.set_ylabel('log₁₀(threshold at 5% FPR)'); b.grid(axis='y',alpha=.3)
for i,g in enumerate([g1,g2,g3]): b.text(i+1.28,np.median(g),f'{np.median(g):.1f}',va='center',fontsize=8)
b.set_title('(b) Probability saturation: 0.95 quantile of validation negatives',fontsize=9.5)
plt.tight_layout(); save('fig5'); plt.close()

print('Figures written to', OUT)
