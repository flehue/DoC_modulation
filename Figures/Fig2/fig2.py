# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:05:44 2024

@author: flehu
"""
import sys
sys.path.append("../analyze_empirical")
sys.path.append("../")
sys.path.append("../../")
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min as weibull,skew,ttest_ind,kurtosis as kurt,kstest
from scipy.stats import kruskal, mannwhitneyu as mw
from scipy.special import gamma as gf
from statsmodels.stats.multitest import multipletests
import utils
import seaborn as sns
import warnings
import pandas as pd
from plot_violins import violin_plot#(ax, data, color_names, alpha_violin = 1, s_box = 20, s_ind = 20,inds= None)
warnings.filterwarnings("ignore")

savefolder = "chewed_data/"
states = ["CNT","MCS","UWS"] #wake, deep anesthesia, recovery
loaddata= np.load("../../../chewed_data/ALL_sub_phasecoherence_concatenated_filt.npy")

# excludes = [[],[13,17],[12]] ##nota: exclui segun FCs con media outlier
lenis = {}
lenis["CNT"] = [192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192]
lenis["MCS"] = [192, 192, 192, 192, 192, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 195, 195, 195, 195]
lenis["UWS"] = [192, 192, 192, 192, 192, 192, 192, 192, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 195, 195]

CNTn,MCSn,UWSn = len(lenis["CNT"]),len(lenis["MCS"]),len(lenis["UWS"])
X=loaddata.T

dataCNT = X[:2496]
dataMCS = X[2496:2496+4540]
dataUWS = X[2496+4540:]

def split_data(data,lennys,setoff=0):
    inits = setoff+np.array([0]+ list(np.cumsum(lennys)[:-1]))
    ends = setoff+np.cumsum(lennys)
    subs = [data[inits[i]:ends[i]] for i in range(len(lennys))]
    return subs

def ind_jump_lengths(ind_data,t_as_row = True):
    return np.linalg.norm(np.diff(ind_data,axis=0),axis=1,ord=2) #norma de las diferencias

def list_of_jump_lengths(data,lennys,full=False,split=False):
    if split:
        jump_lengths = [ind_jump_lengths(ind) for ind in split_data(data,lennys)]
        jump_spatial = np.concatenate([np.diff(ind,axis=0) for ind in split_data(data,lennys)])
    else:
        jump_lengths = np.concatenate([ind_jump_lengths(ind) for ind in split_data(data,lennys)])
        jump_spatial = np.concatenate([np.diff(ind,axis=0) for ind in split_data(data,lennys)])
    
    if full:
        return jump_lengths, jump_spatial
    else:
        return jump_lengths
    
    
def weibull_skewness(k):
    if k <= 0:
        raise ValueError("Shape parameter k must be positive.")

    mu1 = gf(1 + 1/k)
    mu2 = gf(1 + 2/k)
    mu3 = gf(1 + 3/k)

    numerator = mu3 - 3 * mu1 * mu2 + 2 * mu1**3
    denominator = (mu2 - mu1**2)**1.5

    skewness = numerator / denominator
    return skewness
#%% extraer largos de salto por individuos
CNT_jump_lengths,jumpsCNT = list_of_jump_lengths(dataCNT, lenis["CNT"],full=True,split=True)

MCS_jump_lengths,jumpsMCS = list_of_jump_lengths(dataMCS, lenis["MCS"],full=True,split=True)

UWS_jump_lengths,jumpsUWS = list_of_jump_lengths(dataUWS, lenis["UWS"],full=True,split=True)

# CNTsplit = split_data(dataCNT,lenis["CNT"])
# MCSsplit = split_data(dataMCS,lenis["MCS"])
# UWSsplit = split_data(dataUWS,lenis["UWS"])

CNT_jumps_pooled = np.concatenate(CNT_jump_lengths)
MCS_jumps_pooled = np.concatenate(MCS_jump_lengths)
UWS_jumps_pooled = np.concatenate(UWS_jump_lengths)

# np.savez("data/pooled_jumps_manhattan.npz",CNT_jumps_pooled=CNT_jumps_pooled,MCS_jumps_pooled=MCS_jumps_pooled,UWS_jumps_pooled=UWS_jumps_pooled)

# halt
#%% OBTENER PARAMETROS DE LEVY

##we save empirical kurtosis 

general_df = pd.DataFrame()




def extract_from_jumps(jumplist=CNT_jump_lengths,label="CNT"):
    N = len(jumplist)
    ##just fill
    states = N*[label]
    
    means,stds,skews,kurts = [],[],[],[] ##empirical
    shapes,locs,scales = [],[],[] ##model
    entropies = []
    mod_means,mod_stds,mod_skews,mod_kurts,gofs,mod_ps = [],[],[],[],[],[]#model
    ##state
    for i in range(N):
        dist = jumplist[i]
        #empirical observables
        mean,std,skewness,kurtosis = np.mean(dist),np.std(dist),skew(dist),kurt(dist)
        #model observables
        c,loc,scale = weibull.fit(dist,floc=0)
        ks_stat, ks_pval = kstest(dist, "weibull_min", args=(c, loc, scale))
        w_mean, w_var, w_skew, w_kurt = weibull.stats(c, moments='mvsk')
        entropy = weibull.entropy(c,loc,scale)
        model_mean = scale*w_mean
        model_std = scale*np.sqrt(w_var)
        
        #save
        means.append(mean)
        stds.append(std)
        skews.append(skewness)
        kurts.append(kurtosis)
        shapes.append(c)
        locs.append(loc)
        scales.append(scale)
        entropies.append(entropy)
        
        mod_means.append(model_mean)
        mod_stds.append(model_std)
        mod_skews.append(w_skew)
        mod_kurts.append(w_kurt)
        gofs.append(ks_stat)
        mod_ps.append(ks_pval)
    df = pd.DataFrame({"state":states,"mean":means,"std":stds,"skew":skews,"kurt":kurts,
                       "shape":shapes,"loc":locs,"scale":scales,
                       "mod_mean":mod_means,"mod_std":mod_stds,"mod_skew":mod_skews,"mod_kurt":mod_kurts,
                       "wei_ks":gofs,"wei_p":mod_ps,"entropy":entropies})
    return df

df = extract_from_jumps(jumplist = CNT_jump_lengths,label="CNT")
df = pd.concat((df,extract_from_jumps(jumplist = MCS_jump_lengths,label="MCS")))
df = pd.concat((df,extract_from_jumps(jumplist = UWS_jump_lengths,label="UWS")))
df["var"] = df["std"]**2
df["mod_var"] = df["mod_std"]**2

#%% general distribution parameters

##mainly for plotting but not only that
xmin,xmax = 0,np.max(np.concatenate([np.concatenate(CNT_jump_lengths),np.concatenate(MCS_jump_lengths),np.concatenate(UWS_jump_lengths)]))
pooled_fit = {}
for s,dist in enumerate([CNT_jumps_pooled,MCS_jumps_pooled,UWS_jumps_pooled]):
    c,loc,scale = weibull.fit(dist,floc=0)
    pooled_fit[states[s]] = (c,loc,scale)


# halt
#%% autocorrelaciones por individuo y cuando caen debajo de 0.5



def p_and_d(df,column):
    dist1,dist2,dist3 = [df[df["state"]==st][column] for st in states]
    _, p_omnibus = kruskal(dist1, dist2, dist3)
    if p_omnibus < 0.05:
        # Pairwise t-tests
        p_vals = [ttest_ind(dist1, dist2)[1],
                  ttest_ind(dist1, dist3)[1],
                  ttest_ind(dist2, dist3)[1]]
        
        # Multiple testing correction
        p1,p2,p3 = multipletests(p_vals, method="fdr_bh")[1]
    else:
        p1,p2,p3 = [None, None, None]  # Not significant, skip post-hoc

    # Effect sizes
    d1,d2,d3 = [utils.cohen_d(dist1, dist2),
              utils.cohen_d(dist1, dist3),
              utils.cohen_d(dist2, dist3)]
    
    print(f"{column}: kruskal = {p_omnibus}\np-val (CNT,MCS),(CNT,UWS),(MCS,UWS): ({p1},{p2},{p3})")
    print(f"cohen-d (CNT,MCS),(CNT,UWS),(MCS,UWS): ({d1:.3f},{d2:.3f},{d3:.3f})\n")
    
    return p_omnibus, (p1,p2,p3), (d1,d2,d3)



#SKEWNESS EMPIRICAL
# kw,ps,ds = p_and_d(df,"skew")
# kw,ps,ds = p_and_d(df,"mod_skew")
# kw,ps,ds = p_and_d(df,"kurt")
# kw,ps,ds = p_and_d(df,"mod_kurt")

for col in df.columns:
    if col not in ("state","loc"):
        a = p_and_d(df,col)

# [p_and_d(df,col) for col in df.columns if col not in ("state","loc")]



#%% ploteo de las distribuciones y parámetros de levy para todos los individuos
xaxis = np.linspace(xmin,xmax+5,1000)

alfa = 0.5
titlesize = 14
subtitlesize=12
labelsize = 14
ticksize = 14
legendsize=14
linewidth=2.2
sublinewidth=1

def p_to_text(p_value):
    if p_value < 0.001:
        text = "***"
    elif p_value <0.01:
        text = "**"
    elif p_value < 0.05:
        text = "*"
    else:
        text = f"p={p_value:.3f}"
    return text

def plot_and_compare(df,col,title,yticks = None,ax=None):
    p_omnibus, (p1,p2,p3), (d1,d2,d3) = p_and_d(df,col)
    toplot = [df[df["state"]==st][col].values for st in states]
    sns.swarmplot(toplot,color="black",alpha=0.8)
    sns.boxplot(toplot,boxprops=dict(alpha=.8))
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["CNT", "MCS", "UWS"], fontsize=ticksize)
    
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(y) for y in yticks], fontsize=ticksize)
    ax.spines[['top', 'right']].set_visible(False)
    
    deltay = (df[col].max()-df[col].min())*0.08
    if p_omnibus < 0.05:
        title +=p_to_text(p_omnibus)
        if p1 < 0.05:
            h = df[col].max()+deltay
            ax.hlines(h,0,1,color="black")
            ax.text(0.5,h,p_to_text(p1),ha="center")
        if p2 < 0.05:
            h = df[col].max()+2*deltay
            ax.hlines(h,0,2,color="black")
            ax.text(1,h,p_to_text(p2),ha="center")
        if p3 < 0.05:
            h = df[col].max()+3*deltay
            ax.hlines(h,1,2,color="black")
            ax.text(1.5,h,p_to_text(p3),ha="center")
    ax.set_title(title,weight="bold",fontsize=labelsize)    



plt.figure(1,figsize=(11.69,8.27))
plt.clf()

####################GOTTA TRY SOME DISTRIBUTIONS
#general pooled distributions
ax=plt.subplot2grid((1,4),(0,0),colspan=2)
ax.set_title("Pooled distributions",fontsize=titlesize,weight="bold")
dist = CNT_jumps_pooled
c,loc,scale = pooled_fit["CNT"]
print(f"CNT\nmean\tmedian\tstd\tCV\n{dist.mean():.4f}\t{np.median(dist):.4f}\t{dist.std():.4f}\t{dist.std()/dist.mean():.4f}")
ax.hist(dist,label="CNT",density=True,alpha=alfa,bins=50,color="tab:blue")
pdf = weibull.pdf(xaxis,c=c,loc=loc,scale=scale)
ax.plot(xaxis,pdf,color="tab:blue",linewidth=linewidth)


dist = MCS_jumps_pooled
c,loc,scale = pooled_fit["MCS"]
print(f"MCS\n{dist.mean():.4f}\t{np.median(dist):.4f}\t{dist.std():.4f}\t{dist.std()/dist.mean():.4f}")
ax.hist(dist,label="MCS",density=True,alpha=alfa,bins=50,color="tab:orange")
pdf = weibull.pdf(xaxis,c=c,loc=loc,scale=scale)
ax.plot(xaxis,pdf,color="tab:orange",linewidth=linewidth)

dist = UWS_jumps_pooled
c,loc,scale = pooled_fit["UWS"]
print(f"UWS\n{dist.mean():.4f}\t{np.median(dist):.4f}\t{dist.std():.4f}\t{dist.std()/dist.mean():.4f}")
ax.hist(dist,label="UWS",density=True,alpha=alfa,bins=50,color="tab:green")
pdf = weibull.pdf(xaxis,c=c,loc=loc,scale=scale)
ax.plot(xaxis,pdf,color="tab:green",linewidth=linewidth)

ax.legend(loc="upper left",fontsize=legendsize)
ax.set_xticks((0,10,20,30,40,50,60),(0,10,20,30,40,50,60),fontsize=ticksize)
ax.set_yticks((0,0.02,0.04,0.06,0.08),(0,0.02,0.04,0.06,0.08),fontsize=ticksize)
ax.set_ylabel("Normalized count",size=labelsize)
ax.set_xlabel("Jump length",size=labelsize)
ax.spines[['top', 'right']].set_visible(False)
##distributions by state
x0,y0,width,height,space = 0.7,0.73,0.2,0.2,0.28
# left,top,vspace = 0.7,0.8,0.25
xtext,ytext = 0.7,0.8
alfasub= 0.6


#######GOTTA MAKE INSET AXIS FOR INDIVIDUAL DISTRIBUTIONS
subax = ax.inset_axes([x0, y0-0*space, width, height])
subax.text(xtext,ytext, 'CNT', transform=subax.transAxes,fontsize=subtitlesize,weight="bold")
for i in range(13):
    dist = CNT_jump_lengths[i]
    # c,loc,scale,_,_,_,_ = CNT_params[i] ##tiene 5
    c,loc,scale = df[df["state"]=="CNT"].iloc[i][["shape","loc","scale"]]
    pdf = weibull.pdf(xaxis,c=c,loc=loc,scale=scale)
    subax.plot(xaxis,pdf,color="black",alpha=alfasub,linewidth=sublinewidth)
    subax.hist(dist,label=i,density=True,alpha=alfa,bins=20)
subax.set_xlim(xmin,xmax)
subax.spines[['top', 'right']].set_visible(False)
subax.set_xticks((0,30,60),(0,30,60),fontsize=ticksize)
subax.set_yticks((0,0.05,0.1),(0,0.05,0.1),fontsize=ticksize)
# plt.legend()

subax = ax.inset_axes([x0, y0-1*space, width, height])
subax.text(xtext,ytext, 'MCS', transform=subax.transAxes,fontsize=subtitlesize,weight="bold")
for i in range(25):
    dist = MCS_jump_lengths[i]
    # c,loc,scale,_,_,_,_ = MCS_params[i] ##tiene 5
    c,loc,scale = df[df["state"]=="MCS"].iloc[i][["shape","loc","scale"]]
    pdf = weibull.pdf(xaxis,c=c,loc=loc,scale=scale)
    subax.plot(xaxis,pdf,color="black",alpha=alfasub,linewidth=sublinewidth)
    subax.hist(dist,label=i,density=True,alpha=alfa,bins=20)
subax.set_xlim(xmin,xmax)
subax.spines[['top', 'right']].set_visible(False)
subax.set_xticks((0,30,60),(0,30,60),fontsize=ticksize)
subax.set_yticks((0,0.05,0.1),(0,0.05,0.1),fontsize=ticksize)

subax = ax.inset_axes([x0, y0-2*space, width, height])
subax.text(xtext,ytext, 'UWS', transform=subax.transAxes,fontsize=subtitlesize,weight="bold")
for i in range(20):
    dist = UWS_jump_lengths[i]
    # c,loc,scale,_,_,_,_ = UWS_params[i] ##tiene 5
    c,loc,scale = df[df["state"]=="UWS"].iloc[i][["shape","loc","scale"]]
    pdf = weibull.pdf(xaxis,c=c,loc=loc,scale=scale)
    subax.plot(xaxis,pdf,color="black",alpha=alfasub,linewidth=sublinewidth)
    subax.hist(dist,label=i,density=True,alpha=alfa,bins=20)
subax.set_xlim(xmin,xmax)
subax.spines[['top', 'right']].set_visible(False)
subax.set_xticks((0,30,60),(0,30,60),fontsize=ticksize)
subax.set_yticks((0,0.05,0.1),(0,0.05,0.1),fontsize=ticksize)
#parameters



###########empirical stuff: std, skewness and GoF
ax=plt.subplot2grid((3,4),(0,2))
plot_and_compare(df,"skew","Empirical Skewness",yticks= (-0.5,0,0.5,1),ax=ax)

ax=plt.subplot2grid((3,4),(0,3))
plot_and_compare(df,"mod_skew","Weibull skewness",yticks=(-0.4,-0.2,0,0.2,0.4,0.6),ax=ax)

ax=plt.subplot2grid((3,4),(1,2))
plot_and_compare(df,"var","Variance",yticks=(25,50,75,100,125,150),ax=ax)

ax = plt.subplot2grid((3,4),(1,3))
plot_and_compare(df,"entropy","Weibull Entropy",yticks=(3,3.5,10),ax=ax)

ax=plt.subplot2grid((3,4),(2,2))
plot_and_compare(df,"wei_ks","KS distance fit",yticks=(0,0.02,0.04,0.06,0.08,0.1),ax=ax)

ax=plt.subplot2grid((3,4),(2,3))
plot_and_compare(df,"shape","Weibull shape param",yticks=(2,3,4,5,6,7),ax=ax)

plt.tight_layout()
# plt.subplots_adjust(wspace=0.4, hspace=0.6)
plt.savefig("fig2_new.svg",dpi=300,transparent=True)
plt.show()

#%% print parameters
plt.figure(2)
plt.clf()
plt.subplot(111)
for d in range(3):
    param = params[d]
    c,scale = param[:,0],param[:,2]
    plt.scatter(c,scale,label=states[d])
plt.legend()
plt.xlabel("shape 'c'")
plt.ylabel("scale")
plt.show()
    


#%% analyze outliers of MCS jump kurtosis

out_shapes = np.argsort((6/params[1][:,0]))[-4:] ## extreme shapes
outvals_shapes = np.sort((6/params[1][:,0]))[-4:]

out_scales = np.argsort((params[1][:,2]))[-4:] ## extreme scales
outvals_scales = np.sort((params[1][:,2]))[-4:]
###here the outliers are [0 4 6 18]
###########NOTE THAT THERE ARE SOME EXCLUDED
print(out_shapes,outvals_shapes)
print(out_scales,outvals_scales)


