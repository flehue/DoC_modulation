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
from scipy.special import gamma as gf
from statsmodels.stats.multitest import multipletests
import utils
import seaborn as sns
import warnings
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

CNT_stats = np.zeros((13,4)) ##mean,std,skew,kurt
MCS_stats = np.zeros((25,4)) 
UWS_stats = np.zeros((20,4)) 

CNT_params = np.zeros((13,7)) ##shape,loc,scale,skew,kurt,gof
MCS_params = np.zeros((25,7)) 
UWS_params = np.zeros((20,7))

##CNT
for i in range(13):
    # print(i)
    dist = CNT_jump_lengths[i]
    
    CNT_stats[i,:] = np.mean(dist),np.var(dist),skew(dist),kurt(dist)
    
    c,loc,scale = weibull.fit(dist,floc=0)
    #KS test
    ks_stat, ks_pval = kstest(dist, "weibull_min", args=(c, loc, scale))
    print(i,ks_pval)
    
    w_mean, w_var, w_skew, w_kurt = weibull.stats(c, moments='mvsk')
    CNT_params[i,:] = c,loc,scale,scale*w_mean, scale**2*w_var, w_skew, ks_stat
    # print(ks_stat)

##MCS
for i in range(25):
    dist = MCS_jump_lengths[i]
    
    MCS_stats[i,:] = np.mean(dist),np.var(dist),skew(dist),kurt(dist)
    
    c,loc,scale = weibull.fit(dist,floc=0)
    #KS test
    ks_stat, ks_pval = kstest(dist, "weibull_min", args=(c, loc, scale))
    print(i,ks_pval)
    
    w_mean, w_var, w_skew, w_kurt = weibull.stats(c, moments='mvsk')
    MCS_params[i,:] = c,loc,scale,scale*w_mean, scale**2*w_var, w_skew, ks_stat

##UWS
for i in range(20):
    dist = UWS_jump_lengths[i]
    
    UWS_stats[i,:] = np.mean(dist),np.var(dist),skew(dist),kurt(dist)
    
    c,loc,scale = weibull.fit(dist,floc=0)
    #KS test
    ks_stat, ks_pval = kstest(dist, "weibull_min", args=(c, loc, scale))
    print(i,ks_pval)
    
    w_mean, w_var, w_skew, w_kurt = weibull.stats(c, moments='mvsk')
    UWS_params[i,:] = c,loc,scale,scale*w_mean, scale**2*w_var, w_skew, ks_stat

#%% general distribution parameters

##mainly for plotting but not only that
xmin,xmax = 0,np.max(np.concatenate([np.concatenate(CNT_jump_lengths),np.concatenate(MCS_jump_lengths),np.concatenate(UWS_jump_lengths)]))



stats  = [CNT_stats,MCS_stats,UWS_stats]
params = [CNT_params,MCS_params,UWS_params] #inicializamos parametros de levy

pooled_fit = {}
for s,dist in enumerate([CNT_jumps_pooled,MCS_jumps_pooled,UWS_jumps_pooled]):
    c,loc,scale = weibull.fit(dist,floc=0)
    pooled_fit[states[s]] = (c,loc,scale)


# halt
#%% autocorrelaciones por individuo y cuando caen debajo de 0.5



def p_and_d(dist1,dist2,dist3):
    p1 = ttest_ind(dist1,dist2)[1]
    p2 = ttest_ind(dist1,dist3)[1]
    p3 = ttest_ind(dist2,dist3)[1]
    p1,p2,p3 = multipletests((p1,p2,p3),method="fdr_bh")[1]
    
    d1 = utils.cohen_d(dist1,dist2)
    d2 = utils.cohen_d(dist1,dist3)
    d3 = utils.cohen_d(dist2,dist3)
    
    return p1,p2,p3,d1,d2,d3
    

#SKEWNESS EMPIRICAL
p1,p2,p3,d1,d2,d3 = p_and_d(stats[0][:,2],stats[1][:,2],stats[2][:,2])
print(f"EMPIRICAL SKEWNESS\np-val (CNT,MCS),(CNT,UWS),(MCS,UWS): ({p1},{p2},{p3})")
##D de cohen
print(f"cohen-d (CNT,MCS),(CNT,UWS),(MCS,UWS): ({d1:.3f},{d2:.3f},{d3:.3f})")

#SKEWNESS GAMMA
p1,p2,p3,d1,d2,d3 = p_and_d(params[0][:,3],params[1][:,3],params[2][:,3])
print(f"GAMMA SKEWNESS\np-val (CNT,MCS),(CNT,UWS),(MCS,UWS): {p1},{p2},{p3}")
print(f"cohen-d (CNT,MCS),(CNT,UWS),(MCS,UWS): ({d1:.3f},{d2:.3f},{d3:.3f})")

#KURTOSIS EMPIRICAL
p1,p2,p3,d1,d2,d3 = p_and_d(stats[0][:,3],stats[1][:,3],stats[2][:,3])
print(f"\nEMPIRICAL KURTOSIS\np-val (CNT,MCS),(CNT,UWS),(MCS,UWS): ({p1},{p2},{p3})")
print(f"cohen-d (CNT,MCS),(CNT,UWS),(MCS,UWS): ({d1:.3f},{d2:.3f},{d3:.3f})")

#KURTOSIS GAMMA
p1,p2,p3,d1,d2,d3 = p_and_d(params[0][:,4],params[1][:,4],params[2][:,4])
print(f"GAMMA KURTOSIS\np-val (CNT,MCS),(CNT,UWS),(MCS,UWS): ({p1},{p2},{p3})")
print(f"cohen-d (CNT,MCS),(CNT,UWS),(MCS,UWS): ({d1:.3f},{d2:.3f},{d3:.3f})")

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


plt.figure(1)
plt.clf()

####################GOTTA TRY SOME DISTRIBUTIONS
#general pooled distributions
ax=plt.subplot2grid((1,4),(0,0),colspan=2)
plt.title("Pooled distributions",fontsize=titlesize,weight="bold")
dist = CNT_jumps_pooled
c,loc,scale = pooled_fit["CNT"]
print(f"CNT\nmean\tmedian\tstd\tCV\n{dist.mean():.4f}\t{np.median(dist):.4f}\t{dist.std():.4f}\t{dist.std()/dist.mean():.4f}")
plt.hist(dist,label="CNT",density=True,alpha=alfa,bins=50,color="tab:blue")
pdf = weibull.pdf(xaxis,c=c,loc=loc,scale=scale)
plt.plot(xaxis,pdf,color="tab:blue",linewidth=linewidth)


dist = MCS_jumps_pooled
c,loc,scale = pooled_fit["MCS"]
print(f"MCS\n{dist.mean():.4f}\t{np.median(dist):.4f}\t{dist.std():.4f}\t{dist.std()/dist.mean():.4f}")
plt.hist(dist,label="MCS",density=True,alpha=alfa,bins=50,color="tab:orange")
pdf = weibull.pdf(xaxis,c=c,loc=loc,scale=scale)
plt.plot(xaxis,pdf,color="tab:orange",linewidth=linewidth)

dist = UWS_jumps_pooled
c,loc,scale = pooled_fit["UWS"]
print(f"UWS\n{dist.mean():.4f}\t{np.median(dist):.4f}\t{dist.std():.4f}\t{dist.std()/dist.mean():.4f}")
plt.hist(dist,label="UWS",density=True,alpha=alfa,bins=50,color="tab:green")
pdf = weibull.pdf(xaxis,c=c,loc=loc,scale=scale)
plt.plot(xaxis,pdf,color="tab:green",linewidth=linewidth)

plt.legend(loc="upper left",fontsize=legendsize)
plt.xticks((0,10,20,30,40,50,60),(0,10,20,30,40,50,60),fontsize=ticksize)
plt.yticks((0,0.02,0.04,0.06,0.08),(0,0.02,0.04,0.06,0.08),fontsize=ticksize)
plt.ylabel("Normalized count",size=labelsize)
plt.xlabel("Jump length",size=labelsize)
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
    c,loc,scale,_,_,_,_ = CNT_params[i] ##tiene 5
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
    c,loc,scale,_,_,_,_ = MCS_params[i] ##tiene 5
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
    c,loc,scale,_,_,_,_ = UWS_params[i] ##tiene 5
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
plt.title("Empirical Skewness",weight="bold")
toplot = [sta[:,2] for sta in stats]
sns.swarmplot(toplot,color="black")
sns.boxplot(toplot)
plt.xticks([0,1,2],["CNT","MCS","UWS"],fontsize=ticksize)
plt.yticks((-0.5,0,0.5,1),(-0.5,0,0.5,1),fontsize=ticksize)
ax.spines[['top', 'right']].set_visible(False)

ax=plt.subplot2grid((3,4),(1,2))
plt.title("Variance",weight="bold")
toplot = [sta[:,1] for sta in stats]
sns.swarmplot(toplot,color="black")
sns.boxplot(toplot)
plt.xticks([0,1,2],["CNT","MCS","UWS"],fontsize=ticksize)
plt.yticks((25,50,75,100,125,150),(25,50,75,100,125,150),fontsize=ticksize)
ax.spines[['top', 'right']].set_visible(False)

ax=plt.subplot2grid((3,4),(2,2))
plt.title("Weibull fit KS-distance",weight="bold")
toplot = [param[:,-1] for param in params]
sns.swarmplot(toplot,color="black")
sns.boxplot(toplot)
plt.xticks([0,1,2],["CNT","MCS","UWS"],fontsize=ticksize)
plt.yticks((0,0.02,0.04,0.06,0.08,0.1),(0,0.02,0.04,0.06,0.08,0.1),fontsize=ticksize)
ax.spines[['top', 'right']].set_visible(False)


#############fit stuff:
ax=plt.subplot2grid((3,4),(0,3))
plt.title("Weibull Skewness",weight="bold")
toplot = [param[:,5] for param in params]
sns.swarmplot(toplot,color="black")
sns.boxplot(toplot)
plt.xticks([0,1,2],["CNT","MCS","UWS"],fontsize=ticksize)
plt.yticks((-0.4,-0.2,0,0.2,0.4,0.6),(-0.4,-0.2,0,0.2,0.4,0.6),fontsize=ticksize)
ax.spines[['top', 'right']].set_visible(False)

ax=plt.subplot2grid((3,4),(1,3))
plt.title("Weibull Var",weight="bold")
toplot = [param[:,4] for param in params]
sns.swarmplot(toplot,color="black")
sns.boxplot(toplot)
plt.xticks([0,1,2],["CNT","MCS","UWS"],fontsize=ticksize)
plt.yticks((25,50,75,100,125,150),(25,50,75,100,125,150),fontsize=ticksize)
ax.spines[['top', 'right']].set_visible(False)


ax=plt.subplot2grid((3,4),(2,3))
plt.title("Weibull shape parameter",weight="bold")
toplot = [param[:,0] for param in params]
sns.swarmplot(toplot,color="black")
sns.boxplot(toplot)
plt.xticks([0,1,2],["CNT","MCS","UWS"],fontsize=ticksize)
plt.yticks((2,3,4,5,6,7),(2,3,4,5,6,7),fontsize=ticksize)
ax.spines[['top', 'right']].set_visible(False)

# plt.tight_layout()
# plt.subplots_adjust(wspace=0.4, hspace=0.6)
# plt.savefig("fig2.svg",dpi=300,transparent=True)
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


