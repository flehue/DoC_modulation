# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 19:46:29 2024

@author: flehu
"""
import sys
sys.path.append("../analyze_empirical")
sys.path.append("../")
sys.path.append("../../")
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.special import rel_entr as kl #relative entropy = KL distance
from scipy.stats import linregress,spearmanr,mannwhitneyu
from scipy.stats import kstest
import HMA
import pandas as pd
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import FancyBboxPatch
import bct
import matplotlib as mpl
from plot_violins import violin_plot#(ax, data, color_names, alpha_violin = 1, s_box = 20, s_ind = 20,inds= None)
# from scipy.stats import linregress

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


def bar_subax(obs,y, cmap,yticks=None,ax=None,color=None,hidexticks=True):
    my_cmap = plt.get_cmap(cmap)
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y)) ##colorcode
    ax.bar(range(len(obs)), obs, color=my_cmap(rescale(y)))
    
    if yticks:
        ax.set_yticks(yticks[0],yticks[1],fontsize=ticksize)
    ax.legend(fontsize=legendsize,loc="upper center")
    ax.spines[['top', 'right']].set_visible(False)
    if hidexticks:
        ax.tick_params(labelbottom=False)
        ax.xaxis.set_tick_params(length=0)
        
def plot_significance_line(x1, x2, y, p_value, 
                           x_text=None, cliff=None, ax= None,
                           line_width=1.5, text_offset=0.002,linelength=0.001):

    # Draw horizontal line
    ax.plot([x1, x2], [y, y], color='black', lw=line_width)
    # Draw vertical ticks
    ax.plot([x1, x1], [y - linelength, y], color='black', lw=line_width)
    ax.plot([x2, x2], [y - linelength, y], color='black', lw=line_width)

    # Text
    if x_text is None:
        x_text = (x1 + x2) / 2
    
    if p_value < 0.001:
        text = "***"
    elif p_value <0.01:
        text = "**"
    elif p_value < 0.05:
        text = "*"
    else:
        text = f"p={p_value:.3f}"
    ax.text(x_text, y + text_offset, text, ha='center', va='bottom', fontsize=13)

def cohen_d(low,high):
    mean_high = np.mean(high)
    mean_low  = np.mean(low)
    
    var_high = np.var(high)
    var_low  = np.var(low)
    
    n_high = len(high)
    n_low  = len(low)
    # pooled SD per node
    std_pooled = np.sqrt(((n_high - 1) * var_high + (n_low - 1) * var_low) / (n_high + n_low - 2))
    return (mean_high - mean_low) / std_pooled
    

nodes= range(90)
#%%cargar
##cada llave es (G,seed) = (corrs,eucs,jumps_high,counts_high)
with open('../../../fastDMF/output/sweep_summary_seeds0_19.pickle', 'rb') as f:
    data = pickle.load(f)
# del f
#%%

with open('../../../fastDMF/empirical_truth/jump_distributions_dict_filt_ALLDIM.pickle', 'rb') as f:
    jump_dists_all = pickle.load(f)
with open('../../../fastDMF/output/broken_nodes_resweep_analysis_seeds0-19.pickle', 'rb') as f:
    broken_data = pickle.load(f)
del f
# halt
emp_occs_all = np.loadtxt('../../../fastDMF/empirical_truth/occupations_3clusters_alldim_euclidean.txt')
AALlabels = pd.read_csv("../../../sorted_AAL_labels.txt")["label"].values #nombre de las areas
struct = np.loadtxt("../../../structural_Deco_AAL.txt")
#%% join some data by seed
states = ["CNT","MCS","UWS"]

##data
Gs, seeds = [],[]
by_seed = {}
for key in data.keys():
    G,seed = key
    if G not in Gs:
        Gs.append(G)
    if seed not in seeds:
        by_seed[seed] = {}
        seeds.append(seed)
    by_seed[seed][G] = data[(G,seed)]
Gs,seeds = [np.sort(Gs),np.sort(seeds)]
Gs = Gs[Gs<2.6]
nseeds = len(seeds)

####fill the mean occupation euclidean and corr vector
occs = np.zeros((len(Gs),len(("c1","c2","c3"))))
euc = np.zeros((len(Gs),len(states)))
corrs = np.zeros((len(Gs),len(states)))
jumps = np.zeros((len(Gs),nseeds,191)) ###191 es el numero de saltos por simulacion


for seed in seeds:
    dic = by_seed[seed]
    for g,G in enumerate(Gs):
        corrs[g,:] += dic[G][0]
        euc[g,:] += dic[G][1]
        jumps[g,seed,:] =  dic[G][2]
        occs[g,:] += dic[G][3]
        
        
occs /= len(seeds)
euc /= len(seeds)
corrs /= len(seeds)

#sweep data KL occs
klCNT,klMCS,klUWS = [np.array([kl(occs[g,:],emp_occs_all[s]).sum() for g in range(len(Gs))]) for s in range(3)]
klGoCNT,klGoMCS,klGoUWS = [Gs[np.argmin(a)] for a in (klCNT,klMCS,klUWS)]
print("kl optima at",klGoCNT,klGoMCS,klGoUWS)
klminCNT,klminMCS,klminUWS = [np.min(a) for a in (klCNT,klMCS,klUWS)]

###sweep data KS saltos

ksCNT = np.array([np.array([kstest(jumps[g,s,:],jump_dists_all["CNT"])[0] for g in range(len(Gs))]) for s in range(nseeds)]).mean(axis=0)
ksMCS = np.array([np.array([kstest(jumps[g,s,:],jump_dists_all["MCS"])[0] for g in range(len(Gs))]) for s in range(nseeds)]).mean(axis=0)
ksUWS = np.array([np.array([kstest(jumps[g,s,:],jump_dists_all["UWS"])[0] for g in range(len(Gs))]) for s in range(nseeds)]).mean(axis=0)
ksGoCNT,ksGoMCS,ksGoUWS = [Gs[np.argmin(a)] for a in (ksCNT,ksMCS,ksUWS)]
print("ks optima at",ksGoCNT,ksGoMCS,ksGoUWS)
ksminCNT,ksminMCS,ksminUWS = [np.min(a) for a in (ksCNT,ksMCS,ksUWS)]




#%%broken data


##this is only occupations
occs_broken = np.zeros((len(nodes),3))
jumps_broken = np.zeros((len(nodes),191))

G_CNT_kl = 2.48
G_CNT_ks = 2.52
for key in broken_data.keys():
    G,node,seed = key
    if G == G_CNT_kl:
        occs_broken[node,:] = broken_data[key][3]
    if G == G_CNT_ks:
        jumps_broken[node,:] = broken_data[key][2]
        
klsCNT_broken = np.array([kl(occs_broken[n,:],emp_occs_all[0]).sum() for n in range(len(nodes))]) ##todos los kl del sweep
klsMCS_broken = np.array([kl(occs_broken[n,:],emp_occs_all[1]).sum() for n in range(len(nodes))]) ##todos los kl del sweep
klsUWS_broken = np.array([kl(occs_broken[n,:],emp_occs_all[2]).sum() for n in range(len(nodes))]) ##todos los kl del sweep



ksCNT_broken = np.array([kstest(jumps_broken[n,:],jump_dists_all["CNT"])[0] for n in range(len(nodes))])
ksMCS_broken = np.array([kstest(jumps_broken[n,:],jump_dists_all["MCS"])[0] for n in range(len(nodes))])
ksUWS_broken = np.array([kstest(jumps_broken[n,:],jump_dists_all["UWS"])[0] for n in range(len(nodes))])


#%%data for scatter

forces = struct.sum(axis=1)
Clus_num,Clus_size,H_all = HMA.Functional_HP(struct)
Hin,Hse = HMA.Balance(struct, Clus_num, Clus_size)
preHin_node,preHse_node = HMA.nodal_measures(struct, Clus_num, Clus_size)
Hin_node = preHin_node/preHin_node.max()
Hse_node= preHse_node/preHse_node.max()


betweenness_cent = bct.centrality.betweenness_wei(struct)

#%% some eyeballing
colors = ("tab:blue","tab:orange","tab:green")
##experiment using two levels of integration
#violin_plot(ax, data, color_names, alpha_violin = 1, s_box = 20, s_ind = 20,inds= None)

alfa = 0.5
scatteralfa = 0.5
titlesize = 18
subtitlesize=12
labelsize = 14
ticksize = 11
legendsize=10
height=4/13
std = 0.05;xpos1 = 0;xpos2 = 2;tickangle=0
epsilon=1e-2


def create_filter_iqr(x,k=3):
    return x > np.quantile(x, 0.75) + 1.5 * (np.quantile(x, 0.75) - np.quantile(x, 0.25))

def create_filter_std(x,k=2):
    return np.abs(x-np.mean(x)) > k*np.std(x)

def create_filter_dumb(x,k=0.4):
    return x > k

def create_filter_mad(x, k=2):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return (x - med) > k * mad

def create_violin(obs,filtered,ax=None):
    color_names = ("tab:red","tab:blue")
    violin_plot(ax,[obs[filtered],obs[~filtered]],color_names = color_names,alpha_violin=0.7)
    
def stripplot_Hin_force(ax):
    val = forces[~outmask]/np.max(forces)
    ax.scatter(xpos1+np.random.normal(scale=std,size=(len(val))),val,color="grey",alpha=alfa-0.1)
    val = forces[outmask]/np.max(forces)
    ax.scatter(xpos1+np.random.normal(scale=std,size=(len(val))),val,marker="x",color="crimson")

    val = Hin_node[outmask]/np.max(Hin_node)
    ax.scatter(xpos2+np.random.normal(scale=std,size=(len(val))),val,marker="x",color="crimson",alpha=1,label="High Hin")
    val = Hin_node[~outmask]/np.max(Hin_node)
    ax.scatter(xpos2+np.random.normal(scale=std,size=(len(val))),val,color="grey",alpha=alfa+0.1,label="Low Hin")

    ax.hlines(k,xpos2-0.2,xpos2+0.2,color="crimson",linestyle="dashed")
    ###
    ax.set_xticks([xpos1,xpos2],("Node strength","Hin node"),rotation=tickangle,fontsize=ticksize)
    ax.set_xlim(-0.4,xpos2+0.4)
    ax.set_yticks((0,0.5,1),(0,0.5,1),fontsize=ticksize)
    ax.set_ylabel("Normalized value",fontsize=labelsize)
    ax.legend(fontsize=legendsize,loc="upper center")
    
    
def recreate_the_two(x, y, minn,outmask,
                     xticks=None,yticks=None,colors=("gray", "tab:blue"),
                     ax=None,loc=None,alpha=alfa,p=None,d=None):

    # --- Subaxis (clean data, no legend) ---
    subax1 = ax.inset_axes((0,0,2/3-epsilon,1))
    xclean = x[~outmask];yclean = y[~outmask]
    xdirt =  x[outmask] ;ydirt  = y[outmask]
    slope, intercept, r_value, p_value, std_err = linregress(xclean, yclean)
    subax1.scatter(xclean, yclean, color=colors[0],alpha=alpha)
    # ax.plot(np.sort(xclean), slope * np.sort(xclean) + intercept, color=colors[0],linestyle="dashed")

    # --- Main axis (all data) ---
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    subax1.scatter(xdirt, ydirt, color=colors[1],marker="x")
    subax1.plot(np.sort(x), slope * np.sort(x) + intercept, color=colors[1],label=f"r={r_value:.3f}")
    ###baseline
    subax1.hlines(minn,0,1,color="tab:red",linestyle="dashed",alpha=0.7,label="baseline")
    subax1.legend(loc=loc,fontsize=legendsize)
    
    
    
    subax2 = ax.inset_axes((2/3+epsilon,0,1/3-epsilon,1))
    bp1=subax2.boxplot(yclean, positions=(0,), widths=0.6,patch_artist=True, showfliers=False)
    subax2.scatter(0+np.random.normal(scale=std,size=len(yclean)),yclean,color="gray",alpha=alfa)
    
    subax2.scatter(1+np.random.normal(scale=std,size=len(ydirt)),ydirt,color=colors[1],marker="x",alpha=1)
    bp2=subax2.boxplot(ydirt, positions=(1,), widths=0.6,patch_artist=True, showfliers=False)
    for patch in list(bp1['boxes'])+list(bp2['boxes']):
        patch.set_facecolor("none")  # transparent
        patch.set_edgecolor("black")
    if yticks:
        ticklabels = [f"{val:.2f}" for val in yticks]
        subax1.set_yticks(yticks,ticklabels,fontsize=ticksize)
    subax2.set_yticks((yticks))
    subax2.set_yticklabels(len(yticks)*[""])
    if not xticks:
        subax1.set_xticks(())
        # subax1.xaxis.set_tick_params(length=0)
        subax2.set_xticks(())
        # subax2.xaxis.set_tick_params(length=0)
    else:
        subax1.set_xticks((0,0.5,1),(0,0.5,1),fontsize=ticksize)
        subax1.set_xlabel("Hin node",fontsize=labelsize)
        subax2.set_xticks((0,1),("Low\nHin", "High\nHin"),fontsize=labelsize)
    text = ""
    if p and d:
        if p < 0.001:
            add = "***"
        elif p < 0.01:
            add = "**"
        elif p < 0.05:
            add = "*"
        text+=f"D({d:.2f})"  + add
    subax2.text(0.5,0.7, text,fontsize=legendsize,
            ha='center', va='center',transform=subax2.transAxes,rotation=90)
    
    ##ylims
    scale = np.max(y)-np.min(y)
    lower = np.min((np.min(y),minn))-scale*0.03
    subax1.set_ylim(bottom=lower)
    subax2.set_ylim(bottom=lower)
    

    
values = Hin_node/Hin_node.max()
def plot_three_groups(vals,mins,outmask,yticks=None,colors=colors,ax=None):
    v1,v2,v3 = vals
    m1,m2,m3 = mins
    if yticks:
        yt1,yt2,yt3 = yticks
    else:
        yt1,yt2,yt3 = None,None,None
    
    ####comparisons
    l1,h1 = v1[~outmask],v1[outmask]
    l2,h2 = v2[~outmask],v2[outmask]
    l3,h3 = v3[~outmask],v3[outmask]
    _,p1, = mannwhitneyu(l1,h1)
    _,p2, = mannwhitneyu(l2,h2)
    _,p3, = mannwhitneyu(l3,h3)
    p1,p2,p3 = multipletests((p1,p2,p3), method='fdr_bh')[1]
    
    ##cohen ds
    d1 = cohen_d(l1,h1)
    d2 = cohen_d(l2,h2)
    d3 = cohen_d(l3,h3)
    
    ax1 = ax.inset_axes((0,2/3,1,height))
    recreate_the_two(values,v1,m1,outmask,colors = ("gray",colors[0]),
                      yticks=yt1,ax=ax1,p=p1,d=d1)
    
    ax2 = ax.inset_axes((0,1/3,1,height))
    recreate_the_two(values,v2,m2,outmask,colors = ("gray",colors[1]),
                      yticks=yt2,ax=ax2,p=p2,d=d2)
    # ax2.hlines(klminMCS,0,1,color="red",linestyle="dashed")
    
    ax3 = ax.inset_axes((0,0,1,height))
    recreate_the_two(values,v3,m3,outmask,colors = ("gray",colors[2]),
                      yticks=yt3,ax=ax3,xticks=True,p=p3,d=d3)
    # ax3.hlines(klminUWS,0,1,color="red",linestyle="dashed")
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines[:].set_visible(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines[:].set_visible(False)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines[:].set_visible(False)
    
    
# def res_res(y,x1,x2):
    


slope, intercept, r_value, p_value, std_err = linregress(forces, Hin_node) ##here x is going to be force
residuals = Hin_node-(slope * forces + intercept) ##integration - what force predicts

k = 0.4
outmask = create_filter_dumb(Hin_node,k=k)
# outmask = create_filter_std(Hin_node,k=2)

    #%% figure
fig=plt.figure(99)
plt.clf()

##baseline occupations
ax = plt.subplot2grid((3,3),(0,0))
ax.set_title("Distance to Empirical Occupancies",fontsize=labelsize+1)
ax.plot(Gs,klCNT,color=colors[0])
ax.plot(Gs,klMCS,color=colors[1])
ax.plot(Gs,klUWS,color=colors[2])
ax.vlines(klGoCNT,-0.03,0.2,linestyle="dashed",color=colors[0],label=f"CNTo={klGoCNT:.2f}")
ax.vlines(klGoMCS,-0.03,0.2,linestyle="dashed",color=colors[1],label=f"MCSo={klGoMCS:.2f}")
ax.vlines(klGoUWS,-0.03,0.2,linestyle="dashed",color=colors[2],label=f"UWSo={klGoUWS:.2f}")
ax.set_xticks((1.5,1.75,2,2.25,2.5),(1.5,1.75,2,2.25,2.5),fontsize=ticksize)
ax.set_yticks((0,0.5,1),(0,0.5,1),fontsize=ticksize);ax.set_ylim([-0.05,1])
ax.set_xlabel(r"Coupling $G$",fontsize=labelsize)
ax.set_ylabel("KL distance",fontsize=labelsize)
#legend
leg=ax.legend(fontsize=legendsize, loc="upper left",framealpha=1)
leg.get_frame().set_facecolor("white")   # solid background
leg.get_frame().set_edgecolor("black")   # border color
leg.get_frame().set_linewidth(1.2)  


####baseline jumps
ax = plt.subplot2grid((3,3),(0,1))
ax.set_title("Distance to Empirical Jumps",fontsize=labelsize+1)
ax.plot(Gs,ksCNT,color=colors[0])
ax.plot(Gs,ksMCS,color=colors[1])
ax.plot(Gs,ksUWS,color=colors[2])
ax.vlines(ksGoCNT,0.1,0.2,linestyle="dashed",color=colors[0],label=f"CNTo={ksGoCNT:.2f}")
ax.vlines(ksGoMCS,0.1,0.2,linestyle="dashed",color=colors[1],label=f"MCSo={ksGoMCS:.2f}")
ax.vlines(ksGoUWS,0.1,0.2,linestyle="dashed",color=colors[2],label=f"UWSo={ksGoUWS:.2f}")
ax.set_xticks((1.5,1.75,2,2.25,2.5),(1.5,1.75,2,2.25,2.5),fontsize=ticksize)
ax.set_yticks((0.1,0.3,0.5),(0.1,0.3,0.5),fontsize=ticksize)
ax.set_xlabel(r"Coupling $G$",fontsize=labelsize)
ax.set_ylabel("KS distance",fontsize=labelsize)
leg=ax.legend(fontsize=legendsize, loc="upper left",framealpha=1)
leg.get_frame().set_facecolor("white")   # solid background
leg.get_frame().set_edgecolor("black")   # border color
leg.get_frame().set_linewidth(1.2)  


xticks = (0,0.5,1)


############integration vs node strength
ax = plt.subplot2grid((3,6),(0,4))
stripplot_Hin_force(ax)


########break things occupation
ax = plt.subplot2grid((3,3),(1,0),rowspan=2)
ax.set_title("Fit to occupations excluding areas",fontsize=labelsize+1)
plot_three_groups((klsCNT_broken,klsMCS_broken,klsUWS_broken),mins = (klminCNT,klminMCS,klminUWS),
                  outmask=outmask,yticks = ((0,0.04,0.08),(0,0.08,0.16),(0.1,0.25,0.4)),
                  colors = colors,ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.spines[:].set_visible(False)
# ax.set_ylabel("Distance to Jumps (KS)",fontsize=labelsize,labelpad=50)
for i, label in enumerate(['CNT', 'MCS', 'UWS']):
    ax.text(-0.11, (5/6,3/6,1/6)[i], label, fontsize=labelsize,
            ha='right', va='center', transform=ax.transData,rotation=90)
ax.spines[['top','left','bottom','right']].set_visible(False)

#########break things jump
ax = plt.subplot2grid((3,3),(1,1),rowspan=2)
ax.set_title("Fit to jumps excluding areas",fontsize=labelsize+1)
plot_three_groups((ksCNT_broken,ksMCS_broken,ksUWS_broken),mins = (ksminCNT,ksminMCS,ksminUWS),
                  outmask=outmask,yticks = ((0.1,0.16),(0.1,0.16),(0.1,0.2)),
                  colors = colors,ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.spines[:].set_visible(False)
for i, label in enumerate(['CNT', 'MCS', 'UWS']):
    ax.text(-0.11, (5/6,3/6,1/6)[i], label, fontsize=labelsize,
            ha='right', va='center', transform=ax.transData,rotation=90)
ax.spines[['top','left','bottom','right']].set_visible(False)

####brains
ax = plt.subplot2grid((3,6),(1,4),rowspan=2)
ax.set_title("Excluded areas KL",fontsize=labelsize+1)
ax.set_xticks([])
ax.set_yticks([])
ax.spines[:].set_visible(False)
for i, label in enumerate(['CNT', 'MCS', 'UWS']):
    ax.text(-0.11, (5/6,3/6,1/6)[i], label, fontsize=labelsize,
            ha='right', va='center', transform=ax.transData,rotation=90)
ax.spines[['top','left','bottom','right']].set_visible(False)

ax = plt.subplot2grid((3,6),(1,5),rowspan=2)
ax.set_title("Excluded areas KS",fontsize=labelsize+1)
ax.set_xticks([])
ax.set_yticks([])
ax.spines[:].set_visible(False)
# for i, label in enumerate(['CNT', 'MCS', 'UWS']):
#     ax.text(-0.1, (5/6,3/6,1/6)[i], label, fontsize=labelsize,
#             ha='right', va='center', transform=ax.transData,rotation=90)
ax.spines[['top','left','bottom','right']].set_visible(False)




plt.tight_layout()
# plt.savefig("prefig4_better.svg",dpi=300,transparent=True)
plt.show()

#%%
# norm = lambda x: (x-x.min())/(x.max()-x.min())
# to_save = {"Hin_node":norm(Hin_node),
#            "klsCNT_broken":norm(klsCNT_broken),
#            "klsMCS_broken":norm(klsMCS_broken),
#            "klsUWS_broken":norm(klsUWS_broken),
#            "ksCNT_broken":norm(ksCNT_broken),
#            "ksMCS_broken":norm(ksMCS_broken),
#            "ksUWS_broken":norm(ksUWS_broken)}
# np.savez_compressed("maps2brain.npz",**to_save)



#%% neurosynth

from scipy.stats import pearsonr,spearmanr
# from brainsmash.mapgen.base import Base

def reorder(mapp):
    left = mapp[::2]
    right = mapp[1::2][::-1]
    return np.concatenate((left,right))


neurosynth_folder="../"
parcellated_data = np.load(neurosynth_folder+'parcellated_data.npy')[:,:90]#[my_terms,:]
cognitive_terms = np.load(neurosynth_folder+'cognitive_terms.npy')#[my_terms]
# print(cognitive_terms)
for c,term in enumerate(cognitive_terms): #71 es consciousness
    mapp = reorder(parcellated_data[c])
    # r,p = pearsonr(klsCNT_broken,mapp)
    r,p = pearsonr(Hse_node,mapp)
    if p < 0.001:
        print(f"r={r:.3f},p={p}",term)


#%%surveying the Hin node and node strength topographic distribution        

outmask = create_filter_dumb(Hin_node,k=k)


###rsn
names = ("Vis","ES","Aud","SM","DM","EC")
rsn = np.loadtxt(neurosynth_folder+"RSN_AAL_Enzo.txt")
for n,name in enumerate(names): #########SE NOTA QUE COINCIDE CON LA DM network
    if name == "DM":
        print(name,":", AALlabels[rsn[:,n]==1])
print("\noutliers: ",AALlabels[outmask])


full_df = pd.DataFrame()
full_df["label"] = AALlabels
full_df["Hin_node"] = Hin_node/Hin_node.max()
full_df["force"] = forces/forces.max()
full_df["klCNT_broken"] = klsCNT_broken/klsCNT_broken.max()

##let's look for those exceptions





force_th,Hin_th = 0.75,0.4
mask =  (full_df["force"] > 0.7) & (full_df["Hin_node"]<= 0.4) 
print("\nhigh force but low Hin:",full_df[mask])

plt.figure(4)
plt.clf()
plt.subplot(221)
plt.scatter(full_df["force"],full_df["Hin_node"],alpha=0.7,marker="o")
plt.fill_between(np.arange(0.7,1.01,0.01),0,0.4,color="gray",alpha=0.4)
for a,area in enumerate(AALlabels):
    if mask[a]:
        # plt.text(float(full_df["force"].iloc[a]),
        #           float(full_df["Hin_node"].iloc[a]),
        #           area, rotation=np.random.uniform(low=0,high=90))
        plt.scatter([float(full_df["force"].iloc[a])],
                  [float(full_df["Hin_node"].iloc[a])],
                  label=area)
    if "Thalamus" in area:
        plt.scatter([float(full_df["force"].iloc[a])],
                  [float(full_df["Hin_node"].iloc[a])],
                  label=area,color="black")
plt.legend()

plt.xlabel("force")
plt.ylabel("Hin node")
plt.show()











