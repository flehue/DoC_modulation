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
from scipy.stats import linregress,spearmanr
from scipy.stats import kstest
import HMA
import pandas as pd
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import bct
import matplotlib as mpl
from plot_violins import violin_plot#(ax, data, color_names, alpha_violin = 1, s_box = 20, s_ind = 20,inds= None)
# from scipy.stats import linregress

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

def scatter_subax(x,y,yticks=None,ax=None,color=None,hidexticks=True):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line = slope * x + intercept
    #spearman
    rho,pval = spearmanr(x,y)
    ax.plot(x,line,linestyle=linestyle,color=color,alpha=1,label=f"ρ={rho:.3f}")
    ax.scatter(x,y,alpha=scatteralfa,color=color)
    if yticks:
        ax.set_yticks(yticks[0],yticks[1],fontsize=ticksize)
    ax.legend(fontsize=legendsize,loc="upper center")
    ax.spines[['top', 'right']].set_visible(False)
    if hidexticks:
        ax.tick_params(labelbottom=False)
        ax.xaxis.set_tick_params(length=0)


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

nodes= range(90)
#%%cargar
##cada llave es (G,seed) = (corrs,eucs,jumps_high,counts_high)
with open('../../../fastDMF/output/sweep_summary_seeds0_19.pickle', 'rb') as f:
    data = pickle.load(f)
# del f
#%%

with open('../../../fastDMF/empirical_truth/jump_distributions_dict_filt_ALLDIM.pickle', 'rb') as f:
    jump_dists_all = pickle.load(f)
    
####FORMATO output_dic[(G,node,seed)] = (corrs,eucs,jumps_high,counts_high)
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
legendsize=11
height=4/13


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
    
    
def recreate_the_two(x, y, outmask,
                     yticks=None,colors=("gray", "tab:blue"),
                     ax=None,loc=None,alpha=alfa):
    from matplotlib.lines import Line2D

    # --- Subaxis (clean data, no legend) ---
    # ax1 = ax.inset_axes((0.18, 0.8, 0.2, 0.15))
    xclean = x[~outmask]
    yclean = y[~outmask]
    slope, intercept, r_value, p_value, std_err = linregress(xclean, yclean)
    ax.scatter(xclean, yclean, color=colors[0],alpha=alpha)
    ax.plot(np.sort(xclean), slope * np.sort(xclean) + intercept, color=colors[0],linestyle="dashed")

    # --- Main axis (all data) ---
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    ax.scatter(x[outmask], y[outmask], color=colors[1],marker="x")
    ax.plot(np.sort(x), slope * np.sort(x) + intercept, color=colors[1],linestyle="dashed")
    ax.xaxis.set_tick_params(length=0)
    ax.set_xticks(())
    if yticks:
        ticklabels = [f"{val:.2f}" for val in yticks]
        ax.set_yticks(yticks,ticklabels,fontsize=ticksize)
        
    # --- Legend (color-matched) ---
    labels = [
        f"r={linregress(xclean, yclean).rvalue:.3f}   no outlier ",
        f"r={r_value:.3f}   all"
    ]
    handles = [
        Line2D([0], [0], color=colors[0], lw=2),  # clean (red)
        Line2D([0], [0], color=colors[1], lw=2)   # all (blue)
    ]
    ax.legend(handles, labels,loc=loc,fontsize=legendsize)
    
# def res_res(y,x1,x2):
    


slope, intercept, r_value, p_value, std_err = linregress(forces, Hin_node) ##here x is going to be force
residuals = Hin_node-(slope * forces + intercept) ##integration - what force predicts
outmask = create_filter_dumb(Hin_node,k=0.4)
# outmask = create_filter_std(Hin_node,k=2)


fig=plt.figure(99)
plt.clf()
plt.suptitle("when breaking nodes")


############integration vs node strength
ax = plt.subplot2grid((4,3),(0,0))
# sns.stripplot(Hin_node,color="gray")
# outless = 
# ax.scatter(np.random.normal(scale=0.3,size=(len(Hin_node))), )
# ax.set_xticks((0,),["Hin_node"])
ax.set_xlim(-0.3,1)



ax1 = ax.inset_axes((0.5,0.4,0.5,1/2))
ax1.scatter(forces[~outmask],Hin_node[~outmask],color="gray",alpha=0.6)
ax1.scatter(forces[outmask],Hin_node[outmask],color="crimson",marker="x")
ax1.set_xlabel("node strength");ax.set_ylabel("Hin_node")
ax1.set_xticks((1.5,2.5,3.5),(1.5,2.5,3.5),fontsize=ticksize)
ax1.set_yticks((0,0.5,1),(0,0.5,1),fontsize=ticksize)
ax1.spines[['top', 'right']].set_visible(False)


##baseline occupations
ax = plt.subplot2grid((4,3),(0,1))
ax.set_title("Distance to Empirical Occupancies",fontsize=labelsize)
ax.plot(Gs,klCNT,color=colors[0])
ax.plot(Gs,klMCS,color=colors[1])
ax.plot(Gs,klUWS,color=colors[2])
ax.vlines(klGoCNT,-0.03,0.2,linestyle="dashed",color=colors[0],label=f"CNTo={klGoCNT:.2f}")
ax.vlines(klGoMCS,-0.03,0.2,linestyle="dashed",color=colors[1],label=f"MCSo={klGoMCS:.2f}")
ax.vlines(klGoUWS,-0.03,0.2,linestyle="dashed",color=colors[2],label=f"UWSo={klGoUWS:.2f}")
ax.set_xticks((1.5,1.75,2,2.25,2.5),(1.5,1.75,2,2.25,2.5),fontsize=ticksize)
ax.set_yticks((0,0.5,1),(0,0.5,1),fontsize=ticksize);ax.set_ylim([-0.05,1])
ax.legend(
    loc='upper left',
    # bbox_to_anchor=(0.65, 0.7),  # (x, y) in axes coordinates
    bbox_transform=ax.transAxes,
    fontsize=legendsize,
    framealpha=1
)
ax.set_xlabel(r"Coupling $G$",fontsize=labelsize)
ax.set_ylabel("KL distance",fontsize=labelsize)

####baseline jumps
ax = plt.subplot2grid((4,3),(0,2))
ax.set_title("Distance to Empirical Jumps",fontsize=labelsize)
ax.plot(Gs,ksCNT,color=colors[0])
ax.plot(Gs,ksMCS,color=colors[1])
ax.plot(Gs,ksUWS,color=colors[2])
ax.vlines(ksGoCNT,0.1,0.2,linestyle="dashed",color=colors[0],label=f"CNTo={ksGoCNT:.2f}")
ax.vlines(ksGoMCS,0.1,0.2,linestyle="dashed",color=colors[1],label=f"MCSo={ksGoMCS:.2f}")
ax.vlines(ksGoUWS,0.1,0.2,linestyle="dashed",color=colors[2],label=f"UWSo={ksGoUWS:.2f}")
ax.set_xticks((1.5,1.75,2,2.25,2.5),(1.5,1.75,2,2.25,2.5),fontsize=ticksize)
ax.set_yticks((0.1,0.3,0.5),(0.1,0.3,0.5),fontsize=ticksize)
ax.legend(
    loc='upper left',
    # bbox_to_anchor=(0.65, 0.7),  # (x, y) in axes coordinates
    bbox_transform=ax.transAxes,
    fontsize=legendsize,
    framealpha=1
)
ax.set_xlabel(r"Coupling $G$",fontsize=labelsize)
ax.set_ylabel("KS distance",fontsize=labelsize)

###break things occupation
ax = plt.subplot2grid((4,3),(1,1),rowspan=3)
ax1 = ax.inset_axes((0,2/3,1,height))
recreate_the_two(residuals,klsCNT_broken,outmask,colors = ("gray",colors[0]),
                 yticks=(0,0.04,0.08),ax=ax1)
ax2 = ax.inset_axes((0,1/3,1,height))
recreate_the_two(residuals,klsMCS_broken,outmask,colors = ("gray",colors[1]),
                 yticks=(0,0.075,0.15),ax=ax2)
ax3 = ax.inset_axes((0,0,1,height))
recreate_the_two(residuals,klsUWS_broken,outmask,colors = ("gray",colors[2]),
                 yticks=(0,0.2,0.4),ax=ax3)
ax3.set_xlabel("Hin node @ node strength",fontsize=labelsize)
xticks = np.arange(-0.5,0.7,0.25)
ax3.set_xticks(xticks,[f"{val:.2f}" for val in xticks],fontsize=ticksize)

ax.yaxis.set_tick_params(length=0)
ax.set_yticks(())
ax.xaxis.set_tick_params(length=0)
ax.set_xticks(())
ax.set_ylabel("Distance to Occupations (KL)",fontsize=labelsize,labelpad=60)
for i, label in enumerate(['CNT', 'MCS', 'UWS']):
    ax.text(-0.12, (5/6,3/6,1/6)[i], label, fontsize=labelsize,
            ha='right', va='center', transform=ax.transData,rotation=90)
ax.spines[['top','left','bottom','right']].set_visible(False)


#########break things jump
ax = plt.subplot2grid((4,3),(1,2),rowspan=3)
ax1 = ax.inset_axes((0,2/3,1,height))
recreate_the_two(residuals,ksCNT_broken,outmask,colors = ("gray",colors[0]),
                 yticks=list(np.arange(0.1,0.19,0.04)),ax=ax1)
ax2 = ax.inset_axes((0,1/3,1,height))
recreate_the_two(residuals,ksMCS_broken,outmask,colors = ("gray",colors[1]),
                 yticks=list(np.arange(0.05,0.16,0.05)),ax=ax2)
ax3 = ax.inset_axes((0,0,1,height))
recreate_the_two(residuals,ksUWS_broken,outmask,colors = ("gray",colors[2]),
                 yticks=list(np.arange(0.1,0.21,0.05)),ax=ax3)
ax3.set_xlabel("Hin node @ node strength",fontsize=labelsize)
ax3.set_xticks(xticks,[f"{val:.2f}" for val in xticks],fontsize=ticksize)

ax.yaxis.set_tick_params(length=0)
ax.set_yticks(())
ax.xaxis.set_tick_params(length=0)
ax.set_xticks(())
ax.set_ylabel("Distance to Jumps (KS)",fontsize=labelsize,labelpad=60)
for i, label in enumerate(['CNT', 'MCS', 'UWS']):
    ax.text(-0.12, (5/6,3/6,1/6)[i], label, fontsize=labelsize,
            ha='right', va='center', transform=ax.transData,rotation=90)
ax.spines[['top','left','bottom','right']].set_visible(False)


fig.subplots_adjust(wspace=0.4, # horizontal space between axes
                    hspace=0.5  # vertical space between axes
                    )

# plt.tight_layout()
plt.show()


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
        
###rsn
names = ("Vis","ES","Aud","SM","DM","EC")
rsn = np.loadtxt(neurosynth_folder+"RSN_AAL_Enzo.txt")
for n,name in enumerate(names): #########SE NOTA QUE COINCIDE CON LA DM network
    print(name,":", AALlabels[rsn[:,n]==1])

print("\noutliers: ",AALlabels[outmask])



#%% plot
colors = ("tab:blue","tab:orange","tab:green")
hlinecolor = "tab:red"

alfa = 0.5
scatteralfa = 0.5
titlesize = 18
subtitlesize=12
labelsize = 14
ticksize = 14
legendsize=14

###scatters

linestyle = "solid"



plt.figure(96)
plt.clf()
cmap="Reds"
ax = plt.subplot2grid((5,3),(2,2),rowspan=3)
ax.set_title("Integration residuals",fontsize=titlesize)

ax1 = ax.inset_axes((0,2/3,1,height))
bar_subax(obs=klsCNT_broken,y=residuals, cmap=cmap,ax=ax1)
ax2 = ax.inset_axes((0,1/3,1,height))
bar_subax(obs=klsMCS_broken,y=residuals, cmap=cmap,ax=ax2)
ax3 = ax.inset_axes((0,0,1,height))
bar_subax(obs=klsUWS_broken,y=residuals, cmap=cmap,ax=ax3)


ax.yaxis.set_tick_params(length=0)
ax.set_yticks(())
ax.xaxis.set_tick_params(length=0)
ax.set_xticks(())
ax.set_ylabel("KL distance",fontsize=labelsize,labelpad=70)
for i, label in enumerate(['to CNT', 'to MCS', 'to UWS']):
    ax.text(-0.17, (5/6,3/6,1/6)[i], label, fontsize=labelsize,
            ha='right', va='center', transform=ax.transData,rotation=90)
for spine in ax.spines.values():
    spine.set_visible(False)



plt.subplots_adjust(wspace=0.4, hspace=0.4)
# plt.tight_layout()
# plt.savefig("figures/unfinished_fig4.svg",transparent=True,dpi=300)
plt.show()

#%% generate colorbar
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Customization parameters
vmin = 0
vmax = 1
tick_fontsize = 12
label_fontsize = 14
label_text = 'Normalized value'

# Create a ScalarMappable for the colorbar
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = cm.get_cmap('hot_r')  # Reversed 'hot' colormap
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

# Create the figure and vertical colorbar
fig, ax = plt.subplots(figsize=(1.5, 6))
fig.subplots_adjust(left=0.5)
cbar = fig.colorbar(sm, ax=ax, orientation='vertical')

# Customize ticks and label
cbar.ax.tick_params(labelsize=tick_fontsize)
cbar.set_label(label_text, fontsize=label_fontsize)

# Remove the main axes (just keep the colorbar)
ax.remove()
# plt.savefig("figures/colorbar.svg",dpi=300,transparent= True)
plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def check_incremental_effect(y, x1, x2):
    """
    Visualizes whether x2 adds information to predicting y
    beyond what x1 explains.
    """
    # --- Plot y vs x1 and y vs x2
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # y vs x1
    axes[0].scatter(x1, y, alpha=0.7)
    slope, intercept, _, _, _ = linregress(x1, y)
    axes[0].plot(np.sort(x1), intercept + slope*np.sort(x1), color="red")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("y")
    axes[0].set_title("y vs x1")
    
    # y vs x2
    axes[1].scatter(x2, y, alpha=0.7, color="orange")
    slope, intercept, _, _, _ = linregress(x2, y)
    axes[1].plot(np.sort(x2), intercept + slope*np.sort(x2), color="red")
    axes[1].set_xlabel("x2")
    axes[1].set_ylabel("y")
    axes[1].set_title("y vs x2")
    
    # Residuals from y ~ x1 vs x2
    slope, intercept, _, _, _ = linregress(x1, y)
    y_hat = intercept + slope*x1
    resid = y - y_hat
    axes[2].scatter(x2, resid, alpha=0.7, color="green")
    slope_r, intercept_r, _, _, _ = linregress(x2, resid)
    axes[2].plot(np.sort(x2), intercept_r + slope_r*np.sort(x2), color="red")
    axes[2].axhline(0, color="k", linestyle="--")
    axes[2].set_xlabel("x2")
    axes[2].set_ylabel("Residuals (y - ŷ from x1)")
    axes[2].set_title("Do residuals depend on x2?")
    
    plt.tight_layout()
    plt.show()
    
check_incremental_effect(y=klsCNT_broken, x1=forces, x2=Hin_node)


