# -*- coding: utf-8 -*-
"""
IN THIS SCRIPT WE ANALYZE MOVING THE INTERCEPT OF JFIC OF EACH AREA.
Created on Sun Apr 28 19:46:29 2024

@author: flehu
"""

import matplotlib.colors as mcolors
from scipy.stats import linregress
import bct
import sys
sys.path.append("../");sys.path.append("../../")
import pandas as pd
import HMA
from scipy.special import rel_entr as kl  # relative entropy = KL distance
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import kstest,mannwhitneyu,pearsonr,spearmanr
import sys
sys.path.append("../../")

nodes = range(90)
# cada tupla es (corrs,eucs,jumps_high,counts_high)


with open('../../../fastDMF/output/change_JFIC_per_node_fromUWS_seeds0_19.pickle', 'rb') as f:
    by_seed = pickle.load(f)

with open('../../../fastDMF/output/change_JFIC_per_node_fromUWS_extended_seeds0_19.pickle', 'rb') as f:
    by_seed2 = pickle.load(f)

for seed in range(20):
    by_seed[seed].update(by_seed2[seed])


# halt
# %%
with open('../../../fastDMF/empirical_truth/DoC_mean_FCs.pickle', 'rb') as f:
    emp_fcs = pickle.load(f)
with open('../../../fastDMF/empirical_truth/jump_distributions_dict_filt_ALLDIM.pickle', 'rb') as f:
    jump_dists_all = pickle.load(f)

del f

emp_occs_all = np.loadtxt(
    '../../../fastDMF/empirical_truth/occupations_3clusters_alldim_euclidean.txt')
preroi = list(pd.read_csv("../../../sorted_AAL_labels.txt")
              ["label"].values)  # nombre de las areas
ROIlabels = [f"{i}-{preroi[i].replace('_', ' ')}" for i in range(90)]
struct = np.loadtxt("../../../structural_Deco_AAL.txt")
# %%
states = ["CNT", "MCS", "UWS"]

nodes = list(range(90))
seeds = list(range(20))
Cs = []

for key in by_seed[0].keys():
    C, node = key
    if C not in Cs:
        Cs.append(C)
Cs = np.sort(Cs)
Cs = Cs[(Cs >= -2) & (Cs <= 2)]  # only makes sense from there on
nseeds = 20

# fill the mean occupation euclidean and corr vector
occs = np.zeros((len(nodes), len(Cs), len(("c1", "c2", "c3"))))
euc = np.zeros((len(nodes), len(Cs), len(states)))
corrs = np.zeros((len(nodes), len(Cs), len(states)))
jumps = np.zeros((len(nodes),len(Cs),191)) ###191 es el numero de saltos por simulacion

for seed in seeds:
    dic = by_seed[seed]
    for n in nodes:
        for c, C in enumerate(Cs):
            corrs[n, c, :] += dic[C, n][0]
            euc[n, c, :]   += dic[C, n][1]
            jumps[n,c,:]   += dic[C, n][2]
            occs[n, c, :]  += dic[C, n][3]
nseeds = len(seeds)
occs /= nseeds
euc /= nseeds
corrs /= nseeds
jumps /= nseeds
# sweep data
# klCNT = np.vstack([np.array([kl(occs[n, c, :], emp_occs_all[0]).sum()
#                   for c in range(len(Cs))]) for n in nodes])
# klMCS = np.vstack([np.array([kl(occs[n, c, :], emp_occs_all[1]).sum()
#                   for c in range(len(Cs))]) for n in nodes])
# klUWS = np.vstack([np.array([kl(occs[n, c, :], emp_occs_all[2]).sum()
#                   for c in range(len(Cs))]) for n in nodes])

klCNT = np.vstack([1/2*np.array([kl(occs[n, c, :], emp_occs_all[0]).sum() +kl(emp_occs_all[0],occs[n, c, :]).sum()
                  for c in range(len(Cs))]) for n in nodes])
klMCS = np.vstack([1/2*np.array([kl(occs[n, c, :], emp_occs_all[1]).sum() +kl(emp_occs_all[1],occs[n, c, :]).sum()
                  for c in range(len(Cs))]) for n in nodes])
klUWS = np.vstack([1/2*np.array([kl(occs[n, c, :], emp_occs_all[2]).sum() +kl(emp_occs_all[2],occs[n, c, :]).sum()
                  for c in range(len(Cs))]) for n in nodes])



ksCNT = np.vstack([np.array([kstest(jumps[n,c,:],jump_dists_all["CNT"])[0]
                             for c in range(len(Cs))]) for n in nodes])
ksMCS = np.vstack([np.array([kstest(jumps[n,c,:],jump_dists_all["MCS"])[0]
                             for c in range(len(Cs))]) for n in nodes])
ksUWS = np.vstack([np.array([kstest(jumps[n,c,:],jump_dists_all["UWS"])[0]
                             for c in range(len(Cs))]) for n in nodes])

# ksGoCNT,ksGoMCS,ksGoUWS = [Gs[np.argmin(a)] for a in (ksCNT,ksMCS,ksUWS)]


whereCNT, whereMCS, whereUWS = np.where(klCNT == klCNT.min()), np.where(
                                        klMCS == klMCS.min()), np.where(klUWS == klUWS.min())

noCNT, CoCNT = np.array(nodes)[whereCNT[0]], Cs[whereCNT[1]]
noMCS, CoMCS = np.array(nodes)[whereMCS[0]], Cs[whereMCS[1]]
noUWS, CoUWS = np.array(nodes)[whereUWS[0]], Cs[whereUWS[1]]


# optimal
print("kl optima at", (noCNT, CoCNT), (noMCS, CoMCS), (noUWS, CoUWS))
minCNT, minMCS, minUWS = [np.min(a) for a in (klCNT, klMCS, klUWS)]

# %% integration segregation data from structural matrix

forces = struct.sum(axis=1)
Clus_num, Clus_size, H_all = HMA.Functional_HP(struct)
Hin, Hse = HMA.Balance(struct, Clus_num, Clus_size)
Hin_node, Hse_node = HMA.nodal_measures(struct, Clus_num, Clus_size)

betweenness_cent = bct.centrality.betweenness_wei(struct)


# %% plot for a fixed C value
C_base = -0.8
C_base_id = int(np.where(Cs == C_base)[0])
ceteris_paribus = {states[i]: (klCNT, klMCS, klUWS)[
    i][:, C_base_id] for i in range(3)}

no_move_id = int(np.where(Cs == 0)[0])
no_move = {states[i]: (klCNT, klMCS, klUWS)[i]
           [:, no_move_id].mean() for i in range(3)}
# optimals = {states[i]:np.min((klCNT,klMCS,klUWS)[i],axis=1) for i in range(3)}
# for n in nodes:
norm = lambda x: (x-x.min())/(x.max()-x.min())

colors = ("tab:blue", "tab:orange", "tab:green")
hlinecolor = "tab:red"
cmap = "Reds"

alfa = 0.3
scatteralfa = 0.5
titlesize = 15
subtitlesize = 12
labelsize = 14
ticksize = 14
legendsize = 13
ylabel = "alfa (base at 0.75)"
xlabel = "G"
std = 0.05;xpos1 = 0;xpos2 = 2;tickangle=0
epsilon=1e-2

#plot how low they get vs when do they get there.

x = Cs
klys = {"CNT": klCNT,
       "MCS": klMCS,
       "UWS": klUWS}
ksys = {"CNT": ksCNT,
        "MCS": ksMCS,
        "UWS": ksUWS}

values = Hin_node/Hin_node.max()

def create_filter_dumb(x,k=0.4):
    return x > k
k = 0.4
outmask = create_filter_dumb(values,k=k)

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
    
def plotbar(x,ys,outmask,C_base,colors=("gray","tab:blue"),
            yticks=None,xticks=True,  ax=None):
    
    C_base_id = int(np.where(Cs == C_base)[0])
    
    yclean = ys[~outmask,C_base_id]
    ydirt  = ys[outmask,C_base_id]
    
    
    bp1 = ax.boxplot(yclean, positions=(0,), widths=0.6,patch_artist=True, showfliers=False)
    ax.scatter(0+np.random.normal(scale=std,size=len(yclean)),yclean,color="gray",alpha=alfa)
    
    ax.scatter(1+np.random.normal(scale=std,size=len(ydirt)),ydirt,color=colors[1],marker="x",alpha=1)
    bp2 = ax.boxplot(ydirt, positions=(1,), widths=0.6,patch_artist=True, showfliers=False)
    for patch in list(bp1['boxes'])+list(bp2['boxes']):
        patch.set_facecolor("none")  # transparent
        patch.set_edgecolor("black")
    if yticks:
        ticklabels = [f"{val:.2f}" for val in yticks]
        ax.set_yticks(yticks,ticklabels,fontsize=ticksize)
        ax.set_yticklabels(len(yticks)*[""])
    if not xticks:
        ax.set_xticks(())
    else:
        # ax.set_xlabel("Hin node",fontsize=labelsize)
        ax.set_xticks((0,1),("Low\nHin", "High\nHin"),fontsize=labelsize)
        
    p = mannwhitneyu(yclean,ydirt)[1]
    print(p)
    d = cohen_d(yclean,ydirt)
    text = ""
    if p and d:
        if p < 0.001:
            add = "***"
        elif p < 0.01:
            add = "**"
        elif p < 0.05:
            add = "*"
        text+=f"D({d:.2f})"  + add
    ax.text(0.5,0.7, text,fontsize=legendsize,
            ha='center', va='center',transform=ax.transAxes,rotation=90)
    ax.spines[["top","right"]].set_visible(False)
    
def plot_with_color(x, ys,outmask,colors = ("gray","tab:blue"),title= "",ylabel="",
                    yticks = None,xticks=None, ax = None,yminmax=None):
    
    ax1 = ax.inset_axes((0,0,2/3-epsilon,1))
    ax1.set_title(title,fontsize=titlesize)
    
    first_high = True
    first_low = True
    for n, node in enumerate(nodes):
        if outmask[node]:  # high
            if first_high:
                ax1.plot(x, ys[node], color=colors[1], label="High Hin")
                first_high = False
            else:
                ax1.plot(x, ys[node], color=colors[1])
        else:  # low
            if first_low:
                ax1.plot(x, ys[node], color=colors[0], alpha=alfa, label="Low Hin")
                first_low = False
            else:
                ax1.plot(x, ys[node], color=colors[0], alpha=alfa)
                
    ymin = ys.min()
    ymax = ys.max()#ymin + (ys.max()-ymin)/2
    ax1.vlines(C_base, ymin,ymax,linestyle="dashed",color="black",label=f"C={C_base:.1f}")
    ax1.spines[["top","right"]].set_visible(False)
    ax1.set_ylabel(ylabel,fontsize=labelsize)
    
    ticklabels = [f"{val}" for val in xticks]
    ax1.set_xticks(xticks,ticklabels,fontsize=ticksize)
    ax1.set_xlabel("JFIC intercept",fontsize=labelsize)
    ax1.legend(fontsize=legendsize,loc="lower right")
    # ax2.set_xticks((0,1),("Low\nHin", "High\nHin"),fontsize=labelsize)
    
    ###boxx
    ax2 = ax.inset_axes((2/3+epsilon,0,1/3-epsilon,1))
    ax2.set_title(f"fit at C={C_base:.1f}",fontsize=titlesize)
    plotbar(x,ys,outmask,C_base,colors=colors,
                yticks=yticks,xticks=True,  ax=ax2)
    
    ticklabels = [f"{val:.2f}" for val in yticks]
    ax1.set_yticks(yticks,ticklabels,fontsize=ticksize)
    ax2.set_yticks((yticks))
    ax2.set_yticklabels(len(yticks)*[""])
    
    if yminmax:
        mini,maxi = yminmax
        ax1.set_ylim((mini,maxi))
        ax2.set_ylim((mini,maxi))
    
    ##clean big axis
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    

    
def joint_scatter(x1, x2, y, outmask=None, ax=None, yticks=None, ylabel="", colors=("crimson", "gray")):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    ydirt = y[outmask]
    yclean = y[~outmask]
    x1dirt = x1[outmask]; x1clean = x1[~outmask]
    x2dirt = x2[outmask]; x2clean = x2[~outmask]

    # --- Layout parameters ---
    height = 0.45
    pad = 0.05

    # --- Upper inset (scatter 1) ---
    ax_top = ax.inset_axes((0, height + pad, 1, height))
    ax_top.scatter(x1clean, yclean, color=colors[0], alpha=0.7, label="Hin node")
    ax_top.scatter(x1dirt, ydirt, color=colors[0], alpha=1, marker="x")
    ax_top.set_xticks((0, 0.2, 0.4, 0.6, 0.8, 1), 6 * [""])
    ax_top.set_yticks(yticks)
    ax_top.set_yticklabels([f"{val:.2f}" for val in yticks], fontsize=ticksize)
    ax_top.set_ylabel(ylabel, fontsize=labelsize)
    ax_top.spines[["top", "right"]].set_visible(False)

    # --- Lower inset (scatter 2) ---
    ax_bottom = ax.inset_axes((0, 0, 1, height))
    ax_bottom.scatter(x2clean, yclean, color=colors[1], alpha=0.9, label="Node strength")
    ax_bottom.scatter(x2dirt, ydirt, color=colors[1], alpha=1, marker="x")
    ax_bottom.set_xticks((0, 0.2, 0.4, 0.6, 0.8, 1))
    ax_bottom.set_xticklabels((0, 0.2, 0.4, 0.6, 0.8, 1), fontsize=ticksize)
    ax_bottom.set_yticks(yticks)
    ax_bottom.set_yticklabels([f"{val:.2f}" for val in yticks], fontsize=ticksize)
    ax_bottom.set_xlabel("Normalized value", fontsize=labelsize)
    ax_bottom.set_ylabel(ylabel, fontsize=labelsize)
    ax_bottom.spines[["top", "right"]].set_visible(False)

    # --- Single combined legend in the parent axis ---
    handles, labels = [], []
    for a in (ax_top, ax_bottom):
        h, l = a.get_legend_handles_labels()
        handles += h
        labels += l
    ax.legend(handles, labels, fontsize=legendsize, loc="upper right")

    # --- Clean parent axis ---
    ax.spines[:].set_visible(False)
    ax.set_xticks(()); ax.set_yticks(())

    return ax_top, ax_bottom

    # return ax_top, ax_bottom


# plot
plt.figure(1)
plt.clf()

# CN KLS
ax = plt.subplot2grid((2, 4), (0, 0),colspan=2)
plot_with_color(x, klys["CNT"], outmask,ax=ax,
                title = "Fit to empirical CNT occupations",ylabel="KL distance",
                yticks=(0.1,0.25,0.4),xticks=(-2,-1,0,1,2),yminmax=(0.03,0.45))

ax = plt.subplot2grid((2,4),(0,2),colspan=2)
plot_with_color(x,ksys["CNT"],outmask,ax=ax,
                title = "Fit to empirical CNT Jumps",ylabel="KS distance",
                yticks=(0.5,0.55,0.6),xticks=(-2,-1,0,1,2),yminmax=(0.48,0.61))
###brain 
ax = plt.subplot2grid((2,4),(1,1))
ax.set_title("KL distance CNT",fontsize=titlesize)
ax.spines[:].set_visible(False)
ax.set_xticks(())
ax.set_yticks(())

#######
ax = plt.subplot2grid((2,4),(1,0))
ax.set_title("KL-CNT vs nodal measures",fontsize=titlesize)
joint_scatter(norm(Hin_node),norm(forces),klys["CNT"][:,C_base_id],outmask=outmask,
              ax= ax, yticks = (0.05,0.15,0.25),ylabel="KL distance")

###
ax = plt.subplot2grid((2,4),(1,2))
ax.set_title("KS-CNT vs nodal measures",fontsize=titlesize)
joint_scatter(norm(Hin_node),norm(forces),ksys["CNT"][:,C_base_id],outmask=outmask,
              ax= ax, yticks = (0.5,0.53,0.56),ylabel="KS distance")

ax = plt.subplot2grid((2,4),(1,3))
ax.set_title("KS distance CNT",fontsize=titlesize)
ax.spines[:].set_visible(False)
ax.set_xticks(())
ax.set_yticks(())

plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.tight_layout()
plt.savefig("fig5_symmetrized.svg",dpi=300,transparent=True)
plt.show()


halt

#%%
print("node strength",pearsonr(Hin_node,klys["CNT"][:,C_base_id]))
print("node integration",pearsonr(forces,klys["CNT"][:,C_base_id]))



print("node strength",spearmanr(Hin_node,klys["CNT"][:,C_base_id]))
print("node integration",spearmanr(forces,klys["CNT"][:,C_base_id]))


#%% save two files for the 

norm = lambda x: (x-x.min())/(x.max()-x.min())
to_save = {"klys":klys["CNT"][:,C_base_id],
           "ksys":ksys["CNT"][:,C_base_id]}
np.savez_compressed(f"maps2brain_C={C_base:.1f}.npz",**to_save)



#%% for a fixed C, how close are we to each state?
fig = plt.figure(3, figsize=(12, 6))
plt.clf()
fig.suptitle(f"KL distance with intercept variation C={C_base}", fontsize=titlesize)

# to CNT
plt.subplot(311)
plt.title("CNT")
plt.bar(nodes, ceteris_paribus["CNT"], alpha=0.7)
plt.hlines(no_move["CNT"], 0, 90, color="tab:red",
           linestyle="dashed", label="baseline")
plt.xticks(())
plt.legend()

# to MCS
plt.subplot(312)
plt.title("MCS")
plt.bar(nodes, ceteris_paribus["MCS"], alpha=0.7)
plt.hlines(no_move["MCS"], 0, 90, color="tab:red",
           linestyle="dashed", label="baseline")
plt.xticks(())
plt.legend()

# to UWS
plt.subplot(313)
plt.title("UWS")
plt.bar(nodes, ceteris_paribus["UWS"], alpha=0.7)
plt.hlines(no_move["UWS"], 0, 90, color="tab:red",
           linestyle="dashed", label="baseline")
plt.xticks(nodes, ROIlabels, rotation=90)
plt.legend()

plt.tight_layout()
plt.show()

# %% print how close do we get per area

min_args_CNT = np.argmin(klCNT, axis=1)
min_args_MCS = np.argmin(klMCS, axis=1)
min_args_UWS = np.argmin(klUWS, axis=1)
min_c_CNT = Cs[min_args_CNT]
min_c_MCS = Cs[min_args_MCS]
min_c_UWS = Cs[min_args_UWS]


# final FIC?
Fics = 0.75*2.12*struct.sum(axis=1)+1
important_nodes = (17, 21, 22, 23, 24, 33,  # left
                   56, 65, 66, 67, 68, 72)  # right
print([str(np.array(ROIlabels)[i]) for i in important_nodes])


# here we plot the fit to each state moving the
fig = plt.figure(2)
plt.clf()
plt.suptitle("moving C per area")
# CNT

###########################
baseline = np.where(Cs == 0)[0][0]
plt.subplot2grid((4, 1), (1, 0))
plt.imshow(klCNT.T, aspect="auto", interpolation="none")
plt.hlines(baseline, 0, 90, color="red", linestyle="dashed")
# plt.colorbar(label="KL dist to CNT")
plt.plot(range(90), min_args_CNT, color="white")

plt.yticks(range(len(Cs))[::5], Cs[::5])
plt.xticks(())
plt.ylabel("optimal c")
plt.xlim([-0.5, 89.5])

plt.subplot2grid((4, 1), (2, 0))
plt.imshow(klMCS.T, aspect="auto", interpolation="none")
plt.hlines(baseline, 0, 90, color="red", linestyle="dashed")
# plt.colorbar(label="KL dist to MCS")
plt.plot(range(90), min_args_MCS, color="white")
plt.yticks(range(len(Cs))[::5], Cs[::5])
plt.xticks(())
plt.ylabel("optimal c")
plt.xlim([-0.5, 89.5])

plt.subplot2grid((4, 1), (3, 0))
plt.imshow(klUWS.T, aspect="auto", interpolation="none")
plt.hlines(baseline, 0, 90, color="red", linestyle="dashed")
# plt.colorbar(label="KL dist to UWS")
plt.plot(range(90), min_args_UWS, color="white")
plt.yticks(range(len(Cs))[::5], Cs[::5])
plt.xticks(range(90), ROIlabels, rotation=90)
plt.ylabel("optimal c")
plt.xlim([-0.5, 89.5])

plt.tight_layout()
plt.show()




# %%figure with mins


mins_CNT = np.min(klCNT, axis=1)
mins_MCS = np.min(klMCS, axis=1)
mins_UWS = np.min(klUWS, axis=1)


fig = plt.figure(4)
plt.clf()
plt.suptitle("minimum attained")

plt.subplot2grid((3, 1), (0, 0))
plt.bar(range(90), mins_CNT)
plt.xticks(())

plt.subplot2grid((3, 1), (1, 0))
plt.bar(range(90), mins_MCS)
plt.xticks(())

plt.subplot2grid((3, 1), (2, 0))
plt.bar(range(90), mins_UWS)
plt.xticks(range(90), ROIlabels, rotation=90)

plt.tight_layout()
plt.show()

# %%figure with ranges of variation

stds_CNT = np.std(klCNT, axis=1)
stds_MCS = np.std(klMCS, axis=1)
stds_UWS = np.std(klUWS, axis=1)

mins_euc_CNT = np.min(euc[:, :, 0], axis=1)
mins_euc_MCS = np.min(euc[:, :, 1], axis=1)
mins_euc_UWS = np.min(euc[:, :, 2], axis=1)


fig = plt.figure(5)
plt.clf()
plt.suptitle(
    "top: Hin node vs optimal fit to state, force below. Bottom: node sensitivity")


# variation vs Hin node
plt.subplot2grid((5, 3), (0, 0))
plt.scatter(Hin_node, mins_CNT)
plt.xlabel("Hin_node")
plt.ylabel("mins KL")
plt.xscale("log")

plt.subplot2grid((5, 3), (0, 1))
plt.scatter(Hin_node, mins_MCS)
plt.xlabel("Hin_node")
plt.ylabel("mins KL")
plt.xscale("log")

plt.subplot2grid((5, 3), (0, 2))
plt.scatter(Hin_node, mins_UWS)
plt.xlabel("Hin_node")
plt.ylabel("mins KL")
plt.xscale("log")

plt.subplot2grid((5, 3), (1, 0))
plt.scatter(Hin_node, mins_euc_CNT)
plt.xlabel("Hin_node")
plt.ylabel("mins EUC")
plt.xscale("log")

plt.subplot2grid((5, 3), (1, 1))
plt.scatter(Hin_node, mins_euc_MCS)
plt.xlabel("Hin_node")
plt.ylabel("mins EUC")
plt.xscale("log")

plt.subplot2grid((5, 3), (1, 2))
plt.scatter(Hin_node, mins_euc_UWS)
plt.xlabel("Hin_node")
plt.ylabel("mins EUC")
plt.xscale("log")


# node strength
plt.subplot2grid((5, 3), (3, 0))
plt.scatter(forces, mins_CNT)
plt.xlabel("node strength")
plt.ylabel("mins")

plt.subplot2grid((5, 3), (3, 1))
plt.scatter(forces, mins_MCS)
plt.xlabel("node strength")
plt.ylabel("mins")

plt.subplot2grid((5, 3), (3, 2))
plt.scatter(forces, mins_UWS)
plt.xlabel("node strength")
plt.ylabel("mins")


plt.subplot2grid((5, 3), (4, 0))
plt.scatter(forces, mins_euc_CNT)
plt.xlabel("node strength")
plt.ylabel("mins")

plt.subplot2grid((5, 3), (4, 1))
plt.scatter(forces, mins_euc_MCS)
plt.xlabel("node strength")
plt.ylabel("mins")

plt.subplot2grid((5, 3), (4, 2))
plt.scatter(forces, mins_euc_UWS)
plt.xlabel("node strength")
plt.ylabel("mins")

plt.tight_layout()
plt.show()

# %% ver la distirbution de minimos
plt.figure(6)
plt.clf()
plt.suptitle("minimal KL attained sweeping c!")


plt.subplot2grid((3, 1), (0, 0))
plt.title("CNT")
plt.bar(range(90), mins_CNT)
plt.xticks(())

plt.subplot2grid((3, 1), (1, 0))
plt.title("MCS")
plt.bar(range(90), mins_MCS)
plt.xticks(())

plt.subplot2grid((3, 1), (2, 0))
plt.title("UWS")
plt.bar(range(90), mins_UWS)
plt.xticks(range(90), ROIlabels, rotation=90)


plt.tight_layout()
plt.show()

# %%%%%%%%%%

struct_df = pd.DataFrame(index=ROIlabels)
struct_df["Hin"] = Hin_node
struct_df["Hse"] = Hse_node
struct_df["force"] = forces
struct_df["stds_CNT"] = stds_CNT
struct_df["stds_MCS"] = stds_MCS
struct_df["stds_UWS"] = stds_UWS

print(struct_df[(struct_df["force"] > 3) & (struct_df["stds_CNT"] < 0.01)])
