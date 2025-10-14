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

Cs = Cs[(Cs >= -3) & (Cs <= 3)]  # only makes sense from there on

# fill the mean occupation euclidean and corr vector
occs = np.zeros((len(nodes), len(Cs), len(("c1", "c2", "c3"))))
euc = np.zeros((len(nodes), len(Cs), len(states)))
corrs = np.zeros((len(nodes), len(Cs), len(states)))

for seed in seeds:
    dic = by_seed[seed]
    for n in nodes:
        for c, C in enumerate(Cs):
            occs[n, c, :] += dic[C, n][-1]
            euc[n, c, :] += dic[C, n][1]
            corrs[n, c, :] += dic[C, n][0]
occs /= len(seeds)
euc /= len(seeds)
corrs /= len(seeds)
# sweep data
# klCNT,klMCS,klUWS = [np.array([kl(occs[g,:],emp_occs_all[s]).sum() for g in range(len(Gs))]) for s in range(3)]
klCNT = np.vstack([np.array([kl(occs[n, c, :], emp_occs_all[0]).sum()
                  for c in range(len(Cs))]) for n in nodes])
klMCS = np.vstack([np.array([kl(occs[n, c, :], emp_occs_all[1]).sum()
                  for c in range(len(Cs))]) for n in nodes])
klUWS = np.vstack([np.array([kl(occs[n, c, :], emp_occs_all[2]).sum()
                  for c in range(len(Cs))]) for n in nodes])

# halt

# GoCNT,GoMCS,GoUWS = [Gs[np.argmin(a)] for a in (klCNT,klMCS,klUWS)]
whereCNT, whereMCS, whereUWS = np.where(klCNT == klCNT.min()), np.where(
    klMCS == klMCS.min()), np.where(klUWS == klUWS.min())
noCNT, CoCNT = np.array(nodes)[whereCNT[0]], Cs[whereCNT[1]]
noMCS, CoMCS = np.array(nodes)[whereMCS[0]], Cs[whereMCS[1]]
noUWS, CoUWS = np.array(nodes)[whereUWS[0]], Cs[whereUWS[1]]

# halt
# optimal
print("kl optima at", (noCNT, CoCNT), (noMCS, CoMCS), (noUWS, CoUWS))
minCNT, minMCS, minUWS = [np.min(a) for a in (klCNT, klMCS, klUWS)]

# halt
# %% integration segregation data from structural matrix

forces = struct.sum(axis=1)
Clus_num, Clus_size, H_all = HMA.Functional_HP(struct)
Hin, Hse = HMA.Balance(struct, Clus_num, Clus_size)
Hin_node, Hse_node = HMA.nodal_measures(struct, Clus_num, Clus_size)

betweenness_cent = bct.centrality.betweenness_wei(struct)


# %% plot for a fixed C value
C_base = 2
C_base_id = int(np.where(Cs == C_base)[0])
ceteris_paribus = {states[i]: (klCNT, klMCS, klUWS)[
    i][:, C_base_id] for i in range(3)}
no_move_id = int(np.where(Cs == 0)[0])
no_move = {states[i]: (klCNT, klMCS, klUWS)[i]
           [:, no_move_id].mean() for i in range(3)}
# optimals = {states[i]:np.min((klCNT,klMCS,klUWS)[i],axis=1) for i in range(3)}
# for n in nodes:


colors = ("tab:blue", "tab:orange", "tab:green")
hlinecolor = "tab:red"
cmap = "Reds"

alfa = 0.5
scatteralfa = 0.5
titlesize = 18
subtitlesize = 12
labelsize = 14
ticksize = 14
legendsize = 14
ylabel = "alfa (base at 0.75)"
xlabel = "G"

# for a fixed C, how close are we to each state?
fig = plt.figure(1, figsize=(12, 6))
plt.clf()
fig.suptitle(f"KL distance with intercept variation C={
             C_base}", fontsize=titlesize)

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

# plt.xlim((-4,0))


# plt.show()

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

# %% plot how low they get vs when do they get there.

x = Cs
ys = {"CNT": klCNT,
      "MCS": klMCS,
      "UWS": klUWS}


def plot_with_color(x, ys, colorvec=Hin_node, label="Integration"):
    norm = mcolors.Normalize(vmin=colorvec.min(), vmax=colorvec.max())
    cmap = plt.cm.Reds
    for n, node in enumerate(nodes):
        value = colorvec[node]  # the value you want to map to color
        line_color = cmap(norm(value))
        if node in important_nodes:
            plt.plot(x, ys[node], color=line_color,
                     label=ROIlabels[node].split("-")[1])
        else:
            plt.plot(x, ys[node], color=line_color, alpha=0.2)


# th =-1
# important_CNT = np.argwhere(xs["CNT"]>th).flatten()
# print([ROIlabels[i] for i in important_CNT])
color1 = Hin_node
color2 = forces


# plot
plt.figure(3)
plt.clf()
plt.suptitle("sweeping C per area, lower is higher excitability needed")


# plot with different color according to integration

# CN
plt.subplot2grid((3, 2), (0, 0))
plt.title("CNT")
plot_with_color(x, ys["CNT"], colorvec=Hin_node)

# MCS
plt.subplot2grid((3, 2), (1, 0))
plt.title("MCS")
plot_with_color(x, ys["MCS"], colorvec=Hin_node)

# UWS
plt.subplot2grid((3, 2), (2, 0))
plt.title("UWS")
plot_with_color(x, ys["UWS"], colorvec=Hin_node)

# colormap
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# plt.colorbar(sm, ax=plt.gcf().get_axes(), label=label)


# plot with different color according to node strength
# CN
plt.subplot2grid((3,2),(0,1))
plt.title("CNT")
plot_with_color(x,ys["CNT"],colorvec=forces)

##MCS
plt.subplot2grid((3,2),(1,1))
plt.title("MCS")
plot_with_color(x,ys["MCS"],colorvec=forces)

##UWS
plt.subplot2grid((3,2),(2,1))
plt.title("UWS")
plot_with_color(x,ys["UWS"],colorvec=forces)


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
