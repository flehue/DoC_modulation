# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 19:46:29 2024

@author: flehu
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
# import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp as ks,gamma,iqr
from scipy.special import rel_entr as kl #relative entropy = KL distance

#%%cargar
##cada tupla es (corrs,eucs,jumps_high,counts_high)
with open('../fastDMF/output/sweep_summary_seeds0_19.pickle', 'rb') as f:
    data = pickle.load(f)
# del f
#%%
with open('../fastDMF/empirical_truth/DoC_mean_FCs.pickle', 'rb') as f:
    emp_fcs = pickle.load(f)
with open('../fastDMF/empirical_truth/jump_distributions_dict_filt_ALLDIM.pickle', 'rb') as f:
    jump_dists_all = pickle.load(f)
del f

emp_occs_all = np.loadtxt('../fastDMF/empirical_truth/occupations_3clusters_alldim_euclidean.txt')


#%%
states = ["CNT","MCS","UWS"]

Gs, seeds = [],[]

for key in data.keys():
    G,seed = key
    if G not in Gs:
        Gs.append(G)
    if seed not in seeds:
        seeds.append(seed)
Gs,seeds = [np.sort(Gs),np.sort(seeds)]
Gs = Gs[Gs<2.6]

by_seed = {s:{} for s in seeds}
for key in data.keys():
    G,seed = key
    by_seed[seed][G] = data[(G,seed)]


#%%
occs = np.zeros((len(Gs),len(("c1","c2","c3"))))
euc = np.zeros((len(Gs),len(states)))
corrs = np.zeros((len(Gs),len(states)))

for seed in seeds:
    dic = by_seed[seed]
    for g,G in enumerate(Gs):
        occs[g,:] += dic[G][-1]
        euc[g,:] += dic[G][1]
        corrs[g,:] += dic[G][0]
occs /= len(seeds)
euc /= len(seeds)
corrs /= len(seeds)

#%%

klCNT,klMCS,klUWS = [np.array([kl(occs[g,:],emp_occs_all[s]).sum() for g in range(len(Gs))]) for s in range(3)]
GoCNT,GoMCS,GoUWS = [Gs[np.argmin(a)] for a in (klCNT,klMCS,klUWS)]
print("kl optima at",GoCNT,GoMCS,GoUWS)
minCNT,minMCS,minUWS = [np.min(a) for a in (klCNT,klMCS,klUWS)]

plt.figure(1)
plt.clf()


##occupancies
plt.subplot(331)
plt.title("KL to empirical occupancies")
plt.plot(Gs,klCNT,label=f"CNT, Go={GoCNT:.2f}")
plt.plot(Gs,klMCS,label=f"MCS, Go={GoMCS:.2f}")
plt.plot(Gs,klUWS,label=f"UWS, Go={GoUWS:.2f}")
plt.vlines([GoCNT,GoMCS,GoUWS],-0.1,0.2,linestyle="dashed",colors=["tab:blue","tab:orange","tab:green"])
plt.legend()


###euclidean to FC
GoCNT,GoMCS,GoUWS = [Gs[np.argmin(euc[:,s])] for s in range(len(states))]
minCNT,minMCS,minUWS = [np.min(euc[:,s]) for s in range(len(states))]
print("euclidean optima at",GoCNT,GoMCS,GoUWS)
plt.subplot(332)
plt.title("euclidean to empirical FC")
plt.plot(Gs,euc[:,0],label=f"CNT, Go={GoCNT:.2f}")
plt.plot(Gs,euc[:,1],label=f"MCS, Go={GoMCS:.2f}")
plt.plot(Gs,euc[:,2],label=f"UWS, Go={GoUWS:.2f}")
plt.vlines([GoCNT,GoMCS,GoUWS],10,17,linestyle="dashed",colors=["tab:blue","tab:orange","tab:green"])
plt.legend()

GoCNT,GoMCS,GoUWS = [Gs[np.argmax(corrs[:,s])] for s in range(len(states))]
minCNT,minMCS,minUWS = [np.max(corrs[:,s]) for s in range(len(states))]

plt.subplot(333)
plt.title("corr to empirical FC")
plt.plot(Gs,corrs[:,0],label=f"CNT, Go={GoCNT:.2f}")
plt.plot(Gs,corrs[:,1],label=f"MCS, Go={GoMCS:.2f}")
plt.plot(Gs,corrs[:,2],label=f"UWS, Go={GoUWS:.2f}")
plt.legend()


plt.tight_layout()
plt.show()


