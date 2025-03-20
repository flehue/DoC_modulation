# -*- coding: utf-8 -*-
"""
Created on Mon May 13 22:34:52 2024

@author: flehu
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp as ks
from scipy.special import rel_entr as kl #relative entropy = KL distance
import pandas as pd
from scipy.stats import linregress as LR
import HMA
import bct 

"""como comparar las FC al mover G y ADEMAS romper las conexiones!, ver si mejoran... 
la idea es que obtengo un cerebro ya dañado. Y qué pasa si luego de esto mismo le muevo G?
Obviamente obtendré un mejor ajuste a 
"""

with open('../fastDMF/output/broken_nodes_resweep_analysis_seeds0-19.pickle', 'rb') as f:
    data = pickle.load(f)
    
    
#%%
with open('../fastDMF/empirical_truth/DoC_mean_FCs.pickle', 'rb') as f:
    emp_fcs = pickle.load(f)
with open('../fastDMF/empirical_truth/jump_distributions_dict_filt_ALLDIM.pickle', 'rb') as f:
    jump_dists_all = pickle.load(f)
with open('../fastDMF/output/optimal_fits_Gsweep_50seeds.pickle', 'rb') as f:
    fitdic = pickle.load(f)

emp_occs_all = np.loadtxt('../fastDMF/empirical_truth/occupations_3clusters_alldim_euclidean.txt')

AALlabels = list(pd.read_csv("../sorted_AAL_labels.txt")["label"].values) #nombre de las areas
struct = np.loadtxt("../structural_Deco_AAL.txt")

Clus_num,Clus_size,H_all = HMA.Functional_HP(struct)
Hin,Hse = HMA.Balance(struct, Clus_num, Clus_size)
Hin_node,Hse_node = HMA.nodal_measures(struct, Clus_num, Clus_size)

#%% extract parameter and seed values



states = ["CNT","MCS","UWS"]

Gs,nodes,seeds = [],[],[] ##initialize
for key in data.keys(): #fill with what is present
    G,node,seed = key
    if G not in Gs and G < 2.7: ##we'll set a limit for G
        Gs.append(G)
    if node not in nodes:
        nodes.append(node)
    if seed not in seeds:
        seeds.append(seed)

Gs,nodes,seeds = [np.sort(ar) for ar in (Gs,nodes,seeds)]

many_dics = {s:{} for s in seeds} ##we separate by seed
for key in data.keys():
    G,node,seed = key
    many_dics[seed][(G,node)] = data[(G,node,seed)]

#%%
occs = np.zeros((len(Gs),len(nodes),3)) ##last dimension is the number of clusters

for seed in seeds: #generate the respective array for each seed
    print(seed)
    dic = many_dics[seed]
    for g,G in enumerate(Gs):
        for n,node in enumerate(nodes):
            occs[g,n,:] += dic[(G,node)][-1]
occs = occs/len(seeds) #average over seeds

#%%plot out KL

out_node = np.zeros((len(nodes),3))
x = Gs
for n,node in enumerate(nodes):
    
    klsCNT = np.array([kl(occs[g,n,:],emp_occs_all[0]).sum() for g in range(len(Gs))])
    out_node[n,0] = np.min(klsCNT)
    
    
    klsMCS = np.array([kl(occs[g,n,:],emp_occs_all[1]).sum() for g in range(len(Gs))])
    out_node[n,1] = np.min(klsMCS)
    
    klsUWS = np.array([kl(occs[g,n,:],emp_occs_all[2]).sum() for g in range(len(Gs))])
    out_node[n,2] = np.min(klsUWS)
    
    if n ==56: #precuneus
        y1,y2,y3 = klsCNT,klsMCS,klsUWS;x=Gs

##let's test precuneus sweep
plt.figure(3)
plt.clf()
plt.plot(x,y1,label="CNT")
plt.plot(x,y2,label="MCS")
plt.plot(x,y3,label="UWS")
plt.show()

#%%
plt.figure(1)
plt.clf()
plt.subplot(311)
plt.bar(range(90),out_node[:,0],label="CNT")
plt.xticks([])
plt.subplot(312)
plt.bar(range(90),out_node[:,1],label="MCS")
plt.xticks([])
plt.subplot(313)
plt.bar(range(90),out_node[:,2],label="UWS")
plt.xticks(range(90),[f"{i}-{AALlabels[i]}" for i in range(90)],rotation=90)
plt.tight_layout()
plt.show()

##hacemos algunos scatters entre los ajustes de distintas areas
plt.figure(2)
plt.clf()
plt.subplot(331)
x=out_node[:,0]
y=out_node[:,1]
plt.title("CNT vs MCS")
plt.scatter(x,y)
plt.xlabel("CNT");plt.ylabel("MCS")

plt.subplot(335)
x=out_node[:,0]
y=out_node[:,2]
plt.title("CNT vs UWS")
plt.scatter(x,y)
plt.xlabel("CNT");plt.ylabel("UWS")

plt.subplot(339)
x=out_node[:,1]
y=out_node[:,2]
plt.title("MCS vs UWS")
plt.scatter(x,y)
plt.xlabel("MCS");plt.ylabel("UWS")
plt.show()

