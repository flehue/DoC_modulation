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


####FORMATO output_dic[(G,node)] = (corrs,eucs,jumps_high,counts_high)
with open('../fastDMF/output/broken_nodes_resweep_analysis_seeds0-19.pickle', 'rb') as f:
    data = pickle.load(f)
    
with open('../fastDMF/output/optimal_fits_Gsweep_50seeds.pickle', 'rb') as f:
    untouched_data = pickle.load(f)
Go_at_own = (2.48,2.34,2.12)
base_at_own = (0.0006292009131088887, 0.0014041455652912694, 0.0018323863855336832)  
# base_at_CNT = 

    
#%%
with open('../fastDMF/empirical_truth/DoC_mean_FCs.pickle', 'rb') as f:
    emp_fcs = pickle.load(f)
with open('../fastDMF/empirical_truth/jump_distributions_dict_filt_ALLDIM.pickle', 'rb') as f:
    jump_dists_all = pickle.load(f)
with open('../fastDMF/output/optimal_fits_Gsweep_50seeds.pickle', 'rb') as f:
    fitdic = pickle.load(f)
    
##cluster euclidiano en los datos
emp_occs_all = np.loadtxt('../fastDMF/empirical_truth/occupations_3clusters_alldim_euclidean.txt') 

AALlabels = list(pd.read_csv("../sorted_AAL_labels.txt")["label"].values) #nombre de las areas
struct = np.loadtxt("../structural_Deco_AAL.txt")


forces = struct.sum(axis=1)
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
# many_dics_euclidian = {s:{} for s in seeds}

for key in data.keys():
    G,node,seed = key
    many_dics[seed][(G,node)] = data[(G,node,seed)]

#%%
occs = np.zeros((len(Gs),len(nodes),len(["c1","c2","c3"]))) ##last dimension is the number of clusters
euclidians = np.zeros((len(Gs),len(nodes),len(states)))

for seed in seeds: #generate the respective array for each seed
    print(seed)
    dic = many_dics[seed]
    for g,G in enumerate(Gs):
        for n,node in enumerate(nodes):
            occs[g,n,:] += dic[(G,node)][-1] ##occupancies, as many as clusters
            euclidians[g,n,:] += dic[(G,node)][1] ##euclidian, as many as states
occs /= len(seeds) #average over seeds
euclidians /= len(seeds)

#%%plot out KL


Go_by_node= np.zeros((len(nodes),len(states),2)) ##one for KL and other for euclidian
kl_node = np.zeros((len(nodes),len(states)))
euc_node = np.zeros((len(nodes),len(states)))

# occs_opt_reswept = np.zeros((len(nodes),3))

kl_no_resweep = np.zeros((len(nodes),len(states)))
occs_optCNT_resweep = np.zeros((len(nodes),3))
occs_optMCS_resweep = np.zeros((len(nodes),3))
occs_optUWS_resweep = np.zeros((len(nodes),3))
euc_no_resweep = np.zeros((len(nodes),len(states)))
G_CNT_kl = 2.48 ##for seeing broken nodes without resweeping
G_CNT_euc = 2.48

for n,node in enumerate(nodes):
    
    klsCNT = np.array([kl(occs[g,n,:],emp_occs_all[0]).sum() for g in range(len(Gs))]) ##todos los kl del sweep
    kl_node[n,0] = np.min(klsCNT) ##minimo kl
    eucCNT = euclidians[:,n,0] 
    euc_node[n,0] = np.min(eucCNT) #minimo euc
    Go_by_node[n,0,0] = Gs[np.argmin(klsCNT)] ##G donde se alcanza el minimo kl
    Go_by_node[n,0,1] = Gs[np.argmin(eucCNT)] ##G donde se alcanza el minimo euc
    
    klsMCS = np.array([kl(occs[g,n,:],emp_occs_all[1]).sum() for g in range(len(Gs))])
    kl_node[n,1] = np.min(klsMCS)
    eucMCS = euclidians[:,n,1]
    euc_node[n,1] = np.min(eucMCS)
    Go_by_node[n,1,0] = Gs[np.argmin(klsMCS)]
    Go_by_node[n,1,1] = Gs[np.argmin(eucMCS)]
    
    klsUWS = np.array([kl(occs[g,n,:],emp_occs_all[2]).sum() for g in range(len(Gs))])
    kl_node[n,2] = np.min(klsUWS)
    eucUWS = euclidians[:,n,2]
    euc_node[n,2] = np.min(eucUWS)
    Go_by_node[n,2,0] = Gs[np.argmin(klsUWS)]
    Go_by_node[n,2,1] = Gs[np.argmin(eucUWS)]
    
    if n ==56: #precuneus
        y1,y2,y3 = klsCNT,klsMCS,klsUWS;x=Gs
        
    kl_no_resweep[n,0] = klsCNT[Gs == G_CNT_kl] ##no reswept kl
    kl_no_resweep[n,1] = klsMCS[Gs == G_CNT_kl]
    kl_no_resweep[n,2] = klsUWS[Gs == G_CNT_kl]
    
    occs_optCNT_resweep[n,:] = occs[Gs==Gs[np.argmin(klsCNT)],n,:]
    occs_optMCS_resweep[n,:] = occs[Gs==Gs[np.argmin(klsMCS)],n,:]
    occs_optUWS_resweep[n,:] = occs[Gs==Gs[np.argmin(klsUWS)],n,:]
    
    euc_no_resweep[n,0] = euclidians[Gs==G_CNT_euc,n,0] ##no reswept euc
    euc_no_resweep[n,1] = euclidians[Gs==G_CNT_euc,n,1]
    euc_no_resweep[n,2] = euclidians[Gs==G_CNT_euc,n,2]

##let's test precuneus sweep
plt.figure(3)
plt.clf()
plt.plot(x,y1,label="CNT")
plt.plot(x,y2,label="MCS")
plt.plot(x,y3,label="UWS")
plt.show()

#%% occupancies
plt.figure(1)
plt.clf()
plt.suptitle("best KL obtained when reswept with broken node")
plt.subplot(311)
plt.bar(range(90),kl_node[:,0],label="CNT")
plt.hlines(base_at_own[0],0,90,linestyle="dashed",color="red",label="opt_at_own")
plt.xticks([])

plt.subplot(312)
plt.bar(range(90),kl_node[:,1],label="MCS")
plt.hlines(base_at_own[1],0,90,linestyle="dashed",color="red",label="opt_at_own")
plt.xticks([])

plt.subplot(313)
plt.bar(range(90),kl_node[:,2],label="UWS")
plt.hlines(base_at_own[2],0,90,linestyle="dashed",color="red",label="opt_at_own")
plt.xticks(range(90),[f"{i}-{AALlabels[i]}" for i in range(90)],rotation=90)

plt.tight_layout()
plt.show()
#%% euclidian vs broken

plt.figure(2)
plt.clf()
plt.subplot(311)
plt.bar(range(90),euc_node[:,0],label="CNT")
plt.xticks([])

plt.subplot(312)
plt.bar(range(90),euc_node[:,1],label="MCS")
plt.xticks([])

plt.subplot(313)
plt.bar(range(90),euc_node[:,2],label="UWS")
plt.xticks(range(90),[f"{i}-{AALlabels[i]}" for i in range(90)],rotation=90)

plt.tight_layout()
plt.show()

#%% what value of G gives you the optimal

plt.figure(4)
plt.clf()
plt.subplot(311)
plt.bar(range(90),Go_by_node[:,0,0],label="CNT")
# plt.hlines(base_at_own[1][0],0,90,linestyle="dashed",color="red",label="opt_at_own")
plt.xticks([])

plt.subplot(312)
plt.bar(range(90),Go_by_node[:,1,0],label="MCS")
# plt.hlines(base_at_own[1][1],0,90,linestyle="dashed",color="red",label="opt_at_own")
plt.xticks([])

plt.subplot(313)
plt.bar(range(90),Go_by_node[:,2,0],label="UWS")
# plt.hlines(base_at_own[1][2],0,90,linestyle="dashed",color="red",label="opt_at_own")
plt.xticks(range(90),[f"{i}-{AALlabels[i]}" for i in range(90)],rotation=90)

plt.tight_layout()
plt.show()

#%% occupancies broken nodes at the optimal of CNT without resweeping 

plt.figure(4)
plt.clf()
plt.suptitle("no resweep, just break (fixed G at CNT optimal)")

plt.subplot(311)
plt.bar(range(90),kl_no_resweep[:,0],label="CNT")
plt.hlines(base_at_own[0],0,90,linestyle="dashed",color="red",label="opt_at_own")
plt.xticks([])

plt.subplot(312)
plt.bar(range(90),kl_no_resweep[:,1],label="MCS")
plt.hlines(base_at_own[1],0,90,linestyle="dashed",color="red",label="opt_at_own")
plt.xticks([])

plt.subplot(313)
plt.bar(range(90),kl_no_resweep[:,2],label="UWS")
plt.hlines(base_at_own[2],0,90,linestyle="dashed",color="red",label="opt_at_own")
plt.xticks(range(90),[f"{i}-{AALlabels[i]}" for i in range(90)],rotation=90)

plt.tight_layout()
plt.show()

#%%occupations vs broken node
suboccs = occs[Gs==G_CNT_kl,:,:].reshape(90,3)

plt.figure(5)
plt.clf()
plt.subplot(311)
plt.title("occupations when broken node, fixed G at optimal CNT")
bottom = np.zeros(90)
for c in range(3): ##number of clusters
    toplot = suboccs[:,c]
    plt.bar(range(90),toplot,bottom=bottom,label=f"c{c}")
    bottom+=toplot
    
plt.subplot(312)
plt.title("occupations when broken node, reswept")
bottom = np.zeros(90)
for c in range(3): ##number of clusters
    toplot = suboccs[:,c]
    plt.bar(range(90),toplot,bottom=bottom,label=f"c{c}")
    bottom+=toplot
plt.xticks(range(90),[f"{i}-{AALlabels[i]}" for i in range(90)],rotation=90)
plt.ylabel("occupation")
# plt.legend()

plt.subplot(338)
plt.title("empirical occupations")
bottom = np.zeros(3)
for c in range(3): ##number of states:
    toplot = emp_occs_all[:,c]
    plt.bar(range(3),toplot,bottom=bottom,label=f"c{c}")
    bottom+=toplot
plt.xticks(range(3),states)
plt.tight_layout()
plt.show()

#%%occupations of untouched broken vs HMA shit
plt.figure(6)
plt.clf()
plt.suptitle("REMEMBER WE ARE BREAKING THINGS AND LOWER IS BETTER")
#CNT
plt.subplot(431)
plt.title("KL")
plt.scatter(Hin_node,kl_no_resweep[:,0],alpha=0.3)
plt.ylabel("fit to CNT when broken node")
plt.xlabel("integration of node")

plt.subplot(434)
plt.scatter(forces,kl_no_resweep[:,0],alpha=0.3)
plt.ylabel("fit to CNT when broken node")
plt.xlabel("node strength")

#MCS
plt.subplot(432)
plt.scatter(Hin_node,kl_no_resweep[:,1],alpha=0.3)
plt.ylabel("fit to MCS when broken node")
plt.xlabel("integration of node")

plt.subplot(435)
plt.scatter(forces,kl_no_resweep[:,1],alpha=0.3)
plt.ylabel("fit to MCS when broken node")
plt.xlabel("node strength")

#UWS
plt.subplot(433)
plt.scatter(Hin_node,kl_no_resweep[:,2],alpha=0.3)
plt.ylabel("fit to UWS when broken node")
plt.xlabel("integration of node")

plt.subplot(436)
plt.scatter(forces,kl_no_resweep[:,2],alpha=0.3)
plt.ylabel("fit to UWS when broken node")
plt.xlabel("node strength")


#################EUCLIDEAN
plt.subplot(437)
plt.title("euclidian")
plt.scatter(Hin_node,euc_no_resweep[:,0],alpha=0.3)
plt.ylabel("fit to CNT when broken node")
plt.xlabel("integration of node")

plt.subplot(4,3,10)
plt.scatter(forces,euc_no_resweep[:,0],alpha=0.3)
plt.ylabel("fit to CNT when broken node")
plt.xlabel("node strength")

#MCS
plt.subplot(4,3,8)
plt.scatter(Hin_node,euc_no_resweep[:,1],alpha=0.3)
plt.ylabel("fit to MCS when broken node")
plt.xlabel("integration of node")

plt.subplot(4,3,11)
plt.scatter(forces,euc_no_resweep[:,1],alpha=0.3)
plt.ylabel("fit to MCS when broken node")
plt.xlabel("node strength")

#UWS
plt.subplot(4,3,9)
plt.scatter(Hin_node,euc_no_resweep[:,2],alpha=0.3)
plt.ylabel("fit to UWS when broken node")
plt.xlabel("integration of node")

plt.subplot(4,3,12)
plt.scatter(forces,euc_no_resweep[:,2],alpha=0.3)
plt.ylabel("fit to UWS when broken node")
plt.xlabel("node strength")

plt.tight_layout()
plt.show()

#%%occupations of reswept broken vs HMA shit

plt.figure(7)
plt.clf()
plt.suptitle("REMEMBER WE ARE BREAKING THINGS AND LOWER IS BETTER")
#CNT
plt.subplot(431)
plt.title("no resweep")
plt.scatter(Hin_node,kl_node[:,0],alpha=0.3)
plt.ylabel("fit to CNT when broken node")
plt.xlabel("integration of node")

plt.subplot(434)
plt.scatter(forces,kl_node[:,0],alpha=0.3)
plt.ylabel("fit to CNT when broken node")
plt.xlabel("node strength")

#MCS
plt.subplot(432)
plt.scatter(Hin_node,kl_node[:,1],alpha=0.3)
plt.ylabel("fit to MCS when broken node")
plt.xlabel("integration of node")

plt.subplot(435)
plt.scatter(forces,kl_node[:,1],alpha=0.3)
plt.ylabel("fit to MCS when broken node")
plt.xlabel("node strength")

#UWS
plt.subplot(433)
plt.scatter(Hin_node,kl_node[:,2],alpha=0.3)
plt.ylabel("fit to UWS when broken node")
plt.xlabel("integration of node")

plt.subplot(436)
plt.scatter(forces,kl_node[:,2],alpha=0.3)
plt.ylabel("fit to UWS when broken node")
plt.xlabel("node strength")


plt.tight_layout()
plt.show()