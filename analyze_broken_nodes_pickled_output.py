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
# dictt = {}
# for i in range(50):
#     with open(f'output/broken_node_HOMOTOPIC_sim_analysis_seed{i}.pickle', 'rb') as f:
#         dictt[i] = pickle.load(f)

# with open('output/broken_node_HOMOTOPIC_sim_analysis_50seeds.pickle', 'wb') as f:
#     pickle.dump(dictt,f)


"""como comparar las FC al mover G y ADEMAS romper las conexiones!, ver si mejoran... 
la idea es que obtengo un cerebro ya dañado. Y qué pasa si luego de esto mismo le muevo G?
Obviamente obtendré un mejor ajuste a 
"""
    
with open('../fastDMF/output/broken_node_sim_analysis_50seeds.pickle', 'rb') as f:
    data = pickle.load(f)
    
#%%
with open('../fastDMF/empirical_truth/DoC_mean_FCs.pickle', 'rb') as f:
    emp_fcs = pickle.load(f)
with open('../fastDMF/empirical_truth/jump_distributions_dict_filt_PCA15.pickle', 'rb') as f:
    jump_dists_15 = pickle.load(f)
with open('../fastDMF/empirical_truth/jump_distributions_dict_filt_ALLDIM.pickle', 'rb') as f:
    jump_dists_all = pickle.load(f)
with open('../fastDMF/output/optimal_fits_Gsweep_50seeds.pickle', 'rb') as f:
    fitdic = pickle.load(f)
KL_at_optimal_CNT = [0.026417709157886354, 0.1603670411573415, 0.4496975412541529]
    
with open('../fastDMF/output/out_in_wards_entries_100reps_2mayo_0percentile.pickle', 'rb') as f:
    broken_dic = pickle.load(f)


with open('../fastDMF/empirical_truth/modules_degrees_symmetrized45-45.pickle', 'rb') as f:
    degrees_dic = pickle.load(f)
    
emp_occs_15 = np.loadtxt('../fastDMF/empirical_truth/occupations_3clusters_15dim_euclidean.txt')#,allow_pickle=True)

emp_occs_all = np.loadtxt('../fastDMF/empirical_truth/occupations_3clusters_alldim_euclidean.txt')
# emp_occs_all = np.loadtxt('empirical_truth/occupations_3clusters_alldim_euclidean.txt')

AALlabels = list(pd.read_csv("../sorted_AAL_labels.txt")["label"].values) #nombre de las areas
struct = np.loadtxt("../structural_Deco_AAL.txt")

Clus_num,Clus_size,H_all = HMA.Functional_HP(struct)
Hin,Hse = HMA.Balance(struct, Clus_num, Clus_size)
Hin_node,Hse_node = HMA.nodal_measures(struct, Clus_num, Clus_size)

#%%
states = ["CNT","MCS","UWS"]
nodes = np.sort(list(data[0].keys()))
# mask = Gs<2.6

ks_jumps_fits_low = np.zeros((len(nodes),3,50))
means_jumps_low = np.zeros((len(nodes),50))
stds_jumps_low = np.zeros((len(nodes),50))

# ks_jumps_fits_high = np.zeros((len(nodes),3,50))
# means_jumps_high = np.zeros((len(nodes),50))
# stds_jumps_high = np.zeros((len(nodes),50))

kl_occs_fits_low = np.zeros((len(nodes),3,50))
kl_occs_fits_high = np.zeros((len(nodes),3,50))

euc_fits = np.zeros((len(nodes),3,50))
corr_fits = np.zeros((len(nodes),3,50))

# ks_occs_fits = np.zeros((len(Gs),3,50))

# opt_CNT_dist = np.zeros((191,50))
all_counts_low = np.zeros((len(nodes),3,50))
for seed in range(50):
    print(seed)
    subdata = data[seed]
    for i,node in enumerate(nodes):
        corrs,eucs,jumps_high,counts_high,jumps_low,counts_low = subdata[node]
        all_counts_low[i,:,seed] = counts_low
        
        # print(counts_low.shape)
        ks_jumps_fits_low[i,:,seed] = [ks(jumps_low,jump_dists_15[s])[0] for s in states]
        means_jumps_low[i,seed] = np.mean(jumps_low)
        stds_jumps_low[i,seed] = np.std(jumps_low)
        
        euc_fits[i,:,seed] = eucs
        corr_fits[i,:,seed] = corrs
        
        
        # ks_occsCNT,ks_occsMCS,ks_occsUWS = [ks(counts_low,emp_occs_15[i])[0] for i in range(len(states))]
        kl_occs_fits_low[i,:,seed] = [kl(counts_low,emp_occs_15[i]).sum() for i in range(len(states))]
        kl_occs_fits_high[i,:,seed] = [kl(counts_high,emp_occs_all[i]).sum() for i in range(len(states))]
        # ks_occsCNT,ks_occsMCS,ks_occsUWS = [np.linalg.norm(counts_low-emp_occs_15[i]) for i in range(len(states))]

colors = ["tab:blue","tab:orange","tab:green"]
all_counts_low = all_counts_low[:,:,5]



#%% cosas dinamicas

fits_jumps_means_low = ks_jumps_fits_low.mean(axis=2) ##G versus la media de los saltos
fits_jumps_stds_low = ks_jumps_fits_low.std(axis=2) #G versus la std de los saltos
# fits_jumps_means_high = ks_jumps_fits_high.mean(axis=2) ##G versus la media de los saltos
# fits_jumps_stds_high = ks_jumps_fits_high.std(axis=2) #G versus la std de los saltos

#%%
alfa = 0.3
plt.figure(1)
plt.clf()
plt.suptitle("jump KS fit to data, low=15dim and all dimensions, 50 seeds")

nodeo = [nodes[np.argmin(fits_jumps_means_low[:,i])] for i in range(3)]
for i in range(3):
    plt.subplot2grid((3,3),(i,0),colspan=2)
    # plt.title(f"ks jumps {states[i]}")
    plt.bar(nodes,fits_jumps_means_low[:,i],label=states[i],color=colors[i]) #G vs fiteo KS por estado
    plt.legend()
    if i==2:
        plt.xticks(range(90),AALlabels,rotation=90)
    else:
        plt.xticks([])
    

plt.subplot2grid((3,3),(0,2))
plt.title("jumps mean ")
plt.bar(nodes,means_jumps_low.mean(axis=1))

plt.subplot2grid((3,3),(1,2))
plt.title("jumps std ")
plt.bar(nodes,stds_jumps_low.mean(axis=1))
plt.tight_layout()
plt.show()

    
#%% cosas estaticas

euc_fits_mean = euc_fits.mean(axis=2)
corr_fits_mean = corr_fits.mean(axis=2)

C1 = 0.3
euccorr_fits_mean = euc_fits_mean/(corr_fits_mean+C1)

plt.figure(2)
plt.clf()
#ahora ploteamos fiteo FC, 50 semillas

#euclideana
nodeo = [nodes[np.argmin(euc_fits_mean[:,i])] for i in range(3)]
print("euclidean FC minima at node=",nodeo)
for s,state in enumerate(states):
    plt.subplot2grid((3,1),(s,0)) ###############ajuste euclidea
    basal = fitdic["FC_euc"][1][s]
    plt.hlines(basal,0,90,color="red",linestyle="dashed")
    # plt.title("euclidean(empFC,simFC)")
    plt.bar(nodes,euc_fits_mean[:,s],color="tab:blue",label=f"euclidean {state}")
    plt.legend()
    plt.xticks([])

# nodeo = [nodes[np.argmax(corr_fits_mean[:,i])] for i in range(3)]
# print("correlation FC max at node=",nodeo)
# for s,state in enumerate(states):
#     plt.subplot2grid((6,1),(s+3,0)) ##########ajuste corr
#     # plt.title("corr(empFC,simFC)")    
#     basal = fitdic["FC_corr"][1][s]
#     plt.hlines(basal,0,90,color="red",linestyle="dashed")
#     plt.bar(nodes,corr_fits_mean[:,s],color="tab:green",label=f"correlation {state}")
#     plt.xticks([])
#     # plt.vlines(nodeo,0.2,0.5,color=colors,linestyle="dashed")
#     plt.legend()
#     if s ==2:
#         plt.xticks(range(90),AALlabels,rotation=90)
#     else:
#         plt.xticks([])


# nodeo = [nodes[np.argmin(euccorr_fits_mean[:,i])] for i in range(3)]
# print("euccorr FC min at node=",nodeo)
# for s,state in enumerate(states):
#     plt.subplot2grid((9,1),(s+3+3,0)) ####################ajuste euccorr con C1 
#     # plt.title("euccorr(empFC,simFC)")
#     plt.bar(nodes,euccorr_fits_mean[:,s],color="tab:blue",label=f"euclidean {state}")
#     # plt.vlines(nodeo,10,60,color=colors,linestyle="dashed")
    
    
plt.tight_layout()
plt.show()
#%%
alfa = 0.8
fits_occs_low = kl_occs_fits_low.mean(axis=2)
nodeo_low = [nodes[np.argmin(fits_occs_low[:,i])] for i in range(3)]

fits_occs_high = kl_occs_fits_high.mean(axis=2)
nodeo_high = [nodes[np.argmin(fits_occs_high[:,i])] for i in range(3)]

x = nodes

basals = fitdic["KL_occs_high"][1]

plt.figure(3)
plt.clf()
plt.suptitle("KL distance empirical-simulated. LOWER IS BETTER")
for s,state in enumerate(states):
    plt.subplot(3,1,s+1)
    plt.bar(nodes,fits_occs_low[:,s],label=state,alpha=alfa)
    plt.hlines(basals[s],x.min(),x.max(),color="red",linestyle="dashed",label=f"base at {state} optimal")
    # if s>0:
    plt.hlines(KL_at_optimal_CNT[s],x.min(),x.max(),color="blue",linestyle="dashed",label=f"base at CNT optimal")
    if s ==2:
        plt.xticks(range(90),AALlabels,rotation=90)
    plt.legend()
plt.tight_layout()
plt.show()
#%%
alfa=0.7
plt.figure(4,figsize=(6,5))
plt.clf()
plt.subplot(5,1,1)
basal = fitdic["KL_occs_high"][1][0]
plt.bar(nodes,fits_occs_low[:,0],label="CNT",alpha=alfa)
plt.hlines(basal,-2,91,color="black",linestyle="dashed",label="optimal basal fit")
plt.yticks([0,0.04,0.08,0.12],[0,0.04,0.08,0.12],fontsize=15)
plt.xticks([])
plt.ylabel("KL CNT",fontsize=15)
plt.xlim([-2,91])
plt.xticks([])

plt.subplot(5,1,2)
basal = fitdic["KL_occs_high"][1][2]
plt.bar(nodes,fits_occs_low[:,2],label="UWS",alpha=alfa,color="tab:green")
plt.hlines(basal,-2,91,color="black",linestyle="dashed",label="optimal basal fit")
plt.yticks([0,0.2,0.4],[0,0.2,0.4],fontsize=15)
plt.xticks([])
plt.xlim([-2,91])
plt.ylabel("KL UWS",fontsize=15)


plt.subplot(5,1,3)
plt.bar(nodes,struct.sum(axis=1),label="node force",alpha=alfa,color="grey")
plt.yticks([0,1,2,3,4],fontsize=15)
plt.xticks([])
plt.xlim([-2,91])
plt.ylabel("node force",fontsize=15)


plt.subplot(5,1,4)
plt.bar(nodes,bct.clustering_coef_wu(struct),label="clustering coef",alpha=alfa,color="grey")
plt.yticks(fontsize=15)
plt.xticks([])
plt.xlim([-2,91])
plt.ylabel("clustering\ncoef",fontsize=15)

plt.subplot(5,1,5)
# basal = fitdic["KL_occs_high"][1][2]
plt.bar(nodes,Hin_node,label="Hin",alpha=alfa,color="grey")
# plt.hlines(basal,x.min()-2,x.max()+2,color="black",linestyle="dashed",label="optimal basal fit")
plt.xticks(range(90),AALlabels,rotation=90,fontsize=13)
plt.yticks([0,0.005,0.01],fontsize=15)
# plt.xticks([])
plt.xlabel("removed area ID",fontsize=15)
plt.xlim([-2,91])
plt.ylabel("integration\ncomponent",fontsize=15)


plt.tight_layout()
# plt.savefig("../../ICMNS2024_Dublin/node_vs_brokenCNT",dpi=300)
plt.show()



#los cambios mas grandes de distribucion son con el precuneo izquierdo y derecho
#tambien es interesantisimo lo que ocurre con el cingulado posterior izquierdo y derecho
#tambien el cuneo y el calcarino


#%%correlacionando ajuste a UWS con degrees
deg_in,deg_out,deg,c = [degrees_dic[key] for key in ("deg_in","deg_out","deg_all","c")]
names = ("strength inwards broken","strength outwards broken","strength overall broken","binarized deg  broken")

deg_centrality = np.sum(struct>0,axis=1)


props = dict(boxstyle='round', facecolor='white', alpha=0.8)
plt.figure(5)
plt.clf()
plt.suptitle("fit vs degree of nodes. LOWER IS BETTER")
for s,state in enumerate(states):
    y = fits_occs_high[:,s]
    basal = fitdic["KL_occs_high"][1][s]
    for j in range(4):
        x = [deg_in,deg_out,deg,deg_centrality][j]
    
        ax=plt.subplot2grid((3,4),(s,j))
        
        plt.scatter(x,y)
        plt.hlines(basal,x.min(),x.max(),color="red",linestyle="dashed",label=f"sweep fit {state} (basal)")
    
        res = LR(x,y)
        plt.plot(x,res.intercept+x*res.slope,linestyle="dashed",color="black")
        textstr = '\n'.join([f"r={res.rvalue:.3f}",f"p={res.pvalue:.3f}"])
        ax.text(0.1, 0.3, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='center', bbox=props)
        
        plt.legend(loc="upper right")
        if s !=2:
            plt.xticks([])
        else:
            plt.xlabel(names[j],fontsize=15)
            plt.xticks(fontsize=12)
        if j !=0:
            plt.yticks([])
        else:
            plt.ylabel(f"fit to {state}",fontsize=15)
            plt.yticks(fontsize=12)
            
plt.tight_layout()
plt.show()

print("KL distance LOWDIM minima at node=",nodeo_low)
print("KL distance HIGHDIM minima at node=",nodeo_high)


#vemos que cuando mayor es el residuo del grado saliendo del nodo frente al grado,
#osea, mientras más excepcional su grado saliendo es. Quitarlo empeora el ajuste a MCS

#%% integration segregation HMA per node

print("\nmost binarized degree areas\n",[AALlabels[i] for i in np.argsort(deg_centrality)[::-1][:10]])
print("\nmost outside connected areas\n",[AALlabels[i] for i in np.argsort(deg_out)[::-1][:10]])
print("\nmost within connected areas\n",[AALlabels[i] for i in np.argsort(deg_in)[::-1][:10]])
print("\nmost CONNECTED areas\n",[AALlabels[i] for i in np.argsort(deg)[::-1][:10]])
print("\nmost integrated areas\n",[AALlabels[i] for i in np.argsort(Hin_node)[::-1][:10]])
print("\nmost segregated areas\n",[AALlabels[i] for i in np.argsort(Hse_node)[::-1][:10]])


plt.figure(7)
plt.clf()
plt.suptitle("integration segregation HMA analysis. Red is best fit sweeping G")

#precuneus es 33 y 56
#cingulum es 17 y 72

for s,state in enumerate(states):
    basal = fitdic["KL_occs_high"][1][s]
    y = fits_occs_high[:,s]
    for j in range(2):
        x = [Hin_node,Hse_node][j]
        ax=plt.subplot2grid((3,2),(s,j))
        plt.scatter(x,y)
        plt.hlines(basal,x.min(),x.max(),color="red",linestyle="dashed")
        res = LR(x,y)
        plt.plot(x,res.intercept+x*res.slope,linestyle="dashed",color="black")
        textstr = '\n'.join([f"r={res.rvalue:.3f}",f"p={res.pvalue:.3f}"])
        ax.text(0.1, 0.1, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='center', bbox=props)
        # plt.ylabel(f"{state} fit",fontsize=15)        
        
        if s !=2:
            plt.xticks([])
        else:
            plt.xlabel(f"{['integration','segregation'][j]} component node",fontsize=15)
            plt.xticks(fontsize=12)
        if j !=0:
            plt.yticks([])
        else:
            plt.ylabel(f"fit to {state}",fontsize=15)
            plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
#%%
alfa = 0.8

plt.figure(8,figsize=(7,3))
plt.clf()

plt.suptitle("Here we are breaking areas")

ax=plt.subplot(261)
x = Hse_node
y = fits_occs_high[:,0]
plt.scatter(x,y,alpha=alfa)
plt.scatter(x[33],y[33],color="red",label="Precuneus")
plt.scatter(x[56],y[56],color="red")
plt.scatter(x[17],y[17],color="darkgoldenrod",label="Posterior Cingulum")
plt.scatter(x[72],y[72],color="darkgoldenrod")
plt.legend(loc="upper left")
plt.hlines(basal,x.min(),x.max(),color="black",linestyle="dashed")
res = LR(x,y)
plt.ylabel("fit to CNT",fontsize=13)
# plt.yticks([0,0.25,0.5],[0,0.25,0.5],fontsize=13)
plt.plot(x,res.intercept+x*res.slope,linestyle="dashed",color="black")
textstr = '\n'.join([f"r={res.rvalue:.3f}",f"p={res.pvalue:.3f}"])
ax.text(0.1, 0.1, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='center', bbox=props)
# plt.ylabel("fit to UWS",fontsize=13)
plt.yticks(fontsize=13)
# plt.yticks([])
plt.xticks([0.2,0.3,0.4,0.5],[0.2,0.3,0.4,0.5],fontsize=13)
plt.xlabel("Segregation component",fontsize=15)


ax=plt.subplot(262)
x = Hin_node
y = fits_occs_high[:,0]
plt.scatter(x,y,alpha=alfa)
plt.scatter(x[33],y[33],color="red",label="Precuneus")
plt.scatter(x[56],y[56],color="red")
plt.scatter(x[17],y[17],color="darkgoldenrod",label="Posterior Cingulum")
plt.scatter(x[72],y[72],color="darkgoldenrod")
plt.legend(loc="upper left")
plt.hlines(basal,x.min(),x.max(),color="black",linestyle="dashed")
res = LR(x,y)
plt.plot(x,res.intercept+x*res.slope,linestyle="dashed",color="black")
textstr = '\n'.join([f"r={res.rvalue:.3f}",f"p={res.pvalue:.3f}"])
ax.text(0.1, 0.1, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='center', bbox=props)
plt.xticks([0,0.002,0.004,0.006,0.008],[0,0.002,0.004,0.006,0.008],fontsize=13)
plt.yticks([])
plt.xlabel("Integration component",fontsize=15)


node_force = struct.sum(axis=1)
ax=plt.subplot(263)
x =node_force
y = fits_occs_high[:,0]
plt.scatter(x,y,alpha=alfa)
plt.scatter(x[33],y[33],color="red",label="Precuneus")
plt.scatter(x[56],y[56],color="red")
plt.scatter(x[17],y[17],color="darkgoldenrod",label="Posterior Cingulum")
plt.scatter(x[72],y[72],color="darkgoldenrod")
plt.legend(loc="upper left")
plt.hlines(basal,x.min(),x.max(),color="black",linestyle="dashed")
res = LR(x,y)
plt.plot(x,res.intercept+x*res.slope,linestyle="dashed",color="black")
textstr = '\n'.join([f"r={res.rvalue:.3f}",f"p={res.pvalue:.3f}"])
ax.text(0.1, 0.1, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='center', bbox=props)
# plt.ylabel("fit to UWS",fontsize=13)
# plt.yticks([0,0.25,0.5],[0,0.25,0.5],fontsize=13)
plt.yticks([])
plt.xticks([1.5,2,2.5,3,3.5,4],fontsize=13)
plt.xlabel("node force",fontsize=15)

#integration regressed by node force 


res = LR(node_force,Hin_node)
x = Hin_node - (res.intercept+node_force*res.slope)
y = fits_occs_high[:,0]
ax = plt.subplot(264)
plt.scatter(x,y,alpha=alfa)
plt.scatter(x[33],y[33],color="red",label="Precuneus")
plt.scatter(x[56],y[56],color="red")
plt.scatter(x[17],y[17],color="darkgoldenrod",label="Posterior Cingulum")
plt.scatter(x[72],y[72],color="darkgoldenrod")
plt.legend(loc="upper left")
plt.hlines(basal,x.min(),x.max(),color="black",linestyle="dashed")
res = LR(x,y)
plt.plot(x,res.intercept+x*res.slope,linestyle="dashed",color="black")
textstr = '\n'.join([f"r={res.rvalue:.3f}",f"p={res.pvalue:.3f}"])
ax.text(0.1, 0.1, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='center', bbox=props)
# plt.ylabel("fit to UWS",fontsize=13)
# plt.yticks([0,0.25,0.5],[0,0.25,0.5],fontsize=13)
plt.yticks([])
plt.xticks([-0.005,-0.0025,0,0.0025,0.005],fontsize=13)
plt.xlabel("Hin node regressed by node force",fontsize=15)

ax=plt.subplot(265)
x = bct.clustering_coef_wu(struct)
y = fits_occs_high[:,0]
plt.scatter(x,y,alpha=alfa)
plt.scatter(x[33],y[33],color="red",label="Precuneus")
plt.scatter(x[56],y[56],color="red")
plt.scatter(x[17],y[17],color="darkgoldenrod",label="Posterior Cingulum")
plt.scatter(x[72],y[72],color="darkgoldenrod")
plt.legend(loc="upper right")
plt.hlines(basal,x.min(),x.max(),color="black",linestyle="dashed")
res = LR(x,y)
plt.ylabel("fit to CNT",fontsize=13)
plt.yticks([0,0.25,0.5],[0,0.25,0.5],fontsize=13)
plt.plot(x,res.intercept+x*res.slope,linestyle="dashed",color="black")
textstr = '\n'.join([f"r={res.rvalue:.3f}",f"p={res.pvalue:.3f}"])
ax.text(0.1, 0.1, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='center', bbox=props)
# plt.ylabel("fit to UWS",fontsize=13)
# plt.yticks([0,0.25,0.5],[0,0.25,0.5],fontsize=13)
# plt.yticks([])
# plt.xticks([0.2,0.3,0.4,0.5],[0.2,0.3,0.4,0.5],fontsize=13)
plt.xlabel("clustering coef",fontsize=15)

ax=plt.subplot(266)
x = bct.participation_coef(struct,ci=broken_dic["c"])
y = fits_occs_high[:,0]
plt.scatter(x,y,alpha=alfa)
plt.scatter(x[33],y[33],color="red",label="Precuneus")
plt.scatter(x[56],y[56],color="red")
plt.scatter(x[17],y[17],color="darkgoldenrod",label="Posterior Cingulum")
plt.scatter(x[72],y[72],color="darkgoldenrod")
plt.legend(loc="upper right")
plt.hlines(basal,x.min(),x.max(),color="black",linestyle="dashed")
res = LR(x,y)
plt.ylabel("fit to CNT",fontsize=13)
plt.yticks([0,0.25,0.5],[0,0.25,0.5],fontsize=13)
plt.plot(x,res.intercept+x*res.slope,linestyle="dashed",color="black")
textstr = '\n'.join([f"r={res.rvalue:.3f}",f"p={res.pvalue:.3f}"])
ax.text(0.1, 0.1, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='center', bbox=props)
# plt.ylabel("fit to UWS",fontsize=13)
# plt.yticks([0,0.25,0.5],[0,0.25,0.5],fontsize=13)
# plt.yticks([])
# plt.xticks([0.2,0.3,0.4,0.5],[0.2,0.3,0.4,0.5],fontsize=13)
plt.xlabel("participation coef",fontsize=15)



############UWS
ax=plt.subplot(267)
x = Hse_node
y = fits_occs_high[:,2]
plt.scatter(x,y,alpha=alfa)
plt.scatter(x[33],y[33],color="red",label="Precuneus")
plt.scatter(x[56],y[56],color="red")
plt.scatter(x[17],y[17],color="darkgoldenrod",label="Posterior Cingulum")
plt.scatter(x[72],y[72],color="darkgoldenrod")
plt.legend(loc="upper right")
plt.hlines(basal,x.min(),x.max(),color="black",linestyle="dashed")
res = LR(x,y)
plt.ylabel("fit to UWS",fontsize=13)
plt.yticks([0,0.25,0.5],[0,0.25,0.5],fontsize=13)
plt.plot(x,res.intercept+x*res.slope,linestyle="dashed",color="black")
textstr = '\n'.join([f"r={res.rvalue:.3f}",f"p={res.pvalue:.3f}"])
ax.text(0.1, 0.1, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='center', bbox=props)
# plt.ylabel("fit to UWS",fontsize=13)
plt.yticks([0,0.25,0.5],[0,0.25,0.5],fontsize=13)
# plt.yticks([])
plt.xticks([0.2,0.3,0.4,0.5],[0.2,0.3,0.4,0.5],fontsize=13)
plt.xlabel("Segregation component",fontsize=15)


ax=plt.subplot(268)
x = Hin_node
y = fits_occs_high[:,2]
plt.scatter(x,y,alpha=alfa)
plt.scatter(x[33],y[33],color="red",label="Precuneus")
plt.scatter(x[56],y[56],color="red")
plt.scatter(x[17],y[17],color="darkgoldenrod",label="Posterior Cingulum")
plt.scatter(x[72],y[72],color="darkgoldenrod")
plt.legend(loc="upper right")
plt.hlines(basal,x.min(),x.max(),color="black",linestyle="dashed")
res = LR(x,y)
plt.plot(x,res.intercept+x*res.slope,linestyle="dashed",color="black")
textstr = '\n'.join([f"r={res.rvalue:.3f}",f"p={res.pvalue:.3f}"])
ax.text(0.1, 0.1, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='center', bbox=props)
plt.xticks([0,0.002,0.004,0.006,0.008],[0,0.002,0.004,0.006,0.008],fontsize=13)
plt.yticks([])
plt.xlabel("Integration component",fontsize=15)


node_force = struct.sum(axis=1)
ax=plt.subplot(269)
x =node_force
y = fits_occs_high[:,2]
plt.scatter(x,y,alpha=alfa)
plt.scatter(x[33],y[33],color="red",label="Precuneus")
plt.scatter(x[56],y[56],color="red")
plt.scatter(x[17],y[17],color="darkgoldenrod",label="Posterior Cingulum")
plt.scatter(x[72],y[72],color="darkgoldenrod")
plt.legend(loc="upper right")
plt.hlines(basal,x.min(),x.max(),color="black",linestyle="dashed")
res = LR(x,y)
plt.plot(x,res.intercept+x*res.slope,linestyle="dashed",color="black")
textstr = '\n'.join([f"r={res.rvalue:.3f}",f"p={res.pvalue:.3f}"])
ax.text(0.1, 0.1, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='center', bbox=props)
# plt.ylabel("fit to UWS",fontsize=13)
# plt.yticks([0,0.25,0.5],[0,0.25,0.5],fontsize=13)
plt.yticks([])
plt.xticks([1.5,2,2.5,3,3.5,4],fontsize=13)
plt.xlabel("node force",fontsize=15)

#integration regressed by node force 


res = LR(node_force,Hin_node)
x = Hin_node - (res.intercept+node_force*res.slope)
y = fits_occs_high[:,2]
ax = plt.subplot(2,6,10)
plt.scatter(x,y,alpha=alfa)
plt.scatter(x[33],y[33],color="red",label="Precuneus")
plt.scatter(x[56],y[56],color="red")
plt.scatter(x[17],y[17],color="darkgoldenrod",label="Posterior Cingulum")
plt.scatter(x[72],y[72],color="darkgoldenrod")
plt.legend(loc="upper right")
plt.hlines(basal,x.min(),x.max(),color="black",linestyle="dashed")
res = LR(x,y)
plt.plot(x,res.intercept+x*res.slope,linestyle="dashed",color="black")
textstr = '\n'.join([f"r={res.rvalue:.3f}",f"p={res.pvalue:.3f}"])
ax.text(0.1, 0.1, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='center', bbox=props)
# plt.ylabel("fit to UWS",fontsize=13)
# plt.yticks([0,0.25,0.5],[0,0.25,0.5],fontsize=13)
plt.yticks([])
plt.xticks([-0.005,-0.0025,0,0.0025,0.005],fontsize=13)
plt.xlabel("Hin node regressed by node force",fontsize=15)

ax=plt.subplot(2,6,11)
x = bct.clustering_coef_wu(struct)
y = fits_occs_high[:,2]
plt.scatter(x,y,alpha=alfa)
plt.scatter(x[33],y[33],color="red",label="Precuneus")
plt.scatter(x[56],y[56],color="red")
plt.scatter(x[17],y[17],color="darkgoldenrod",label="Posterior Cingulum")
plt.scatter(x[72],y[72],color="darkgoldenrod")
plt.legend(loc="upper right")
plt.hlines(basal,x.min(),x.max(),color="black",linestyle="dashed")
res = LR(x,y)
plt.ylabel("fit to UWS",fontsize=13)
plt.yticks([0,0.25,0.5],[0,0.25,0.5],fontsize=13)
plt.plot(x,res.intercept+x*res.slope,linestyle="dashed",color="black")
textstr = '\n'.join([f"r={res.rvalue:.3f}",f"p={res.pvalue:.3f}"])
ax.text(0.1, 0.1, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='center', bbox=props)
# plt.ylabel("fit to UWS",fontsize=13)
# plt.yticks([0,0.25,0.5],[0,0.25,0.5],fontsize=13)
# plt.yticks([])
# plt.xticks([0.2,0.3,0.4,0.5],[0.2,0.3,0.4,0.5],fontsize=13)
plt.xlabel("clustering coef",fontsize=15)

ax=plt.subplot(2,6,12)
x = bct.participation_coef(struct,ci=broken_dic["c"])
y = fits_occs_high[:,2]
plt.scatter(x,y,alpha=alfa)
plt.scatter(x[33],y[33],color="red",label="Precuneus")
plt.scatter(x[56],y[56],color="red")
plt.scatter(x[17],y[17],color="darkgoldenrod",label="Posterior Cingulum")
plt.scatter(x[72],y[72],color="darkgoldenrod")
plt.legend(loc="upper right")
plt.hlines(basal,x.min(),x.max(),color="black",linestyle="dashed")
res = LR(x,y)
plt.ylabel("fit to UWS",fontsize=13)
plt.yticks([0,0.25,0.5],[0,0.25,0.5],fontsize=13)
plt.plot(x,res.intercept+x*res.slope,linestyle="dashed",color="black")
textstr = '\n'.join([f"r={res.rvalue:.3f}",f"p={res.pvalue:.3f}"])
ax.text(0.1, 0.1, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='center', bbox=props)
# plt.ylabel("fit to UWS",fontsize=13)
# plt.yticks([0,0.25,0.5],[0,0.25,0.5],fontsize=13)
# plt.yticks([])
# plt.xticks([0.2,0.3,0.4,0.5],[0.2,0.3,0.4,0.5],fontsize=13)
plt.xlabel("participation coef",fontsize=15)

plt.tight_layout()
plt.savefig("../../ICMNS2024_Dublin/int_seg_UWS",dpi=300)
plt.show()


#%%correlacion entre medidas de integracion y fuerza de nodo
names = ["part","clust","Hin","Hse"]

x0 = node_force
x1 = bct.participation_coef(struct,ci=broken_dic["c"])
x2 = bct.clustering_coef_wu(struct)
x3 = Hin_node
x4 = Hse_node
plt.figure(9)
plt.clf()
for i in range(4):
    corri = np.corrcoef(x0,eval(f'x{i+1}'))[0,1]
    plt.subplot(2,2,i+1)
    plt.title(names[i]+f" corr={corri:.3f}")
    plt.scatter(x0,eval(f"x{i+1}"))
    plt.xlabel("node force")
    plt.ylabel(names[i])
plt.show()

plt.figure(10)
plt.clf()
for i in range(4):
    plt.subplot(4,1,i+1)
    plt.bar(range(90),eval(f'x{i}'))
    plt.ylabel((["node force"]+ names)[i])
    if i < 3:
        plt.xticks([])
    else:
        plt.xticks(range(90),AALlabels,rotation=90)
        # plt.xlabel("node force")
plt.tight_layout()
plt.show()


