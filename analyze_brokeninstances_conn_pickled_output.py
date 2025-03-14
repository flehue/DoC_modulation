# -*- coding: utf-8 -*-
"""
Created on Sat May 25 15:05:50 2024

@author: flehu
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp as ks,gamma
from scipy.special import rel_entr as kl #relative entropy = KL distance
from numba import njit

# dictt = {}
# for i in range(50):
#     with open(f'output/broken_simulation_analysis_DELAYS_seed{i}.pickle', 'rb') as f:
#         dictt[i] = pickle.load(f)

# with open(f'output/broken_simulation_analysis_DELAYS_50seeds.pickle', 'wb') as f:
#     pickle.dump(dictt,f)
    
# halt
#%%cargar
##cada tupla es (corrs,eucs,jumps_outer,counts_outer,jumps_inner,counts_inner)
print(f"KILLING CONNECTIONS")

def entropy_of_gamma(shape,scale):
    from scipy.special import gamma as gammaf,digamma as digammaf
    entropy = shape + np.log(scale)+ np.log(gammaf(shape)) + (1-shape)*digammaf(shape)
    return entropy
    

with open(f'../fastDMF/output/broken_simulation_analysis_INNER_50seeds.pickle', 'rb') as f:
    data_inner = pickle.load(f)
with open(f'../fastDMF/output/broken_simulation_analysis_OUTER_50seeds.pickle', 'rb') as f:
    data_outer = pickle.load(f)
with open('../fastDMF/output/out_in_wards_entries_100reps_2mayo_0percentile.pickle', 'rb') as f:
    broken_dic = pickle.load(f)

with open('../fastDMF/output/optimal_fits_Gsweep_50seeds.pickle', 'rb') as f:
    fitdic = pickle.load(f)
# with open('output/struct_nonzero_sorted_by_DELAY_2mayo_0percentile.pickle', 'rb') as f:
#     broken_dic_delays = pickle.load(f)

# all_strengths_inner = broken_dic[f"inner_strengths"]
all_strengths_inner = broken_dic["inner_strengths"]
all_strengths_outer = broken_dic[f"outer_strengths"]
struct = np.loadtxt("../structural_Deco_AAL.txt")

# del f
#%%
with open('../fastDMF/empirical_truth/DoC_mean_FCs.pickle', 'rb') as f:
    emp_fcs = pickle.load(f)
with open('../fastDMF/empirical_truth/jump_distributions_dict_filt_PCA15.pickle', 'rb') as f:
    jump_dists_15 = pickle.load(f)
with open('../fastDMF/empirical_truth/jump_distributions_dict_filt_ALLDIM.pickle', 'rb') as f:
    jump_dists_all = pickle.load(f)
del f

emp_occs_15 = np.loadtxt('../fastDMF/empirical_truth/occupations_3clusters_alldim_euclidean.txt')#,alinner_pickle=True
# emp_occs_all = np.loadtxt('empirical_truth/occupations_3clusters_alldim_euclidean.txt')


#%%
names = ["all_by_st","outer_by_st","within_by_st","all_rand_1","all_rand_2","all_by_dist"]

with open("../fastDMF/output/broken_instances_st_wheres_24mayo.pickle","rb") as f:
    broken_instances = pickle.load(f)

states = ["CNT","MCS","UWS"]
IDs = range(654) ##todas las distintas estructurales tomadas
IDs_dist = range(1000)
# mask = Gs<2.6
# sts_inner = np.array([0] + list(all_strengths_inner.cumsum()[:len(IDs)-1]))/struct.sum()*100 ##porcentaje de destruccion y weas
# sts_outer = np.array([0] + list(all_strengths_outer.cumsum()[:len(IDs)-1]))/struct.sum()*100 ##porcentaje de destruccion y weas
nseeds = 20



means_jumps = {name:np.zeros((len(IDs),nseeds)) for name in names};means_jumps.update({"all_by_dist":np.zeros((len(IDs_dist),nseeds))})
stds_jumps = {name:np.zeros((len(IDs),nseeds)) for name in names};stds_jumps.update({"all_by_dist":np.zeros((len(IDs_dist),nseeds))})
kl_occs_fits = {name:np.zeros((len(IDs),3,nseeds)) for name in names};kl_occs_fits.update({"all_by_dist":np.zeros((len(IDs_dist),3,nseeds))})
euc_fits = {name:np.zeros((len(IDs),3,nseeds)) for name in names};euc_fits.update({"all_by_dist":np.zeros((len(IDs_dist),3,nseeds))})
corr_fits = {name:np.zeros((len(IDs),3,nseeds)) for name in names};corr_fits.update({"all_by_dist":np.zeros((len(IDs_dist),3,nseeds))})
opt_CNT_dist = {name:np.zeros((191,nseeds)) for name in names}
all_counts = {name:np.zeros((len(IDs),3,nseeds)) for name in names};all_counts.update({"all_by_dist":np.zeros((len(IDs_dist),3,nseeds))})

sts_dic = {}

gamma_params_sim = {n:np.zeros((len(IDs),2,nseeds)) for n in names};gamma_params_sim.update({"all_by_dist":np.zeros((len(IDs_dist),2,nseeds))})

for n,name in enumerate(names):
    
    sts = broken_instances[name][0]
    sts_dic[name] = sts
    if name == "all_by_dist":
        with open(f'../fastDMF/output/broken_conn_ORDER_{name}_analysis.pickle', 'rb') as f:
            broken_dic = pickle.load(f)
        with open(f'../fastDMF/output/broken_conn_ORDER_{name}_analysis_from_654.pickle', 'rb') as f:
            broken_dic2 = pickle.load(f)
        broken_dic.update(broken_dic2) 
        print(len(broken_dic))
        these_IDs = IDs_dist
    else:
        with open(f'../fastDMF/output/broken_conn_ORDER_{name}_analysis.pickle', 'rb') as f:
            broken_dic = pickle.load(f)
        these_IDs = IDs
    print("processing")
    for seed in range(nseeds):
        print(seed)
        for i,ID in enumerate(these_IDs):
            corr,euc,jumps_high,counts_high,jumps_low,counts_low = broken_dic[ID,seed]
            all_counts[name][i,:,seed] = counts_high
            
            a,loc,scale = gamma.fit(jumps_low,floc=0)
            gamma_params_sim[name][i,:,seed] = a,scale
            
            means_jumps[name][i,seed] = np.mean(jumps_low)
            stds_jumps[name][i,seed] = np.std(jumps_low)
            euc_fits[name][i,:,seed] = euc
            corr_fits[name][i,:,seed] = corr
            
            kl_occs_fits[name][i,:,seed] = [kl(counts_high,emp_occs_15[i]).sum() for i in range(len(states))]
    
    
all_counts = {name:all_counts[name].mean(axis=2) for name in names}


#%%
plt.figure(1)
plt.clf()
plt.suptitle("KL OCCS")
for n,name in enumerate(names):
    plt.subplot(3,2,n+1)
    plt.title(name)
    if name =="all_by_dist":
        x = np.cumsum(sts_dic[name][:len(IDs_dist)])/(struct.sum()/2)*100
    else:
        x = np.cumsum(sts_dic[name][:len(IDs)])/(struct.sum()/2)*100
    plt.plot(x,kl_occs_fits[name].mean(axis=2)[:,0],label="CNT")
    plt.plot(x,kl_occs_fits[name].mean(axis=2)[:,1],label="MCS")
    plt.plot(x,kl_occs_fits[name].mean(axis=2)[:,2],label="UWS")
    plt.legend()
    plt.xlabel("broken connectivity %")
plt.tight_layout()
plt.show()

##per state

colors = ["tab:blue","tab:orange","tab:green","red","violet","tab:brown"]

plt.figure(2)
plt.clf()
# plt.suptitle("KL OCCS")
for s,state in enumerate(states):
    
    basal = fitdic["KL_occs_high"][1][s]
    plt.subplot2grid((2,3),(0,s))
    for n,name in enumerate(names):
        plt.title("KL OCC to "+state)
        if name =="all_by_dist":
            x = IDs_dist
        else:
            x = IDs
        y = kl_occs_fits[name].mean(axis=2)[:,s] #corresponde al estado
        
        xo = x[np.argmin(y)]
        plt.vlines(xo,y.min()-0.03,y.min()+0.03,color=colors[n],linestyle="dashed")
        
        
        plt.plot(x,y,label=name,color=colors[n])
        plt.xlabel("n° of broken connections")
    plt.hlines(basal,np.min(x),np.max(x),color="black",linestyle="dashed")
    plt.legend()
    
    
    plt.subplot2grid((2,3),(1,s))
    for n,name in enumerate(names):
        if name =="all_by_dist":
            x = np.cumsum(sts_dic[name][:len(IDs_dist)])/(struct.sum()/2)*100
        else:
            x = np.cumsum(sts_dic[name][:len(IDs)])/(struct.sum()/2)*100
        y = kl_occs_fits[name].mean(axis=2)[:,s] #corresponde al estado
        xo = x[np.argmin(y)]
        plt.vlines(xo,y.min()-0.03,y.min()+0.03,color=colors[n],linestyle="dashed")
        
        
        plt.plot(x,y,label=name,color=colors[n])
        plt.xlabel("broken connectivity %")
        # plt.xlim([0,20])
        # plt.ylim([0,0.1])
    plt.hlines(basal,0,100,color="black",linestyle="dashed")
    plt.legend()
plt.tight_layout()
plt.show()



     

#%%





plt.figure(3)
plt.clf()

plt.subplot(321)
plt.title("mean velocities vs n° connections")
for n,name in enumerate(names):
    y = means_jumps[name].mean(axis=1)
    if name =="all_by_dist":
        x = IDs_dist
    else:
        x = IDs
    plt.plot(x,y,label=name)
plt.legend()
plt.xlabel("broken connections")
plt.subplot(322)
plt.title("mean velocities vs % of broken connectivity")
for n,name in enumerate(names):
    y = means_jumps[name].mean(axis=1)
    if name =="all_by_dist":
        x = np.cumsum(sts_dic[name][:len(IDs_dist)])/(struct.sum()/2)*100
    else:
        x = np.cumsum(sts_dic[name][:len(IDs)])/(struct.sum()/2)*100
    plt.plot(x,y,label=name)
plt.xlabel("broken connectivity %")
plt.legend()

plt.subplot(323)
plt.title("gamma shape parameter vs n° connections")
for n,name in enumerate(names):
    if name =="all_by_dist":
        x = IDs_dist
    else:
        x = IDs
    y = gamma_params_sim[name].mean(axis=2)[:,0]
    plt.plot(x,y,label=name)
plt.xlabel("broken connections")
plt.legend()
plt.subplot(324)
plt.title("gamma shape parameter vs % of connectivity")
for n,name in enumerate(names):
    if name =="all_by_dist":
        x = np.cumsum(sts_dic[name][:len(IDs_dist)])/(struct.sum()/2)*100
    else:
        x = np.cumsum(sts_dic[name][:len(IDs)])/(struct.sum()/2)*100
    y = gamma_params_sim[name].mean(axis=2)[:,0]
    plt.plot(x,y,label=name)
plt.xlabel("broken connectivity")

plt.subplot(325)
plt.title("gamma scale parameter vs n° connections")
for n,name in enumerate(names):
    if name =="all_by_dist":
        x = IDs_dist
    else:
        x = IDs
    y = gamma_params_sim[name].mean(axis=2)[:,1]
    plt.plot(x,y,label=name)
plt.xlabel("broken connections")
plt.legend()
plt.subplot(326)
plt.title("gamma scale parameter vs % of connectivity")
for n,name in enumerate(names):
    if name =="all_by_dist":
        x = np.cumsum(sts_dic[name][:len(IDs_dist)])/(struct.sum()/2)*100
    else:
        x = np.cumsum(sts_dic[name][:len(IDs)])/(struct.sum()/2)*100
    y = gamma_params_sim[name].mean(axis=2)[:,1]
    plt.plot(x,y,label=name)
plt.xlabel("broken connectivity")
plt.tight_layout()
plt.show()

plt.figure(4)
plt.clf()
for n,name in enumerate(names):
    if name == "all_by_dist":
        x = np.cumsum(sts_dic[name][:len(IDs_dist)])/(struct.sum()/2)*100
    else:
        x = np.cumsum(sts_dic[name][:len(IDs)])/(struct.sum()/2)*100
    plt.subplot2grid((6,4),(n,0))
    plt.title("euclidean vs n° "+name)
    y = euc_fits[name].mean(axis=2)
    plt.plot(x,y,label=name)
    
    plt.subplot2grid((6,4),(n,1))
    plt.title("euclidean vs % "+name)
    y = euc_fits[name].mean(axis=2)
    plt.plot(x,y,label=name)
    
    plt.subplot2grid((6,4),(n,2))
    plt.title("corr vs n° "+name)
    y = corr_fits[name].mean(axis=2)
    plt.plot(x,y,label=name)
    
    plt.subplot2grid((6,4),(n,3))
    plt.title("corr vs % "+name)
    y = corr_fits[name].mean(axis=2)
    plt.plot(x,y,label=name)
    
plt.tight_layout()
plt.show()



#%% cosas dinamicas
# fits_jumps_means_inner = ks_jumps_fits_inner.mean(axis=2) ##G versus la media de los saltos
# fits_jumps_stds_inner = ks_jumps_fits_inner.std(axis=2) #G versus la std de los saltos
# fits_jumps_means_outer = ks_jumps_fits_outer.mean(axis=2) ##G versus la media de los saltos
# fits_jumps_stds_outer = ks_jumps_fits_outer.std(axis=2) #G versus la std de los saltos

#%%
colors_inner = ["lightblue","lightcoral","palegreen"]
colors_outer = ["navy","darkred","darkgreen"]


# plt.figure(1)
# plt.clf()
# plt.suptitle(f"jump KS fit to data, inner=15dim {nseeds} seeds")
# plt.subplot(231)
# plt.title("jumps_fit")
# sto_inner = [sts_inner[np.argmin(fits_jumps_means_inner[:,i])] for i in range(3)]
# sto_outer = [sts_outer[np.argmin(fits_jumps_means_outer[:,i])] for i in range(3)]

# for i in range(3):
#     plt.plot(sts_inner,fits_jumps_means_inner[:,i],label=states[i]+"_inner",color=colors_inner[i]) #G vs fiteo KS por estado
# for i in range(3):
#     plt.plot(sts_outer,fits_jumps_means_outer[:,i],label=states[i]+"_outer",color=colors_outer[i])
# plt.vlines(sto_inner,0,0.5,linestyle="dashed",color=colors_inner) #destacar optimos
# print("ks jumps innerDIM KS-minima at st=",[f"{a:.3f}%" for a in sto_inner])
# plt.vlines(sto_outer,0,0.5,linestyle="dashed",color=colors_outer) #destacar optimos
# plt.xlabel("broken_connectivity (% of total)")
# print("ks jumps outerDIM KS-minima at st=",[f"{a:.3f}%" for a in sto_outer])
# plt.legend(loc="lower right")
# plt.subplot(232)
# plt.title("mean jump length")
# plt.plot(sts_inner,means_jumps_inner.mean(axis=1),label="inner")
# plt.vlines(sto_inner,6,9,linestyle="dashed",color=colors_inner)
# plt.plot(sts_outer,means_jumps_outer.mean(axis=1),label="outer")
# plt.vlines(sto_outer,6,9,linestyle="dashed",color=colors_outer)
# plt.xlabel("broken_connectivity (% of total)")
# plt.legend()

# plt.subplot(233)
# plt.title("std jump length")
# plt.plot(sts_inner,stds_jumps_inner.mean(axis=1),label="inner")
# plt.vlines(sto_inner,2,7,linestyle="dashed",color=colors_inner)
# plt.plot(sts_outer,stds_jumps_outer.mean(axis=1),label="outer")
# plt.vlines(sto_outer,2,7,linestyle="dashed",color=colors_outer)
# plt.legend()
# plt.xlabel("broken_connectivity (% of total)")

# argmins_inner = [np.argmin(fits_jumps_means_inner[:,i]) for i in range(3)]
# sto_inner = sts_inner[argmins_inner]#[sts_inner[np.argmin(fits_occs_inner[:,i])] for i in range(3)]

# argmins_outer = [np.argmin(fits_jumps_means_outer[:,i]) for i in range(3)]
# sto_outer = sts_outer[argmins_outer]#[sts_inner[np.argmin(fits_occs_inner[:,i])] for i in range(3)]

# plt.subplot(234)
# plt.title("how much sheer connectivity breaking to get from CNT EUCLIDEAN")
# plt.bar([0,1,3,4],[sto_inner[1],sto_outer[1],
#         sto_inner[2],sto_outer[2]],
#         color= [colors_inner[1],colors_outer[1],colors_inner[2],colors_outer[2]],
#         tick_label = ["MCS\nbreaking inner","MCS\nbreaking outer","UWS\nbreaking inner","UWS\nbreaking outer"])
# plt.ylabel("total connectivity percentage %")

# plt.subplot(235)
# plt.title("how many broken connections to get from CNT EUCLIDEAN")
# plt.bar([0,1,3,4],[argmins_inner[1],argmins_outer[1],
#         argmins_inner[2],argmins_outer[2]],
#         color= [colors_inner[1],colors_outer[1],colors_inner[2],colors_outer[2]],
#         tick_label = ["MCS\nbreaking inner","MCS\nbreaking outer","UWS\nbreaking inner","UWS\nbreaking outer"])
# plt.ylabel("n° of connections")
# plt.tight_layout()
# plt.show()

    
#%% grafico con observables de ajuste de FC estatica a cada estado 

euc_fits_mean_inner = euc_fits_inner.mean(axis=2)
corr_fits_mean_inner = corr_fits_inner.mean(axis=2)

euc_fits_mean_outer = euc_fits_outer.mean(axis=2)
corr_fits_mean_outer = corr_fits_outer.mean(axis=2)

C1 = 0.3
euccorr_fits_mean_inner = euc_fits_mean_inner/(corr_fits_mean_inner+C1)
euccorr_fits_mean_outer = euc_fits_mean_outer/(corr_fits_mean_outer+C1)

plt.figure(3)
plt.clf()
plt.suptitle("FC fit observables (broken inner, broken outer)")
#ahora ploteamos fiteo FC, 50 semillas
plt.subplot(231) ###############ajuste euclidea
plt.title("euclidean(empFC,simFC)")
sto_inner = [sts_inner[np.argmin(euc_fits_mean_inner[:,i])] for i in range(3)]
sto_outer = [sts_outer[np.argmin(euc_fits_mean_outer[:,i])] for i in range(3)]
print("euclidean FC minima at inner st=",[f"{a:.3f}%" for a in sto_inner])
print("euclidean FC minima at outer st=",[f"{a:.3f}%" for a in sto_outer])
for i in range(3):
    plt.plot(sts_inner,euc_fits_mean_inner[:,i],color=colors_inner[i])
for i in range(3):
    plt.plot(sts_outer,euc_fits_mean_outer[:,i],color=colors_outer[i])
plt.vlines(sto_inner,10,20,color=colors_inner,linestyle="dashed")
plt.vlines(sto_outer,10,20,color=colors_outer,linestyle="dashed")
plt.xlim([-0.2,12])

plt.subplot(232) ##########ajuste corr
plt.title("corr(empFC,simFC)")
sto_inner = [sts_inner[np.argmax(corr_fits_mean_inner[:,i])] for i in range(3)]
sto_outer = [sts_outer[np.argmax(corr_fits_mean_outer[:,i])] for i in range(3)]
print("correlation FC max at st=",[f"{a:.3f}%" for a in sto_inner])
for i in range(3):
    plt.plot(sts_inner,corr_fits_mean_inner[:,i],color=colors_inner[i])
for i in range(3):
    plt.plot(sts_outer,corr_fits_mean_outer[:,i],color=colors_outer[i])
plt.vlines(sto_inner,0.2,0.5,color=colors_inner,linestyle="dashed")
plt.vlines(sto_outer,0.2,0.5,color=colors_outer,linestyle="dashed")

plt.subplot(233) ####################ajuste euccorr con C1 
plt.title("euccorr(empFC,simFC)")
sto_inner = [sts_inner[np.argmin(euccorr_fits_mean_inner[:,i])] for i in range(3)]
sto_outer = [sts_outer[np.argmin(euccorr_fits_mean_outer[:,i])] for i in range(3)]
print("euccorr FC min at inner st=",[f"{a:.3f}%" for a in sto_inner])
print("euccorr FC min at outer st=",[f"{a:.3f}%" for a in sto_outer])
for i in range(3):
    plt.plot(sts_inner,euccorr_fits_mean_inner[:,i],color=colors_inner[i])
for i in range(3):
    plt.plot(sts_outer,euccorr_fits_mean_outer[:,i],color=colors_outer[i])
plt.vlines(sto_inner,10,40,color=colors_inner,linestyle="dashed")
plt.vlines(sto_outer,10,40,color=colors_outer,linestyle="dashed")
plt.tight_layout()
plt.show()

# grafico con euclideana y con cuanto hay que romper para llegar a los optimos

argmins_inner = [np.argmin(euc_fits_mean_inner[:,i]) for i in range(3)]
sto_inner = sts_inner[argmins_inner]#[sts_inner[np.argmin(fits_occs_inner[:,i])] for i in range(3)]

argmins_outer = [np.argmin(euc_fits_mean_outer[:,i]) for i in range(3)]
sto_outer = sts_outer[argmins_outer]#[sts_inner[np.argmin(fits_occs_inner[:,i])] for i in range(3)]


plt.subplot(234)
plt.title("how much sheer connectivity breaking to get from CNT EUCLIDEAN")
plt.bar([0,1,3,4],[sto_inner[1],sto_outer[1],
        sto_inner[2],sto_outer[2]],
        color= [colors_inner[1],colors_outer[1],colors_inner[2],colors_outer[2]],
        tick_label = ["MCS\nbreaking inner","MCS\nbreaking outer","UWS\nbreaking inner","UWS\nbreaking outer"])
plt.ylabel("total connectivity percentage %")

plt.subplot(235)
plt.title("how many broken connections to get from CNT EUCLIDEAN")
plt.bar([0,1,3,4],[argmins_inner[1],argmins_outer[1],
        argmins_inner[2],argmins_outer[2]],
        color= [colors_inner[1],colors_outer[1],colors_inner[2],colors_outer[2]],
        tick_label = ["MCS\nbreaking inner","MCS\nbreaking outer","UWS\nbreaking inner","UWS\nbreaking outer"])
plt.ylabel("n° of connections")
plt.tight_layout()
plt.show()





#%% ajuste KL a las ocupaciones
"""cosas interesantes que mostrar
*OBSERVABLE: resta de ajuste WAKE y UWS en el optimo, versus las semillas, como en luppi
    esto 
*OPTIMO: posicion de los optimos
    **en terminos de conectividad rota
    **en terminos de cantidad total de conexiones rotas
"""


fits_occs_inner = ks_occs_fits_inner.mean(axis=2)  ##KL fits, dims:(conexiones rotas, state)
argmins_inner = [np.argmin(fits_occs_inner[:,i]) for i in range(3)]
sto_inner = sts_inner[argmins_inner]#[sts_inner[np.argmin(fits_occs_inner[:,i])] for i in range(3)]


fits_occs_outer = ks_occs_fits_outer.mean(axis=2)
argmins_outer = [np.argmin(fits_occs_outer[:,i]) for i in range(3)]
sto_outer = sts_outer[argmins_outer]#[sts_inner[np.argmin(fits_occs_inner[:,i])] for i in range(3)]

plt.figure(5)
plt.clf()
plt.subplot(211)
plt.title("KL-distance occupancies empirical/simulated (optimals dashed)\nfrom CNT optimal",fontsize=20)
for i in range(3):
    if i !=0:
        plt.plot(sts_inner,fits_occs_inner[:,i],label=states[i]+"_inner",color=colors_inner[i])
    else:
        plt.plot(sts_inner,fits_occs_inner[:,i],color=colors_inner[i])
for i in range(3):
    if i!=0:
        plt.plot(sts_outer,fits_occs_outer[:,i],label=states[i]+"_outer",color=colors_outer[i])
    else:
        plt.plot(sts_outer,fits_occs_outer[:,i],color=colors_outer[i])
plt.vlines(sto_inner,0,0.5,color=colors_inner,linestyle="dashed")
plt.vlines(sto_outer,0,0.5,color=colors_outer,linestyle="dashed")
plt.xlabel("broken_connectivity (% of total)")
plt.legend()
plt.xlim([-0.2,10])

plt.subplot(223)
plt.title("how much sheer connectivity breaking to get from CNT")
plt.bar([0,1,3,4],[sto_inner[1],sto_outer[1],
        sto_inner[2],sto_outer[2]],
        color= [colors_inner[1],colors_outer[1],colors_inner[2],colors_outer[2]],
        tick_label = ["MCS\nbreaking\ninner","MCS\nbreaking\nouter","UWS\nbreaking\ninner","UWS\nbreaking\nouter"])
plt.ylabel("total connectivity percentage %",fontsize=15)

plt.subplot(224)
plt.title("how many broken connections to get from CNT")
plt.bar([0,1,3,4],[argmins_inner[1],argmins_outer[1],
        argmins_inner[2],argmins_outer[2]],
        color= [colors_inner[1],colors_outer[1],colors_inner[2],colors_outer[2]],
        tick_label = ["MCS\nbreaking\ninner","MCS\nbreaking\nouter","UWS\nbreaking\ninner","UWS\nbreaking\nouter"])
plt.ylabel("n° of connections",fontsize=15)

plt.show()

print("KL distance occs minima at inner st=",[f"{a:.3f}%" for a in sto_inner])
print("KL distance occs minima at outer st=",[f"{a:.3f}%" for a in sto_outer])

#%%
## ivan: un tema es que esto no te dice donde parte algo y termina algo
## como que faltaria una interpretacion de lo que está haciendo la partición

#seria interesante ver que tipo de conexiones rompo, no solamente la cantidad... quizas algunas son mas criticas que otras
#podria hacerse el matcheo cortical - subcortical (solo 2 modulos)
#se puede hacer un baseline chance con areas al azar...
#analizar que modulos tienen conexiones inter mas importantes
#se puede identificar que modulo es interesante y cual no, viendo que porcentaje de matar el modulo dentro de si mismo mata y bla bla 
#pregunta: importa donde estoy o importa el numero?, el overall total.
##se puede ver incluso con los que salen
#se pueden comparar nodos hub versus no hub :O, y uno puede hacerlo por cantidad de conexiones o fuerza overall

###que pasa si mato nodos en general

#cosa interesante: comparar el ajuste a MCS y UWS con conectoma sano, versus su respectivo optimo con conectoma roto









