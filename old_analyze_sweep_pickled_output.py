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
# from scipy.stats import ks_2samp as ks,gamma,iqr
# from numba import njit

# dictt = {}
# for i in range(50):
#     with open(f'output/simulation_analysis_seed{i}.pickle', 'rb') as f:
#         dictt[i] = pickle.load(f)

# with open(f'output/simulation_analysis_50seeds.pickle', 'wb') as f:
#     pickle.dump(dictt,f)
#%%cargar
##cada tupla es (corrs,eucs,jumps_high,counts_high,jumps_low,counts_low)
with open('../fastDMF/output/simulation_sweepG_analysis_50seeds.pickle', 'rb') as f:
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
Gs = np.sort(list(data[0].keys()))
mask = Gs<2.6
Gs = Gs[mask]

ks_jumps_fits_low = np.zeros((len(Gs),3,50))
means_jumps_low = np.zeros((len(Gs),50))
stds_jumps_low = np.zeros((len(Gs),50))

ks_jumps_fits_high = np.zeros((len(Gs),3,50))
means_jumps_high = np.zeros((len(Gs),50))
stds_jumps_high = np.zeros((len(Gs),50))

KL_occs_fits_low = np.zeros((len(Gs),3,50))
KL_occs_fits_high = np.zeros((len(Gs),3,50))

euc_fits = np.zeros((len(Gs),3,50))
corr_fits = np.zeros((len(Gs),3,50))

# ks_occs_fits = np.zeros((len(Gs),3,50))




# opt_lowjump_dist = {G:[] for G in Gs}
all_counts_low = np.zeros((len(Gs),3,50))
gamma_params_sim = {G:[] for G in Gs}

for seed in range(50):
    print(seed)
    subdata = data[seed]
    for i,G in enumerate(Gs):
        corrs,eucs,jumps_high,counts_high,_,_ = subdata[G] ##espacios sobrantes son cosas de low dimension
    
        ks_jumps_fits_high[i,:,seed] = [ks(jumps_high,jump_dists_all[s])[0] for s in states]
        means_jumps_high[i,seed] = np.mean(jumps_high)
        stds_jumps_high[i,seed] = np.std(jumps_high)
        
        a,loc,scale = gamma.fit(jumps_high,floc=0)
        gamma_params_sim[G].append((a,scale))
        
        euc_fits[i,:,seed] = eucs
        corr_fits[i,:,seed] = corrs
        KL_occs_fits_high[i,:,seed] = [kl(counts_high,emp_occs_all[i]).sum() for i in range(len(states))]

print(gamma_params_sim)
colors = ["tab:blue","tab:orange","tab:green"]
# all_counts_low = all_counts_low[:,:,5]

#%% cosas dinamicas
fits_jumps_means_high = ks_jumps_fits_high.mean(axis=2) ##G versus la media de los saltos
fits_jumps_stds_high = ks_jumps_fits_high.std(axis=2) #G versus la std de los saltos

#%%


plt.figure(1)
plt.clf()
plt.suptitle("jump KS fit to data, low=15dim and all dimensions, 50 seeds")

##high
plt.subplot(321)
plt.title("jumps_fit_high")
Go_jumps_high = tuple([Gs[np.argmin(fits_jumps_means_high[:,i])] for i in range(3)])
o_jumps_high = tuple([np.min(fits_jumps_means_high[:,i]) for i in range(3)])
for i in range(3):
    plt.plot(Gs,fits_jumps_means_high[:,i],label=states[i],color=colors[i]) #G vs fiteo KS por estado
    plt.fill_between(Gs,
                     fits_jumps_means_high[:,i]-fits_jumps_stds_high[:,i],
                     fits_jumps_means_high[:,i]+fits_jumps_stds_high[:,i],
                     alpha=0.4,color=colors[i])
     #el optimo corresponde al minimo de KS distance
    plt.vlines(Go_jumps_high,0,0.5,linestyle="dashed",color=colors) #destacar optimos
print(f"ks jumps HIGHDIM KS-minima at G={Go_jumps_high}")
plt.legend()
plt.subplot(324)
plt.title("mean jump length high")
plt.plot(Gs,means_jumps_high.mean(axis=1))

plt.vlines(Go_jumps_high,6,9,linestyle="dashed",color=colors)
plt.subplot(326)
plt.title("std jump length high")
plt.plot(Gs,stds_jumps_high.mean(axis=1))
plt.vlines(Go_jumps_high,2,7,linestyle="dashed",color=colors)
plt.tight_layout()
plt.show()

# plt.figure(2)
# plt.clf()
# plt.hist(opt_CNT_dist.flatten(),density=True,alpha=0.5,label="simCNT",bins=50)
# plt.hist(jump_dists_15["CNT"],density=True,alpha=0.5,label="empCNT",bins=50)
# plt.legend()
# plt.show()


    
#%% cosas estaticas

euc_fits_mean = euc_fits.mean(axis=2)
corr_fits_mean = corr_fits.mean(axis=2)

C1 = 0.3
euccorr_fits = euc_fits/(corr_fits+C1)
euccorr_fits_mean = euccorr_fits.mean(axis=2)

plt.figure(2)
plt.clf()
#ahora ploteamos fiteo FC, 50 semillas
plt.subplot(311) ###############ajuste euclidea
plt.title("euclidean(empFC,simFC)")
Go_euc = tuple([Gs[np.argmin(euc_fits_mean[:,i])] for i in range(3)])
o_euc = tuple([np.min(euc_fits_mean[:,i]) for i in range(3)])
print("euclidean FC minima at G=",Go_euc)
plt.plot(Gs,euc_fits_mean)
plt.vlines(Go_euc,10,20,color=colors,linestyle="dashed")

plt.subplot(312) ##########ajuste corr
plt.title("corr(empFC,simFC)")
Go_corr = tuple([Gs[np.argmax(corr_fits_mean[:,i])] for i in range(3)])
o_corr = tuple([np.max(corr_fits_mean[:,i]) for i in range(3)])
print("correlation FC max at G=",Go_corr)
plt.plot(Gs,corr_fits_mean)
plt.vlines(Go_corr,0.2,0.5,color=colors,linestyle="dashed")

plt.subplot(313) ####################ajuste euccorr con C1 
plt.title("euccorr(empFC,simFC)")
Go_euccorr = tuple([Gs[np.argmin(euccorr_fits_mean[:,i])] for i in range(3)])
o_euccorr = tuple([np.min(euccorr_fits_mean[:,i]) for i in range(3)])
print("euccorr FC min at G=",Go_euccorr)
plt.plot(Gs,euccorr_fits_mean)
plt.vlines(Go_euccorr,10,60,color=colors,linestyle="dashed")

plt.tight_layout()
plt.show()
#%%
fits_occs_low = KL_occs_fits_low.mean(axis=2)
Go_KL_low = tuple([Gs[np.argmin(fits_occs_low[:,i])] for i in range(3)])
o_KL_low = tuple([np.min(fits_occs_low[:,i]) for i in range(3)])

fits_occs_high = KL_occs_fits_high.mean(axis=2)
Go_KL_high = tuple([Gs[np.argmin(fits_occs_high[:,i])] for i in range(3)])
o_KL_high = tuple([np.min(fits_occs_high[:,i]) for i in range(3)])

plt.figure(3,figsize=(5,3))
plt.clf()
# plt.suptitle("KL-distance PMS empirical/simulated",fontsize=15)

plt.subplot(111)
plt.plot(Gs,fits_occs_high[:,0],label="CNT")
plt.plot(Gs,fits_occs_high[:,1],label="MCS")
plt.plot(Gs,fits_occs_high[:,2],label="UWS")
plt.vlines(Go_KL_high,0,0.5,color=colors,linestyle="dashed")
plt.legend(fontsize=10)
plt.xticks([1.5,1.75,2,2.25,2.5],[1.5,1.75,2,2.25,2.5],fontsize=13)
plt.yticks([0,0.5,1,1.5],[0,0.5,1,1.5],fontsize=13)
plt.xlabel("coupling G",fontsize=15);plt.ylabel("KL-divergence",fontsize=15)
plt.tight_layout()
# plt.savefig("../../ICMNS2024_Dublin/sweepG.png",dpi=300)
plt.show()

print("KL distance LOWDIM minima at G=",Go_KL_low)
print("KL distance HIGHDIM minima at G=",Go_KL_high)



#%%save shit and compare the metrics 

names = ["jump_l","jump_h","euc","1-corr FC","euccorr","KL_occ"]
names_01 = ["jump_l","jump_h","1-corr FC","KL_occ"]

optimals = (Go_jumps_high,Go_euc,Go_corr,Go_euccorr,Go_KL_high)
optimals = np.array(optimals)
opti_dic = {"CNT":tuple([o[0] for o in optimals]),
            "MCS":tuple([o[1] for o in optimals]),
            "UWS":tuple([o[2] for o in optimals])}


opti_mats = {}
opti_mats_01 = {}
for s,state in enumerate(states):
    
    opti_Gs = opti_dic[state]
    
    opti_mat = np.zeros((6,6)) #la primera columna guarda el valor de G
    opti_mat_01 = np.zeros((4,4)) #la primera columna guarda el valor de G
    
    k = 0
    for i in range(6):
        Go_val = opti_Gs[i]
        
        Go_mask = (Gs==Go_val)
        # jump_low_val = float(ks_jumps_fits_low[Go_mask,:,:][:,s,:].mean())
        jump_high_val = float(ks_jumps_fits_high[Go_mask,:,:][:,s,:].mean())
        euc_val = float(euc_fits[Go_mask,:,:][:,s,:].mean())
        corr_val = 1-float(corr_fits[Go_mask,:,:][:,s,:].mean())
        euccorr_val = float(euccorr_fits[Go_mask,:,:][:,s,:].mean())
        KL_high_val = float(KL_occs_fits_high[Go_mask,:,:][:,s,:].mean())
        
        # opti_mat[i,:] = (jump_low_val,jump_high_val,euc_val,corr_val,euccorr_val,KL_high_val)
        if i not in [2,4]:
            opti_mat_01[k,:] = (jump_low_val,jump_high_val,corr_val,KL_high_val)
            k+=1
    opti_mats[state] = opti_mat
    opti_mats_01[state] = opti_mat_01
    
plt.figure(5)
plt.clf()
plt.suptitle("transfered optimals for observables, rows are optimals for observable",fontsize=15)
for s,state in enumerate(states):
    plt.subplot2grid((3,3),(1,s))
    plt.title(state)
    plt.imshow(opti_mats_01[state],vmin=0)
    plt.colorbar()
    plt.xticks(range(4),names_01,rotation=45);plt.yticks(range(4),names_01)
    plt.xlabel("to");plt.ylabel("from")
    
    
    plt.subplot2grid((3,3),(2,s))
    plt.title(f"overall transfer {state} (LOWER IS BETTER)")
    plt.bar(range(4),opti_mats_01[state].sum(axis=1))
    plt.xticks(range(4),names_01,rotation=0)
    
    plt.subplot2grid((3,3),(0,s))
    plt.title(f"G values {state}")
    # plt.title(state)
    plt.bar(range(6),optimals[:,s])
    plt.xticks(range(6),names,rotation=0)
    plt.ylim([2,2.6])
    
plt.tight_layout()
plt.show()
        
#%%analize gamma at optimal 

shapesCNT = [t[0] for t in gamma_params_sim[2.5]]
shapesMCS = [t[0] for t in gamma_params_sim[2.32]]
shapesUWS = [t[0] for t in gamma_params_sim[2.02]]
scaleCNT = [t[1] for t in gamma_params_sim[2.5]]
scaleMCS = [t[1] for t in gamma_params_sim[2.32]]
scaleUWS = [t[1] for t in gamma_params_sim[2.02]]


allshapes = []
allscales = []
for G in Gs:
    shape = np.mean([gamma_params_sim[G][i][0] for i in (range(50))])
    allshapes.append(shape)
    scale = np.mean([gamma_params_sim[G][i][1] for i in (range(50))])
    allscales.append(scale)


plt.figure(76)
plt.clf()
plt.subplot(221)
plt.ylabel("gamma shape parameter",fontsize=13)
plt.yticks(fontsize=13)
plt.plot(Gs,allshapes)
plt.xticks([])
# plt.xlabel("G")
plt.vlines(Go_KL_low,0,7,color=colors,linestyle="dashed")
plt.subplot(223)
plt.ylabel("gamma scale parameter",fontsize=13)
plt.yticks(fontsize=13)
plt.plot(Gs,allscales)
plt.xlabel("coupling G", fontsize=13)
plt.xticks(fontsize=13)
plt.vlines(Go_KL_low,0,7,color=colors,linestyle="dashed")

plt.subplot(222)
plt.title("shape parameter of jumps at KL optimal",fontsize=13)
sns.boxplot([shapesCNT,shapesMCS,shapesUWS])
sns.swarmplot([shapesCNT,shapesMCS,shapesUWS],color="black")
plt.xticks(range(3),states,fontsize=13)
plt.yticks(fontsize=13)
plt.subplot(224)
plt.title("scale parameter of jumps at KL optimal",fontsize=13)
sns.boxplot([scaleCNT,scaleMCS,scaleUWS])
sns.swarmplot([scaleCNT,scaleMCS,scaleUWS],color="black")
plt.xticks(range(3),states,fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.show()
        
        
        

#%%


# fitdic = {"info":"formato ((G_CNT,G_MCS,G_UWS),(val_CNT,val_MCS,val_UWS)). KL low and high are the same",
#           "KS_jumps_low":(Go_jumps_low,o_jumps_low),
#           "KS_jumps_high":(Go_jumps_high,o_jumps_high),
#           "FC_euc":(Go_euc,o_euc),
#           "FC_corr":(Go_corr,o_corr),
#           "FC_euccorr":(Go_euccorr,o_euccorr),
#           "KL_occs_high": (Go_KL_high,o_KL_high)}
# with open('output/optimal_fits_Gsweep_50seeds.pickle', 'wb') as f:
#     pickle.dump(fitdic,f)


#%%
struct = np.loadtxt("../structural_Deco_AAL.txt")
ranmat = np.zeros((90,90))
ranmat[13,:] = 1;ranmat[:,13] =1
plt.figure(7)
plt.clf()
plt.subplot(121)
plt.title("original structural ")
plt.imshow(struct)
plt.subplot(122)
plt.title("row and column out")
plt.imshow(ranmat)
plt.show()

#%%

G_CNT = 2.47
print([float(fits_occs_high[:,s][Gs==G_CNT]) for s in (0,1,2)])



