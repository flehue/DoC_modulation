# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:59:01 2023

@author: flehu
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt 
import utils
from sklearn.decomposition import PCA

###cargar datos
alldata = loadmat("data/ts_Coma_ParisAALsymm.mat") ##columnas son estados
lower = np.tril_indices(90,k=-1) ##ENTRADAS DE LA SUBDIAGONAL DE MATRIZ DE 90x90

TR=2.4 ##periodo de sampleo, o dt.
states = ["CNT","MCS","UWS"]
CNTn,MCSn,UWSn = 13,27,21
excludes = [[],[13,17],[12]] ##nota: exclui segun FCs con media outlier
lennys = {"CNT":[],"MCS":[],"UWS":[]}


# halt
#%% main loop

# n_components = 10

lenis={}
aall = []
for i,state in enumerate(states): ##estados
    lenis[state] = []
    for ind in range(eval(state+"n")): ##individuos no-excluidos
        if ind not in excludes[i]:
            print(i)
            data = alldata["tseriesDec"][ind,i]
            # pca = PCA(n_components = n_components)
            # data = pca.fit_transform(data.T).T
            
            # print(state,ind,data.shape, f"explained variance = {pca.explained_variance_.sum():.3f}")
            lennys[state].append(data.shape[1])
            
            
            subvecs = utils.get_phase_subFCD(data,filterr=True)
            aall.append(subvecs)
aall = np.concatenate(aall,axis=1) #concatenamos todos los individuos


#%%
# CNT_sub = aall[:,:2496]
# MCS_sub = aall[:,(2496):(2496+4540)]
# UWS_sub = aall[:,(2496+4540):(2496+45400+3676)]


#guardar. Warning: APP 300 MB
# np.save("chewed_data/ALL_sub_phasecoherence_concatenated_filt.npy",aall)

# np.save("chewed_data/all_sub_phasecoherence_no_filt_CNT.npy",CNT_sub)
# np.save("chewed_data/all_sub_phasecoherence_no_filt_MCS.npy",MCS_sub)
# np.save("chewed_data/all_sub_phasecoherence_no_filt_UWS.npy",UWS_sub)

halt
#%% solo para visualizar un poco

plt.figure(1)
plt.clf()
plt.title("submatrices de coherencia de fase, estiraditas")
plt.imshow(aall,cmap="jet")
plt.xticks([0,2496,2496+4540,(2496+4540)+3676],["-","fin CNT","fin MCS","fin UWS"])
# plt.xticks([(0+2496)/2,(2496+2496+4540)/2,((2496+4540)+(2496+4540)+3676)/2],"")
plt.show()

