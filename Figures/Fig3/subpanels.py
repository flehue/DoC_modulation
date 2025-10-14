# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 15:11:55 2025

@author: flehu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
import sys;sys.path.append("../../")
import utils
from numba import njit
from sklearn.decomposition import PCA

lower = np.tril_indices(90,k=-1)

@njit
def coherences(phases,t_as_col = True):
    if not t_as_col:
        phases = phases.T
    lenny,times = phases.shape
    out = np.zeros((lenny,lenny,times))
    for i in range(lenny):
        for j in range(lenny):
            for t in range(times):
                out[i,j,t] = np.cos(phases[i,t]-phases[j,t])
    return out

def get_phase_subFCD(data,filterr=False,t_as_col=True,TR=2.4,full=False):
    
    if not t_as_col: ##time must be in columns
        data = data.T
    n_entries,times = data.shape ###(dimensiones,tiempo)


    lower = np.tril_indices(n_entries,k=-1)
    print(lower[0].shape)
    if filterr:
        a,b = signal.bessel(2,[2 * 0.01 * TR, 2 * 0.1 * TR], btype = 'bandpass') ##banda [0.01,0.1]
        data = signal.filtfilt(a,b,data,axis=1)

    analytic_signal = signal.hilbert(data,axis=1)
    # phases = np.unwrap(np.angle(analytic_signal))
    phases = np.angle(analytic_signal)
    difs_phases = coherences(phases,t_as_col=True) 
    subvecs = np.concatenate([difs_phases[lower][:,t].reshape(-1,1) for t in range(times)],axis=1)
    if full:
        return difs_phases,phases,subvecs
    else:
        return subvecs

def reconstruct_symm(flattened_matrix,N=90,k=-1,what_on_diag = 1):
    tril_indices = np.tril_indices(N,k=k)
    out = np.zeros((N,N))
    out[tril_indices] = flattened_matrix
    out = (out+out.T)
    if k ==1 or k==-1:
        out[np.diag_indices(N)] = what_on_diag
    return out
    

#%% serie de tiempo
TR = 2.4
data = loadmat("ts_Coma_ParisAALsymm.mat")["tseriesDec"][3,0].T#np.load("c=-0.800_seed3_node67_nofilt.npy").T
a,b= signal.bessel(2,[2 * 0.01 * TR, 2 * 0.1 * TR], btype = 'bandpass')
serie = signal.filtfilt(a,b,data,axis=0)

print(serie.shape)


#%% serie de tiempo plot
plt.figure(1)
plt.clf()
ax = plt.subplot(331)
ax.set_title("BOLD signals",fontsize=14)
ax.plot(serie)
ax.spines[["top","right"]].set_visible(False)
ax.set_xticks(np.arange(0,201,50),[f"{int(val)}" for val in np.arange(0,201,50)*TR],fontsize=12)
ax.set_xlabel("Time (seconds)",fontsize=12)
ax.set_yticks((-2,0,2),(-2,0,2),fontsize=12)
# ax.set_yticks(())

plt.savefig("some_BOLD.svg",dpi=300,transparent=True)
plt.show()

#%%hilbert transform 

#link: https://upload.wikimedia.org/wikipedia/commons/a/aa/Visualisation_complex_number_roots.svg

#%% plot phase matrices

# subvec= np.load("../../../chewed_data/ALL_sub_phasecoherence_concatenated_filt.npy")[:,:3000]

difs_phases,phases,subvec = get_phase_subFCD(serie,t_as_col=False,filterr=True,full=True)
subvec_mean = subvec.mean(axis=1)
ies = (7,8,9,10)


plt.figure(2)
plt.clf()
for i,idd in enumerate(ies):
    ax=plt.subplot(2,2,i+1)
    mat = reconstruct_symm(subvec[:,idd],N=90,k=-1,what_on_diag=1)
    plt.imshow(mat,cmap="jet")
    ax.set_xticks(())
    ax.set_yticks(())

plt.savefig("phase_matrices.svg",dpi=300,transparent=True)
plt.show()


#%% plot random walk 

pca = PCA(n_components = 2)
sub = pca.fit_transform(subvec.T)

##jump shit
jumps = np.linalg.norm(np.diff(sub,axis=0),axis=1)
# jump_quantile = 0.9
stds = 2.5
extreme_jump_ids = jumps-jumps.mean() > stds*jumps.std()

plt.figure(3)
plt.clf()

####stream
ax = plt.subplot2grid((2,2),(0,0))
ax.scatter(sub[:,0],sub[:,1],marker="o",alpha=0.8)

first_event = True
first_notevent = True
for e,event in enumerate(extreme_jump_ids):
    if event:
        if first_event:
            ax.plot(sub[:,0][e:e+2],sub[:,1][e:e+2],color="red",linewidth=2,alpha=1,label="long jump")
            first_event = False
        else:
            ax.plot(sub[:,0][e:e+2],sub[:,1][e:e+2],color="red",linewidth=2,alpha=1)
    else:
        if first_notevent:    
            ax.plot(sub[:,0][e:e+2],sub[:,1][e:e+2],color="tab:blue",linewidth=1,alpha=0.7)#,label="jump")
            first_notevent = False
        else:
            ax.plot(sub[:,0][e:e+2],sub[:,1][e:e+2],color="tab:blue",linewidth=1,alpha=0.7)
ax.set_xticks(())
ax.set_yticks(())
ax.spines[["top","right"]].set_visible(False)
ax.legend(fontsize=14)

##jump hist
ax = plt.subplot2grid((2,2),(0,1))
ax.hist(jumps,bins=20,alpha=0.8,density=True)
ax.spines[["top","right"]].set_visible(False)
ax.set_xticks(())
ax.set_yticks(())
ax.set_xlabel("jump length",fontsize=14)
ax.set_ylabel("frequency",fontsize=14)


####toy ML shit
# ax = plt.subplot2grid((2,2),(1,0))
# means = np.random.normal(size=(3,2),scale=3.2)
# for i in range(3):
#     mean = means[i]
#     X = np.random.multivariate_normal(mean=mean,cov = np.eye(2),size=40)
#     x,y = X[:,0],X[:,1]
#     ax.scatter(x,y,label=i)
# ax.spines[["top","right"]].set_visible(False)
# ax.set_xticks(())
# ax.set_yticks(())
# ax.legend()

ax = plt.subplot2grid((2,2),(1,0))
x = np.random.uniform(size=(3,))
x = x/x.sum()
ax.bar((0,1,2),x,color="gray")
ax.set_xticks((0,1,2),("c1","c2","c3"),fontsize=14)
ax.set_yticks((0,0.5,1),(0,0.5,1),fontsize=14)
ax.set_title("occupation",fontsize=16)
ax.spines[["top","right"]].set_visible(False)




plt.savefig("some_pannels.svg",dpi=300,transparent=True)
plt.show()


###################
if event:
    ax.plot(sub[:,0][e:e+2],sub[:,1][e:e+2],color="red",alpha=0.8)#%%see shit against time
plt.figure(99)
plt.clf()

plt.subplot(231)
plt.imshow(serie.T,cmap="jet")
plt.colorbar()

plt.subplot(234)
plt.imshow(phases,cmap="jet")
plt.colorbar()

plt.subplot(232)
plt.imshow(np.corrcoef(serie.T),cmap="jet")
plt.colorbar()

plt.subplot(235)
mat = reconstruct_symm(subvec_mean,N=90,k=-1,what_on_diag=1)
plt.imshow(mat,cmap="jet")
plt.colorbar()


plt.subplot(133)
plt.imshow(subvec,cmap="jet",aspect=0.1)
plt.colorbar()
plt.xlabel("time")
plt.ylabel("connections")

plt.tight_layout()
plt.show()


#%% matrices

plt.figure(3)
plt.clf()

plt.show()




