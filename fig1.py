# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:32:27 2024

@author: flehu
"""
import sys
sys.path.append("../analyze_empirical")
sys.path.append("../")

import numpy as np
import pandas as pd
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric
from scipy.spatial.distance import cdist
from pyclustering.cluster.center_initializer import random_center_initializer
# import pickle
from scipy.special import rel_entr as kl
from scipy.stats import entropy


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import utils
# from scipy.stats import ks_2samp as ks
import warnings
warnings.filterwarnings("ignore")
lower = np.tril_indices(90,k=-1)


savefolder = "chewed_data/"
states = ["CNT","MCS","UWS"] 
loaddata= np.load("../chewed_data/ALL_sub_phasecoherence_concatenated_filt.npy")
distance_measures = {'EUCLIDEAN': 0, 'squared euclidean': 1, 'manhattan': 2, 'chebyshev': 3, 
                    'canberra': 5, 'chi-square': 6}

#numero de individuos
CNTn,MCSn,UWSn = 13,25,20 ##NOTAR QUE YA ESTAN EXCLUIDOS
#largo de registros
lenis = {}
lenis["CNT"] = [192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192]
lenis["MCS"] = [192, 192, 192, 192, 192, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 195, 195, 195, 195]
lenis["UWS"] = [192, 192, 192, 192, 192, 192, 192, 192, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 195, 195]



def dwell_time(label_stream):
    dwellings = []
    dwell = 1
    for t in range(len(label_stream)-1):
        now = label_stream[t]
        after = label_stream[t+1]
        if now == after:
            dwell+=1
        else:
            dwellings.append(dwell)
            dwell=1
    dwellings = np.array(dwellings)
    
    counts = []
    for dwt in range(1,dwellings.max()):
        counts.append((dwellings==dwt).sum())
    return dwellings,counts


def centers_labels(X,n_clusters,dist_measure = 0,export_instance=False): #manhattan por default
    initial_centers = random_center_initializer(X, n_clusters, random_state=13).initialize()
    instanceKm = kmeans(X, initial_centers=initial_centers, metric=distance_metric(dist_measure))
    instanceKm.process()
    pyCenters = instanceKm.get_centers()
    pyLabels = instanceKm.get_clusters()
    if export_instance:
        return pyCenters,pyLabels,instanceKm
    else:
        return pyCenters,pyLabels
    

def predict(X,centroids):
    labels = []
    for x in X:
        dis = [np.linalg.norm(x-c) for c in centroids]
        labels.append(np.argmin(dis))
    return np.array(labels).astype(int)

    


#%% Clasificador
X=loaddata.T

n_clusters = 3

centroids,prelabels,clusterer = centers_labels(X,n_clusters,export_instance=True)
prelabels = utils.process_labels_pyclustering(len(X),prelabels)

orAAL2symm = list(range(0,90,2)) + list(range(1,90,2))[::-1]

##centroideslabels
emes = np.array([utils.reconstruct(c) for c in np.array(centroids)])    
var_emes = np.array([np.var(c) for c in centroids])
struct = np.loadtxt("../structural_Deco_AAL.txt")
lowerstruct = struct[lower]
cors = np.array([np.corrcoef(lowerstruct,m[lower])[0,1] for m in emes])

ordd = np.argsort(cors) ##aqui ordenamos la correlacion de la mas baja a la mas alta
cors = cors[ordd]
emes = emes[ordd]
centroids = np.array(centroids)[ordd]
var_emes = var_emes[ordd]

##reordenar labels
labels = np.zeros_like(prelabels)
for i,position in enumerate(ordd):
    labels[prelabels==position] = i

#%% analizar labels, calcular matrices de transicion y entropia
lens= {'CNT': 2496, 'MCS': 4540, 'UWS': 3676}


labelsCNT= labels[:lens["CNT"]]
labelsMCS = labels[lens["CNT"]:lens["CNT"]+lens["MCS"]]
labelsUWS = labels[lens["CNT"]+lens["MCS"]:]

countsCNT = np.array([(labelsCNT==i).sum() for i in range(n_clusters)])/lens["CNT"]
countsMCS = np.array([(labelsMCS==i).sum() for i in range(n_clusters)])/lens["MCS"]
countsUWS = np.array([(labelsUWS==i).sum() for i in range(n_clusters)])/lens["UWS"]

mat_gen,stat_gen,ent_gen = utils.transition_matrix(labels)
mat_CNT,stat_CNT,ent_CNT = utils.transition_matrix(labelsCNT)
mat_MCS,stat_MCS,ent_MCS = utils.transition_matrix(labelsMCS)
mat_UWS,stat_UWS,ent_UWS = utils.transition_matrix(labelsUWS)

print(f"entropias: CNT{ent_CNT:.3f},  MCS{ent_MCS:.3f},  UWS{ent_UWS:.3f}")

#%%
CNTlabels= labels[:lens["CNT"]]
MCSlabels = labels[lens["CNT"]:lens["CNT"]+lens["MCS"]]
UWSlabels = labels[lens["CNT"]+lens["MCS"]:]

props = {"CNT":[],"MCS":[],"UWS":[]}
predicted_from_kl_to_overall = {"CNT":[],"MCS":[],"UWS":[]}
entropies = {"CNT":[],"MCS":[],"UWS":[]}
starts = (0,lens["CNT"],lens["CNT"]+lens["MCS"])
finishes = (lens["CNT"],lens["CNT"]+lens["MCS"],lens["CNT"]+lens["MCS"]+lens["UWS"])


for i,state in enumerate(states):
    init = starts[i]
    these_lenis = lenis[state]
    for ind in range([CNTn,MCSn,UWSn][i]): #cantidad de individuos
        lenga = these_lenis[i]
        end = init + lenga
        sublabel = labels[init:end]
        # print(len(sublabel))
        subprops = np.array([(sublabel==k).sum()/lenga for k in range(n_clusters)])
        props[state].append(subprops)
        ##en estricto rigor aqui deberia separar training y test set
        subpredicted = np.argmin([kl(subprops,[countsCNT,countsMCS,countsUWS][i]).sum() for i in range(3)])
        predicted_from_kl_to_overall[state].append(subpredicted)
        
        entropi = entropy(subprops)
        entropies[state].append(entropi)
        
        init=end

# colors = ["tab:blue","tab:orange","tab:green"]

dfCNT = pd.DataFrame(np.array(props["CNT"]),columns=[f"pattern{i}" for i in range(n_clusters)])
dfMCS = pd.DataFrame(np.array(props["MCS"]),columns=[f"pattern{i}" for i in range(n_clusters)])
dfUWS = pd.DataFrame(np.array(props["UWS"]),columns=[f"pattern{i}" for i in range(n_clusters)])
#%%

arrayCNT = np.zeros((CNTn*n_clusters,2))
for i in range(n_clusters):
    arrayCNT[CNTn*i:CNTn*(i+1),0] = cors[i]
    arrayCNT[CNTn*i:CNTn*(i+1),1] = dfCNT[f"pattern{i}"]
arrayMCS = np.zeros((MCSn*n_clusters,2))
for i in range(n_clusters):
    arrayMCS[MCSn*i:MCSn*(i+1),0] = cors[i]
    arrayMCS[MCSn*i:MCSn*(i+1),1] = dfMCS[f"pattern{i}"]
arrayUWS = np.zeros((UWSn*n_clusters,2))
for i in range(n_clusters):
    arrayUWS[UWSn*i:UWSn*(i+1),0] = cors[i]
    arrayUWS[UWSn*i:UWSn*(i+1),1] = dfUWS[f"pattern{i}"]

###let's put some noise
scale = 0.015
arrayCNT[:,0] = arrayCNT[:,0]+np.random.normal(scale=scale,size=len(arrayCNT))
arrayMCS[:,0] = arrayMCS[:,0]+np.random.normal(scale=scale,size=len(arrayMCS))
arrayUWS[:,0] = arrayUWS[:,0]+np.random.normal(scale=scale,size=len(arrayUWS))
##occupations dictionary
occ_df = pd.DataFrame()
occ_df["corr val"] = np.concatenate([countsCNT,countsMCS,countsUWS])
occ_df["centroid"] = 3*["c1","c2","c3"]
occ_df["state"] = 3*["CNT"]+3*["MCS"]+3*["UWS"]



#%%THE VERY FIGURE
from mpl_toolkits.axes_grid1 import make_axes_locatable
colors = ("crimson","orange","forestgreen")
colors = ("tab:blue","chocolate","tab:red")
colors = ("tab:blue","tab:orange","tab:green")

titlesize=14
RSNsize=14
ticksize=14
labelsize=14
legendsize=14

cmap_fc = "jet"
cmap_sc = "binary"
cmap_ent = "binary"

demertzi_mats = [utils.symm2RSNd(m,nanear=True) for m in emes]

fig=plt.figure(1)
plt.clf()
subindex = np.tril_indices(91)
for i in range(n_clusters):
    ax = plt.subplot(2, n_clusters + 2, i + 1)  # axis
    
    mat = demertzi_mats[i]
    cor = np.corrcoef(lowerstruct, mat[lower])[0, 1]

    im = ax.imshow(mat, vmin=0, vmax=1, cmap=cmap_fc)  # store the image for the colorbar

    ax.set_title(f"SC-FC corr = {cors[i]:.3f}\n(mean={np.nanmean(mat):.3f})", fontsize=titlesize)
    ax.set_xticks([0, 7, 16, 42, 55, 67, 90])
    ax.set_xticklabels(['Vis', 'ES', 'Aud', 'SM', 'DM', 'EC', ""], fontsize=RSNsize, rotation=90)
    ax.set_yticks([0, 7, 16, 42, 55, 67, 90])
    ax.set_yticklabels(['Vis', 'ES', 'Aud', 'SM', 'DM', 'EC', ""], fontsize=RSNsize)
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('right')    
    ax.spines[['bottom', 'left']].set_visible(False)

    # Add colorbar only to the leftmost axis (i == 0)
    if i == n_clusters-1:
        # Get the position of the current axis
        pos = ax.get_position()
        # Create a new axes on the left side of it (adjust x0 and width as needed)
        cbar_ax = fig.add_axes([pos.x1 + 0.03, pos.y0, 0.015, pos.height])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar_ax.yaxis.set_ticks_position('right')
        cbar_ax.yaxis.set_label_position('right')
        cbar.ax.tick_params(labelsize=ticksize)

##structural connectivity    
ax=plt.subplot(2,n_clusters+1,n_clusters+1)
ax.set_title("Structural Connectivity",fontsize=titlesize)
demertzi_SC = utils.symm2RSNd(struct)
ax.imshow(demertzi_SC,cmap=cmap_sc)
ax.set_xticklabels(['Vis', 'ES', 'Aud', 'SM', 'DM', 'EC', ""], fontsize=RSNsize, rotation=90)
ax.set_yticks([0, 7, 16, 42, 55, 67, 90])
ax.set_yticklabels(['Vis', 'ES', 'Aud', 'SM', 'DM', 'EC', ""], fontsize=RSNsize)
ax.xaxis.set_ticks_position('top')
ax.yaxis.set_ticks_position('right')    
    


ax = plt.subplot2grid((2, 3), (1, 0))
ax.set_title("Transition Matrix", fontsize=titlesize)
im = ax.imshow(mat_gen, cmap=cmap_ent)
for i in range(mat_gen.shape[0]):
    for j in range(mat_gen.shape[1]):
        ax.text(j, i, f"{mat_gen[i, j]:.2f}", ha='center', va='center', color='tab:blue',weight="bold",fontsize=legendsize)
pos = ax.get_position()
cbar_ax = fig.add_axes([pos.x0, pos.y0 - 0.03, pos.width, 0.02])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=ticksize)
ax.set_xticks((0,1,2),[f"c{i}" for i in range(n_clusters)],fontsize=ticksize)
ax.set_yticks((0,1,2),[f"c{i}" for i in range(n_clusters)],fontsize=ticksize)
# ax.set_ylabel("From",fontsize=labelsize)
# ax.set_xlabel("To",fontsize=labelsize)
ax.text(-0.1, 1.05, "FROM \\ TO", ha='center', va='center', fontsize=12, color='black',weight="bold", transform=ax.transAxes)
ax.xaxis.set_ticks_position('top')

##average occupations
ax= plt.subplot2grid((2,3),(1,1))
ax.set_title("Average occupation",fontsize=titlesize)
sns.barplot(occ_df,y="corr val",x="state",hue="centroid",color="grey",ax=ax)
ax.set_xticks((0,1,2),states,fontsize=ticksize,weight="bold")
ax.set_yticks((0,0.2,0.4,0.6,0.8),(0,0.2,0.4,0.6,0.8),fontsize=ticksize)
ax.set_xlabel("state",fontsize=labelsize)
ax.set_ylabel("Time proportion",fontsize=labelsize)
ax.legend(fontsize=legendsize)

alfaind = 0.8
###occupations by individual
ax = plt.subplot2grid((2,3),(1,2))
ax.set_title("Occupation by Individual",fontsize=titlesize)
ax.scatter(arrayCNT[:,0],arrayCNT[:,1],label="CNT",color=colors[0],alpha=alfaind)
lin = LinearRegression()
lin.fit(arrayCNT[:,0].reshape(-1,1),arrayCNT[:,1])
intercept,slope = float(lin.intercept_),float(lin.coef_)
ax.plot(cors,intercept+slope*cors,color=colors[0])

ax.scatter(arrayMCS[:,0],arrayMCS[:,1],label="MCS",color=colors[1],alpha=alfaind)
lin = LinearRegression()
lin.fit(arrayMCS[:,0].reshape(-1,1),arrayMCS[:,1])
intercept,slope = float(lin.intercept_),float(lin.coef_)
ax.plot(cors,intercept+slope*cors,color=colors[1])

ax.scatter(arrayUWS[:,0],arrayUWS[:,1],label="UWS",color=colors[2],alpha=alfaind)
lin = LinearRegression()
lin.fit(arrayUWS[:,0].reshape(-1,1),arrayUWS[:,1])
intercept,slope = float(lin.intercept_),float(lin.coef_)
ax.plot(cors,intercept+slope*cors,color=colors[2])
ax.set_yticks((0,0.2,0.4,0.6,0.8,1),(0,0.2,0.4,0.6,0.8,1),fontsize=ticksize)
# plt.xticks([0,1]+list(cors),[0,1]+[f"c{i}" for i in range(n_clusters)],fontsize=ticksize)
ax.set_xticks(cors,[f"c{i}\n{cors[j]:.3f}" for j in range(3)],fontsize=ticksize)
ax.set_xlabel("SC-FC correlation",fontsize=labelsize)
ax.legend(loc="upper left",fontsize=legendsize)


# plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.savefig("figures/fig1.svg",dpi=300,transparent=True)
plt.show()


#%% distance to centroids:
alfa = 0.6
unique_labels = np.unique(labels)
n_labels = len(unique_labels)
plt.figure(4)
plt.clf()
for i, l in enumerate(unique_labels):
    # Select points with label l
    points_l = X[labels == l]
    # Compute distances to all centroids
    distances = cdist(points_l, centroids)  # shape (n_points_l, n_labels)
    
    ax = plt.subplot(3,3,i+1)
    ax.set_title(f"from points {l} label")
    for j in range(n_labels):    
        ax.hist(distances[:, j], bins=30, color=colors[j], alpha=0.7,label=f"to centroid {j}")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Count")
    ax.legend()
plt.tight_layout()
plt.show()
