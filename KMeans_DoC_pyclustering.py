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
from sklearn.decomposition import PCA
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric
from pyclustering.cluster.center_initializer import random_center_initializer
import pickle
from scipy.special import rel_entr as kl
from scipy.stats import entropy


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import utils
from scipy.stats import ks_2samp as ks
import warnings
warnings.filterwarnings("ignore")
lower = np.tril_indices(90,k=-1)


savefolder = "chewed_data/"
states = ["CNT","MCS","UWS"] #wake, deep anesthesia, recovery
loaddata= np.load("../chewed_data/ALL_sub_phasecoherence_concatenated_filt.npy")
distance_measures = {'EUCLIDEAN': 0, 'squared euclidean': 1, 'manhattan': 2, 'chebyshev': 3, 
                    'canberra': 5, 'chi-square': 6}

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
    initial_centers = random_center_initializer(X, n_clusters, random_state=12).initialize()
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

def cent_dists(X,labels,centroids):
    dists = {l:[] for l in np.unique(labels)}
    for i in range(len(X)):
        this_l = labels[i]
        dists[this_l].append(np.linalg.norm(X[i]-centroids[this_l]))
    return dists

    


#%% Clasificador
X=loaddata.T

n_clusters = 3


##PCA<

pesea = False
if pesea:
    n_components = 15
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X = pca.transform(X)
    print(X.shape, f"explained variance with {n_components} componentes = {pca.explained_variance_ratio_.sum()*100:.3f}%")
    
    precentroids,prelabels,clusterer = centers_labels(X,n_clusters,export_instance=True)
    prelabels = utils.process_labels_pyclustering(len(X),prelabels)
    centroids = [pca.inverse_transform(c) for c in precentroids]    
    
else:
    centroids,prelabels,clusterer = centers_labels(X,n_clusters,export_instance=True)
    prelabels = utils.process_labels_pyclustering(len(X),prelabels)

#%%
dataCNT = X[:2496]
dataMCS = X[2496:2496+4540]
dataUWS = X[2496+4540:]

alldata = {"CNT":dataCNT,"MCS":dataMCS,"UWS":dataUWS}   
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
# centroids = centroids[ordd]
var_emes = var_emes[ordd]
#%%un juguetito matematico
centroids_ordd = np.array(centroids)[ordd]

x = centroids_ordd[1]-centroids_ordd[0];y=centroids_ordd[2]-centroids_ordd[1]#;z=centroids[3]-centroids[2]
scalar = lambda x,y: (x*y).sum()/((x*x).sum()**.5*(y*y).sum()**.5)
print(x.shape,scalar(x,y),f"angulo = {np.arccos(scalar(x,y))*57.2958:.4f} grados")#,scalar(x,z))
#%% reordenamos labels

labels = np.zeros_like(prelabels)
for i,position in enumerate(ordd):
    labels[prelabels==position] = i

#%% graficar clusters?
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
labelsais = 15
ticsais = 13

# orAAL2symm = list(range(0,90,2)) + list(range(1,90,2))[::-1]
plt.figure(1,figsize = (10,4))
plt.clf()
for i in range(n_clusters):
    ax = plt.subplot(1,4,i+1)
    mat = emes[i]
    plt.title(r"centroid {}".format(i+1)+f"\nSC/FC corr = {cors[i]:.3f}",fontsize=labelsais)#,weight="bold")
    im=plt.imshow(mat,vmin=-1,vmax=1,cmap="jet")
    plt.xticks([0,44,89],[1,45,90],fontsize=ticsais)
    plt.yticks([])
    
    plt.xlabel("ROIs",weight="bold",fontsize=labelsais)
    if i==0:
        plt.yticks([0,44,89],[1,45,90],fontsize=ticsais)  
        plt.ylabel("ROIs",weight="bold",fontsize=labelsais)
    
    
    if i+1==3:
        axins1 = inset_axes(ax,
                        width="6.25%",  # width = 50% of parent_bbox width
                        height="100%",  # height : 5%
                        loc='center right', borderpad = -3)
                        # axes_kwargs = {"anchor":(0.5,0)})
        cax = plt.colorbar(im, cax=axins1, orientation="vertical", ticks=[1, 2, 3])
        cax.set_ticks([-1,-0.5,0, 0.5, 1],labels=[-1,-0.5,0, 0.5, 1],fontsize=13)
        ticklabs = cax.ax.get_xticklabels()
        cax.ax.set_xticklabels(ticklabs, fontsize=13)
        
        cax.set_label(r'Pearson Correlation between ROIs', fontsize = 13)
# plt.colorbar()
plt.tight_layout()
# plt.savefig("../../ICMNS2024_Dublin/centroides.png",dpi=300)
plt.show()



#%%GUARDAR CLUSTERER?

# name_to_save = f"KMeans_pyclustering_euclidean_instance_ordd_{n_clusters}clusters_alldim_ALL_filtered"

# with open("../chewed_data/"+name_to_save+'.pickle', 'wb') as f:
#     pickle.dump([clusterer,ordd], f)

#%% analizar labels, calcular matrices de transicion y entropia
lens= {'CNT': 2496, 'MCS': 4540, 'UWS': 3676}

##esto es solo para visualizacion en un heatmap, no importante
labels2D = labels[:,None]*np.ones((len(loaddata.T),800))


labelsCNT= labels[:lens["CNT"]]
labelsMCS = labels[lens["CNT"]:lens["CNT"]+lens["MCS"]]
labelsUWS = labels[lens["CNT"]+lens["MCS"]:]

countsCNT = np.array([(labelsCNT==i).sum() for i in range(n_clusters)])/lens["CNT"]
countsMCS = np.array([(labelsMCS==i).sum() for i in range(n_clusters)])/lens["MCS"]
countsUWS = np.array([(labelsUWS==i).sum() for i in range(n_clusters)])/lens["UWS"]


mat_CNT,stat_CNT,ent_CNT = utils.transition_matrix(labelsCNT)
mat_MCS,stat_MCS,ent_MCS = utils.transition_matrix(labelsMCS)
mat_UWS,stat_UWS,ent_UWS = utils.transition_matrix(labelsUWS)

print(f"entropias: CNT{ent_CNT:.3f},  MCS{ent_MCS:.3f},  UWS{ent_UWS:.3f}")



dists = np.zeros((3,n_clusters))
dists[0,:] = countsCNT;dists[1,:] = countsMCS;dists[2,:] = countsUWS
# np.save(savefolder+f"dists_per_state_filt_{n_clusters}clusters_COMA.npy",dists)

#%% distance to centroids:
alfa = 0.6
distance_to_centroids = cent_dists(X,labels,centroids)
distance_to_centroids_CNT = cent_dists(dataCNT,labelsCNT,centroids)
distance_to_centroids_MCS = cent_dists(dataMCS,labelsMCS,centroids)
distance_to_centroids_UWS = cent_dists(dataUWS,labelsUWS,centroids)



plt.figure(4)
plt.clf()
ax = plt.subplot(111)
for i in range(n_clusters):
    plt.hist(distance_to_centroids[i],density=True,bins=50,label=f"cluster {i}",alpha=alfa)
plt.legend()
plt.show()

distance_centroids_dic = {"distance_general":distance_to_centroids,
                          "CNT":distance_to_centroids_CNT,
                          "MCS":distance_to_centroids_MCS,
                          "UWS":distance_to_centroids_UWS}
# with open("clust_dists_empirical_3centroids_alldim.pickle","wb") as f:
#     pickle.dump(distance_centroids_dic,f)





#%%
CNTlabels= labels[:lens["CNT"]]
MCSlabels = labels[lens["CNT"]:lens["CNT"]+lens["MCS"]]
UWSlabels = labels[lens["CNT"]+lens["MCS"]:]



props = {"CNT":[],"MCS":[],"UWS":[]}
predicted_from_kl_to_overall = {"CNT":[],"MCS":[],"UWS":[]}
entropies = {"CNT":[],"MCS":[],"UWS":[]}
starts = [0,lens["CNT"],lens["CNT"]+lens["MCS"]]
finishes = [lens["CNT"],lens["CNT"]+lens["MCS"],lens["CNT"]+lens["MCS"]+lens["UWS"]]

#numero de individuos
CNTn,MCSn,UWSn = 13,25,20 ##NOTAR QUE YA ESTAN EXCLUIDOS
#largo de registros
lenis = {}
lenis["CNT"] = [192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192]
lenis["MCS"] = [192, 192, 192, 192, 192, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 195, 195, 195, 195]
lenis["UWS"] = [192, 192, 192, 192, 192, 192, 192, 192, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 195, 195]
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

colors = ["tab:blue","tab:orange","tab:green"]

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





#%%
plt.figure(2)
plt.clf()
plt.subplot(311)
plt.title("CNT")
plt.imshow(labels2D[:lens["CNT"]].T)
plt.colorbar()
plt.subplot(312)
plt.title("MCS")
plt.imshow(labels2D[lens["CNT"]:lens["CNT"]+lens["MCS"]].T)
plt.colorbar()
plt.subplot(313)
plt.title("UWS")
plt.imshow(labels2D[lens["CNT"]+lens["MCS"]:].T)
plt.colorbar()
plt.show()


#%%PLOTEAMOS A TODOS LOS INDIVIDUOS
plt.figure(3)
plt.clf()

for i,m in enumerate(emes):
    
    demertzi_m = utils.symm2RSNd(m)
    
    cor = np.corrcoef(lowerstruct,m[lower])[0,1]
    ax = plt.subplot(2,n_clusters+1,i+1)
    
    plt.title(f"p{i}\nSC-FC corr = {cors[i]:.4f}\n(mean={m.mean():.3f})")
    plt.imshow(demertzi_m,vmin=-1,vmax=1,cmap="jet")
    if i==0:
        plt.colorbar(location="left")
    plt.xticks([0,7,16,42,55,67,90],['Vis', 'ES', 'Aud', 'SM', 'DM', 'EC',""],fontsize=8)
    plt.yticks([0,7,16,42,55,67,90],['Vis', 'ES', 'Aud', 'SM', 'DM', 'EC',""],fontsize=8)
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('right')    
plt.subplot(2,n_clusters+1,n_clusters+1)
plt.title("STRUCT")
demertzi_SC = utils.symm2RSNd(struct)
plt.imshow(demertzi_SC,cmap="jet")
plt.xticks([]);plt.yticks([])
    

# plt.subplot(3,1,3)


plt.subplot(245)
plt.title("occupation for CNT")
plt.bar(range(n_clusters),countsCNT,label="CNT",alpha=0.6)
plt.ylim([0,1])
plt.legend()
plt.xticks(range(n_clusters))
plt.subplot(246)
plt.title("occupation for MCS")
plt.bar(range(n_clusters),countsMCS,label="MCS",alpha=0.6)
plt.ylim([0,1])
plt.legend()
plt.xticks(range(n_clusters))
plt.subplot(247)
plt.title("occupation for UWS")
plt.bar(range(n_clusters),countsUWS,label="UWS",alpha=0.6)
plt.ylim([0,1])
plt.xticks(range(n_clusters))
plt.legend()
plt.subplot(248)
plt.scatter(arrayCNT[:,0],arrayCNT[:,1],label="CNT",color=colors[0])
lin = LinearRegression()
lin.fit(arrayCNT[:,0].reshape(-1,1),arrayCNT[:,1])
intercept,slope = float(lin.intercept_),float(lin.coef_)
plt.plot(cors,intercept+slope*cors,color=colors[0])

plt.scatter(arrayMCS[:,0],arrayMCS[:,1],label="MCS",color=colors[1])
lin = LinearRegression()
lin.fit(arrayMCS[:,0].reshape(-1,1),arrayMCS[:,1])
intercept,slope = float(lin.intercept_),float(lin.coef_)
plt.plot(cors,intercept+slope*cors,color=colors[1])

plt.scatter(arrayUWS[:,0],arrayUWS[:,1],label="UWS",color=colors[2])
lin = LinearRegression()
lin.fit(arrayUWS[:,0].reshape(-1,1),arrayUWS[:,1])
intercept,slope = float(lin.intercept_),float(lin.coef_)
plt.plot(cors,intercept+slope*cors,color=colors[2])
plt.xticks([0,1]+list(cors),[0,1]+[f"p{i}" for i in range(n_clusters)],fontsize=18)
plt.legend()
plt.tight_layout()
plt.show()


#%% plot solo las ocupaciones
arrayCNT[:,0] = arrayCNT[:,0]+np.random.normal(scale=0.01/2,size=len(arrayCNT))
arrayMCS[:,0] = arrayMCS[:,0]+np.random.normal(scale=0.01/2,size=len(arrayMCS))
arrayUWS[:,0] = arrayUWS[:,0]+np.random.normal(scale=0.01/2,size=len(arrayUWS))

#%%
occ_df = pd.DataFrame()
occ_df["corr val"] = np.concatenate([countsCNT,countsMCS,countsUWS])
occ_df["centroid"] = 3*["c1","c2","c3"]
occ_df["state"] = 3*["CNT"]+3*["MCS"]+3*["UWS"]

plt.figure(99)
plt.clf()
plt.title("occupation of clusters per state")
sns.barplot(occ_df,y="corr val",x="state",hue="centroid",color="grey")
plt.show()

#%%


alfa = 0.8
plt.figure(5,figsize=(5,3))
plt.clf()
plt.subplot2grid((3,2),(0,1),rowspan=3)
plt.scatter(arrayCNT[:,0],arrayCNT[:,1],label="CNT",color=colors[0],alpha=alfa)
lin = LinearRegression()
lin.fit(arrayCNT[:,0].reshape(-1,1),arrayCNT[:,1])
intercept,slope = float(lin.intercept_),float(lin.coef_)
plt.plot(cors,intercept+slope*cors,color=colors[0])

plt.scatter(arrayMCS[:,0],arrayMCS[:,1],label="MCS",color=colors[1],alpha=alfa)
lin = LinearRegression()
lin.fit(arrayMCS[:,0].reshape(-1,1),arrayMCS[:,1])
intercept,slope = float(lin.intercept_),float(lin.coef_)
plt.plot(cors,intercept+slope*cors,color=colors[1])

plt.scatter(arrayUWS[:,0],arrayUWS[:,1],label="UWS",color=colors[2],alpha=alfa)
lin = LinearRegression()
lin.fit(arrayUWS[:,0].reshape(-1,1),arrayUWS[:,1])
intercept,slope = float(lin.intercept_),float(lin.coef_)
plt.plot(cors,intercept+slope*cors,color=colors[2])
# plt.xticks([0,1]+list(cors),[0,1]+[f"p{i},corr={cors[i]:.2f}" for i in range(n_clusters)],fontsize=13,rotation=45)
plt.xticks([0.25,0.5,0.75],[0.25,0.5,0.75],fontsize=11)
plt.legend()
plt.xlabel("SC-FC cluster correlation",fontsize=13)
plt.yticks([0,0.25,0.5,0.75,1],fontsize=13)
# plt.tight_layout()

plt.subplot2grid((3,2),(0,0))
plt.title("CNT",fontsize=13)
plt.bar(range(n_clusters),countsCNT,label="CNT",alpha=alfa,color="tab:blue")
plt.yticks([0,0.25,0.5],[0,0.25,0.5],fontsize=13)
plt.xticks([0,1,2],["","",""],fontsize=13)

plt.subplot2grid((3,2),(1,0))
plt.title("MCS",fontsize=13)
plt.bar(range(n_clusters),countsMCS,label="MCS",alpha=alfa,color="tab:orange")
plt.xticks([0,1,2],["","",""],fontsize=13)
plt.yticks([0,0.25,0.5],[0,0.25,0.5],fontsize=13)

plt.subplot2grid((3,2),(2,0))
plt.title("UWS",fontsize=13)
plt.bar(range(n_clusters),countsUWS,label="UWS",alpha=alfa,color="tab:green")
plt.xticks([0,1,2],["c1","c2","c3"],fontsize=13)
plt.yticks([0,0.4,0.8],[0,0.4,0.8],fontsize=13)

plt.tight_layout()
plt.savefig("../../ICMNS2024_Dublin/occupancies_clusters",dpi=300)
plt.show()

#%%

plt.figure(100)
plt.clf()
plt.title("variance of patterns")
plt.bar(range(n_clusters),var_emes)
plt.xticks(range(n_clusters))
plt.tight_layout()
plt.show()


#%% saltos y su correlacion con las labels

to_analyze = {"CNT":(dataCNT,labelsCNT),
              "MCS":(dataMCS,labelsMCS),
              "UWS":(dataUWS,labelsUWS),
              "all":(X,labels)}

chosen ="all"
this_data,this_label = to_analyze[chosen]

jumps = np.linalg.norm(np.diff(this_data,axis=0),axis=1,ord=2)
means = np.mean(this_data,axis=1)[1:]
# jumps = np.array([1-np.corrcoef(this_data[i],this_data[i+1])[0,1] for i in range(len(this_data)-1)])

print(chosen,jumps.mean())
observables_df = pd.DataFrame()
observables_df["jump"] = jumps
observables_df["mean"] = means
observables_df["cluster_label"] = this_label[1:]

ks_jump_states = np.zeros((n_clusters,n_clusters))
p_matrix = np.zeros((n_clusters,n_clusters))
for i in range(n_clusters):
    for j in range(n_clusters):
        j1,j2 = observables_df[observables_df["cluster_label"]==i]["jump"],observables_df[observables_df["cluster_label"]==j]["jump"]
        kss,p = ks(j1,j2)
        # print(i,j,p)
        ks_jump_states[i,j] = kss
        p_matrix[i,j] = p



plt.figure(6)
plt.clf()
plt.suptitle(f"saltos comenzando en cada cluster, state={chosen}, ndim = {n_components}")
plt.subplot(221)
sns.violinplot(observables_df,y="jump",x="cluster_label")

plt.subplot(223)
for i in range(n_clusters):
    plt.hist(observables_df[observables_df["cluster_label"]==i]["jump"],label=f"cluster{i}",density=True,alpha=0.5,bins=30)
plt.legend()


ax=plt.subplot(222)
ax.set_title("ks_distance between jump dists")
plt.imshow(ks_jump_states)
for i in range(n_clusters):
    for j in range(n_clusters):
        text = ax.text(j, i, f"{ks_jump_states[i, j]:.3f}\np={p_matrix[i,j]:.3f}",
                       ha="center", va="center", color="w",weight="bold",fontsize=10)
plt.xticks(range(n_clusters))
plt.yticks(range(n_clusters))
plt.xlabel("cluster_label");plt.ylabel("cluster_label")
plt.colorbar()

ax=plt.subplot(224)
plt.title("transition matrix")
mat = eval("mat_"+state)
plt.imshow(mat)
for i in range(n_clusters):
    for j in range(n_clusters):
        text = ax.text(j, i, f"{mat[i, j]:.3f}",
                       ha="center", va="center", color="w",weight="bold",fontsize=10)
plt.xticks(range(n_clusters));plt.yticks(range(n_clusters))
plt.xlabel("cluster_label");plt.ylabel("cluster_label")
plt.colorbar()
plt.show()


plt.figure(6)
plt.clf()
for s,chosen in enumerate(states):
    ax=plt.subplot(1,3,s+1)
    ax.set_title(chosen+" transition matrix")
    mat = eval("mat_"+chosen)
    plt.imshow(mat,vmin=0,vmax=1)
    for i in range(n_clusters):
        for j in range(n_clusters):
            text = ax.text(j, i, f"{mat[i, j]:.3f}",
                           ha="center", va="center", color="w",weight="bold",fontsize=10)
    plt.xticks(range(n_clusters));plt.xlabel("to cluster n°")
    plt.yticks(range(n_clusters));plt.ylabel("from cluster n°")
    plt.colorbar()
plt.show()   

#%%
# dif1 = emes[0].flatten()-emes[1].flatten()
# dif2 = emes[2].flatten()-emes[1].flatten()
                              
# dot = (dif1*dif2).sum()
# n1 = np.linalg.norm(dif1)
# n2 = np.linalg.norm(dif2)
# print(dot/(n1*n2)) 
# #este es el coseno del angulo entre los clusters, si da 1 o -1 son colineales 
# #tambien sirve que la correlacion de las diferencias sea exactamente 1.


# #%%


load1 = pca.components_[0,:]
as_matrix_or1 =utils.reconstruct(load1,diag_fill=np.nan) 
as_matrix1 = utils.symm2RSNd(as_matrix_or1)

load2 = pca.components_[1,:]
as_matrix_or2 =utils.reconstruct(load2,diag_fill=np.nan) 
as_matrix2 = utils.symm2RSNd(as_matrix_or2)


plt.figure(1111)
plt.clf()
plt.subplot(121)
plt.title("weight of entries for pc1")
plt.imshow(as_matrix_or1,cmap="jet")
plt.xticks([0,7,16,42,55,67,90],['Vis', 'ES', 'Aud', 'SM', 'DM', 'EC',""],fontsize=15)
plt.yticks([0,7,16,42,55,67,90],['Vis', 'ES', 'Aud', 'SM', 'DM', 'EC',""],fontsize=15)
plt.colorbar()
plt.subplot(122)
plt.title("weight of entries for pc2")
plt.imshow(as_matrix_or2,cmap="jet")
plt.xticks([0,7,16,42,55,67,90],['Vis', 'ES', 'Aud', 'SM', 'DM', 'EC',""],fontsize=15)
plt.yticks([0,7,16,42,55,67,90],['Vis', 'ES', 'Aud', 'SM', 'DM', 'EC',""],fontsize=15)
plt.colorbar()
plt.show()

#%%
plt.figure(101)
sns.swarmplot(entropies)
plt.show()

print([f'{states[i]}: {(np.array(predicted_from_kl_to_overall[states[i]])==i).sum()}/{len(lenis[states[i]])}' for i in range(3)])


#%% dwell times
alfa = 0.5


dwell_CNT,dwell_counts_CNT = dwell_time(labelsCNT)
dwell_MCS,dwell_counts_MCS = dwell_time(labelsMCS)
dwell_UWS,dwell_counts_UWS = dwell_time(labelsUWS)




plt.figure(99)
plt.clf()
plt.subplot(121)
plt.title("dwelling time for states")
plt.plot(dwell_counts_CNT,"x-",label="CNT")
plt.plot(dwell_counts_MCS,"x-",label="MCS")
plt.plot(dwell_counts_UWS,"x-",label="UWS")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.subplot(122)
plt.boxplot([dwell_CNT,dwell_MCS,dwell_UWS])
plt.yscale("log")
plt.show()





