# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:32:27 2024

@author: flehu
"""
import sys
sys.path.append("../analyze_empirical")
sys.path.append("../")
sys.path.append("../../")
import numpy as np
import pandas as pd
from scipy.special import rel_entr as kl
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress,spearmanr
from statsmodels.stats.multitest import multipletests
import utils
import warnings
warnings.filterwarnings("ignore")
lower = np.tril_indices(90,k=-1)

states = ["CNT","MCS","UWS"] 

struct = np.loadtxt("../../../structural_Deco_AAL.txt")
lowerstruct = struct[lower]


#%% load data

data = np.load("../../data/clustering_output.npz")
prelabels = data['prelabels']
labels = data['labels']
ordd = data['ordd']
cors = data['cors_sorted']
emes = data['emes_sorted']
n_clusters = len(cors)

lenis = {st:data[f"lenis_{st}"] for st in states}
CNTn,MCSn,UWSn = [len(lenis[f"{st}"])  for st in states]


# halt

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

prearrayCNT = np.zeros((CNTn*n_clusters,2))
prearrayMCS = np.zeros((MCSn*n_clusters,2))
prearrayUWS = np.zeros((UWSn*n_clusters,2))

for i in range(n_clusters):
    prearrayCNT[CNTn*i:CNTn*(i+1),0] = cors[i]
    prearrayCNT[CNTn*i:CNTn*(i+1),1] = dfCNT[f"pattern{i}"]
for i in range(n_clusters):
    prearrayMCS[MCSn*i:MCSn*(i+1),0] = cors[i]
    prearrayMCS[MCSn*i:MCSn*(i+1),1] = dfMCS[f"pattern{i}"]

for i in range(n_clusters):
    prearrayUWS[UWSn*i:UWSn*(i+1),0] = cors[i]
    prearrayUWS[UWSn*i:UWSn*(i+1),1] = dfUWS[f"pattern{i}"]

###let's calculate spearman correlations
rhoCNT,pCNT = spearmanr(prearrayCNT[:,0],prearrayCNT[:,1])
rhoMCS,pMCS = spearmanr(prearrayMCS[:,0],prearrayMCS[:,1])
rhoUWS,pUWS = spearmanr(prearrayUWS[:,0],prearrayUWS[:,1])
pCNT,pMCS,pUWS = multipletests((pCNT,pMCS,pUWS), method='fdr_bh')[1]
print(f"CNT:rho={rhoCNT:.4f},p={pCNT:.4f}")
print(f"MCS:rho={rhoMCS:.4f},p={pMCS:.4f}")
print(f"UWS:rho={rhoUWS:.4f},p={pUWS:.4f}")



arrayCNT,arrayMCS,arrayUWS = [np.copy(arrei) for arrei in(prearrayCNT,prearrayMCS,prearrayUWS)]
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
colors = ("crimson","orange","forestgreen")
colors = ("tab:blue","chocolate","tab:red")
colors = ("tab:blue","tab:orange","tab:green")

titlesize=14
RSNsize=14
ticksize=14
labelsize=14
legendsize=12

cmap_fc = "jet"
cmap_sc = "binary"

def plot_transition_mat(mat,title,ax=None,cmap_ent = "binary"):
    ax.set_title(title, fontsize=titlesize)
    im = ax.imshow(mat, cmap=cmap_ent)
    for i in range(mat_gen.shape[0]):
        for j in range(mat_gen.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha='center', va='center', color='tab:blue',weight="bold",fontsize=legendsize)
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

#%%plot the very thing

demertzi_mats = [utils.symm2RSNd(m,nanear=True) for m in emes]
# sc_mat = utils.symm2RSNd(m,nanear=True)

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
        # cbar_ax.yaxis.set_ylabel("correlation")
        cbar.ax.tick_params(labelsize=ticksize)

##structural connectivity    
ax=plt.subplot(2,n_clusters+1,n_clusters+1)
ax.set_title("Structural Connectivity",fontsize=titlesize)
demertzi_SC = utils.symm2RSNd(struct,nanear=True)
ax.imshow(demertzi_SC,cmap=cmap_sc)
# ax.colorbar()
ax.set_xticklabels(['Vis', 'ES', 'Aud', 'SM', 'DM', 'EC', ""], fontsize=RSNsize, rotation=90)
ax.set_yticks([0, 7, 16, 42, 55, 67, 90])
ax.set_yticklabels(['Vis', 'ES', 'Aud', 'SM', 'DM', 'EC', ""], fontsize=RSNsize)
ax.xaxis.set_ticks_position('top')
ax.yaxis.set_ticks_position('right')   
ax.spines[['bottom', 'left']].set_visible(False) 
    


###transition matrix
ax = plt.subplot2grid((2, 3), (1, 0))
plot_transition_mat(mat_gen,"Transition Matrix",ax=ax)

##average occupations
palette = ["tab:blue","tab:orange","tab:green"]
palette = ["silver","gray","dimgray"]
ax= plt.subplot2grid((2,3),(1,1))
ax.set_title("Average occupation",fontsize=titlesize)
sns.barplot(occ_df,y="corr val",x="state",hue="centroid"
            ,palette=palette,
            ax=ax)
ax.set_xticks((0,1,2),states,fontsize=ticksize,weight="bold")
ax.set_yticks((0,0.2,0.4,0.6,0.8),(0,0.2,0.4,0.6,0.8),fontsize=ticksize)
ax.set_xlabel("state",fontsize=labelsize)
ax.set_ylabel("Time proportion",fontsize=labelsize)
ax.legend(fontsize=legendsize)
ax.spines[["top","right"]].set_visible(False)

alfaind = 0.7
###occupations by individual
ax = plt.subplot2grid((2,3),(1,2))
ax.set_title("Occupation by Individual",fontsize=titlesize)
ax.scatter(arrayCNT[:,0],arrayCNT[:,1],color=colors[0],alpha=alfaind)

x = arrayCNT[:,0]#.reshape(-1,1)
y = arrayCNT[:,1]
slope, intercept, r_value, p_value, std_err = linregress(x, y)
ax.plot(cors,intercept+slope*cors,color=colors[0],label=r"CNT, $\rho=$"+f"{rhoCNT:.3f}")

ax.scatter(arrayMCS[:,0],arrayMCS[:,1],color=colors[1],alpha=alfaind)
x = arrayMCS[:,0]#.reshape(-1,1)
y = arrayMCS[:,1]
slope, intercept, r_value, p_value, std_err = linregress(x, y)
ax.plot(cors,intercept+slope*cors,color=colors[1],label=r"MCS, $\rho=$"+f"{rhoMCS:.3f}")

ax.scatter(arrayUWS[:,0],arrayUWS[:,1],color=colors[2],alpha=alfaind)
x = arrayUWS[:,0]#.reshape(-1,1)
y = arrayUWS[:,1]
slope, intercept, r_value, p_value, std_err = linregress(x, y)
ax.plot(cors,intercept+slope*cors,color=colors[2],label=r"UWS, $\rho=$"+f"{rhoUWS:.3f}")
ax.set_yticks((0,0.2,0.4,0.6,0.8,1),(0,0.2,0.4,0.6,0.8,1),fontsize=ticksize)
# plt.xticks([0,1]+list(cors),[0,1]+[f"c{i}" for i in range(n_clusters)],fontsize=ticksize)
ax.set_xticks(cors,[f"c{j+1}\n{cors[j]:.3f}" for j in range(3)],fontsize=ticksize)
ax.set_xlabel("SC-FC correlation",fontsize=labelsize)
ax.legend(loc="upper left",fontsize=legendsize)
ax.spines[["top","right"]].set_visible(False)


plt.tight_layout()
# plt.subplots_adjust(wspace=0.2, hspace=0.2)
# plt.savefig("fig1.svg",dpi=300,transparent=True)
plt.show()
# plt.close()

#%% plot transition matrices per states

plt.figure(2)
plt.clf()
ax= plt.subplot(131)
plot_transition_mat(mat_CNT,"Transition Matrix CNT",ax=ax)
ax= plt.subplot(132)
plot_transition_mat(mat_MCS,"Transition Matrix MCS",ax=ax)
ax= plt.subplot(133)
plot_transition_mat(mat_UWS,"Transition Matrix UWS",ax=ax)

plt.tight_layout()
plt.show()