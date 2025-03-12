# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:05:44 2024

@author: flehu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import gamma,skew,ttest_ind,kurtosis as kurt
from statsmodels.stats.multitest import multipletests
import utils
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

savefolder = "chewed_data/"
states = ["CNT","MCS","UWS"] #wake, deep anesthesia, recovery
loaddata= np.load("../chewed_data/ALL_sub_phasecoherence_concatenated_filt.npy")


lenis = {}
lenis["CNT"] = [192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192]
lenis["MCS"] = [192, 192, 192, 192, 192, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 195, 195, 195, 195]
lenis["UWS"] = [192, 192, 192, 192, 192, 192, 192, 192, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 195, 195]

CNTn,MCSn,UWSn = len(lenis["CNT"]),len(lenis["MCS"]),len(lenis["UWS"])
X=loaddata.T

def split_data(data,lennys,setoff=0):
    inits = setoff+np.array([0]+ list(np.cumsum(lennys)[:-1]))
    ends = setoff+np.cumsum(lennys)
    subs = [data[inits[i]:ends[i]] for i in range(len(lennys))]
    return subs

def ind_jump_lengths(ind_data,t_as_row = True):
    return np.linalg.norm(np.diff(ind_data,axis=0),axis=1,ord=1) #norma de las diferencias
    # return np.array([np.linalg.norm(ind_data[i+1]-ind_data[i]) for i in range(len(ind_data)-1)])

def list_of_jump_lengths(data,lennys,full=False,split=False):
    if split:
        jump_lengths = [ind_jump_lengths(ind) for ind in split_data(data,lennys)]
        jump_spatial = np.concatenate([np.diff(ind,axis=0) for ind in split_data(data,lennys)])
    else:
        jump_lengths = np.concatenate([ind_jump_lengths(ind) for ind in split_data(data,lennys)])
        jump_spatial = np.concatenate([np.diff(ind,axis=0) for ind in split_data(data,lennys)])
    
    if full:
        return jump_lengths, jump_spatial
    else:
        return jump_lengths

def ind_outliers(some_data,th=3,full=False):
    some_data = np.array(some_data)
    z = (some_data-some_data.mean())/some_data.std()
    out_mask = (np.abs(z) > th)
    if full:
        return out_mask,some_data[out_mask] #outliers y deteccion de outliers
    else: 
        return out_mask
    
def list_of_outliers(data,lennys,th=3): ##notar que aqui tienen que entrar los saltos, que son menos
    #entramos los saltos
    out = [list(ind_outliers(ind_jump_lengths(ind),th=th,full=False)) for ind in split_data(data,lennys)]
    return np.array(sum(out,[]))

def autocorr(serie,lags = 30,th=0.3):
    T = len(serie)
    auto = np.array([np.corrcoef(serie[:T-lag],serie[lag:])[1,0] for lag in range(lags)])
    where_fell = np.where(auto<th)[0][0]
    return auto,where_fell,2/np.sqrt(T) #autocorrelations, caida, significancia


#%% aplicar PCA a los datos

# n_components = 100

# pca = PCA(n_components=n_components)
# pca.fit(loaddata.T)

# chose_components = 15 #numero de dimensiones con las cuales nos quedamos
# X = pca.transform(loaddata.T)[:,:chose_components]

# print(f"explained variance with {chose_components} components : {pca.explained_variance_ratio_[:chose_components].sum()*100:.3f}%")

# X = loaddata.T
dataCNT = X[:2496]
dataMCS = X[2496:2496+4540]
dataUWS = X[2496+4540:]
#%% extraer largos de salto por individuos
CNT_jump_lengths,jumpsCNT = list_of_jump_lengths(dataCNT, lenis["CNT"],full=True,split=True)

MCS_jump_lengths,jumpsMCS = list_of_jump_lengths(dataMCS, lenis["MCS"],full=True,split=True)

UWS_jump_lengths,jumpsUWS = list_of_jump_lengths(dataUWS, lenis["UWS"],full=True,split=True)

CNTsplit = split_data(dataCNT,lenis["CNT"])
MCSsplit = split_data(dataMCS,lenis["MCS"])
UWSsplit = split_data(dataUWS,lenis["UWS"])


#%% OBTENER PARAMETROS DE LEVY

##we save empirical kurtosis 

CNT_params = np.zeros((13,4))
MCS_params = np.zeros((25,4))
UWS_params = np.zeros((20,4))

for i in range(13):
    print(i)
    dist = CNT_jump_lengths[i]
    a,loc,scale = gamma.fit(dist,floc=0)
    CNT_params[i,:] = a,loc,scale,kurt(dist)
for i in range(25):
    print(i)
    dist = MCS_jump_lengths[i]
    a,loc,scale = gamma.fit(dist,floc=0)
    MCS_params[i,:] = a,loc,scale,kurt(dist)
for i in range(20):
    print(i)
    dist = UWS_jump_lengths[i]
    a,loc,scale = gamma.fit(dist,floc=0)
    UWS_params[i,:] = a,loc,scale,kurt(dist)
    
#%% autocorrelaciones por individuo y cuando caen debajo de 0.5

params = [CNT_params,MCS_params,UWS_params] #inicializamos parametros de levy

#%% ploteo de los parametros de levy para todos los individuos 


xmin,xmax = 0,np.max(np.concatenate([np.concatenate(CNT_jump_lengths),np.concatenate(MCS_jump_lengths),np.concatenate(UWS_jump_lengths)]))
xaxis = np.linspace(xmin,xmax,1000)

alfa = 0.5

plt.figure(2)
plt.clf()
plt.suptitle("jump length distributions and gamma fit (loc=0 fixed)")
plt.subplot(131)
plt.title("all pooled")
dist = np.concatenate(CNT_jump_lengths)
print(f"CNT\nmean\tmedian\tstd\tCV\n{dist.mean():.4f}\t{np.median(dist):.4f}\t{dist.std():.4f}\t{dist.std()/dist.mean():.4f}")
plt.hist(dist,label="CNT",density=True,alpha=alfa,bins=50,color="tab:blue")
dist = np.concatenate(MCS_jump_lengths)
print(f"MCS\n{dist.mean():.4f}\t{np.median(dist):.4f}\t{dist.std():.4f}\t{dist.std()/dist.mean():.4f}")
plt.hist(dist,label="MCS",density=True,alpha=alfa,bins=50,color="tab:orange")
dist = np.concatenate(UWS_jump_lengths)
print(f"UWS\n{dist.mean():.4f}\t{np.median(dist):.4f}\t{dist.std():.4f}\t{dist.std()/dist.mean():.4f}")
plt.hist(dist,label="UWS",density=True,alpha=alfa,bins=50,color="tab:green")

dist = np.abs(np.random.normal(loc=0,scale=1000,size=4000))
print(f"NORMAL\n{dist.mean():.4f}\t{np.median(dist):.4f}\t{dist.std():.4f}\t{dist.std()/dist.mean():.4f}")
plt.hist(dist,label="NORMAL",density=True,alpha=0.3,bins=50,color="black")


plt.legend()
plt.xlim(xmin,xmax)


plt.subplot(332)
plt.title("CNT individuals, N=13")
for i in range(13):
    # print(i)4
    dist = CNT_jump_lengths[i]
    a,loc,scale,_ = CNT_params[i]
    pdf = gamma.pdf(xaxis,a=a,loc=loc,scale=scale)
    plt.plot(xaxis,pdf,color="black")
    plt.hist(dist,label=i,density=True,alpha=alfa,bins=20)
plt.xlim(xmin,xmax)
# plt.legend()
plt.subplot(335)
plt.title("MCS individuals, N=25")
for i in range(25):
    # print(i)
    dist = MCS_jump_lengths[i]
    a,loc,scale,_ = MCS_params[i]
    pdf = gamma.pdf(xaxis,a=a,loc=loc,scale=scale)
    plt.plot(xaxis,pdf,color="black")
    plt.hist(dist,label=i,density=True,alpha=alfa,bins=20)
# plt.legend()
plt.xlim(xmin,xmax)
plt.subplot(338)
plt.title("UWS individuals, N=20")
for i in range(20):
    # print(i)
    dist = UWS_jump_lengths[i]
    a,loc,scale,_ = UWS_params[i]
    pdf = gamma.pdf(xaxis,a=a,loc=loc,scale=scale)
    plt.plot(xaxis,pdf,color="black")
    plt.hist(dist,label=i,density=True,alpha=alfa,bins=20)
# plt.legend()
plt.xlim(xmin,xmax)

plt.subplot(433)
plt.title("a ('shape')")
sns.swarmplot([sta[:,0] for sta in params],color="black")
# sns.boxplot([sta[:,0] for sta in params])
plt.xticks([0,1,2],["CNT","MCS","UWS"])
plt.subplot(336)

plt.subplot(436)
plt.title("scale")
sns.swarmplot([sta[:,2] for sta in params],color="black")
# sns.boxplot([sta[:,2] for sta in params])
plt.xticks([0,1,2],["CNT","MCS","UWS"])

plt.subplot(439)
plt.title("theoretical kurtosis"+r"  $(2/a)$")
sns.swarmplot([6/sta[:,0] for sta in params],color="black")
# sns.boxplot([6/sta[:,0] for sta in params])
plt.xticks([0,1,2],["CNT","MCS","UWS"])

plt.subplot(4,3,12)
plt.title("empirical kurtosis")
sns.swarmplot([sta[:,3] for sta in params],color="black")
# plt.yticks([])
# sns.boxplot([sta[:,3] for sta in params])
plt.xticks([0,1,2],["CNT","MCS","UWS"])

plt.tight_layout()
plt.show()


#%%kurtosis analysis
CNT_kurt = 6/params[0][:,0]
MCS_kurt = 6/params[1][:,0]
UWS_kurt = 6/params[2][:,0]

##T test
p1 = ttest_ind(CNT_kurt,MCS_kurt)[1]
p2 = ttest_ind(CNT_kurt,UWS_kurt)[1]
p3 = ttest_ind(MCS_kurt,UWS_kurt)[1]
p1,p2,p3=corrected = multipletests((p1,p2,p3),method="fdr_bh")[1]
print(f"p-val (CNT,MCS),(CNT,UWS),(MCS,UWS): {corrected}")

##D de cohen
d1 = utils.cohen_d(CNT_kurt,MCS_kurt)
d2 = utils.cohen_d(CNT_kurt,UWS_kurt)
d3 = utils.cohen_d(MCS_kurt,UWS_kurt)
print(f"cohen-d (CNT,MCS),(CNT,UWS),(MCS,UWS): {(d1,d2,d3)}")

plt.figure(4)
plt.clf()
plt.subplot(111)
plt.title("theoretical kurtosis"+r"  $(6/a)$")
sns.swarmplot([CNT_kurt,MCS_kurt,UWS_kurt],color="black")
sns.boxplot([CNT_kurt,MCS_kurt,UWS_kurt])
plt.xticks([0,1,2],["CNT","MCS","UWS"])
plt.show()



