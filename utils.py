# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 13:03:01 2022

@author: flehu
"""

import numpy as np
import os
from scipy import signal
# import BOLDModel as BD
from scipy import signal 
import pandas as pd
from numba import njit
hilbert = signal.hilbert


# a,b = signal.bessel(2,[2 * 0.01 * 2, 2 * 0.1 * 2], btype = 'bandpass')
lower = np.tril_indices(90,k=-1)

def cohen_d(x,y): ##diferencia entre dos distribuciones en terminos de tamaño de efecto
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def flat_FC(FC):
    lenfc = len(FC)
    return np.concatenate([FC[i,i+1:] for i in range(lenfc)])

def get_FC(data,t_as_col=True,filterr=True,low=0.01,high=0.1,TR=2.4):
    if not t_as_col:
        data= data.T
    if filterr:
        a,b= signal.bessel(2,[2 * low * TR, 2 * high * TR], btype = 'bandpass')
        data = signal.filtfilt(a,b,data)
    fc = np.corrcoef(data)
    return fc





def kuramoto(sign):
    analytic = hilbert(sign,axis=0)
    angle = np.angle(analytic)
    kuramoton = np.abs(np.mean(np.exp(1j*angle),axis=1))
    sync = kuramoton.mean()
    meta = kuramoton.std()
    return sync,meta

def get_all_metrics(sFC,empFC,data_range=2):
    """
    input:
        FC1,FC2: pair of identically shaped (n,n) FC matrices
    output:
        *numbers: corr,euclidean,ssim
    """
    from skimage.metrics import structural_similarity as ssim
    lenny = len(sFC)
    flat_empFC = np.concatenate([empFC[i,i+1:] for i in range(lenny)])
    flat_sFC = np.concatenate([sFC[i,i+1:] for i in range(lenny)])
    this_corr = np.corrcoef(flat_sFC,flat_empFC)[0,1]
    this_euc = np.linalg.norm(flat_empFC-flat_sFC)
    this_ssim = ssim(sFC,empFC,data_range=data_range)

    return (this_corr,this_euc,this_ssim)

thal = [38,51]


subL, subR = [35,36,37,38],[51,52,53,54]
sub = subL+subR
cortex = [i for i in range(90) if i not in sub]
def sub_weight(struct,prop=1): ########RECORDAR QUE SON PROPORCION DE LA ORIGINAL
    
    
    out = np.copy(struct)
    out[:,sub] = prop*struct[:,sub] ###todo lo que llega a la corteza    
    out[sub,:] = prop*struct[sub,:] ##prueba
    for i in sub:
        for j in sub:
            out[i,j] = struct[i,j]
    return out


def cortex_mat(mat90):
    subL, subR = [35,36,37,38],[51,52,53,54]
    sub = subL+subR
    cortex = [i for i in range(90) if i not in sub]
    return mat90[cortex,:][:,cortex]


#############ENTRADAS POR RSN
# RSN_index = []
# RSNs = np.loadtxt("../RSN_AAL_Enzo.txt")
# RSN_labels = ["Vis","ES","Aud","SM","DM","EC"]
# for i in range(6):
#     these_nodes = RSNs[:,i]==1
#     RSN_index.append(np.array(range(90))[these_nodes])
# ##########

# def RSN_profile_FC(FC):
#     ##asumimos que la señal de BOLD viene filtrada
#     profile = []
#     for k,rsn in enumerate(RSN_index):
#         subFC = FC[rsn,:][:,rsn]
#         flat_subFC = flat_FC(subFC)
#         mean = flat_subFC.mean()
#         profile.append(mean)
#     return np.array(profile)


def find_extreme(df,targetval,ex="min", cols=None):
    if ex =="min":
        exval = df[targetval].min()
    else:
        exval = df[targetval].max()
    
    if cols:
        return df[df[targetval] == exval].iloc[0][cols]
    return df[df[targetval] == exval].iloc[0]

def xy2plotcor(x,y,xvals,yvals):
    lennyx,minx,maxx = len(xvals),min(xvals),max(xvals)
    mx = (lennyx-1)/(maxx-minx)
    nx = -minx*mx
    lennyy,miny,maxy = len(yvals),min(yvals),max(yvals)
    my = (lennyy-1)/(maxy-miny)
    ny = -miny*my
    xcor = x*mx+nx
    ycor= y*my+ny
    return xcor,ycor

def fill_missing(df,col1,col2,what=np.nan):
    df2 = df
    idx1,idx2 = list(df.columns).index(col1),list(df.columns).index(col2) ##lugares de las columnas
    vals1,vals2 = df[col1],df[col2] #valores que quiero
    
    present = df[[col1,col2]].values ##los que estan 
    here = [(present[i,0],present[i,1]) for i in range(len(df))]
    
    for x in vals1:
        for y in vals2:
            if (x,y) not in here:
                lis = what*np.ones(df.shape[1]) ##llenamos con basura todo
                lis[idx1],lis[idx2] = x,y ##menos las columnas de interes
                df2 = df2.append(pd.DataFrame(lis[:,None].T,columns=df.columns))
    return df2

thal = [38,51]
not_thal = [i for i in range(90) if i not in thal]
#cada fila es lo que entra a la corteza
def scale_mat(struct,G,subG): ##aqui subG puede ser un vector
    out = np.copy(struct)
    for area in thal:
        out[:,area] = subG*struct[:,area]
    for area in not_thal:
        out[:,area] = G*struct[:,area]
    return out

def reconstruct(lower_tri,lenny=90,diag_fill=1):
    lower = np.tril_indices(lenny,k=-1)
    out = np.zeros((lenny,lenny))
    out[lower] = lower_tri
    for i in range(lenny): ##simetrizamos
        out[i,i+1:] = out[i+1:,i]
    out[np.diag_indices(lenny)] = diag_fill
    return out

def reconstruct_leida(v,lenny=68):
    v = v.reshape((lenny,1))
    return np.dot(v,v.T)


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
    # for t in range(times):
    #     for i in range(lenny):
    #         for j in range(lenny):
    #             out[t,i,j] = np.cos(phases[t,i]-phases[t,j])
                # out[t,i,j] = phases[t,i] - phases[t,j]
    return out



def construct_FCD_phase(difs_phases):
    outdim = len(difs_phases)
    out = np.zeros((outdim,outdim))
    for i in range(outdim):
        for j in range(outdim):
            if i <= j:
                mati = difs_phases[i,:,:][lower]
                matj = difs_phases[j,:,:][lower]
                val = np.corrcoef(mati,matj)[0,1] ##aqui deberia usar manhattan
                # mati = difs_phases[i,:,:]
                # matj = difs_phases[j,:,:]
                # val = ssim(mati,matj)
                out[i,j] = val
                out[j,i] = val
    return out



def get_phase_subFCD(data,filterr=False,t_as_col=True,TR=2.4,full=False):
    if not t_as_col:
        data = data.T
    n_entries,times = data.shape ###(dimensiones,tiempo)


    lower = np.tril_indices(n_entries,k=-1)
    if filterr:
        a,b = signal.bessel(2,[2 * 0.01 * TR, 2 * 0.1 * TR], btype = 'bandpass') ##banda [0.01,0.1]
        data = signal.filtfilt(a,b,data,axis=1)

    analytic_signal = signal.hilbert(data,axis=1)
    phases = np.unwrap(np.angle(analytic_signal))
    difs_phases = coherences(phases,t_as_col=True) 
    subvecs = np.concatenate([difs_phases[lower][:,t].reshape(-1,1) for t in range(times)],axis=1)
    if full:
        return difs_phases,phases,subvecs
    else:
        return subvecs


def LEiDA(mat):
    w,v=np.linalg.eig(mat)
    w,v = np.real(w),np.real(v)
    idd = np.argsort(w)[::-1]
    w = w[idd]
    v = v[:,idd]
    v1 = v[:,0]
    return v1

def get_LEiDAs(data,lendata,filterr=False):
    if filterr:
        data = signal.filtfilt(a,b,data,axis=0)
        jilber = signal.hilbert(data,axis=0)
        analytic_signal = jilber[lendata:lendata*2]
        phases = np.unwrap(np.angle(analytic_signal))
        
        difs_phases = coherences(phases)
        # subvecs = np.concatenate([difs_phases[i][lower].reshape(-1,1) for i in range(lendata)],axis=1)
        LEiDAs = np.concatenate([LEiDA(difs_phases[i]).reshape(-1,1) for i in range(lendata)],axis=1)
    else:
        jilber = signal.hilbert(data,axis=0)
        analytic_signal = jilber[lendata:lendata*2]
        phases = np.unwrap(np.angle(analytic_signal)) ##wrap??
        
        difs_phases = coherences(phases)
        # subvecs = np.concatenate([difs_phases[i][lower].reshape(-1,1) for i in range(lendata)],axis=1)
        LEiDAs = np.concatenate([LEiDA(difs_phases[i]).reshape(-1,1) for i in range(lendata)],axis=1)
    return LEiDAs




def kl_divergence(p, q, sym=True):
    if sym:
        val = (np.sum(np.where(p != 0, p * np.log(p / q), 0))+np.sum(np.where(q != 0, q * np.log(q / p), 0)))/2
    else:
        val = np.sum(np.where(p != 0, p * np.log(p / q), 0))
    return val

def entr(p):
    if p==0:
        return 0
    else:
        return -p*np.log(p)
ent = np.vectorize(entr)

def transition_matrix(states): ##deben estaro enumerados de 0 en adelante
    n = 1+ max(states) #number of states
    M = np.zeros((n,n))

    for (i,j) in zip(states,states[1:]):
        M[i][j] += 1
        
    for i in range(n):
        row = M[i,:]
        s = np.sum(row)
        if s>0:
            M[i,:] = row/s
    #distribucion estacionaria
    w,v = np.linalg.eig(M.T)
    dist_stat = v[:,np.round(w,4)==1] #esto es cercano a 1
    dist_stat = (dist_stat/dist_stat.sum()).flatten()
    entro = 0
    for i in range(n):
        Si = dist_stat[i]*np.sum(ent(M[i,:]))
        entro+=Si
        
    return M,dist_stat,entro

def FCD_speeds(FCD,order = 2): ##notese que entrega una speed distinta para cada tiempo
    times = len(FCD)
    out = np.zeros((times-order,order))
    for i in range(times-order):
        out[i] = FCD[i,i:i+order]
    return out ##notese que entrega una speed distinta para cada tiempo

def speeds(data,lendata):
    data = signal.filtfilt(a,b,data,axis=0)
    jilber = signal.hilbert(data,axis=0)
    analytic_signal = jilber[lendata:lendata*2]
    phases = np.angle(analytic_signal)
    
    difs_phases = coherences(phases)
    fcd_phase = phase_FCD(difs_phases) 
    meta = np.var(fcd_phase)
    # fcd_LEiDA,vecs = LEiDA(difs_phases)

    all_speeds = FCD_speeds(fcd_phase,order=2)
    all_speeds = 1-all_speeds[:,1]
    return all_speeds,meta

def process_labels_pyclustering(lon,pyLabels):
    labels = np.zeros(lon)
    for i in range(len(pyLabels)):
        labels[pyLabels[i]] = i
    return labels.astype(int)


def symm2RSNd(FC):
    new_entries = [21, 22, 23, 24, 66, 67, 68,
               23, 24, 25, 26, 29, 60, 63, 64, 65,
               5, 6, 8, 14, 15,16, 20, 31, 36, 37, 38, 39, 40, 47, 48, 49, 50, 51, 52, 53, 58, 69, 72, 73, 75, 81,
               8, 16, 28, 29, 34, 39, 50, 55, 60, 61, 73, 81, 89,
               12, 15, 17, 32, 33, 38, 49, 51, 56, 57, 72, 77,
               0, 1, 3, 9, 11, 12, 14, 15, 16, 31, 33, 41, 48, 54, 56, 73, 74, 75, 77, 78, 80, 86, 88, 89]
    return FC[new_entries,:][:,new_entries]

def split_data(data,lennys,setoff=0,t_as_col=True):
    if not t_as_col:
        data=data.T
    inits = setoff+np.array([0]+ list(np.cumsum(lennys)[:-1]))
    ends = setoff+np.cumsum(lennys)
    subs = [data[:,inits[i]:ends[i]] for i in range(len(lennys))]
    return subs

def ind_jump_lengths(ind_data,t_as_col = True):
    if not t_as_col:
        ind_data = ind_data.T
    diffs= np.diff(ind_data,axis=1)
    # print(diffs.shape)
    return np.linalg.norm(diffs,axis=0,ord=2) #norma de las diferencias
    # return np.array([1-np.corrcoef(ind_data[:,i],ind_data[:,i+1])[0,1] for i in range(ind_data.shape[1]-1)])

def list_of_jump_lengths(data,lennys,full=False,split=False,t_as_col=True):
    if not t_as_col:
        data = data.T
    if split:
        jump_lengths = [ind_jump_lengths(ind) for ind in split_data(data,lennys,t_as_col=True)]
        jump_spatial = np.concatenate([np.diff(ind,axis=1) for ind in split_data(data,lennys,t_as_col=True)],axis=1)
    else:
        jump_lengths = np.concatenate([ind_jump_lengths(ind) for ind in split_data(data,lennys)])
        jump_spatial = np.concatenate([np.diff(ind,axis=1) for ind in split_data(data,lennys)],axis=1)
    
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

def reord(data):
    "toma un AAL90 con (1L,1R,2L,2R,...) intercalado y lo simetriza a (1L,2L,...,44L,45L,45R,44R,...,2R,1R)"
    left = range(0,90,2)
    right = range(1,90,2)
    ordd = list(left) + list(right[::-1])
    if len(data.shape)==2:
        return data[ordd,:][:,ordd]
    elif len(data.shape)==1:
        return data[ordd]

