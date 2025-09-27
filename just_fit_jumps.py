# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:05:44 2024

@author: flehu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import skew,ttest_ind,kstest,kurtosis as kurt
from scipy.stats import gamma,levy,lognorm,levy_stable,weibull_min,gengamma
from statsmodels.stats.multitest import multipletests
import utils
import seaborn as sns
import warnings
import levy as leby
warnings.filterwarnings("ignore")

savefolder = "chewed_data/"
states = ["CNT","MCS","UWS"] #wake, deep anesthesia, recovery

npzfile = np.load("data/pooled_jumps_euclidean.npz")
print(npzfile.files)
CNT_jumps_pooled = npzfile["CNT_jumps_pooled"]
MCS_jumps_pooled = npzfile["MCS_jumps_pooled"]
UWS_jumps_pooled = npzfile["UWS_jumps_pooled"]

# excludes = [[],[13,17],[12]] ##nota: exclui segun FCs con media outlier
lenis = {}
lenis["CNT"] = [192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192]
lenis["MCS"] = [192, 192, 192, 192, 192, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 195, 195, 195, 195]
lenis["UWS"] = [192, 192, 192, 192, 192, 192, 192, 192, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 195, 195]


xmin,xmax = 0,np.max(np.concatenate([CNT_jumps_pooled,MCS_jumps_pooled,UWS_jumps_pooled]))
xaxis = np.linspace(xmin,xmax,1000)

colors = ("tab:blue","tab:orange","tab:green");alfa = 0.5
#%%plot all


plt.figure(1)
plt.clf()


###levy generalizada
plt.subplot(331)
plt.title("stable general")
print("stable general alpha,beta,mu,sigma")
for d,dist in enumerate((CNT_jumps_pooled,MCS_jumps_pooled,UWS_jumps_pooled)):
    pars,ll = leby.fit_levy(dist)
    alpha,beta,mu,sigma = pars.get()
    pdf = leby.levy(xaxis,alpha=alpha,beta=beta,mu=mu,sigma=sigma)
    plt.hist(dist,label="CNT",density=True,alpha=alfa,bins=50,color=colors[d])
    plt.plot(xaxis,pdf,color=colors[d])
    print(states[d],(alpha,beta,mu,sigma))

#gamma
plt.subplot(332)
plt.title("gamma")
print("gamma a,loc,scale")
for d,dist in enumerate((CNT_jumps_pooled,MCS_jumps_pooled,UWS_jumps_pooled)):
    a,loc,scale = gamma.fit(dist,floc=0)
    pdf = gamma.pdf(xaxis,a=a,loc=loc,scale=scale)
    plt.hist(dist,label="CNT",density=True,alpha=alfa,bins=50,color=colors[d])
    plt.plot(xaxis,pdf,color=colors[d])
    print(states[d],(a,loc,scale))
    
##levy
plt.subplot(333)
plt.title("levy")
print("levy loc,scale")
for d,dist in enumerate((CNT_jumps_pooled,MCS_jumps_pooled,UWS_jumps_pooled)):
    loc,scale = levy.fit(dist,floc=0)
    pdf = levy.pdf(xaxis,loc=loc,scale=scale)
    plt.hist(dist,label="CNT",density=True,alpha=alfa,bins=50,color=colors[d])
    plt.plot(xaxis,pdf,color=colors[d])
    print(states[d],(loc,scale))
    
##lognormal
plt.subplot(334)
plt.title("lognormal")
print("lognormal s,loc,scale")
for d,dist in enumerate((CNT_jumps_pooled,MCS_jumps_pooled,UWS_jumps_pooled)):
    s,loc,scale = lognorm.fit(dist,floc=0)
    pdf = lognorm.pdf(xaxis,s=s,loc=loc,scale=scale)
    plt.hist(dist,label="CNT",density=True,alpha=alfa,bins=50,color=colors[d])
    plt.plot(xaxis,pdf,color=colors[d])
    print(states[d],(s,loc,scale))
    

##weibull
plt.subplot(335)
plt.title("weibull")
print("weibull c,loc,scale")
for d,dist in enumerate((CNT_jumps_pooled,MCS_jumps_pooled,UWS_jumps_pooled)):
    c,loc,scale = weibull_min.fit(dist,floc=0)
    pdf = weibull_min.pdf(xaxis,c=c,loc=loc,scale=scale)
    plt.hist(dist,label="CNT",density=True,alpha=alfa,bins=50,color=colors[d])
    plt.plot(xaxis,pdf,color=colors[d])
    print(states[d],(c,loc,scale))
    
##generalized gamma
plt.subplot(336)
plt.title("generalized gamma")
print("generalized gamma a,c,loc,scale")
for d,dist in enumerate((CNT_jumps_pooled,MCS_jumps_pooled,UWS_jumps_pooled)):
    a,c,loc,scale = gengamma.fit(dist,floc=0)
    pdf = gengamma.pdf(xaxis,a=a,c=c,loc=loc,scale=scale)
    plt.hist(dist,label="CNT",density=True,alpha=alfa,bins=50,color=colors[d])
    plt.plot(xaxis,pdf,color=colors[d])
    print(states[d],(a,c,loc,scale))
    
    
plt.show()
    
#%%

plt.figure(2)
plt.clf()
plt.subplot(111)
# plt.title("weibull fit",fontsize=14)
# print("weibull c,loc,scale")
for d,dist in enumerate((CNT_jumps_pooled,MCS_jumps_pooled,UWS_jumps_pooled)):
    c,loc,scale = weibull_min.fit(dist,floc=0)
    mean, var, skew, kurt = weibull_min.stats(c, moments='mvsk')
    weibull_min.stats
    print(c,loc,scale)
    print(mean,var,skew,kurt)
    
    # pdf = weibull_min.pdf(xaxis,c=c,loc=loc,scale=scale)
    plt.hist(dist,label=states[d],density=True,alpha=alfa,bins=50,color=colors[d])
    # plt.plot(xaxis,pdf,color=colors[d])
plt.xticks((0,10,20,30,40,50,60),(0,10,20,30,40,50,60),fontsize=12)
plt.yticks((0,0.01,0.02,0.03,0.04,0.05,0.06),(0,0.01,0.02,0.03,0.04,0.05,0.06),fontsize=12)
plt.legend(fontsize=14)
plt.show()
    # print(states[d],(c,loc,scale))