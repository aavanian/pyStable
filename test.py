# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 09:57:22 2016

@author: pakitochus
"""
import os
import matplotlib.pyplot as plt
import numpy as np
os.chdir('/home/pakitochus/Dropbox/Investigacion/Experimentos/201604-Brain_Simulator/pylevy')
import stable as stb
os.chdir('..')

est = stb.StableDist(alpha=0.5, beta=-0.5, mu=1, sigma=0.1)
#est._make_data_file()
#est._make_approx_data_file()

# TEST NORMAL DISTRIBUTION. 
#plt.plot(ejex, est._approximate_pdf(ejex, 0.5, 1))
x = np.random.randn(3000)
from scipy.stats import norm
ejex=np.arange(-5,5,0.01)
plt.plot(ejex,norm.pdf(ejex), label='Original')

est.fit(x)
plt.plot(ejex,est.levy(ejex), label='Estimated')
plt.legend()
# Assertions for normal distribution.  
assert np.abs(2.-est.alpha)<1e-5/len(x)
assert np.abs(0.-est.mu)<0.1
assert np.abs(1.-np.sqrt(2)*est.sigma)<2/np.sqrt(len(x))
#Variance of a alfa stable is 2*sigma**2
#%% TEST ALPHA RANDOM NUMBERS. 
est = stb.StableDist(alpha=0.6, beta=1, mu=0, sigma=1)
plt.plot(ejex,est.levy(ejex), label='Initial')
x = est.sample(50000)
est = stb.StableDist(alpha=1,beta=0)
est.fit(x,maxiter=1e9)
plt.plot(ejex,est.levy(ejex), label='Estimated')
plt.legend()



#%%
plt.style.use('ggplot')
alpha=0.5
sigma=1
mu = 0
xvals = np.arange(-4, 4, 0.01)
for beta in [1, 0.5, 0]:
    pdf = []
    cdf = []
    for el in xvals:
        pdf.append(integrate.nquad(_calculate_phi, [[0.0, np.inf]], args=(el,alpha, beta, mu, sigma))[0])

    ax = plt.plot(xvals, np.array(pdf)/(2*np.pi), label=r'$\beta$=%.2f'%beta)
plt.legend()
plt.title(r'PDF for $\alpha$=%.2f'%alpha)