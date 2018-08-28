import pandas as pd
import os
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from numpy import linalg as LA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from statsmodels.graphics import utils

os.chdir('/home/felipe/astro/IrregularAR/PackageIAR_Py')
from FunctionsIATS import IAR_sample,IAR_phi_loglik,IAR_loglik,ajuste,gentime

#Generating IAR sample
np.random.seed(6713)
sT=gentime(n=100)
y,sT =IAR_sample(0.99,100,sT)

#Compute Phi
phi=IAR_loglik(y,sT)
phi

#Compute the standard deviation of innovations
n=len(y)
d=np.hstack((0,np.diff(sT)))
phi1=phi**d
yhat=phi1*np.hstack((0,y[0:(n-1)]))
sigma=np.var(y,ddof=1)
nu=np.hstack((sigma,sigma*(1-phi1**(2))[1:n]))
tau=nu/sigma
sigmahat=np.mean((y-yhat)**2/tau)
nuhat=sigmahat*(1-phi1**(2))
nuhat2=np.sqrt(nuhat)
ciar=(1-phi1**(2))
nuhat3=np.sqrt(sigma*ciar)

####Fitting Regular Models

arma_mod10 = sm.tsa.ARMA(y, (1,0)).fit(trend='nc')
#arma_mod10.summary()
syar=arma_mod10.sigma2/(1-arma_mod10.params[0]**2)
car=(1-arma_mod10.params[0]**2)
sear=np.sqrt(sigma*car)


pdf = matplotlib.backends.backend_pdf.PdfPages("outputAR.pdf")
ax=None
fig, ax = utils.create_mpl_ax(ax)
ax.vlines(sT[1:100], ymin=np.zeros(100), ymax=nuhat3[1:100])
ax.scatter(sT[1:100],nuhat3[1:100],c='green')
ax.axhline(np.sqrt(sigma),color='red')
ax.axhline(sear,color='blue')
ax.axhline(np.mean(nuhat3[1:100]),color='black')
ax.set_ylim([0, 1])
pdf.savefig(1)
pdf.close()
