import pandas as pd
import os
import numpy as np
import statsmodels.formula.api as sm
import scipy
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

os.chdir('/home/felipe/astro/IrregularAR/PackageIAR_Py')
from FunctionsIATS import IAR_phi_loglik,IAR_loglik,ajuste,IAR_Test

os.chdir('/home/felipe/astro/IrregularAR/PackageIAR_Py')
namefile='c-54101.hip'
data = np.genfromtxt(namefile, unpack=True,delimiter='',filling_values=np.nan)
t=data[0,]
m=data[1,]
f1=0.060033386
res,sT=ajuste(t,m,f1)
y=res/np.sqrt(np.var(res,ddof=1))
#Standarized Data
phi=IAR_loglik(y,sT,True)
#Unstandardized Data
#phi=IAR_loglik(res,sT,False) 
phi,norm,z0,pvalue=IAR_Test(m,sT,f1,phi,plot=True,xlim=(-10,0.5),nameP='output.pdf')
#phi,norm,z0,pvalue=IAR_Test(m,sT,f1,phi,plot=False) #Without Plot
print(phi)
print(norm)
print(z0)
print(pvalue)

