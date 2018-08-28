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
from FunctionsIATS import IAR_phi_loglik,IAR_loglik,ajuste,IAR_Test2

os.chdir('/home/felipe/astro/IrregularAR/PackageIAR_Py')
namefile='ResidualPlanet.dat'
data = np.genfromtxt(namefile, unpack=True,delimiter='',filling_values=np.nan)
t=data[0,]
luz=data[1,]
m=luz/np.sqrt(np.var(luz,ddof=1))
#Standarized Data 
phi=IAR_loglik(m,t,True)
#Unstandardized Data
#phi=IAR_loglik(luz,t,False)                                                                                                                                                                               
phi,norm,z0,pvalue=IAR_Test2(m,t,phi,plot=True,xlim=(-10,-9),nameP='outputP.pdf')
#phi,norm,z0,pvalue=IAR_Test2(m,t,phi,plot=False) #Without Plot    
print(phi)
print(phi**(np.mean(np.diff(t))))
print(norm)
print(z0)
print(pvalue)
