#export LD_LIBRARY_PATH=/home/felipe/astro/Software/boost/lib:${LD_LIBRARY_PATH}
#export LD_LIBRARY_PATH=/home/felipe/astro/Software/armadillo/lib:${LD_LIBRARY_PATH}

import os
import numpy as np
from scipy.optimize import minimize
import scipy
import carmcmc as cm

os.chdir('/home/felipe/astro/IrregularAR/PackageIAR_Py')
from FunctionsIATS import IARg_sample,IAR_phi_gamma,gentime,IAR_gamma

np.random.seed(6713)
sT=gentime(n=300)
y,sT =IARg_sample(0.9,300,sT,1,1)
phi,mu,sigma,ll=IAR_gamma(y,sT)

ysig = np.zeros(300)
carma_model = cm.CarmaModel(sT, y, ysig)
pmax = 1
MLE, pqlist, AICc_list = carma_model.choose_order(pmax)
carma_sample = carma_model.run_mcmc(50000)
print carma_sample.parameters
carma_sample.get_samples('log_omega')
