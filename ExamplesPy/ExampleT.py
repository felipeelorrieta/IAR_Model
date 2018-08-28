#export LD_LIBRARY_PATH=/home/felipe/astro/Software/boost/lib:${LD_LIBRARY_PATH}
#export LD_LIBRARY_PATH=/home/felipe/astro/Software/armadillo/lib:${LD_LIBRARY_PATH}

import os
import numpy as np
from scipy.optimize import minimize
import scipy
import carmcmc as cm

os.chdir('/home/felipe/astro/IrregularAR/PackageIAR_Py')
from FunctionsIATS import IARt_sample,IAR_phi_t,gentime,IAR_t

np.random.seed(6713)
sT=gentime(n=300)
y,sT =IARt_sample(0.9,300,sT,sigma2=1,nu=5)
phi,sigma,ll=IAR_t(y,sT,nu=5)

ysig = np.zeros(300)
carma_model = cm.CarmaModel(sT, y, ysig)
pmax = 1
MLE, pqlist, AICc_list = carma_model.choose_order(pmax)
carma_sample = carma_model.run_mcmc(50000)
print carma_sample.parameters
carma_sample.get_samples('log_omega')
