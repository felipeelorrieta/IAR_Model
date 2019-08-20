import pandas as pd
import numpy as np
import scipy
from scipy.optimize import minimize_scalar
import statsmodels.formula.api as sm
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

def IAR_sample(phi,n,sT):
    Sigma=np.zeros(shape=(n,n))
    for i in range(np.shape(Sigma)[0]):
        d=sT[i]-sT[i:n]
        Sigma[i,i:n]=phi**abs(d)
        Sigma[i:n,i]=Sigma[i,i:n]
    b,v=LA.eig(Sigma)
    A=np.dot(np.dot(v,np.diag(np.sqrt(b))),v.transpose())
    e=np.random.normal(0, 1, n)
    y=np.dot(A,e)
    return y, sT

def IAR_phi_loglik(x,y,sT,delta,include_mean=False,standarized=True):
    n=len(y)
    sigma=1
    mu=0
    if standarized == False:
        sigma=np.var(y,ddof=1)
    if include_mean == True:
        mu=np.mean(y)
    d=np.diff(sT)
    delta=delta[1:n]
    phi=x**d
    yhat=mu+phi*(y[0:(n-1)]-mu)
    y2=np.vstack((y[1:n],yhat))
    cte=0.5*n*np.log(2*np.pi)
    s1=cte+0.5*np.sum(np.log(sigma*(1-phi**2)+delta**2)+(y2[0,]-y2[1,])**2/(sigma*(1-phi**2)+delta**2))
    return s1

def IAR_loglik(y,sT,delta,include_mean=False,standarized=True):
    out=minimize_scalar(IAR_phi_loglik,args=(y,sT,delta,include_mean,standarized),bounds=(0,1),method="bounded",tol=0.0001220703)
    return out.x


def ajuste(t,m,f):
    ws = pd.DataFrame({
    'x': m,
    't': t})
    ols_fit=sm.ols('x ~ t', data=ws).fit()
    m = ols_fit.resid
    ws = pd.DataFrame({
    'x': m,
    'y1': np.sin(2*np.pi*t*f),
    'y2': np.cos(2*np.pi*t*f),
    'y3': np.sin(4*np.pi*t*f),
    'y4': np.cos(4*np.pi*t*f),
    'y5': np.sin(6*np.pi*t*f),
    'y6': np.cos(6*np.pi*t*f),
    'y7': np.sin(8*np.pi*t*f),
    'y8': np.cos(8*np.pi*t*f)
    })
    ols_fit=sm.ols('x ~ y1+y2+y3+y4+y5+y6+y7+y8', data=ws).fit()
    res = ols_fit.resid
    return res,t

def gentime(n,lambda1=130,lambda2=6.5,w1=0.15,w2=0.85):
    aux1=np.random.exponential(scale=lambda1,size=n)
    aux2=np.random.exponential(scale=lambda2,size=n)
    aux = np.hstack((aux1,aux2))
    prob = np.hstack((np.repeat(w1, n),np.repeat(w2, n)))/n
    dT=np.random.choice(aux,n,p=prob)
    sT=np.cumsum(dT)
    return sT

#def IAR_Test(y,sT,f,phi,plot='TRUE',xlim=np.arange(-1,0.1,1)):
#    aux=np.arange(2.5,48,2.5)
#    aux=np.hstack((-aux,aux))
#    aux=np.sort(aux)
#    f0=f*(1+aux/100)
#    f0=np.sort(f0)
#   l1=len(f0)
#    bad=np.zeros(l1)
#    m=y
#    for j in range(l1):
#        res,sT=ajuste(sT,m,f0[j])
#        y=res/np.sqrt(np.var(res,ddof=1))
#        res3=IAR_loglik(y,sT)
#        bad[j]=res3
#    mubf=np.mean(np.log(bad))
#    sdbf=np.std(np.log(bad),ddof=1)
#    z0=np.log(phi)
#    pvalue=scipy.stats.norm.cdf(z0,mubf,sdbf)
#    norm=np.hstack((mubf,sdbf))
#    if plot=='TRUE':
#       fig = plt.figure()
#       density = gaussian_kde(np.log(bad))
#       xs = np.linspace(xlim[0],xlim[1],1000)
#       print(z0)
#       density.covariance_factor = lambda : .25
#       density._compute_covariance()
#       plt.plot(xs,density(xs))
#       plt.axis([xlim[0],xlim[1], 0, np.max(density(xs))+0.01,])
#       #ax = plt.add_subplot(111)
#       plt.plot(z0, np.max(density(xs))/100, 'o')
#       plt.show()
#    return phi,norm,z0,pvalue

def IAR_Test2(y,sT,phi,iter=100,plot=True,xlim=np.arange(-1,0.1,1),bw=0.15,nameP='output.pdf'):
    phi2=np.zeros(iter)
    for i in range(iter):
        order=np.random.choice(range(len(y)),len(y),replace=False)
        y1=y[order]
        phi2[i]=IAR_loglik(y1,sT)
    mubf=np.mean(np.log(phi2))
    sdbf=np.std(np.log(phi2),ddof=1)
    z0=np.log(phi)
    pvalue=scipy.stats.norm.cdf(z0,mubf,sdbf)
    norm=np.hstack((mubf,sdbf))
    if plot==True:
       pdf = matplotlib.backends.backend_pdf.PdfPages(nameP)
       fig = plt.figure()
       xs = np.linspace(xlim[0],xlim[1],1000)
       density = kde_sklearn(np.log(phi2),xs,bandwidth=bw)
       print(z0)
       plt.plot(xs,density)
       plt.axis([xlim[0],xlim[1], 0, np.max(density)+0.01,])
       #ax = plt.add_subplot(111)                                                                                                                                                                          
       plt.plot(z0, np.max(density)/100, 'o')
       plt.show()
       pdf.savefig(1)
       pdf.close()
    return phi,norm,z0,pvalue

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

def IAR_Test(y,sT,f,phi,plot=True,xlim=np.arange(-1,0.1,1),bw=0.15,nameP='output.pdf'):
    aux=np.arange(2.5,48,2.5)
    aux=np.hstack((-aux,aux))
    aux=np.sort(aux)
    f0=f*(1+aux/100)
    f0=np.sort(f0)
    l1=len(f0)
    bad=np.zeros(l1)
    m=y
    for j in range(l1):
        res,sT=ajuste(sT,m,f0[j])
        y=res/np.sqrt(np.var(res,ddof=1))
        res3=IAR_loglik(y,sT)
        bad[j]=res3
    mubf=np.mean(np.log(bad))
    sdbf=np.std(np.log(bad),ddof=1)
    z0=np.log(phi)
    pvalue=scipy.stats.norm.cdf(z0,mubf,sdbf)
    norm=np.hstack((mubf,sdbf))
    if plot==True:
       pdf = matplotlib.backends.backend_pdf.PdfPages(nameP) 
       fig = plt.figure()
       xs = np.linspace(xlim[0],xlim[1],1000)
       density = kde_sklearn(np.log(bad),xs,bandwidth=bw)
       #print(z0)
       plt.plot(xs,density)
       plt.axis([xlim[0],xlim[1], 0, np.max(density)+0.01,])
       #ax = plt.add_subplot(111)                                                                                                                                                                          
       plt.plot(z0, np.max(density)/100, 'o')
       pdf.savefig(1)
       pdf.close()
    return phi,norm,z0,pvalue

def IAR_phi_kalman(x,y,yerr,t,zero_mean=True,standarized=True,c=0.5):
    n=len(y)
    Sighat=np.zeros(shape=(1,1))
    Sighat[0,0]=1
    if standarized == False:
         Sighat=np.var(y)*Sighat
    if zero_mean == False:
         y=y-np.mean(y)
    xhat=np.zeros(shape=(1,n))
    delta=np.diff(t)
    Q=Sighat
    phi=x
    F=np.zeros(shape=(1,1))
    G=np.zeros(shape=(1,1))
    G[0,0]=1
    sum_Lambda=0
    sum_error=0
    if np.isnan(phi) == True:
        phi=1.1
    if abs(phi) < 1:
        for i in range(n-1):
            Lambda=np.dot(np.dot(G,Sighat),G.transpose())+yerr[i+1]**2 
            if (Lambda <= 0) or (np.isnan(Lambda) == True):
                sum_Lambda=n*1e10
                break
            phi2=phi**delta[i]
            F[0,0]=phi2
            phi2=1-phi**(delta[i]*2)
            Qt=phi2*Q
            sum_Lambda=sum_Lambda+np.log(Lambda)
            Theta=np.dot(np.dot(F,Sighat),G.transpose())
            sum_error= sum_error + (y[i]-np.dot(G,xhat[0:1,i]))**2/Lambda
            xhat[0:1,i+1]=np.dot(F,xhat[0:1,i])+np.dot(np.dot(Theta,inv(Lambda)),(y[i]-np.dot(G,xhat[0:1,i])))
            Sighat=np.dot(np.dot(F,Sighat),F.transpose()) + Qt - np.dot(np.dot(Theta,inv(Lambda)),Theta.transpose())
        yhat=np.dot(G,xhat)
        out=(sum_Lambda + sum_error)/n
        if np.isnan(sum_Lambda) == True:
            out=1e10
    else:
        out=1e10
    return out


def IAR_kalman(y,sT,delta=0,zero_mean=True,standarized=True):
    if np.sum(delta)==0:
        delta=np.zeros(len(y))
    out=minimize_scalar(IAR_phi_kalman,args=(y,delta,sT,zero_mean,standarized),bounds=(0,1),method="bounded",tol=0.0001220703)
    return out.x

