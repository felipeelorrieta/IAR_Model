import pandas as pd
import numpy as np
import scipy
from scipy.optimize import minimize,minimize_scalar
import statsmodels.formula.api as sm
from numpy import linalg as LA
import matplotlib
matplotlib.use('Agg')
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

def IAR_phi_loglik(x,y,sT,standarized=True):
    n=len(y)
    sigma=1
    if standarized == False:
        sigma=np.var(y,ddof=1)
    d=np.diff(sT)
    phi=x**d
    yhat=phi*y[0:(n-1)]
    y2=np.vstack((y[1:n],yhat))
    cte=0.5*n*np.log(2*np.pi)
    s1=cte+0.5*np.sum(np.log(sigma*(1-phi**2))+(y2[0,]-y2[1,])**2/(sigma*(1-phi**2)))
    return s1

def IAR_loglik(y,sT,standarized=True):
    out=minimize_scalar(IAR_phi_loglik,args=(y,sT,standarized),bounds=(0, 1),method='bounded',tol=0.0001220703)
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

def IARg_sample(phi,n,sT,sigma2,mu):
    d=np.diff(sT)
    y=np.zeros(n)
    y[0]=np.random.gamma(shape=1, scale=1, size=1)
    shape=np.zeros(n)
    scale=np.zeros(n)
    yhat=np.zeros(n)
    for i in range(n-1):
        phid=phi**(d[i])
        yhat[i+1]=mu+phid * y[i]
        gL = sigma2*(1-phid**(2))
        shape[i+1]=yhat[i+1]**2/gL
        scale[i+1]=(gL/yhat[i+1])
        y[i+1]=np.random.gamma(shape=shape[i+1], scale=scale[i+1], size=1)
    return y, sT

def IAR_phi_gamma(x,y,sT):
    mu=x[1]
    sigma=x[2]
    x=x[0]
    d=np.diff(sT)
    n=len(y)
    phi=x**d
    yhat=mu+phi*y[0:(n-1)]
    gL=sigma*(1-phi**2)
    beta=gL/yhat
    alpha=yhat**2/gL
    s1=np.sum(-alpha*np.log(beta) - scipy.special.gammaln(alpha) - y[1:n]/beta + (alpha-1) * np.log(y[1:n])) - y[0]
    s1=-s1
    return s1

def IAR_gamma(y,sT):
    aux=1e10
    value=1e10
    br=0
    for i in range(20):
        phi=np.random.uniform(0,1,1).mean()
        mu=np.mean(y)*np.random.uniform(0,1,1).mean()
        sigma=np.var(y)*np.random.uniform(0,1,1).mean()
        bnds = ((0, 0.9999), (0.0001, np.mean(y)),(0.0001, np.var(y)))
        out=minimize(IAR_phi_gamma,np.array([phi, mu, sigma]),args=(y,sT),bounds=bnds,method='L-BFGS-B')
        value=out.fun
        if aux > value:
            par=out.x
            aux=value
            br=br+1
        if aux <= value and br>5 and i>10:
            break
        #print br
    if aux == 1e10:
       par=np.zeros(3)
    return par[0],par[1],par[2],aux

def IARt_sample(phi,n,sT,sigma2,nu):
    d=np.diff(sT)
    y=np.zeros(n)
    y[0]=np.random.normal(loc=0, scale=1, size=1)
    yhat=np.zeros(n)
    for i in range(n-1):
        phid=phi**(d[i])
        yhat[i+1]=phid * y[i]
        gL = sigma2*(1-phid**(2))
        y[i+1]=np.random.standard_t(df=nu,size=1)*np.sqrt(gL*(nu-2)/nu)+yhat[i+1]
    return y, sT

def IAR_phi_t(x,y,sT,nu):
    sigma=x[1]
    x=x[0]
    d=np.diff(sT)
    n=len(y)
    phi=x**d
    yhat=phi*y[0:(n-1)]
    gL=sigma*(1-phi**2)*(nu-2)/nu
    cte=(n-1)*np.log((scipy.special.gamma((nu+1)/2)/(scipy.special.gamma(nu/2)*np.sqrt(nu*np.pi))))
    stand=((y[1:n]-yhat)/np.sqrt(gL))**2
    s1=np.sum(0.5*np.log(gL))
    s2=np.sum(np.log(1 + (1/nu)*stand))
    out=cte-s1-((nu+1)/2)*s2 -0.5*(np.log(2*np.pi) + y[0]**2)
    out=-out
    return out

def IAR_t(y,sT,nu):
    aux=1e10
    value=1e10
    br=0
    for i in range(20):
        phi=np.random.uniform(0,1,1)[0]
        sigma=np.var(y)*np.random.uniform(0,1,1)[0]
        nu=float(nu)
        bnds = ((0, 0.9999), (0.0001, 2*np.var(y)))
        out=minimize(IAR_phi_t,np.array([phi, sigma]),args=(y,sT,nu),bounds=bnds,method='L-BFGS-B')
        value=out.fun
        if aux > value:
            par=out.x
            aux=value
            br=br+1
        if aux <= value and br>5 and i>10:
            break
        #print br                                                                               
    if aux == 1e10:
        par=np.zeros(2)
    return par[0],par[1],aux
