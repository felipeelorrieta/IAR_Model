IAR.sample=function(phi,n=100,sT){
 Sigma=matrix(0,ncol=n,nrow=n)
 for( i in 1:n){
      d<-sT[i]-sT[i:n]
      Sigma[i,i:n]=phi**abs(d)
      Sigma[i:n,i]=Sigma[i,i:n]
 }
 b=eigen(Sigma)
 V <- b$vectors
 A=V %*% diag(sqrt(b$values)) %*% t(V)
 e=rnorm(n)
 y=as.vector(A%*%e)
 out=list(series=y,times=sT)
 return(out)
}

IAR.phi.loglik=function(x,y,sT,standarized='TRUE')
{
 sigma=1
 if(standarized=='FALSE')
	sigma=var(y)
 n=length(y)
 d<-diff(sT)
 phi=x**d
 yhat=phi*y[-n]
 cte=(n/2)*log(2*pi)
 s1=cte+0.5*sum(log(sigma*(1-phi**2))+(y[-1]-yhat)**2/(sigma*(1-phi**2)))
 return(s1)
}

IAR.loglik=function(y,sT,standarized='TRUE'){
out=optimize(IAR.phi.loglik,interval=c(0,1),y=y,sT=sT,standarized=standarized)
 phi=out$minimum
 ll=out$objective
 return(list(phi=phi,loglik=ll))
}

IAR.unif.phi=function(y,sT){
 out=optimize(x.fun.3,interval=c(0,1),y=y,sT=sT)$minimum
 return(out)
}

x.fun.3=function(x,y,sT){
 n=length(y)
 phi=x
 Sigma=matrix(0,ncol=n,nrow=n)
 for( i in 1:n){
 d<-sT[i]-sT[i:n]
 Sigma[i,i:n]=phi**abs(d)
 Sigma[i:n,i]=Sigma[i,i:n]
 }
 if(det(Sigma)>0){
  b=eigen(Sigma)
  B=solve(Sigma)
  out=log(prod(b$values))+t(y)%*%B%*%y
  }
  else out=1e10
 return(out) # log-likelihood
}

IAR.Test=function(y,sT,f,phi,plot='TRUE',xlim=c(-1,0))
{
      aux=seq(2.5,47.5,by=2.5)
      aux=c(-aux,aux)
      aux=sort(aux)
      f0=f*(1+aux/100)
      f0<-sort(f0)
      l1<-length(f0)
      bad<-rep(0,l1)
      data<-cbind(sT,y)
      for(j in 1:l1)
      {
        results=harmonicfit(file=data,f1=f0[j])
        y=results$res/sqrt(var(results$res))
        sT=results$t
        res3=IAR.loglik(y,sT)[1]
        bad[j]=res3$phi
      }
      mubf<-mean(log(bad))
      sdbf<-sd(log(bad))
      z0<-log(phi)
      pvalue<-pnorm(z0,mubf,sdbf)
      out<-NULL
      if(plot=='TRUE')
      {
      	phi2=bad
      	phi2<-as.data.frame(phi2)
		phi<-as.data.frame(phi)
		out<-ggplot(phi2,aes(log(phi2)))+geom_density(adjust=2)+xlab("")		+ylab("")+theme_bw()+ggtitle("")+geom_point(data = phi,aes(log(phi)), y = 0, size = 4,col='red',shape=17) + xlim(xlim[1],xlim[2])+
        theme(plot.title = element_text(face="bold", size=20),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank())
        out
      }
     return(list(phi=phi,norm=c(mubf,sdbf),z0=z0,pvalue=pvalue,out=out))
}

IAR.Test2=function(y,sT,iter=100,phi,plot='TRUE',xlim=c(-1,0))
{
	phi2=rep(0,iter)
	for(i in 1:iter)
	{
		ord<-sample(1:length(y))
		y1<-y[ord]
		phi2[i]=IAR.loglik(y=y1,sT=sT)$phi
	}
    mubf<-mean(log(phi2))
    sdbf<-sd(log(phi2))
    z0<-log(phi)
    pvalue<-pnorm(z0,mubf,sdbf)
    out<-NULL
    if(plot=='TRUE')
    {
    	phi2<-as.data.frame(phi2)
		phi<-as.data.frame(phi)
		out<-ggplot(phi2,aes(log(phi2)))+geom_density(adjust=2)+xlab("")		+ylab("")+theme_bw()+ggtitle("")+geom_point(data = phi,aes(log(phi)), y = 0, size = 4,col='red',shape=17) + xlim(xlim[1],xlim[2])+
        theme(plot.title = element_text(face="bold", size=20),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank())
        out
      } 
return(list(phi=phi,norm=c(mubf,sdbf),z0=z0,pvalue=pvalue,out=out))
}

harmonicfit <-function(file,f1,nham=4,weights=NULL,print=FALSE){
        mycurve=file
        t=mycurve[,1]
        y=residuals(lm(mycurve[,2]~t))
        y=y
        for(i in 1:nham)
        {
        	y=cbind(y,sin(2*i*pi*f1*t),cos(2*i*pi*f1*t))	
        }
        df=data.frame(y)
        model=lm(y~.,data=df)
        if(length(weights)>0)
        		model=lm(y~.,data=df,weights=weights)
        if(print == TRUE)
        		print(summary(model))
        res=residuals(model)     
        R2<-summary(model)$adj.r.squared
        MSE<-anova(model)$"Mean Sq"[nham*2+1]
	    return(list(res=res,t=t,R2=R2,MSE=MSE))
}

foldlc<-function(file,f1)
{
	mycurve=file
	t=mycurve[,1]
    	m=mycurve[,2]
	merr=mycurve[,3]
    	P<-1/f1
    	fold<-(t-t[1])%%(P)/P
    	fold<-c(fold,fold+1)
    	m<-rep(m,2)
    	merr<-rep(merr,2)
    	dat1<-cbind(fold,m,merr)
    	dat1<-as.data.frame(dat1)
    	require(ggplot2)
    	out<-ggplot(dat1, aes(x=fold, y=m)) +
    	geom_errorbar(aes(ymin=m-merr, ymax=m+merr), width=.01,col='red') +
    	geom_point()+scale_y_reverse()+xlab("")+ylab("")+theme_bw()+
    	theme(plot.title = element_text(face="bold", size=20),
    	panel.grid.major = element_blank(),
    	panel.grid.minor = element_blank(),
    	panel.background = element_blank())
    	out
}

gentime<-function(n,lambda1=130,lambda2=6.5,p1=0.15,p2=0.85)
{
	dT<-rep(0,n)
	a<-sample(c(lambda1,lambda2),size=n,prob=c(p1,p2),replace=TRUE)
	dT[which(a==lambda1)]=rexp(length(which(a==lambda1)),rate=1/lambda1)
	dT[which(a==lambda2)]=rexp(length(which(a==lambda2)),rate=1/lambda2)
	sT=cumsum(dT)
	return(sT)
}

IARg.sample<-function(n,phi,st,sigma2=1,mu=1)
{
        delta<-diff(st) #Times differences
        y <- vector("numeric", n)
        y[1] <- rgamma(1,shape=1,scale=1) #initialization
        shape<-rep(0,n)
        scale<-rep(0,n)
        yhat<-rep(0,n)
        for (i in 2:n)
        {
        phid=phi**(delta[i-1]) #Expected Value Conditional Distribution
        yhat[i] = mu+phid * y[i-1]  #Mean of conditional distribution
        gL=sigma2*(1-phid**(2)) #Variance Value Conditional Distribution
        shape[i]=yhat[i]**2/gL
        scale[i]=(gL/yhat[i])
        y[i] <- rgamma(1,shape=shape[i], scale=scale[i])#Conditional Gamma IAR
        }
        #ts.plot(yhat)
        return(list(y=y,st=st))
}

IAR.phi.gamma<-function (x, y, sT) #Minus Log Full Likelihood Function
{
    mu=x[2]
    sigma=x[3]
    x=x[1]
    n = length(y)
    d <- diff(sT)
    xd=x**d
    yhat = mu+xd * y[-n]  #Mean of conditional distribution
    gL=sigma*(1-xd**(2))  #Variance of conditional distribution
    beta=gL/yhat #Beta parameter of gamma distribution
    alpha=yhat**2/gL #Alpha parameter of gamma distribution
    out=sum((-alpha)*log(beta) - lgamma(alpha) - y[-1]/beta + (alpha-1)*log(y[-1])) - y[1]  #Full Log Likelihood
    out=-out #-Log Likelihood (We want to minimize it)
    return(out)
}

IAR.gamma<-function(y, sT)
{
        aux<-1e10
        value<-1e10
        br<-0
        for(i in 1:20)
        {
                phi=runif(1)
                mu=mean(y)*runif(1)
                sigma=var(y)*runif(1)
			optim<-nlminb(start=c(phi,mu,sigma),obj=IAR.phi.gamma,y=y,sT=sT,lower=c(0,0.0001,0.0001),upper=c(0.9999,mean(y),var(y)))
                value<-optim$objective
                if(aux>value)
                {
                        par<-optim$par
                        aux<-value
                        br<-br+1
                }
                if(aux<=value & br>5 & i>10)
                        break;
        }
        if(aux==1e10)
                par<-c(0,0,0)
        return(list(phi=par[1],mu=par[2],sigma=par[3],ll=aux))
}


IAR.phi.t<-function (x, y, sT, nu=3) #Minus Log Full Likelihood Function
{
    sigma=x[2]
    x=x[1]
    n = length(y)
    d <- diff(sT)
    xd=x**d
    yhat = xd * y[-n]  #Mean of conditional distribution
    gL=sigma*(1-xd**(2))*((nu-2)/nu)  #Variance of conditional distribution
    cte = (n-1)*log((gamma((nu+1)/2)/(gamma(nu/2)*sqrt(nu*pi))))
    stand=((y[-1]-yhat)/sqrt(gL))**2
    s1=sum(0.5*log(gL))
    s2=sum(log(1 + (1/nu)*stand))
    out= cte - s1 - ((nu+1)/2)*s2 -0.5*(log(2*pi) + y[1]**2)
    out=-out #-Log Likelihood (We want to minimize it)
    return(out)
}

IAR.t<-function (y, sT,nu=3) #Find minimum of IAR.phi.gamma
{
        aux<-1e10
        value<-1e10
        br<-0
        for(i in 1:20)
        {
                phi=runif(1)
                sigma=var(y)*runif(1)
                optim<-nlminb(start=c(phi,sigma),obj=IAR.phi.t,y=y,sT=sT,nu=nu,lower=c(0,0.0001),upper=c(0.9999,2*var(y)))
                value<-optim$objective
                #print(c(optim$objective,optim$par,aux>value))
                if(aux>value)
                {
                        par<-optim$par
                        aux<-value
                        br<-br+1
                }
                if(aux<=value & br>10 & i>15)
                        break;
        }
        if(aux==1e10)
        par<-c(0,0)
        return(list(phi=par[1],sigma=par[2],ll=aux))
}
####Generating T-std IAR

IARt.sample<-function(n,phi,st,sigma2=1,nu=3)
{
        delta<-diff(st) #Times differences
	y <- vector("numeric", n)
        y[1] <- rnorm(1) #initialization
        for (i in 2:n)
        {
                phid=phi**(delta[i-1]) #Expected Value Conditional Distribution
                yhat = phid * y[i-1]  #Mean of conditional distribution
                gL=sigma2*(1-phid**(2)) #Variance Value Conditional Distribution
                y[i] <- rt(1, df=nu)*sqrt(gL * (nu-2)/nu) + yhat #Conditional-t IAR
        }
        return(list(y=y,st=st))
}
