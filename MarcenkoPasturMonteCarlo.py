import numpy as np, pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity

# function to generate the Marcenk-Pastur distribution for a given variance and dimension
def mpPDF(var,q,pts):
    # q=T/N for a TxN array of observations
    eMin,eMax=var*(1-(1./q)**.5)**2,var*(1+(1./q)**.5)**2   # min and max expected eigenvals
    eVal=np.linspace(eMin,eMax,pts)     # x-axis (eigenvals)
    pdf=q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5      # pdf as array
    pdf=pd.Series(pdf,index=eVal)   # pdf as pandas series
    return pdf

# function to get eigenvals and eigenvectors from a Hermitian or real symmetric matrix
def getPCA(matrix):
    eVal,eVec=np.linalg.eigh(matrix)
    indices=eVal.argsort()[::-1]    # arguments for sorting eVal desc
    eVal,eVec=eVal[indices],eVec[:,indices]
    eVal=np.diagflat(eVal)
    return eVal,eVec

# function to fit a kernel density estimator to observed data
# x is the array of values on which the fit KDE will be evaluated
def fitKDE(obs,bWidth=0.3,kernel="gaussian",x=None):
    if len(obs.shape)==1:
        obs=obs.reshape(-1,1)
    kde=KernelDensity(kernel=kernel,bandwidth=bWidth).fit(obs)
    if x is None:
        x=np.unique(obs).reshape(-1,1)
    if len(x.shape)==1:
        x=x.reshape(-1,1)
    logProb=kde.score_samples(x) # log(density)
    pdf=pd.Series(np.exp(logProb),index=x.flatten())
    return pdf

# generate a random covariance matrix
T = 50000
N = 1000
x=np.random.normal(size=(T,N))
eVal0,eVec0=getPCA(np.corrcoef(x,rowvar=False))
pdf0=mpPDF(1.,q=x.shape[0]/float(x.shape[1]),pts=1000)
pdf1=fitKDE(np.diag(eVal0),bWidth=.01) # empirical pdf

# plots
plt.figure()
plt.plot(pdf0,label="Theoretical distribution with random x")
plt.plot(pdf1,label="Kernel estimate of observed random data")
plt.legend()
plt.show()
