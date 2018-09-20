import numpy as np

t = np.arange(1,5)
ys = np.array([12 * t**2.5,14 * t **3.12]).T
def fit(x,a,m): # power-law fit (based on previous studies)
    return a*(x**m)
a1,a2= np.linalg.lstsq(np.vstack([np.ones(len(ys)), np.log(t)]).T, np.log(ys))[0] # calculating fitting coefficients (a,m)
y_predict = fit(np.repeat(np.arange(1,5)[:,None],2,axis=1),np.exp(a1),a2) # prediction based of fitted model
print y_predict - ys