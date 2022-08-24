import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from PdfGaussian import GaussianPDF

def GaussianEM(X,k:int,n_iteration:int):
    means=np.random.choice(X, k)
    variances=np.random.random_sample(size=k)
    weights=np.ones((k))/k

    epsilon=1e-9

    for iteration in range(n_iteration):
        
        #Maximum likelihood of each x_i
        likelh = []

        #Estimation Step (E-step)
        for j in range(k):
            likelh.append(GaussianPDF(X, means[j], np.sqrt(variances[j])))
        likelh = np.array(likelh)

        #M-step
        l = []
        for j in range(k):
            #Calculate the probabilities of each data to belong from either gaussian  
            l.append((likelh[j] * weights[j]) / (np.sum([likelh[i] * weights[i] for i in range(k)], axis=0)+epsilon))
  
            #Update the parameters
            means[j] = np.sum(l[j] * X) / (np.sum(l[j]+epsilon))
            variances[j] = np.sum(l[j] * np.square(X - means[j])) / (np.sum(l[j]+epsilon))
            weights[j] = np.mean(l[j])
        
        return(means,variances)

