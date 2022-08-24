from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from PdfGaussian import GaussianPDF
from SimulationOfGMM import SimulateGMM

def GaussianEM(X,k:int,n_iteration:int):
    means=np.random.choice(X, k)
    variances=np.random.random_sample(size=k)
    weights=np.ones((k))/k
    X=np.array(X)

    epsilon=1e-9
    means_all=[]
    var_all=[]

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
            means_all.append(np.sum(l[j] * X) / (np.sum(l[j]+epsilon)))
            var_all.append(np.sum(l[j] * np.square(X - means[j])) / (np.sum(l[j]+epsilon)))
            weights[j] = np.mean(l[j])
    
    means_all=np.array(means_all)
    var_all=np.array(var_all)

    means_one=means_all[::2]
    means_two=means_all[1::2]

    means_total=np.column_stack((means_one,means_two))

    var_one=var_all[::2]
    var_two=var_all[1::2]

    var_total=np.column_stack((var_one, var_two))
    
    return[means_total,var_total,n_iteration]

GaussianEM(SimulateGMM(100,10,4,-5,3),2,15)