import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from gaussianfunctions import GaussianPDF

def GaussianEM_for(X,n_components:int,n_iteration:int):
    means=np.random.choice(X, n_components)
    variances=np.random.random_sample(size=n_components)
    weights=np.ones((n_components))/n_components
    X=np.array(X)

    epsilon=1e-8
    
    means_all=[]
    var_all=[]

    for iteration in range(n_iteration):
        
        #Maximum likelihood of each x_i
        likelh = []

        #Estimation Step (E-step)
        for j in range(n_components):
            likelh.append(GaussianPDF(X, means[j], np.sqrt(variances[j])))
        likelh = np.array(likelh)

        #M-step
        l = []
        for j in range(n_components):
            #Calculate the probabilities of each data to belong from either gaussian  
            l.append((likelh[j] * weights[j]) / (np.sum([likelh[i] * weights[i] for i in range(n_components)], axis=0)+epsilon))
  
            #Update the parameters
            means_all.append(np.sum(l[j] * X) / (np.sum(l[j]+epsilon)))
            var_all.append(np.sum(l[j] * np.square(X - means[j])) / (np.sum(l[j]+epsilon)))
            weights[j] = np.mean(l[j])
    
    #Make an array of all means and variance
    means_all=np.array(means_all)
    var_all=np.array(var_all)

    #Seperate the means and variances of each component and then combine them again into n_component
    #number of column
    means_one=means_all[::2]
    means_two=means_all[1::2]
    means_total=np.column_stack((means_one,means_two))

    var_one=var_all[::2]
    var_two=var_all[1::2]
    var_total=np.column_stack((var_one, var_two))
    
    return means_total,var_total,n_iteration

#try using while loop
def GaussianEM_while(X,n_components:int):
    
    means=np.random.choice(X, n_components)
    variances=np.random.random_sample(size=n_components)
    weights=np.ones((n_components))/n_components
    X=np.array(X)

    epsilon=1e-8 #to avoid singularities
    #stopping condition
    mean_delta=1e-4 
    var_delta=1e-2
    weight_delta=1e-8

    max_iteration=10000
    iteration=0
    means_all=[]
    var_all=[]
    weights_all=[]

    while iteration<max_iteration:
        px_j=[]
        for j in range(n_components):
            px_j.append(GaussianPDF(X, means[j], np.sqrt(variances[j])))
        px_j = np.array(px_j)
        
        posterior = []
        means_new=[]
        variances_new=[]
        weights_new=[]
        for j in range(n_components):
            #Calculate the probabilities of each data to belong from either gaussian  
            posterior.append((px_j[j] * weights[j]) / (np.sum([px_j[i] * weights[i] for i in range(n_components)], axis=0)+epsilon))
        
        #Maximisation step (M-step):
            #Update the parameters
            means_new.append(np.sum(posterior[j] * X) / (np.sum(posterior[j]+epsilon)))
            means_all.append(means_new)
            
            variances_new.append(np.sum(posterior[j] * np.square(X - means[j])) / (np.sum(posterior[j]+epsilon)))
            var_all.append(variances_new)
            
            weights_new.append(np.mean(posterior[j]))
            weights_all.append(weights_new)
        if np.all(np.abs(np.subtract((means_new),(means))))<mean_delta\
            or np.all(np.abs(np.subtract((variances_new),(variances))))<var_delta\
            or np.all(np.abs(np.subtract((weights_new),(weights))))<weight_delta :
            break
        iteration+=1
        means=means_new
        variances=variances_new
        weights=weights_new
    
    #Make an array of all means and variance
    means_all=np.array(means_all)
    var_all=np.array(var_all)

    #Seperate the means and variances of each component and then combine them again into n_component
    #number of column
    means_one=means_all[::2]
    means_two=means_all[1::2]
    means_total=np.column_stack((means_one,means_two))

    var_one=var_all[::2]
    var_two=var_all[1::2]
    var_total=np.column_stack((var_one, var_two))
    
    return means_total,var_total, iteration
