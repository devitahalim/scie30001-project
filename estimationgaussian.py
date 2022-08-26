import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from gaussianfunctions import GaussianPDF

def GaussianEM(X,n_components:int,n_iteration:int):
    means=np.random.choice(X, n_components)
    variances=np.random.random_sample(size=n_components)
    weights=np.ones((n_components))/n_components
    X=np.array(X)

    epsilon=1e-9
    
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

def PlotGMM(X,means,variances,n_iteration,iter_plot:int):
    gmm_datapoints=np.linspace(np.min(X),np.max(X),100)
    for i in range(n_iteration):
        if i%iter_plot==0:
            #Set figure size, title, and plot the data points
            plt.figure(figsize=(8,5))
            plt.title=("Iteration {}".format(i))
            plt.scatter(X, [0.005] * len(X), color='mediumslateblue', s=15, marker="|", label="Data points")
            
            #Plot the estimated pdf
            plt.plot(gmm_datapoints, GaussianPDF(gmm_datapoints, means[i][0], variances[i][0]), color='red', label="Distribution 1")
            plt.plot(gmm_datapoints, GaussianPDF(gmm_datapoints, means[i][1], variances[i][1]), color='green', label="Distribution 2")
            
            #Set the x and y label
            plt.xlabel=("x")
            plt.ylabel=("Probability Density Function (PDF)")
            plt.legend(loc="upper left")

            plt.show()
