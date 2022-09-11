from statistics import variance
import matplotlib.pyplot as plt
import numpy as np
from gaussianfunctions import GaussianPDF

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