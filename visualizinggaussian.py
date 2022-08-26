from statistics import variance
import matplotlib.pyplot as plt
import numpy as np
from pdfgaussian import GaussianPDF
from SimulationOfGMM import SimulateGMM
from estimationgaussian import GaussianEM

def PlotGMM(X,means,variances,n_iteration,plot:int):
    gmm_datapoints=np.linspace(np.min(X),np.max(X),100)
    for iteration in range(n_iteration):
        if iteration%plot==0:
            plt.figure(figsize=(8,5))
            plt.scatter(X, [0.005] * len(X), color='mediumslateblue', s=15, marker="|", label="Data points")
            
            #Plot the estimated pdf
            plt.plot(gmm_datapoints, GaussianPDF(gmm_datapoints, means[0], variances[0]), color='red', label="Distribution 1")
            plt.plot(gmm_datapoints, GaussianPDF(gmm_datapoints, means[1], variances[1]), color='green', label="Distribution 2")
            
            #Set the x and y label
            plt.xlabel=("x")
            plt.ylabel=("Probability Density Function (PDF)")
            plt.legend(loc="upper left")

            plt.show()