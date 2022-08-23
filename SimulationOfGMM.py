import matplotlib.pyplot as plt
from PdfGaussian import GaussianPDF
import numpy as np
from scipy.stats import multivariate_normal



def SimulateGMM(n, mean1, sig1, mean2, sig2):
    X1=np.random.normal(mean1,np.sqrt(sig1),n)
    X2=np.random.normal(mean2,np.sqrt(sig2),n)
    X = np.array(list(X1) + list(X2))
    np.random.shuffle(X)
    
    #Plot the Gaussian and the data points
    gmm_datapoints=np.linspace(np.min(X),np.max(X),100)
    plt.figure(figsize=(8,5))

    plt.scatter(X, [0.005] * len(X), color='mediumslateblue', s=15, marker="|", label="Data points")
    plt.plot(gmm_datapoints, GaussianPDF(gmm_datapoints, mean1, sig1), color='black', label="True pdf")
    plt.plot(gmm_datapoints, GaussianPDF(gmm_datapoints, mean2, sig2), color='black')
    
    #Set the x and y label
    plt.xlabel=("x")
    plt.ylabel=("Probability Density Function (PDF)")
    plt.legend()

    plt.show()
    return(X)
