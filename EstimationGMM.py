import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from PdfGaussian import GaussianPDF

def GaussianEM(X,k,n_iteration):
    means=np.random.choice(X, k)
    variances=np.random.random_sample(size=k)
    weights=np.ones((k))/k

    epsilon=1e-9
    gmm_datapoints=np.linspace(np.min(X),np.max(X),100)

    for iteration in range(n_iteration):
        if iteration%5==0:
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
