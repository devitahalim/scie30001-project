import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

def GaussianPDF(data, mean:float, var:float):
    pdf_gauss=(1/(np.sqrt(2*np.pi*var)))*np.exp(-(np.square(data - mean)/(2*var)))
    return pdf_gauss

def SimulateGMM(n:int, mean1:float, sig1:float, mean2:float, sig2:float):
    X1=np.random.normal(mean1,np.sqrt(sig1),n)
    X2=np.random.normal(mean2,np.sqrt(sig2),n)
    X = np.array(list(X1) + list(X2))
    np.random.shuffle(X)
    
    #Plot the Gaussian and the data points
    gmm_datapoints=np.linspace(np.min(X),np.max(X),100)
    plt.figure(figsize=(7,4))

    plt.scatter(X, [0.005] * len(X), color='mediumslateblue', s=15, marker="|", label="Data points")
    plt.plot(gmm_datapoints, GaussianPDF(gmm_datapoints, mean1, sig1), color='black', label="True pdf")
    plt.plot(gmm_datapoints, GaussianPDF(gmm_datapoints, mean2, sig2), color='black')
    
    #Set the x and y label
    plt.xlabel=("x")
    plt.ylabel=("Probability Density Function (PDF)")
    plt.legend()

    plt.show()
    return(X)

def GaussianEMPlot(X,k:int,n_iteration:int, iter_plot:int):
    means=np.random.choice(X, k)
    variances=np.random.random_sample(size=k)
    weights=np.ones((k))/k
    X=np.array(X)

    epsilon=1e-9
    gmm_datapoints=np.linspace(np.min(X),np.max(X),100)
    

    for iteration in range(n_iteration):
        if iteration%iter_plot==0:
            plt.figure(figsize=(7,4))
            plt.title("Iteration {}".format(iteration))
            plt.scatter(X, [0.005] * len(X), color='mediumslateblue', s=15, marker="|", label="Data points")
                
            #Plot the estimated pdf
            plt.plot(gmm_datapoints, GaussianPDF(gmm_datapoints, means[0], variances[0]), color='red', label="Distribution 1")
            plt.plot(gmm_datapoints, GaussianPDF(gmm_datapoints, means[1], variances[1]), color='green', label="Distribution 2")
                
            #Set the x and y label
            plt.xlabel=("x")
            plt.ylabel=("Probability Density Function (PDF)")
            plt.legend()

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
            means[j]=np.sum(l[j] * X) / (np.sum(l[j]+epsilon))
            variances[j]=np.sum(l[j] * np.square(X - means[j])) / (np.sum(l[j]+epsilon))
            weights[j] = np.mean(l[j])