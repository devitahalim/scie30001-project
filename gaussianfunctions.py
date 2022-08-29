import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

def GaussianPDF(data, mean:float, var:float):
    pdf_gauss=(1/(np.sqrt(2*np.pi*var)))*np.exp(-(np.square(data - mean)/(2*var)))
    return pdf_gauss

def SimulateGMM(n:int, mean1:float, sig1:float, mean2:float, sig2:float):
    X1=np.random.normal(mean1,np.sqrt(sig1),n)
    X2=np.random.normal(mean2,np.sqrt(sig2),n)
    X = np.array(list(X1) + list(X2))
    np.random.shuffle(X)
    return(X)

def GaussianEM(X,n_components:int):
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
    
    return means_total,var_total, iteration, n_components

def PlotGMM(X,means,variances,n_iteration:int, n_components:int,iter_plot:int):
    c=['red','green','blue']
    gmm_datapoints=np.linspace(np.min(X),np.max(X),100)
    for i in range(n_iteration):
        if i%iter_plot==0:
            #Set figure size, title, and plot the data points
            plt.figure(figsize=(8,5))
            plt.title("Iteration {}".format(i))
            plt.scatter(X, [0.005] * len(X), color='mediumslateblue', s=15, marker="|", label="Data points")
            
            #Plot the estimated pdf
            for k in range(n_components):
                plt.plot(gmm_datapoints, GaussianPDF(gmm_datapoints, means[i][k], variances[i][k]), color=c[k], label="Distribution {}".format(k))
            
            #Set the x and y label
            plt.xlabel=("x")
            plt.ylabel=("Probability Density Function (PDF)")
            plt.legend(loc="upper left")

            plt.show()

def BIC_gmm(X):
    X.reshape(-1,1)
    n_components=np.arange(1, 11)

    gmm_models=[None for k in range(len(n_components))]
    for k in range(len(n_components)):
        gmm_models[k]=(GaussianMixture(n_components[k]).fit(X))

    for models in gmm_models:
        BIC=[models.bic(X)]
    
    plt.figure(figsize=(8,5))
    plt.title("The Gradient of BIC")
    plt.plot(n_components,BIC)
    plt.xlabel=("Number of Distribution")
    plt.ylabel=("Gradient of BIC")
    plt.legend()
    plt.show()