import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import pandas as pd

def GaussianPDF(data, mean:float, var:float):

    pdf_gauss=(1/(np.sqrt(2*np.pi*abs(var))))*np.exp(-(np.square(data - mean)/(2*var)))
    return pdf_gauss

def SimulateGMM(n:int, mean1:float, sig1:float, mean2:float, sig2:float):

    X1=np.random.normal(mean1,np.sqrt(sig1),n)
    X2=np.random.normal(mean2,np.sqrt(sig2),n)
    X = np.array(list(X1) + list(X2))
    np.random.shuffle(X)

    return(X)

def PlotTrue(X,mean,sd):

    gmm_datapoints = np.linspace(np.min(X),np.max(X),100)

    plt.figure(figsize=(8,5))
    plt.scatter(X, [0.005] * len(X), color='mediumslateblue', s=15, marker="|", label="Data points")
    
    for i in range(len(mean)):
        plt.plot(gmm_datapoints, GaussianPDF(gmm_datapoints, mean[i], sd[i]), color='black', label="True pdf {}".format(i))
    plt.xlabel('$x$')
    plt.ylabel('pdf')
    plt.legend()
    plt.show()

def GaussianEM(X,n_components:int, initial_param):
    
    if initial_param==[]:
        def get_spaced_elm(X, n_components):
            spaced_elm = X[np.round(np.linspace(0, len(X)-1, n_components)).astype(int)]
            return spaced_elm
        
        initial_means=get_spaced_elm(np.sort(X),n_components)
        
        initial_param=list()
        for i in range(n_components):
            init_params={
            'Mean':initial_means[i],
            'Variance': 0.1,
            'Weight': 1/n_components
        }
            initial_param.append(init_params)

    epsilon=-1e-6 #to avoid singularities
    #stopping condition
    mean_delta=1e-6
    var_delta=1e-4
    weight_delta=1e-8

    max_iteration=50
    iteration=0

    iteration_param=list()

    while iteration<max_iteration:
        
        px_j=[]
        for j in range(n_components):
            px_j.append(GaussianPDF(X, initial_param[j]['Mean'], initial_param[j]['Variance']))
        px_j = np.array(px_j)
        
        posterior = []
        new_parameters=list()
        for j in range(n_components):
            #Calculate the probabilities of each data to belong from either gaussian  
            posterior.append((px_j[j] * initial_param[j]['Weight']) / (np.sum([px_j[i] * initial_param[i]['Weight'] for i in range(n_components)], axis=0)+epsilon))
        
        #Maximisation step (M-step):
            #Update the parameters
            new_param={
                'Mean':np.sum(posterior[j] * X) / (np.sum(posterior[j]+epsilon)),
                'Variance':np.sum(posterior[j] * np.square(X - initial_param[j]['Mean'])) / (np.sum(posterior[j]+epsilon)),
                'Weight':np.mean(posterior[j])
            }
            new_parameters.append(new_param)
        #Calculate difference between parameters
        mean_diff=list()
        for i in range (n_components):
            mean_diff_indiv=(abs(new_parameters[i]['Mean']-initial_param[i]['Mean'])/(abs(initial_param[i]['Mean']+epsilon)))
            mean_diff.append(mean_diff_indiv)

        var_diff=list()
        for i in range(n_components):
            var_diff_indiv=(abs(new_parameters[i]['Variance']-initial_param[i]['Variance'])/abs(initial_param[i]['Variance']))
            var_diff.append(var_diff_indiv)

        weight_diff=list()
        for i in range(n_components):
            weight_diff_indv=(abs(new_parameters[i]['Weight']-initial_param[i]['Weight'])/abs(initial_param[i]['Weight']))
            weight_diff.append(weight_diff_indv)
        
        def check_less_than(list, value): 
            for x in list: 
                if value <= x: 
                    return False
            return True

        if check_less_than(mean_diff,mean_delta) is True\
            and check_less_than(var_diff,var_delta) is True\
            and check_less_than(weight_diff, weight_delta) is True:
            break
        iteration+=1
        initial_param=new_parameters

        iteration_param.append(new_parameters)
    return(iteration_param)

def findThreshold(X,n_components,iteration_data):

    X_lastiter=iteration_data[-1]
    thresholds=[]

    for k in range(n_components-1):
        X_space=np.linspace(min(X),max(X),100)
        zscore_1=list()
        zscore_2=list()
        for i in range(len(X_space)):
            zscore_1.append((X_space[i]-X_lastiter[k]['Mean'])/(np.sqrt(X_lastiter[k]['Variance'])))
            zscore_2.append((X_space[i]-X_lastiter[k+1]['Mean'])/(np.sqrt(X_lastiter[k+1]['Variance'])))
            
            def area_left(zscore):
                return norm.cdf(zscore)
            def area_right(zscore):
                return 1-norm.cdf(zscore)

            area_gaussone = area_right(zscore_1)
            area_gausstwo = area_left(zscore_2)

            thres = X_space[np.argmin(area_gaussone+area_gausstwo)]
        thresholds.append(thres)
    return(thresholds)

def PlotGMM(X,iteration_data,plotper_iter:int,thresholds):
    c=['red','green','blue','magenta']
    gmm_datapoints=np.linspace(np.min(X),np.max(X),100)
    for i in range(len(iteration_data)):
        if i%plotper_iter==0 or i==len(iteration_data)-1:
            #Set figure size, title, and plot the data points
            plt.figure(figsize=(8,5))
            plt.title("Iteration {}".format(i))
            plt.scatter(X, [0.005] * len(X), color='mediumslateblue', s=15, marker="|", label="Data points")
            
            #Plot the estimated pdf
            for k in range(len(iteration_data[i])):
                plt.plot(gmm_datapoints, GaussianPDF(gmm_datapoints, iteration_data[i][k]['Mean'], iteration_data[i][k]['Variance']), color=c[k], label="Distribution {}".format(k))
            
            plt.ylim(0,15)
            #Set the x and y label
            plt.xlabel("x")
            plt.ylabel("Probability Density Function (PDF)")
            plt.legend(loc="upper left")
            
            if i==len(iteration_data)-1:
                for i in range (len(thresholds)):
                    plt.axvline(thresholds[i],c='red',ls='--',lw=0.5)

            plt.show()

def BIC_gmm(X):
    X=X.reshape(-1,1)
    n_components=np.arange(1, 11)

    gmm_models=[None for k in range(len(n_components))]
    for k in range(len(n_components)):
        gmm_models[k]=(GaussianMixture(n_components[k]).fit(X))

    BIC=[models.bic(X) for models in gmm_models]

    # plt.figure(figsize=(8,5))
    # plt.title("BIC")
    # plt.plot(n_components,BIC)

    # plt.xlabel("Number of Distribution")
    # plt.ylabel("BIC Score")

    # plt.show()
    return ((BIC.index(np.amin(BIC)))+1)


    