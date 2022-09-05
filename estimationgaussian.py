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

def GaussianEM2(X, n_components:int):

    initial_param=list()
    for i in range(n_components):
        init_params={
            'Mean':np.random.choice(X),
            'Variance': 0.1,
            'Weight': (1/n_components)
        }
        initial_param.append(init_params)

    epsilon=1e-8 #to avoid singularities
    #stopping condition
    mean_delta=1e-4 
    var_delta=1e-2
    weight_delta=1e-8

    max_iteration=50
    iteration=0

    iteration_param=list()

    while iteration<max_iteration:
        px_j=[]
        for j in range(n_components):
            px_j.append(GaussianPDF(X, initial_param[j]['Mean'], np.sqrt(initial_param[j]['Variance'])))
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

        if np.all(np.abs(np.subtract((new_param['Mean']),(init_params['Mean']))))<mean_delta\
            or np.all(np.abs(np.subtract((new_param['Variance']),(init_params['Variance']))))<var_delta\
            or np.all(np.abs(np.subtract((new_param['Weight']),(init_params['Variance']))))<weight_delta :
            break
        iteration+=1
        initial_param=new_parameters

        iteration_param.append(new_parameters)
    return(iteration_param)

def Plot_GMM2(X, iteration_data, plotper_iter):
    c=['red','green','blue']
    gmm_datapoints=np.linspace(np.min(X),np.max(X),100)
    for i in range(len(iteration_data)):
        if i%plotper_iter==0:
            #Set figure size, title, and plot the data points
            plt.figure(figsize=(8,5))
            plt.title("Iteration {}".format(i))
            plt.scatter(X, [0.005] * len(X), color='mediumslateblue', s=15, marker="|", label="Data points")
            
            #Plot the estimated pdf
            for k in range(len(iteration_data[i])):
                plt.plot(gmm_datapoints, GaussianPDF(gmm_datapoints, iteration_data[i][k]['Mean'], iteration_data[i][k]['Variance']), color=c[k], label="Distribution {}".format(k))
            
            #Set the x and y label
            plt.xlabel("x")
            plt.ylabel("Probability Density Function (PDF)")
            plt.legend(loc="upper left")

            plt.show()

def GaussianEM1(X,n_components:int):
    means=np.random.choice(X,n_components)
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
    
    means_total=np.column_stack(means_all)
    var_total=np.column_stack(var_all)
    
    return means_total,var_total, iteration, n_components

def PlotGMM1(X,means,variances,n_iteration:int, n_components:int,iter_plot:int):
    c=['red','green','blue','magenta']
    gmm_datapoints=np.linspace(np.min(X),np.max(X),100)
    for i in range(n_iteration):
        if i%iter_plot==0:
            #Set figure size, title, and plot the data points
            plt.figure(figsize=(8,5))
            plt.title("Iteration {}".format(i))
            plt.scatter(X, [0.005] * len(X), color='mediumslateblue', s=15, marker="|", label="Data points")
            
            #Plot the estimated pdf
            for k in range(n_components):
                plt.plot(gmm_datapoints, GaussianPDF(gmm_datapoints, means[k][i], variances[k][i]), color=c[k], label="Distribution {}".format(k))
            
            #Set the x and y label
            plt.xlabel("x")
            plt.ylabel("Probability Density Function (PDF)")
            plt.legend(loc="upper left")

            plt.show()