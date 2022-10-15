import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import pandas as pd

def GaussianPDF(data, mean:float, var:float):

    pdf_gauss=(1/(np.sqrt(2*np.pi*var)))*np.exp(-((np.square(data - mean))/(2*var)))
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
        
        def get_elm(X,n_components):
            b=np.linspace(min(X),max(X),n_components)
            return b
            
        initial_means=get_elm(X,n_components)
        
        initial_param=list()
        for i in range(n_components):
            init_params={
            'Mean':initial_means[i],
            'Variance': 0.1,
            'Weight': 1/n_components
        }
            initial_param.append(init_params)

    epsilon=-1e-200 #to avoid singularities
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
            mu=np.sum(posterior[j] * X) / (np.sum(posterior[j]+epsilon))
            new_param={
                'Mean':mu,
                'Variance':np.sum(posterior[j] * np.square(X - mu)) / (np.sum(posterior[j]+epsilon)),
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

def EMGMM_varconstraint(X,n_components:int, initial_param):
    
    if initial_param==[]:
        def get_spaced_elm(X, n_components):
            spaced_elm = X[np.round(np.linspace(0, len(X)-1, n_components)).astype(int)]
            return spaced_elm
        
        def get_elm(X,n_components):
            b=np.linspace(min(X),max(X),n_components)
            return b

        initial_means=get_elm(X,n_components)
        
        initial_param=list()
        for i in range(n_components):
            init_params={
            'Mean':initial_means[i],
            'Variance': 0.005,
            'Weight': 1/n_components
        }
            initial_param.append(init_params)

    epsilon=-1e-200 #to avoid singularities
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
            mu=np.sum(posterior[j] * X) / (np.sum(posterior[j]+epsilon))
            
            # Constraint the variance
            newvar_temp=np.sum(posterior[j] * np.square(X - mu)) / (np.sum(posterior[j]+epsilon))
            if newvar_temp<0.01:
                newvar=newvar_temp
            else:
                newvar=initial_param[j]['Variance']

            # New parameters
            new_param={
                'Mean':mu,
                'Variance':newvar,
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
        
        #Check if difference between iterations is less than the stopping criterion
        def check_less_than(list, value): 
            for x in list: 
                if value <= x: 
                    return False
            return True

        # Stop iterations if already satisfy the stopping condition
        if check_less_than(mean_diff,mean_delta) is True\
            and check_less_than(var_diff,var_delta) is True\
            and check_less_than(weight_diff, weight_delta) is True:
            break

        iteration+=1
        initial_param=new_parameters

        iteration_param.append(new_parameters)
    return(iteration_param,px_j)

def findThreshold1(X,n_components,iteration_data):

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

def findThreshold2(iteration_data):
    
    X_lastiter=iteration_data[-1]
    n_components=len(X_lastiter)

    thresholds=[]
    for i in range(n_components-1):
        a=(X_lastiter[i+1]['Variance']-X_lastiter[i]['Variance'])
        b=2*((X_lastiter[i]['Variance']*X_lastiter[i+1]['Mean'])-(X_lastiter[i+1]['Variance']*X_lastiter[i]['Mean']))
        c=(X_lastiter[i+1]['Variance']*(X_lastiter[i]['Mean']**2))-(X_lastiter[i]['Variance']*(X_lastiter[i+1]['Mean']**2))\
            -(2*X_lastiter[i]['Variance']*X_lastiter[i+1]['Variance']*np.log((X_lastiter[i]['Weight']*np.sqrt(X_lastiter[i+1]['Variance']))\
                /(X_lastiter[i+1]['Weight']*np.sqrt(X_lastiter[i]['Variance']))))

        dis = abs((b**2) - (4*a*c))

        thres1= (-b-np.sqrt(dis))/(2*a)
        thres2= (-b+np.sqrt(dis))/(2*a)

        if X_lastiter[i]['Mean']<thres1<X_lastiter[i+1]['Mean']:
            thresholds.append(thres1)
        elif X_lastiter[i]['Mean']<thres2<X_lastiter[i+1]['Mean']:
            thresholds.append(thres2)
        
        
    return(thresholds)

def PlotGMM(X,iteration_data,plotper_iter:int,thresholds,ylimit):
    c=['red','green','blue','magenta','darkorange','slategray']
    gmm_datapoints=np.linspace(np.min(X),np.max(X),100)
    for i in range(len(iteration_data)):
        if  i==len(iteration_data)-1:
            # i%plotper_iter==0
            #Set figure size, title, and plot the data points
            plt.figure(figsize=(8,5))
            plt.title("Iteration {}".format(i))
            plt.scatter(X, [0.005] * len(X), color='mediumslateblue', s=15, marker="|", label="Data points")
            plt.hist(X,bins=75,density=True)

            #Plot the estimated pdf
            for k in range(len(iteration_data[i])):
                plt.plot(gmm_datapoints, GaussianPDF(gmm_datapoints, iteration_data[i][k]['Mean'], iteration_data[i][k]['Variance']), color=c[k], label="Distribution {}".format(k))
            
            if ylimit==[]:
                pass
            else:
                plt.ylim(0,ylimit)
                
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
    n_components=np.arange(1, 6)

    gmm_models=[None for k in range(len(n_components))]
    for k in range(len(n_components)):
        gmm_models[k]=(GaussianMixture(n_components[k]).fit(X))

    BIC=[models.bic(X) for models in gmm_models]

    return ((BIC.index(np.amin(BIC)))+1)


def check_prob(pxj):
    
    prob_indv= [None for i in range(len(pxj[0]))]

    for i in range(len(pxj[0])):
        prob_indv[i]=[pxj[j][i] for j in range (len(pxj))]

    # Assess the probability
    
    low_prob_on=list()
    for i in range (len(pxj[0])):
        if any(j>0.01 for j in prob_indv[i])==False:
            low_prob_on.append(i)
    return (len(low_prob_on))

# This one consider distance from mean instead of fixed likelihood value
def check_prob2(pxj,iteration_data):
    
    last_iter=iteration_data[-1]
    # Rearrange array into individual likelihoof of each data points
    prob_indv= [None for i in range(len(pxj[0]))]
    for i in range(len(pxj[0])):
        prob_indv[i]=[pxj[j][i] for j in range (len(pxj))]

    # Calculate the minimum likelihood for each gaussian (99.7%)
    min_likelihood=list()
    for j in range(len(last_iter)):
        indv_likelihood=GaussianPDF((last_iter[j]['Mean']+(2.967738*np.sqrt(last_iter[j]['Variance']))),last_iter[j]['Mean'],last_iter[j]['Variance'])
        min_likelihood.append(indv_likelihood)
    
    # Create list of elements that have high likelihood of belonging to any gaussian
    highprob=list()
    for i in range (len(pxj[0])):
        indv_high_prob=[l1 for l1,l2 in zip(prob_indv[i],min_likelihood) if l1>l2]
        highprob.append(indv_high_prob)
    
    # Identify the one with no adequate likelihood of belonging to any gaussian
    n_lowprob=list()
    for i in range(len(highprob)):
        if highprob[i]==[]:
            n_lowprob.append(i)

    return (len(n_lowprob))

def check_mean_dis(iteration_data):
    # Create list of means
    means_list=list()
    for i in range (len(iteration_data[-1])):
        means_list.append(iteration_data[-1][i]['Mean'])
    means_list.sort()
    
    # Compute the difference between means of adjacent Gaussians
    means_diff=list()
    for i in range (len(means_list)-1):
        diff= np.subtract(means_list[i+1],means_list[i])
        means_diff.append(diff)
    
    # Check if distance between means is less than 0.4 or not, if yes, return False.
    def all_meansdiff(means_diff):
        for i in means_diff:
            if i <0.2:
                return False
        return True
    return (all_meansdiff(means_diff))