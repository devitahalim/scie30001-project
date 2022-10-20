import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd
import functions.gaussianfunctions as gauss
import csv

B=pd.read_csv('/Users/devitahalim/SCIE30001_Project/GMM/locusRatioFrame.csv')
a=list(B)
KIR=a[1:]
KIR.remove("KIR3DL3")
sample_n=B.iloc[:,0].to_numpy()

threshold_all=list()
data_lowprob_all=list()
for i in KIR:
    X=B["{}".format(i)].to_numpy()
    
    def main(X):

        n_components=gauss.BIC_gmm(X)
        low_prob_proportion=len(X)
        while n_components<6:
            em,pxj=gauss.EMGMM_varconstraint(X,n_components,[])
            low_prob_proportion,data_lowprob=gauss.check_prob2(pxj,em)
            if low_prob_proportion>=(0.03*len(X)):
                n_components=n_components+1
            elif gauss.check_mean_dis(em)==False:
                n_components=n_components-1
            else:
                break
        return(em,pxj)
    
    em,pxj=main(X)
    
    # Thresholds
    thresholds=gauss.findThreshold2(em)

    mean_ghost=[]
    var_ghost=[]
    # Check threshold of 0 copies
    if gauss.check_gaps_below(em)==(False,0):
        mean_ghost.append(1e-4)
        var_ghost.append(1e-6)
        thresholds.append(0.07277145626097836)
    elif gauss.check_gaps_below(em)==(False,1):
        last_iter=em[-1]
        mean_ghost.append(1e-4)
        var_ghost.append(1e-6)
        thresholds.append(0.07277145626097836)
        
        new_thres=last_iter[0]['Mean']-(3.89*np.sqrt(last_iter[0]['Variance']))
        mean_ghost.append((min(thresholds)+new_thres)/2)
        var_ghost.append((last_iter[0]['Variance']))
        thresholds.append(new_thres)   

    # Check if there is "double gap", add threshold if yes.
    if gauss.check_gaps_means(em)[0]==True:
        gauss_index=gauss.check_gaps_means(em)[1]
        last_iter=em[-1]
        extra_threshold_1=last_iter[gauss_index]['Mean']+(3.89*np.sqrt(last_iter[gauss_index]['Variance']))
        dis_thres=extra_threshold_1-thresholds[-1]
        extra_threshold_2=last_iter[gauss_index+1]['Mean']-(3.89*np.sqrt(last_iter[gauss_index+1]['Variance']))
        thresholds=thresholds[:-1]

        mean_ghost.append((extra_threshold_1+extra_threshold_2)/2)
        var_ghost.append(((extra_threshold_2-extra_threshold_1)/6.58)**2)

        thresholds.append(extra_threshold_1)
        thresholds.append(extra_threshold_2)
    thresholds.sort()

    threshold_all.append(thresholds)

    # Sample numbers with low probability
    prop_lowprob,data_lowprob=gauss.check_prob2(pxj,em)
    lowprob_sample_number=list()
    for k in data_lowprob:
        lowprob_sample_number.append(sample_n[k])
        
    data_lowprob_all.append(lowprob_sample_number)
    
    #Output the figures
    fig=gauss.PlotGMM(X,em,50,thresholds,i,10,mean_ghost,var_ghost)
    output_path= "/Users/devitahalim/Documents/GitHub/scie30001-project/output/plots/plot_"
    fig.savefig(output_path+(i))

# CSV for thresholds
for i in range (len(KIR)):
    threshold_all[i].insert(0,KIR[i])

# CSV for thresholds
threshold_header=["","0-1","1-2","2-3","3-4","4-5","5-6"]
threshold_data=threshold_all
with open('/Users/devitahalim/Documents/GitHub/scie30001-project/output/output_threshold.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(threshold_header)
    writer.writerows(threshold_data)

# CSV for low probability
lowprob_header=["","Sample number with low likelihood"]
lowprob_data=list()
for i in range(len(KIR)):
    k=KIR[i],data_lowprob_all[i]
    lowprob_data.append(k)

with open('/Users/devitahalim/Documents/GitHub/scie30001-project/output/output_outliers.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(lowprob_header)
    writer.writerows(lowprob_data)