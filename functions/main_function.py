import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd
import gaussianfunctions as gauss

data=pd.read_csv('/Users/devitahalim/SCIE30001_Project/GMM/locusRatioFrame.csv')


def main(X):

    n_components=gauss.BIC_gmm(X)
    X.sort()
    low_prob_proportion=len(X)
    while n_components<6:
        em,pxj=gauss.EMGMM_varconstraint(X,n_components,[])
        low_prob_proportion=gauss.check_prob2(pxj,em)
        if low_prob_proportion>=(0.05*len(X)):
            n_components=n_components+1
        elif gauss.check_mean_dis(em)==False:
            n_components=n_components-1
        else:
            break
    return(em)