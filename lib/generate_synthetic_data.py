import math
from random import seed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

SEED = 1122334455
seed(SEED)
np.random.seed(SEED)

def generate_synthetic_data(plot_data=False):

    """
    Code for generating and plotting the synthetic data.
        
    We will generate a dataset containing 2000 points (1000 points from class 1 and 1000 points from class -1).
    For each point, we will have 2 non-sensitive features, 1 sensitive feature, and 1 class label (1/-1).
    The sensitive feature is correlated with the non-sensitive features.
        
    A sensitive feature value of 0.0 means the example is in protected group (e.g., female) 
    and 1.0 means it's in unprotected group (e.g., male).
        
    A data point is more likely to be in the unprotected group if a rotation of the non-sensitive features 
    is also more likely to be in the positive class label.
    
    PARAMETERS
    - plot_data: if True, plot the generated synthetic data
    
    RETURN
    - X_non_sensitive: 2000 by 2 array containing non-sensitive features
    - y: array of class labels
    - sensitive_feat_dict: dictionary of sensitive features for each data point in X_non_sensitive

    """
   
    # generate 2 non-sensitive features for each data point from Gaussian distributions
    X_non_sensitive, y, pos_dist, neg_dist = generate_non_sensitive_feat()
    
    # generate 1 sensitive feature that is correlated with the non-sensitive features
    sensitive_feat = generate_sensitive_feat(X_non_sensitive, pos_dist, neg_dist)
    sensitive_feat_array = np.array(sensitive_feat)
    
    if plot_data:
        plot_synthetic_data(X_non_sensitive, y, sensitive_feat_array)
          
    return X_non_sensitive, y, sensitive_feat

def generate_non_sensitive_feat():

    """ 
    Randomly generate 2 non-sensitive features values for each data point. 
    The features will be drawn from 2 multivariate Gaussian distributions (one for each class label).
    
    RETURN
    - X_non_sensitive: 2000 by 2 array containing non-sensitive features
    - y: array of class labels
    - pos_dist: distribution of non-sensitive features for the positive class
    - neg_dist: distribution of non-sensitive features for the negative class
    
    """
    def gen_gaussian(mean_in, cov_in, class_label):
        nv = multivariate_normal(mean = mean_in, cov = cov_in)
        X = nv.rvs(n_samples)
        y = np.ones(n_samples, dtype=float) * class_label
        return nv,X,y
    
    # generate 1000 data points per class label
    n_samples = 1000
    
    # draw non-sensitive features for class label 1 (positive class)
    mu1, sigma1 = [2, 2], [[5, 1], [1, 5]]
    pos_dist, X_non_sensitive_positive, y_positive = gen_gaussian(mu1, sigma1, 1)
    
    # draw non-sensitive features for class label -1 (negative class)
    mu2, sigma2 = [-2,-2], [[10, 1], [1, 3]]
    neg_dist, X_non_sensitive_negative, y_negative = gen_gaussian(mu2, sigma2, -1)

    # combine the positive and negative class
    X_non_sensitive = np.vstack((X_non_sensitive_positive, X_non_sensitive_negative))
    y = np.hstack((y_positive, y_negative))

    # randomly shuffle the data
    idx = np.arange(0, n_samples * 2)
    np.random.shuffle(idx)
    X_non_sensitive = X_non_sensitive[idx]
    y = y[idx]
    
    return X_non_sensitive, y, pos_dist, neg_dist

def generate_sensitive_feat(X_non_sensitive, pos_dist, neg_dist, rotation_factor=math.pi/8.0):
    
    """ 
    Generate 1 sensitive feature by rotating the non-sensitive features according to a rotation factor.
    The sensitive feature is more correlated with non-sensitive features when the rotation factor is close to 0.
    
    PARAMETERS
    - X_non_sensitive: 2000 by 2 array containing non-sensitive features
    - pos_dist: distribution of non-sensitive features for the positive class
    - neg_dist: distribution of non-sensitive features for the negative class
    - rotation_factor: parameter that determines rotation of sensitive feature; closer to 0 means the sensitive 
                       feature will be more correlated with the non-sensitive features
    RETURN
    - sensitive_feat: list of sensitive features for each data point in X_non_sensitive
    
    """
    
    # introducing bias by rotating the non-sensitive features
    rotation_matrix = np.array([[math.cos(rotation_factor), -math.sin(rotation_factor)], 
                              [math.sin(rotation_factor), math.cos(rotation_factor)]])
    X_rotated = np.dot(X_non_sensitive, rotation_matrix)


    # use X_rotated to generate a prob. distribution to draw the sensitive feature from
    sensitive_feat = []
    
    for i in range (0, len(X_rotated)):
        x = X_rotated[i]

        # chance that the point belongs to the distribution for the positive class label
        prob_positive = pos_dist.pdf(x)
        # chance that the point belongs to the distribution for the negative class label
        prob_negative = neg_dist.pdf(x)
        
        # normalize the probabilities from 0 to 1
        p = prob_positive / (prob_positive + prob_negative)
        
        # simulate drawing from a Bernoulli distribution where 
        # a data point is more likely to be in the unprotected group (sensitive feature = 1) 
        # if the rotated point is also more likely to be in the positive class label (y = 1). 
        
        r = np.random.uniform() # generate a random number from 0 to 1

        if r < p:
            sensitive_feat.append(1.0) # 1 means unprotected (e.g. male)
        else:
            sensitive_feat.append(0.0) # 0 means protected (e.g. female)
    
    return sensitive_feat
    

def plot_synthetic_data(X_non_sensitive, y, sensitive_feat_array):
    
    """ 
    Plot the synthetic data. 
    The protected class will be marked on the plot by "x" and the unprotected class by "o".
    The positive class will be colored green on the plot and the negative class will be colored red.
    
    PARAMETERS
    - X_non_sensitive: 2000 by 2 array containing non-sensitive features
    - y: array of class labels
    - sensitive_feat_array: array of sensitive features for each data point in X_non_sensitive
    
    """
    num_to_draw = 200 # only draw a small number of points to avoid clutter
    x_draw = X_non_sensitive[:num_to_draw]
    y_draw = y[:num_to_draw]
    sensitive_feat_draw = sensitive_feat_array[:num_to_draw]

    X_s_0 = x_draw[sensitive_feat_draw == 0.0]
    X_s_1 = x_draw[sensitive_feat_draw == 1.0]
    y_s_0 = y_draw[sensitive_feat_draw == 0.0]
    y_s_1 = y_draw[sensitive_feat_draw == 1.0]
    plt.figure(figsize=(8, 6))
    plt.scatter(X_s_0[y_s_0==1.0][:, 0], X_s_0[y_s_0==1.0][:, 1], color='green', marker='x', s=30, linewidth=1.5, label= "Protected (female), Hired")
    plt.scatter(X_s_0[y_s_0==-1.0][:, 0], X_s_0[y_s_0==-1.0][:, 1], color='red', marker='x', s=30, linewidth=1.5, label = "Protected (female), Not Hired")
    plt.scatter(X_s_1[y_s_1==1.0][:, 0], X_s_1[y_s_1==1.0][:, 1], color='green', marker='o', facecolors='none', s=30, label = "Unprotected (male), Hired")
    plt.scatter(X_s_1[y_s_1==-1.0][:, 0], X_s_1[y_s_1==-1.0][:, 1], color='red', marker='o', facecolors='none', s=30, label = "Unprotected (male), Not Hired")
    
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    plt.legend(loc=2, fontsize=15)
    plt.xlim((-15,10))
    plt.ylim((-10,15))
    plt.xlabel("Prior Income")
    plt.ylabel("Years of Work Experience")
    plt.show()
