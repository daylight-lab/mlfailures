import numpy as np
from random import seed
from scipy.optimize import minimize # for loss func minimization
from multiprocessing import Pool, Process, Queue
from collections import defaultdict
from copy import deepcopy
import matplotlib.pyplot as plt # for plotting stuff
import sys

SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)


def train_model(x, y, x_control, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma=None):

    """

    Function that trains the model subject to various fairness constraints.
    If no constraints are given, then simply trains an unaltered classifier.
    Example usage in: "synthetic_data_demo/decision_boundary_demo.py"

    ----

    Inputs:

    X: (n) x (d+1) numpy array -- n = number of examples, d = number of features, one feature is the intercept
    y: 1-d numpy array (n entries)
    x_control: dictionary of the type {"s": [...]}, key "s" is the sensitive feature name, and the value is a 1-d list with n elements holding the sensitive feature values
    loss_function: the loss function that we want to optimize -- for now we have implementation of logistic loss, but other functions like hinge loss can also be added
    apply_fairness_constraints: optimize accuracy subject to fairness constraint (0/1 values)
    apply_accuracy_constraint: optimize fairness subject to accuracy constraint (0/1 values)
    sep_constraint: apply the fine grained accuracy constraint
        for details, see Section 3.3 of arxiv.org/abs/1507.05259v3
        For examples on how to apply these constraints, see "synthetic_data_demo/decision_boundary_demo.py"
    Note: both apply_fairness_constraints and apply_accuracy_constraint cannot be 1 at the same time
    sensitive_attrs: ["s1", "s2", ...], list of sensitive features for which to apply fairness constraint, all of these sensitive features should have a corresponding array in x_control
    sensitive_attrs_to_cov_thresh: the covariance threshold that the classifier should achieve (this is only needed when apply_fairness_constraints=1, not needed for the other two constraints)
    gamma: controls the loss in accuracy we are willing to incur when using apply_accuracy_constraint and sep_constraint

    ----

    Outputs:

    w: the learned weight vector for the classifier

    """

    assert((apply_accuracy_constraint == 1 and apply_fairness_constraints == 1) == False) # both constraints cannot be applied at the same time

    max_iter = 100000 # maximum number of iterations for the minimization algorithm

    if apply_fairness_constraints == 0:
        constraints = []
    else:
        constraints = get_constraint_list_cov(x, y, x_control, sensitive_attrs, sensitive_attrs_to_cov_thresh)      

    if apply_accuracy_constraint == 0: #its not the reverse problem, just train w with cross cov constraints

        f_args=(x, y)
        w = minimize(fun = loss_function,
            x0 = np.random.rand(x.shape[1],),
            args = f_args,
            method = 'SLSQP',
            options = {"maxiter":max_iter},
            constraints = constraints
            )

    else:

        # train on just the loss function
        w = minimize(fun = loss_function,
            x0 = np.random.rand(x.shape[1],),
            args = (x, y),
            method = 'SLSQP',
            options = {"maxiter":max_iter},
            constraints = []
            )

        old_w = deepcopy(w.x)
        

        def constraint_gamma_all(w, x, y,  initial_loss_arr):
            gamma_arr = np.ones_like(y) * gamma # set gamma for everyone
            new_loss = loss_function(w, x, y)
            old_loss = sum(initial_loss_arr)
            return ((1.0 + gamma) * old_loss) - new_loss

        def constraint_protected_people(w,x,y): # dont confuse the protected here with the sensitive feature protected/non-protected values -- protected here means that these points should not be misclassified to negative class
            return np.dot(w, x.T) # if this is positive, the constraint is satisfied
        
        def constraint_unprotected_people(w,ind,old_loss,x,y):
            new_loss = loss_function(w, np.array([x]), np.array(y))
            return ((1.0 + gamma) * old_loss) - new_loss

        constraints = []
        predicted_labels = np.sign(np.dot(w.x, x.T))
        unconstrained_loss_arr = loss_function(w.x, x, y, return_arr=True)

        if sep_constraint == True: # separate gemma for different people
            for i in range(0, len(predicted_labels)):
                if predicted_labels[i] == 1.0 and x_control[sensitive_attrs[0]][i] == 1.0: # for now we are assuming just one sensitive attr for reverse constraint, later, extend the code to take into account multiple sensitive attrs
                    c = ({'type': 'ineq', 'fun': constraint_protected_people, 'args':(x[i], y[i])}) # this constraint makes sure that these people stay in the positive class even in the modified classifier             
                    constraints.append(c)
                else:
                    c = ({'type': 'ineq', 'fun': constraint_unprotected_people, 'args':(i, unconstrained_loss_arr[i], x[i], y[i])})                
                    constraints.append(c)
        else: # same gamma for everyone
            c = ({'type': 'ineq', 'fun': constraint_gamma_all, 'args':(x,y,unconstrained_loss_arr)})
            constraints.append(c)

        def cross_cov_abs_optm_func(weight_vec, x_in, x_control_in_arr):
            cross_cov = (x_control_in_arr - np.mean(x_control_in_arr)) * np.dot(weight_vec, x_in.T)
            return float(abs(sum(cross_cov))) / float(x_in.shape[0])

        w = minimize(fun = cross_cov_abs_optm_func,
            x0 = old_w,
            args = (x, x_control[sensitive_attrs[0]]),
            method = 'SLSQP',
            options = {"maxiter":100000},
            constraints = constraints
            )

    try:
        assert(w.success == True)
    except:
        print("Optimization problem did not converge.. Check the solution returned by the optimizer.")
        print("Returned solution is:")
        print(w)

    return w.x

def log_logistic(X):
    """ This function is used from scikit-learn source code. Source link below """

    """Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.
    This implementation is numerically stable because it splits positive and
    negative values::
        -log(1 + exp(-x_i))     if x_i > 0
        x_i - log(1 + exp(x_i)) if x_i <= 0

    Parameters
    ----------
    X: array-like, shape (M, N)
        Argument to the logistic function

    Returns
    -------
    out: array, shape (M, N)
        Log of the logistic function evaluated at every point in x
    Notes
    -----
    Source code at:
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py
    -----

    See the blog post describing this implementation:
    http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
    """
    if X.ndim > 1: raise Exception("Array of samples cannot be more than 1-D!")
    out = np.empty_like(X) # same dimensions and data types

    idx = X>0
    out[idx] = -np.log(1.0 + np.exp(-X[idx]))
    out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
    return out

def _logistic_loss(w, X, y, return_arr=None):
    """Computes the logistic loss.

    This function is used from scikit-learn source code

    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape (n_samples,)
        Array of labels.

    """

    yz = y * np.dot(X,w)
    # Logistic loss is the negative of the log of the logistic function.
    if return_arr == True:
        out = -(log_logistic(yz))
    else:
        out = -np.sum(log_logistic(yz))
    return out


def print_classifier_fairness_stats(acc_arr, correlation_dict_arr, cov_dict_arr, s_attr_name):
    
    correlation_dict = get_avg_correlation_dict(correlation_dict_arr)
    non_prot_pos = correlation_dict[s_attr_name][1][1]
    prot_pos = correlation_dict[s_attr_name][0][1]
    p_rule = (prot_pos / non_prot_pos) * 100.0
    
    print("Accuracy: %0.2f" % (np.mean(acc_arr)))
    print("Protected/non-protected in +ve class: %0.0f%% / %0.0f%%" % (prot_pos, non_prot_pos))
    print("P-rule achieved: %0.0f%%" % (p_rule))
    print("Covariance between sensitive feature and decision from distance boundary : %0.3f" % (np.mean([v[s_attr_name] for v in cov_dict_arr])))
    print()
    return p_rule


def compute_p_rule(x_control, class_labels):

    """ Compute the p-rule based on doctrine of disparate impact """

    non_prot_all = sum(x_control == 1.0) # non-protected group
    prot_all = sum(x_control == 0.0) # protected group
    non_prot_pos = sum(class_labels[x_control == 1.0] == 1.0) # non_protected in positive class
    prot_pos = sum(class_labels[x_control == 0.0] == 1.0) # protected in positive class
    frac_non_prot_pos = float(non_prot_pos) / float(non_prot_all)
    frac_prot_pos = float(prot_pos) / float(prot_all)
    p_rule = (frac_prot_pos / frac_non_prot_pos) * 100.0
    print()
    print("Total data points: %d" % (len(x_control)))
    print("# non-protected examples: %d" % (non_prot_all))
    print("# protected examples: %d" % (prot_all))
    print("Non-protected in positive class: %d (%0.0f%%)" % (non_prot_pos, non_prot_pos * 100.0 / non_prot_all))
    print("Protected in positive class: %d (%0.0f%%)" % (prot_pos, prot_pos * 100.0 / prot_all))
    print("P-rule is: %0.0f%%" % ( p_rule ))
    return p_rule


def check_accuracy(model, x_train, y_train, x_test, y_test, y_train_predicted, y_test_predicted):

    """
    returns the train/test accuracy of the model
    we either pass the model (w) else we pass y_predicted
    """
    if model is not None and y_test_predicted is not None:
        print("Either the model (w) or the predicted labels should be None")
        raise Exception("Either the model (w) or the predicted labels should be None")

    if model is not None:
        y_test_predicted = np.sign(np.dot(x_test, model))
        y_train_predicted = np.sign(np.dot(x_train, model))

    def get_accuracy(y, Y_predicted):
        correct_answers = (Y_predicted == y).astype(int) # will have 1 when the prediction and the actual label match
        accuracy = float(sum(correct_answers)) / float(len(correct_answers))
        return accuracy, sum(correct_answers)

    train_score, correct_answers_train = get_accuracy(y_train, y_train_predicted)
    test_score, correct_answers_test = get_accuracy(y_test, y_test_predicted)

    return train_score, test_score, correct_answers_train, correct_answers_test

def test_sensitive_attr_constraint_cov(model, x_arr, y_arr_dist_boundary, x_control, thresh, verbose):

    
    """
    The covariance is computed b/w the sensitive attr val and the distance from the boundary
    If the model is None, we assume that the y_arr_dist_boundary contains the distace from the decision boundary
    If the model is not None, we just compute a dot product or model and x_arr
    for the case of SVM, we pass the distace from bounday becase the intercept in internalized for the class
    and we have compute the distance using the project function

    this function will return -1 if the constraint specified by thresh parameter is not satifsified
    otherwise it will reutrn +1
    if the return value is >=0, then the constraint is satisfied
    """

    assert(x_arr.shape[0] == x_control.shape[0])
    if len(x_control.shape) > 1: # make sure we just have one column in the array
        assert(x_control.shape[1] == 1)
    
    arr = []
    if model is None:
        arr = y_arr_dist_boundary # simply the output labels
    else:
        arr = np.dot(model, x_arr.T) # the product with the weight vector -- the sign of this is the output label
    
    arr = np.array(arr, dtype=np.float64)
    cov = np.dot(x_control - np.mean(x_control), arr ) / float(len(x_control))

    ans = thresh - abs(cov) # will be <0 if the covariance is greater than thresh -- that is, the condition is not satisfied
    # ans = thresh - cov # will be <0 if the covariance is greater than thresh -- that is, the condition is not satisfied
    if verbose is True:
        print("Covariance is", cov)
        print("Diff is:", ans)
        print()
    return ans

def print_covariance_sensitive_attrs(model, x_arr, y_arr_dist_boundary, x_control, sensitive_attrs):


    """
    reutrns the covariance between sensitive features and distance from decision boundary
    """

    arr = []
    if model is None:
        arr = y_arr_dist_boundary # simplt the output labels
    else:
        arr = np.dot(model, x_arr.T) # the product with the weight vector -- the sign of this is the output label

    sensitive_attrs_to_cov_original = {}
    for attr in sensitive_attrs:

        attr_arr = x_control[attr]
        attr_arr_transformed = np.array(attr_arr, dtype=int)

        thresh = 0
        cov = thresh - test_sensitive_attr_constraint_cov(None, x_arr, arr, np.array(attr_arr), thresh, False)
        sensitive_attrs_to_cov_original[attr] = cov
            
    return sensitive_attrs_to_cov_original


def get_correlations(model, x_test, y_predicted, x_control_test, sensitive_attrs):

    """
    returns the fraction in positive class for sensitive feature values
    """

    if model is not None:
        y_predicted = np.sign(np.dot(x_test, model))
        
    y_predicted = np.array(y_predicted)
    
    out_dict = {}
    for attr in sensitive_attrs:

        attr_val = []
        for v in x_control_test[attr]: attr_val.append(v)
        assert(len(attr_val) == len(y_predicted))


        total_per_val = defaultdict(int)
        attr_to_class_labels_dict = defaultdict(lambda: defaultdict(int))

        for i in range(0, len(y_predicted)):
            val = attr_val[i]
            label = y_predicted[i]

            # val = attr_val_int_mapping_dict_reversed[val] # change values from intgers to actual names
            total_per_val[val] += 1
            attr_to_class_labels_dict[val][label] += 1

        class_labels = set(y_predicted.tolist())

        local_dict_1 = {}
        for k1,v1 in attr_to_class_labels_dict.items():
            total_this_val = total_per_val[k1]

            local_dict_2 = {}
            for k2 in class_labels: # the order should be the same for printing
                v2 = v1[k2]

                f = float(v2) * 100.0 / float(total_this_val)

                local_dict_2[k2] = f
            local_dict_1[k1] = local_dict_2
        out_dict[attr] = local_dict_1

    return out_dict



def get_constraint_list_cov(x_train, y_train, x_control_train, sensitive_attrs, sensitive_attrs_to_cov_thresh):

    """
    get the list of constraints to be fed to the minimizer
    """

    constraints = []
    
    for attr in sensitive_attrs:

        attr_arr = x_control_train[attr]
        attr_arr_transformed = np.array(attr_arr, dtype=int)
        thresh = sensitive_attrs_to_cov_thresh[attr]
        c = ({'type': 'ineq', 'fun': test_sensitive_attr_constraint_cov, 'args':(x_train, y_train, attr_arr_transformed,thresh, False)})
        constraints.append(c)

    return constraints



def split_into_train_test(x_all, y_all, x_control_all, train_fold_size):
    split_point = int(round(float(x_all.shape[0]) * train_fold_size))
    x_all_train = x_all[:split_point]
    x_all_test = x_all[split_point:]
    y_all_train = y_all[:split_point]
    y_all_test = y_all[split_point:]
    x_control_all_train = x_control_all[:split_point]
    x_control_all_test = x_control_all[split_point:]
#     x_control_all_train = {}
#     x_control_all_test = {}
#     for k in x_control_all.keys():
#         x_control_all_train[k] = x_control_all[k][:split_point]
#         x_control_all_test[k] = x_control_all[k][split_point:]

    return x_all_train, y_all_train, x_control_all_train, x_all_test, y_all_test, x_control_all_test


def get_avg_correlation_dict(correlation_dict_arr):
    # make the structure for the correlation dict
    correlation_dict_avg = {}
    # print correlation_dict_arr
    for k,v in correlation_dict_arr[0].items():
        correlation_dict_avg[k] = {}
        for feature_val, feature_dict in v.items():
            correlation_dict_avg[k][feature_val] = {}
            for class_label, frac_class in feature_dict.items():
                correlation_dict_avg[k][feature_val][class_label] = []

    # populate the correlation dict
    for correlation_dict in correlation_dict_arr:
        for k,v in correlation_dict.items():
            for feature_val, feature_dict in v.items():
                for class_label, frac_class in feature_dict.items():
                    correlation_dict_avg[k][feature_val][class_label].append(frac_class)

    # now take the averages
    for k,v in correlation_dict_avg.items():
        for feature_val, feature_dict in v.items():
            for class_label, frac_class_arr in feature_dict.items():
                correlation_dict_avg[k][feature_val][class_label] = np.mean(frac_class_arr)

    return correlation_dict_avg


def train_test_classifier(X_train, y_train, x_sensitive_train, X_test, y_test, x_sensitive_test, sensitive_attrs,
                          loss_function=_logistic_loss, apply_fairness_constraints=0, 
                          apply_accuracy_constraint=0, sep_constraint=0, 
                          sensitive_attrs_to_cov_thresh={}, gamma=None):
    """
    Train the classifer on the training data and evaluate on the test data.
    
    PARAMETERS
    - X_train, y_train, x_sensitive_train: the non-sensitive features, class labels, and sensitive features in the training set
    - X_test, y_test, x_sensitive_test: the non-sensitive features, class labels, and sensitive features in the test set
    - sensitive_attrs: a list of sensitive feature names
    - apply_accuracy_constraint: 0/1, whether or not to apply accuracy constraints
    - gamma: accuracy constraint
    
    RETURN
    - theta: the parameters that the model learned
    - p-rules: the p-rule for each sensitive feature
    - test-score: the accuracy of the classifier evaluated on the test data
    """
    
    theta = train_model(X_train, y_train, x_sensitive_train, loss_function, 
                       apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, 
                       sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)
    train_score, test_score, _, _ = check_accuracy(theta, X_train, y_train, X_test, y_test, None, None)
    distances_from_decision_boundary = np.dot(X_test, theta).tolist()
    predicted_class_labels = np.sign(distances_from_decision_boundary)
    correlation_dict_test = get_correlations(None, None, predicted_class_labels, x_sensitive_test, sensitive_attrs)
    cov_dict_test = print_covariance_sensitive_attrs(None, X_test, distances_from_decision_boundary, x_sensitive_test, 
                                                        sensitive_attrs)
    p_rules = [print_classifier_fairness_stats([test_score], [correlation_dict_test], [cov_dict_test], i)
               for i in sensitive_attrs]
    
    return theta, p_rules, test_score


def plot_boundaries(X, y, x_sensitive_array, theta1, p_rule1, accuracy1, theta2=None, p_rule2=None, accuracy2=None):
    
    """
    Plot the data and the classifier's decision boundary.
    """
    num_to_draw = 200 # draw a small number of points to avoid clutter
    X_draw = X[:num_to_draw]
    y_draw = y[:num_to_draw]
    x_sensitive_draw = x_sensitive_array[:num_to_draw]
    
    # plot the data by class and sensitive group
    X_protected = X_draw[x_sensitive_draw == 0.0]
    X_unprotected = X_draw[x_sensitive_draw == 1.0]
    y_protected = y_draw[x_sensitive_draw == 0.0]
    y_unprotected = y_draw[x_sensitive_draw == 1.0]
    plt.scatter(X_protected[y_protected==1.0][:, 1], X_protected[y_protected==1.0][:, 2], color='green', marker='x', s=30, linewidth=1.5, label= "Protected (female), Hired")
    plt.scatter(X_protected[y_protected==-1.0][:, 1], X_protected[y_protected==-1.0][:, 2], color='red', marker='x', s=30, linewidth=1.5, label = "Protected (female), Not Hired")
    plt.scatter(X_unprotected[y_unprotected==1.0][:, 1], X_unprotected[y_unprotected==1.0][:, 2], color='green', marker='o', facecolors='none', s=30, label = "Unprotected (male), Hired")
    plt.scatter(X_unprotected[y_unprotected==-1.0][:, 1], X_unprotected[y_unprotected==-1.0][:, 2], color='red', marker='o', facecolors='none', s=30, label = "Unprotected (male), Not Hired")

    # draw the decision boundary
    def get_line_coordinates(w, x1, x2):
        y1 = (-w[0] - (w[1] * x1)) / w[2]
        y2 = (-w[0] - (w[1] * x2)) / w[2]    
        return y1, y2
    
    x1,x2 = max(X_draw[:,1]), min(X_draw[:,1])
    y1,y2 = get_line_coordinates(theta1, x1, x2)
    plt.plot([x1,x2], [y1,y2], 'c-', linewidth=3, label = "Acc=%0.2f; p%% rule=%0.0f%% - Original"%(accuracy1, p_rule1))
    if theta2 is not None:
        y1,y2 = get_line_coordinates(theta2, x1, x2)
        plt.plot([x1,x2], [y1,y2], 'b--', linewidth=3, label = "Acc=%0.2f; p%% rule=%0.0f%% - Constrained"%(accuracy2, p_rule2))

    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    plt.legend(loc=(1.01, 0.3), fontsize=15)
    plt.xlim((-15,10))
    plt.ylim((-10,15))
    plt.xlabel("Prior Income")
    plt.ylabel("Years of Work Experience")
    plt.show()
