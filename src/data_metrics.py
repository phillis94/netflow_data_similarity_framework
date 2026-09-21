import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import scipy.stats

from collections import Counter
import scipy.stats as ss
import math
from sklearn.ensemble import IsolationForest



NUMERICAL_ATTRIBUTES = ['IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS']
CATEGORICAL_ATTRIBUTES = [ 'IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT',
                            'PROTOCOL', 
                            #'L7_PROTO', 
                            #'IN_BYTES', 'OUT_BYTES', 
                            #'IN_PKTS', 'OUT_PKTS',
                            'TCP_FLAGS', 
                            #'FLOW_DURATION_MILLISECONDS', 
                            'Label', 
                            #'Attack' #unequal amount of labels, to prevent errors for pcd dont use them
                        ]


def kernel_density_estimation(s1):
    #https://stackoverflow.com/questions/66903255/retrieve-values-from-scipy-gaussian-kde
    density = scipy.stats.gaussian_kde(s1)
    return density

def calculate_wasserstein_distance(s1:Series, s2:Series):
    return scipy.stats.wasserstein_distance(s1,s2)

def calculate_ks_test(s1:Series, s2:Series):
    results = scipy.stats.kstest(s1,s2, N=100)
    print(results)
    quit()


#https://stackoverflow.com/questions/56811525/calculate-probability-vector-from-sample-data
def calculate_jenson_shennon_divergence(s1:Series, s2:Series):
    s1 = s1.to_numpy().reshape(1, -1) #cdist requires 2d-arrays
    s2 = s2.to_numpy().reshape(1, -1) #cdist requires 2d-arrays
    #print(s1.shape, s2.shape)
    assert (s1.shape == s2.shape)
    return scipy.spatial.distance.cdist(s1, s2, 'jensenshannon')

# population ########################################################

def calculate_pearson(s1:Series, s2:Series):
    s1 = s1[NUMERICAL_ATTRIBUTES]
    s2 = s2[NUMERICAL_ATTRIBUTES]
    s1_corr = s1.corr()
    s2_corr = s2.corr()
    return s1_corr, s2_corr

#https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta

def calculate_correlation_ratio(s1:Series, s2:Series):
    df_corr_matrix_s1 = pd.DataFrame(index=CATEGORICAL_ATTRIBUTES, columns=NUMERICAL_ATTRIBUTES)
    df_corr_matrix_s2 = pd.DataFrame(index=CATEGORICAL_ATTRIBUTES, columns=NUMERICAL_ATTRIBUTES)

    #calc correlation ratio per attribute pair; filter by non existing pairs
    for cat in CATEGORICAL_ATTRIBUTES:
        #print('CAT')
        #print(len(s1[cat]), len(s2[cat]))
        if (len(s1[cat])>0) and (len(s2[cat])>0):
            for num in NUMERICAL_ATTRIBUTES:
                #check if they are available in s1 and s2
                #print('NUM')
                #print(len(s1[num]), len(s2[num]))
                if (len(s1[num])>0) and (len(s2[num])>0):
                    c_ratio_s1 = correlation_ratio(s1[cat], s1[num])
                    #print('c_ratio_s1', c_ratio_s1)
                    df_corr_matrix_s1.loc[cat,num] = c_ratio_s1
                    c_ratio_s2 = correlation_ratio(s2[cat], s2[num])
                    #print('c_ratio_s1', c_ratio_s1)
                    df_corr_matrix_s2.loc[cat,num] = c_ratio_s2
        
    return df_corr_matrix_s1, df_corr_matrix_s2

#FIXME
#https://www.kaggle.com/code/akshay22071995/alone-in-the-woods-using-theil-s-u-for-survival
def conditional_entropy(x,y):
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy

def theil_u(x,y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x

def calculate_theil_u(s1:Series, s2:Series):
    #calc for each combination of categorical variables
    df_corr_matrix_s1 = pd.DataFrame(index=CATEGORICAL_ATTRIBUTES, columns=CATEGORICAL_ATTRIBUTES)
    df_corr_matrix_s2 = pd.DataFrame(index=CATEGORICAL_ATTRIBUTES, columns=CATEGORICAL_ATTRIBUTES)

    for row in CATEGORICAL_ATTRIBUTES:
        if len(s1[row])>0 and len(s2[row])>0:
            for col in CATEGORICAL_ATTRIBUTES:
                #check if they are available in s1 and s2
                if len(s1[col])>0 and len(s2[col])>0:
                    corr = theil_u(s1[row], s1[col])
                    df_corr_matrix_s1.loc[row,col] = corr
                    df_corr_matrix_s1.loc[col,row] = corr #correlation is equal here (symmetric)

                    corr = theil_u(s2[row], s2[col])
                    df_corr_matrix_s2.loc[row,col] = corr
                    df_corr_matrix_s2.loc[col,row] = corr #correlation is equal here (symmetric)
                
    return df_corr_matrix_s1, df_corr_matrix_s2


#https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html#sphx-glr-auto-examples-miscellaneous-plot-anomaly-comparison-py
#https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_outlier_detection_bench.html#sphx-glr-auto-examples-miscellaneous-plot-outlier-detection-bench-py

def test_isolation_forest_dsicriminator_raw(x_ds_1, x_ds_2):
    y_ds_1 = np.ones((x_ds_1.shape[0],1))*-1
    y_ds_2 = np.ones((x_ds_2.shape[0],1))*-1

    #fit ds_1, test ds_2
    #clf_ds_1 = IsolationForest(random_state=42, n_estimators=1000, max_features=1.0, max_samples=1.0, bootstrap=False, contamination=0.0001, n_jobs=2).fit(X=x_ds_1)
    clf_ds_1 = IsolationForest(random_state=42, n_jobs=2).fit(X=x_ds_1)
    anomaly_scores_ds_1 = clf_ds_1.predict(x_ds_2) 

    #fit ds_2, test_ds_1
    #clf_ds_2 = IsolationForest(random_state=42, n_estimators=1000, max_features=1.0, max_samples=1.0, bootstrap=False, contamination=0.0001, n_jobs=2).fit(X=x_ds_2)
    clf_ds_2 = IsolationForest(random_state=42, n_jobs=2).fit(X=x_ds_2)
    anomaly_scores_ds_2 = clf_ds_2.predict(x_ds_1) 

    y_ds_1 = pd.Series(y_ds_1.flatten())
    y_ds_2 = pd.Series(y_ds_2.flatten())

    anomaly_scores_ds_1 = pd.Series(anomaly_scores_ds_1.flatten())
    anomaly_scores_ds_2 = pd.Series(anomaly_scores_ds_2.flatten())

    return {'if_disc_y_ds1':y_ds_1, 'if_disc_tr-ds1_tst-ds2':anomaly_scores_ds_1,
            'if_disc_y_ds2':y_ds_2, 'if_disc_tr-ds2_tst-ds1':anomaly_scores_ds_2}


def split_task_frames(ds_1, ds_2, ds_eval=None):
    """Build the (train, eval) frames for the task metrics.

    Without ds_eval this is the original TRTS setup: train on one dataset's benign flows,
    score the other dataset in full, and grade against that dataset's own labels.

    With ds_eval it becomes TSTR, which is what an unsupervised AD pipeline actually needs.
    The CIDDS generators in ../mp-lissmann are trained on the attack-free train split by
    design, so both ds_1 (synthetic) and ds_2 (real train reference) are 100 % Label==0 and
    the original setup degenerates: it trains on one class and grades against one class, so
    F1 carries no information and the Domain Dissimilarity Score built on it is undefined.

    Instead, both detectors are trained on their own benign flows and evaluated on the *same*
    real, attack-carrying sample. The synthetic-trained score against the real-trained score
    is then a direct answer to "how much detection quality is lost by substituting synthetic
    data", which is the question RQ4 needs and which no distribution distance answers.

    Sharing one eval frame across both directions is what makes the two numbers a paired
    comparison; giving each its own sample would let the difference between them come from
    the draw rather than from the training data.
    """
    x_ds_1_train = ds_1[ds_1['Label'] == 0].reset_index(drop=True).drop(columns=['Label'])
    x_ds_2_train = ds_2[ds_2['Label'] == 0].reset_index(drop=True).drop(columns=['Label'])

    if ds_eval is None:
        y_for_ds_1_model, x_for_ds_1_model = ds_2['Label'], ds_2.drop(columns=['Label'])
        y_for_ds_2_model, x_for_ds_2_model = ds_1['Label'], ds_1.drop(columns=['Label'])
    else:
        y_for_ds_1_model = y_for_ds_2_model = ds_eval['Label']
        x_for_ds_1_model = x_for_ds_2_model = ds_eval.drop(columns=['Label'])

    #range from [-1,1]; benign -> +1 (inlier), attack -> -1, matching predict()
    y_for_ds_1_model = ((y_for_ds_1_model*2)-1)*-1
    y_for_ds_2_model = ((y_for_ds_2_model*2)-1)*-1

    return (x_ds_1_train, x_for_ds_1_model, y_for_ds_1_model,
            x_ds_2_train, x_for_ds_2_model, y_for_ds_2_model)


def test_isolation_forest_task_raw(ds_1, ds_2, ds_eval=None):
    #x_ds_1, y_ds_1, x_ds_2, y_ds_2
    (x_ds_1_train, x_ds_2, y_ds_2,
     x_ds_2_train, x_ds_1, y_ds_1) = split_task_frames(ds_1, ds_2, ds_eval)

    #fit ds_1, test ds_2
    #clf_ds_1 = IsolationForest(random_state=42, n_estimators=1000, max_features=1.0, max_samples=1.0, bootstrap=False, contamination=0.0001, n_jobs=2).fit(X=x_ds_1_train)
    clf_ds_1 = IsolationForest(random_state=42, n_jobs=2).fit(X=x_ds_1_train)
    anomaly_scores_ds_1 = clf_ds_1.predict(x_ds_2) 

    #fit ds_2, test_ds_1
    #clf_ds_2 = IsolationForest(random_state=42, n_estimators=1000, max_features=1.0, max_samples=1.0, bootstrap=False, contamination=0.0001, n_jobs=2).fit(X=x_ds_2_train)
    clf_ds_2 = IsolationForest(random_state=42, n_jobs=2).fit(X=x_ds_2_train)
    anomaly_scores_ds_2 = clf_ds_2.predict(x_ds_1) 

    anomaly_scores_ds_1 = pd.Series(anomaly_scores_ds_1.flatten())
    anomaly_scores_ds_2 = pd.Series(anomaly_scores_ds_2.flatten())
    
    return {'if_task_y_ds1':y_ds_1, 'if_task_tr-ds1_tst-ds2':anomaly_scores_ds_1,
            'if_task_y_ds2':y_ds_2, 'if_task_tr-ds2_tst-ds1':anomaly_scores_ds_2}


from sklearn.svm import OneClassSVM


def test_ocsvm_dsicriminator_raw(x_ds_1, x_ds_2):
    y_ds_1 = np.ones((x_ds_1.shape[0],1))*-1
    y_ds_2 = np.ones((x_ds_2.shape[0],1))*-1

    #fit ds_1, test ds_2
    clf_ds_1 = OneClassSVM().fit(X=x_ds_1)
    anomaly_scores_ds_1 = clf_ds_1.predict(x_ds_2) #in style of prpensity score good models should at least have 0.5?    

    #fit ds_2, test_ds_1
    clf_ds_2 =OneClassSVM().fit(X=x_ds_2)
    anomaly_scores_ds_2 = clf_ds_2.predict(x_ds_1) #in style of prpensity score good models should at least have 0.5?

    y_ds_1 = pd.Series(y_ds_1.flatten())
    y_ds_2 = pd.Series(y_ds_2.flatten())

    anomaly_scores_ds_1 = pd.Series(anomaly_scores_ds_1.flatten())
    anomaly_scores_ds_2 = pd.Series(anomaly_scores_ds_2.flatten())

    return {'ocsvm_disc_y_ds1':y_ds_1, 'ocsvm_disc_tr-ds1_tst-ds2':anomaly_scores_ds_1,
            'ocsvm_disc_y_ds2':y_ds_2, 'ocsvm_disc_tr-ds2_tst-ds1':anomaly_scores_ds_2}

def test_ocsvm_task_raw(ds_1, ds_2, ds_eval=None):
    #x_ds_1, y_ds_1, x_ds_2, y_ds_2
    (x_ds_1_train, x_ds_2, y_ds_2,
     x_ds_2_train, x_ds_1, y_ds_1) = split_task_frames(ds_1, ds_2, ds_eval)

    #fit ds_1, test ds_2
    clf_ds_1 = OneClassSVM().fit(X=x_ds_1_train)
    anomaly_scores_ds_1 = clf_ds_1.predict(x_ds_2)   

    #fit ds_2, test_ds_1
    clf_ds_2 = OneClassSVM().fit(X=x_ds_2_train)
    anomaly_scores_ds_2 = clf_ds_2.predict(x_ds_1) 


    anomaly_scores_ds_1 = pd.Series(anomaly_scores_ds_1.flatten())
    anomaly_scores_ds_2 = pd.Series(anomaly_scores_ds_2.flatten())
    
    return {'ocsvm_task_y_ds1':y_ds_1, 'ocsvm_task_tr-ds1_tst-ds2':anomaly_scores_ds_1,
            'ocsvm_task_y_ds2':y_ds_2, 'ocsvm_task_tr-ds2_tst-ds1':anomaly_scores_ds_2}


#import xgboost as xgb

# def test_xgboost_task_raw(ds_1, ds_2):
#     #x_ds_1, y_ds_1, x_ds_2, y_ds_2
#     y_ds_1 = ds_1['Label']
#     x_ds_1 = ds_1.drop(columns=['Label'])
#
#     y_ds_2 = ds_2['Label']
#     x_ds_2 = ds_2.drop(columns=['Label'])
#
#     #y_ds_1 = ((y_ds_1*2)-1)*-1 #range from [-1,1]
#     #y_ds_2 = ((y_ds_2*2)-1)*-1 #range from [-1,1]
#
#     #fit ds_1, test ds_2
#     clf_ds_1 = xgb.XGBClassifier(n_jobs=2).fit(x_ds_1, y_ds_1)
#     anomaly_scores_ds_1 = clf_ds_1.predict(x_ds_2)
#
#     #fit ds_2, test_ds_1
#     clf_ds_2 = xgb.XGBClassifier(n_jobs=2).fit(x_ds_2, y_ds_2)
#     anomaly_scores_ds_2 = clf_ds_2.predict(x_ds_1)
#
#
#     anomaly_scores_ds_1 = pd.Series(anomaly_scores_ds_1.flatten())
#     anomaly_scores_ds_2 = pd.Series(anomaly_scores_ds_2.flatten())
#
#     return {'xgb_task_y_ds1':y_ds_1, 'xgb_task_tr-ds1_tst-ds2':anomaly_scores_ds_1,
#             'xgb_task_y_ds2':y_ds_2, 'xgb_task_tr-ds2_tst-ds1':anomaly_scores_ds_2}