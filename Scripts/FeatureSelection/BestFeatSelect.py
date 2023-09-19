""" 
BestFeatSelect.py
1. PCA reduce input layer features to 100 components
2. For each iteration (behavioural dimension, leave item out fold, k features),
get best features (via RFE, with linear regression as estimator) and assign to 
output 4D matrix (out_mat) of size:
    80 folds * 100 components * k feature sets * k behavioural dimensions
"""

#--- Imports
import time
import os
# import math
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE

#--- User input

targ_dims = ["V_DL_1", "V_DL_2", "V_DL_3", "V_DL_4", "V_DL_5",
             "M_DL_1","M_DL_2","M_DL_3","M_DL_4","M_DL_5","M_DL_6",
             "F_DL_1","F_DL_2","F_DL_3","F_DL_4"]

best_feat_sizes = np.concatenate((np.array([1,5]),np.arange(10,51,10))) # number of best_k_features/components

n_comp = 100 # N PCA components to start with

main_path = "/home/jon/GitRepos/OK_CLIP/"
dnn_feats = (main_path + "Data/EightyTools/features.npy")
dim_data = (main_path + 
                "Data/BehaviouralDimensions/SelectedDimensions.csv")
out_path = (main_path + "temp/BestFeatures/")

##

#--- Functions

def load_dim_data(dim_data):
    """ Load dimension names and values"""
    df = pd.read_csv(dim_path) 
    dim_names = list(df.columns[1:])
    dim_vals = df.iloc[:,1:].to_numpy()    
    return dim_names, dim_vals

def reorder_dim_data(targ_dims, dim_names, dim_vals):
    """ Reorder dimension names and values based on target dimension 
    ordering"""
    td_idx = [dim_names.index(i) for i in targ_dims]
    dim_vals = np.stack(dim_vals[:,td_idx]).astype(None) # convert to regular array    
    return dim_vals

def custom_cv_split(n_exemp, n_item, n_fold):
    """ Create cross-validation splits, specifying 
    n_folds, n_items, n_exemplars"""  
    te_idx = [np.arange(i,i+n_exemp) 
              for i in np.arange(0,n_item*n_exemp,n_exemp)]
    tr_idx = [np.setdiff1d(np.arange(n_item*n_exemp), i) 
              for i in te_idx]
    custom_cv = [i for i in zip(tr_idx, te_idx)] 
    return custom_cv

def pca_feats(feats, n_comp, custom_cv, fold):
    """PCA over all samples of the current CV fold"""
    
    # Subset training items for current fold
    feats_pca = np.copy(feats)
    feats_pca = feats_pca[custom_cv[fold][0],:]
    
    # Scale + PCA
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("PCA", PCA(n_components=n_comp,svd_solver='full'))])
    feats_pca = pipe.fit_transform(feats_pca)
    return feats_pca
    

# !!! use targ_dims instead of dim_names going forward

# Load dnn_feats
feats = np.load(dnn_feats)

# Load dim_data
dim_names, dim_vals = load_dim_data(dim_data)

# Reorder dimensions based on ordering of targ_dims 
dim_vals = reorder_dim_data(targ_dims, dim_names, dim_vals)

# Standardize dimension values
dim_vals = StandardScaler().fit_transform(dim_vals)

# Get cross-validation splits
n_item = len(dim_vals)
n_exemp = int(len(feats) / n_item)
n_fold = n_item
custom_cv = custom_cv_split(n_exemp, n_item, n_fold)

# Initialize output matrix and headers
out_mat = np.zeros((n_fold, n_comp, len(best_feat_sizes), len(targ_dims)))
out_heads = targ_dims

# Get index-value pairs for best_feat_sizes
out_bk_idx_val = [(i_idx,i) for i_idx, i in enumerate(best_feat_sizes)]


# Loop here!



# Loop
out_mat = np.zeros((n_fold, n_comp, len(best_feat_sizes), len(targ_dims)))
out_heads = targ_dims
out_bk_idx_val = [(i_idx,i) for i_idx, i in enumerate(best_feat_sizes)]
for cnn_name, layer_name in cnn_layer_names:
    tic = time.time()        
    
    # #--- Load layer features, reshape to 2D
    # feats = np.load(os.path.join(cnn_layer_path,cnn_name, layer_name,'features.npy'))
    # if len(feats.shape) > 2:
    #     feats = feats.reshape(feats.shape[0], np.prod(np.array(feats.shape[1:])))
 
    for best_k_idx, best_k_feats in enumerate(best_feat_sizes):  

        #--- PC regression 
        for td, td_name in enumerate(targ_dims):
            for f in np.arange(n_fold):
                
                # #--- Scale and PCA
                # feats_pca = np.copy(feats)
                # feats_pca = feats_pca[custom_cv[f][0],:]
                # feats_pca = sc.fit_transform(feats_pca)
                # pca = PCA(n_components=n_comp,svd_solver='full')
                # feats_pca = pca.fit_transform(feats_pca)
                
                # Training data
                train_X = feats_pca
                train_y = np.repeat(dim_vals[:,td],n_exemp)[custom_cv[f][0]]                 
                                     
                # Assign RFE feature rankings
                rfe = RFE(estimator=LinearRegression(), 
                          n_features_to_select=best_k_feats, 
                          importance_getter = "coef_")
                rfe.fit(train_X,train_y)                 
                out_mat[f,:,best_k_idx, td] = rfe.ranking_                       
               
    out_name = "RFE_BestFeatures_%s_%s_%dComp_Incremental_C" % (cnn_name, layer_name, n_comp)
    np.savez(os.path.join(out_path, out_name), out_mat = out_mat, 
             out_heads = out_heads, out_bk_idx_val = out_bk_idx_val)
    print("Layer time: %.3f" % (time.time()-tic))

