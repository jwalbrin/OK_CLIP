""" 
BestFeatSelect.py
1. PCA reduce input DNN layer features to 100 components
2. For each iteration (behavioural dimension, leave item out fold, k features),
get best features (via RFE, with linear regression as estimator) and assign to 
output 4D matrix (out_mat) of size:
    80 folds * 100 components * k feature sets * k behavioural dimensions
"""

#--- Imports
import time
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

#--- User input

# Dimensions to test
targ_dims = ["V_DL_1", "V_DL_2", "V_DL_3", "V_DL_4", "V_DL_5",
             "M_DL_1","M_DL_2","M_DL_3","M_DL_4","M_DL_5","M_DL_6",
             "F_DL_1","F_DL_2","F_DL_3","F_DL_4"]

targ_dims = ["V_DL_1", "V_DL_2", "V_DL_3"]

# Chosen string for dnn (for output name)
dnn_output_name = "ViT_CLIP"

# name of features (either "EightyTools" or "ThingsImages")
feat_name = "EightyTools"

# Paths
main_path = "/home/jon/GitRepos/OK_CLIP/"
dnn_feats = (main_path + "Data/" + feat_name + "/features.npy")
dim_data = (main_path + 
                "Data/BehaviouralDimensions/SelectedDimensions.csv")
out_path = (main_path + "temp/BestFeatures/")

# number of best_k_features/components to test
best_feat_sizes = np.concatenate((np.array([1,5]),np.arange(10,51,10))) 
best_feat_sizes = np.array([1,5]) 

# N PCA components to start with
n_comp = 100 

#--- Functions

def load_dim_data(dim_data):
    """ Load dimension names and values"""
    df = pd.read_csv(dim_data) 
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

#--- Main

if os.path.exists(out_path) == False:
    os.mkdir(out_path)

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

# RFE (for each best_feat_size, targ_dim, fold)
for best_k_idx, best_k_feats in enumerate(best_feat_sizes): 
    tic = time.time()
    for td, td_name in enumerate(targ_dims):
        for f in np.arange(n_fold):
            
            # PCA --> training features
            train_X = pca_feats(feats, n_comp, custom_cv, f)
            
            # Get training targets
            train_y = np.repeat(dim_vals[:,td],n_exemp)[custom_cv[f][0]]                 
                                 
            # Assign RFE feature rankings
            rfe = RFE(estimator=LinearRegression(), 
                      n_features_to_select=best_k_feats, 
                      importance_getter = "coef_")
            rfe.fit(train_X,train_y)                 
            out_mat[f,:,best_k_idx, td] = rfe.ranking_ 
            
    print("Best %i features run time: %.2f seconds" 
          % (best_k_feats, time.time()-tic))

# Save output
out_name = "RFE_BestFeatures_%s_%s_%dComp" % (dnn_output_name, 
                                              feat_name,n_comp)
np.savez(os.path.join(out_path, out_name), out_mat = out_mat, 
         out_heads = out_heads, out_bk_idx_val = out_bk_idx_val)

