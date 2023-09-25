""" 
BestFeatSelect.py
1. PCA reduce input DNN layer features to 100 components
2. For each iteration (behavioural dimension (y), leave item out fold, k features),
get best features (via RFE, with linear regression as estimator) and assign to 
output 4D matrix (out_mat) of size:
    80 folds * 100 components * feature sets * behavioural dimensions
"""

#--- Imports
import time
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

import sys
scripts_path = "/home/jon/GitRepos/OK_CLIP/Scripts/"
sys.path.append(scripts_path)
from Functions.dataformatting import *

#--- User input

# Dimensions to test
targ_dims = ["V_DL_1", "V_DL_2", "V_DL_3", "V_DL_4", "V_DL_5",
             "M_DL_1","M_DL_2","M_DL_3","M_DL_4","M_DL_5","M_DL_6",
             "F_DL_1","F_DL_2","F_DL_3","F_DL_4"]

# Chosen string for dnn (for output name)
dnn_output_name = "ViT_CLIP"

# Features (either "EightyTools" or "ThingsImages")
feat_name = "EightyTools"

# Paths
main_path = "/home/jon/GitRepos/OK_CLIP/"
dnn_feats = os.path.join(main_path, "Data", feat_name, "features.npy")
out_path = os.path.join(main_path, "temp/BestFeatures/")
dim_data = os.path.join(main_path, 
                        "Data/BehaviouralDimensions/",
                        "SelectedDimensions.csv")

# Best k features/components to test
best_feat_sizes = np.concatenate((np.array([1,5]),np.arange(10,51,10))) 

# N PCA components 
n_comp = 100 

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
    for td_i in np.arange(len(targ_dims)):
        for f in np.arange(n_fold):
            
            # PCA --> training features
            train_X = pca_feats(feats, n_comp, custom_cv, f)
            
            # Get training targets
            train_y,_ = select_repeat_y(dim_vals[:,td_i], n_exemp, 
                                        custom_cv[f][0], custom_cv[f][1])                 
             
            # Assign RFE feature rankings
            rfe = RFE(estimator=LinearRegression(), 
                      n_features_to_select=best_k_feats, 
                      importance_getter = "coef_")
            rfe.fit(train_X,train_y)                 
            out_mat[f,:,best_k_idx, td_i] = rfe.ranking_ 
            
    print((f"Best {best_k_feats} features run time: " +
          f"{time.time()-tic: .02f} seconds"))

# Save output
out_name = "RFE_BestFeatures_%s_%s_%dComp" % (dnn_output_name, 
                                              feat_name,n_comp)
np.savez(os.path.join(out_path, out_name), out_mat = out_mat, 
         out_heads = out_heads, out_bk_idx_val = out_bk_idx_val)

