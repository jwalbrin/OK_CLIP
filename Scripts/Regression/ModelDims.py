""" 
ModelDims.py
Leave-one-item-out (LIO) linear regression to model 
behavioural dimensions (y) with DNN image representations (X)
For each iteration (dimension, CV fold):
a. PCA is applied to DNN representations (e.g. 100 components)
b. Best k features/components are selected (e.g. 10 best 
features as determined in BestFeatSelect.py)
c. Evaluate and assigned measure (e.g. R-square) to
output matrix (n exemplar images * n target dimensions)
Note: there are 800 input samples (80 objects * 10 exemplar images),
resulting in 10 model fits - each based on 80 items, for an arbitrary
selection of exemplars (e.g. exemplar 1 = the first exemplar for each 
of the 80 objects)
"""

#--- Imports

import os
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import sys
scripts_path = "/home/jon/GitRepos/OK_CLIP/Scripts/"
sys.path.append(scripts_path)
from Functions.dataformatting import *

#--- User input

# Name of features (either "EightyTools" or "ThingsImages")
feat_name = "EightyTools"

# Paths
main_path = "/home/jon/GitRepos/OK_CLIP/"
dnn_feats = os.path.join(main_path, "Data", feat_name, "features.npy")
out_path = os.path.join(main_path, "temp/ModelOutput/")
dim_data = os.path.join(main_path, 
                        "Data/BehaviouralDimensions/",
                        "SelectedDimensions.csv")

# Best k features data (output of BestFeatSelect.py)
bkf_path = os.path.join(main_path,"temp/BestFeatures/")
bkf_stem = "RFE_BestFeatures_ViT_CLIP_%s_%dComp.npz"

# N components (as in best k features data), k best features
n_comp = 100
best_k_feats = 10

# Dimensions to test
targ_dims = ["V_DL_1", "V_DL_2", "V_DL_3", "V_DL_4", "V_DL_5",
             "M_DL_1","M_DL_2","M_DL_3","M_DL_4","M_DL_5","M_DL_6",
             "F_DL_1","F_DL_2","F_DL_3","F_DL_4"]

# Model evaluation metric
mod_eval_metric = "r2" # "r2", "r2adj", "MSE", "RMSE", "MAE"

#--- Main

# Get model evaluation metric (e.g. R-square)
eval_score, y_label, y_lims = get_eval_score_function(mod_eval_metric)

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

# Load best k features data
bkf_full_path = os.path.join(bkf_path, bkf_stem % (feat_name, n_comp))
bkf_mat, bkf_idx_val = load_bkf_reordered(bkf_full_path, targ_dims)

# Find data matrix index for best k, then indices of those k components
k_idx = int(np.where(np.array(bkf_idx_val)[:,1]==best_k_feats)[0])

# Initialize data matrix 
data_mat = np.zeros((n_exemp,len(targ_dims)))

# Regression
for td_i, td in enumerate(targ_dims):
    
    # Temporary matrix for n exemp * n fold predictions
    temp_pred_mat = np.zeros((n_exemp,n_fold))     
    
    # Time it
    tic = time.time()
    for f in np.arange(n_fold):
    
        
        # Get train and test features
        train_X, test_X = tr_te_split(feats, custom_cv[f][0], custom_cv[f][1])
    	
    	# PCA fit and transform	to n components
       	train_X, test_X = pca_tr_te(train_X, test_X, n_comp)
    	
    	# Slice best k features 
        bkf_idx = np.where(bkf_mat[f,:,k_idx,td_i] == 1)[0] 
       	train_X = train_X[:,bkf_idx]
       	test_X = test_X[:,bkf_idx]    
    	
    	# Get train, test y (repeat across exemplars)
       	train_y, test_y = select_repeat_y(dim_vals[:,td_i], n_exemp,
                                           					custom_cv[f][0], 
                                           					custom_cv[f][1])

        # Fit regression
        linreg = LinearRegression()  
        linreg.fit(train_X,train_y)
        
        # Collect predictions
        temp_pred_mat[:,f] = linreg.predict(test_X) 
        
    # Evaluate for each exemplar split
    for e in np.arange(n_exemp):        
        data_mat[e,td_i] = eval_score(dim_vals[:,td_i], 
                                   temp_pred_mat[e,:], 
                                   best_k_feats)
                                   
    print(f"Dimension {td} run time: {time.time() - tic: .02f} seconds")

# Bar plot (mean model fits across exemplar splits)
mod_fit_bars(data_mat, best_k_feats, targ_dims, y_label, y_lims)






