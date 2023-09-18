""" 
Best features
For each behavioural dimension (e.g. vision 1: V_DL_1),
for each fold (i.e. 80 leave-one-item-out folds),
get best features (via PCA, then RFE)
Stores all ks in a single npz per layer
"""
import time
import os
# import math
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE

# Settings

cnn_layer_names = [("ViT_CLIP", "FCNorm")]

targ_dims = ["V_DL_1", "V_DL_2", "V_DL_3", "V_DL_4", "V_DL_5",
             "M_DL_1","M_DL_2","M_DL_3","M_DL_4","M_DL_5","M_DL_6",
             "F_DL_1","F_DL_2","F_DL_3","F_DL_4"]

best_k_feats_iter = np.concatenate((np.array([1,5]),np.arange(10,51,10))) # number of best_k_features/components

n_comp = 100 # N PCA components to start with

# cnn_layer_path = "/mnt/EXT4_DRIVE/MRIData/ETDR/Analysis/CNN_Comp_Models/CNN/ExtractedLayers/Older/"
cnn_layer_path = "/mnt/EXT4_DRIVE/MRIData/ETDR/Analysis/CNN_Comp_Models/CNN/ExtractedLayers/"

beh_dim_path = "/mnt/EXT4_DRIVE/MRIData/ETDR/Analysis/DimensionExtraction/MDSSelectedDimensions/"
out_path = "/home/jon/GitRepos/Modelling_Object_Dimensionality/temp/BestFeatures/Final/"

##

n_exemp = 10 
n_item = 80
n_fold = n_item

#--- Custom CV splits
# te_idx = [np.arange(i,n_item*n_exemp,n_exemp) 
#           for i in np.arange(n_exemp)]
# tr_idx = [np.setdiff1d(np.arange(n_item*n_exemp), i) 
#           for i in te_idx]
# custom_cv = [i for i in zip(tr_idx, te_idx)] 

#--- Load feature data + image names
dim_mat = loadmat(os.path.join(beh_dim_path,"jorgeSelectedDimensions.mat"))
dim_vals = dim_mat["selectDims"][1:,1:]
dim_heads = dim_mat["selectDims"][0,1:]
dim_heads = [str(i[0]) for i in dim_heads]
del dim_mat

#--- Filter and format dim data
td_idx = [dim_heads.index(i) for i in targ_dims]
dim_heads = np.array(dim_heads)[td_idx]
dim_heads = [str(i) for i in dim_heads]
dim_vals = dim_vals[:,td_idx]
dim_vals = np.stack(dim_vals).astype(None) # convert to regular array

#--- Standardize dims
sc = StandardScaler()
dim_vals = sc.fit_transform(dim_vals)

#--- Custom CV splits
te_idx = [np.arange(i,i+n_exemp) 
          for i in np.arange(0,n_item*n_exemp,n_exemp)]
tr_idx = [np.setdiff1d(np.arange(n_item*n_exemp), i) 
          for i in te_idx]
custom_cv = [i for i in zip(tr_idx, te_idx)] 

# Loop
out_mat = np.zeros((n_fold, n_comp, len(best_k_feats_iter), len(targ_dims)))
out_heads = dim_heads
out_bk_idx_val = [(i_idx,i) for i_idx, i in enumerate(best_k_feats_iter)]
for cnn_name, layer_name in cnn_layer_names:
    tic = time.time()        
    
    #--- Load layer features, reshape to 2D
    feats = np.load(os.path.join(cnn_layer_path,cnn_name, layer_name,'features.npy'))
    if len(feats.shape) > 2:
        feats = feats.reshape(feats.shape[0], np.prod(np.array(feats.shape[1:])))
 
    for best_k_idx, best_k_feats in enumerate(best_k_feats_iter):  

        #--- PC regression 
        for td, td_name in enumerate(targ_dims):
            for f in np.arange(n_fold):
                
                #--- Scale and PCA
                feats_pca = np.copy(feats)
                feats_pca = feats_pca[custom_cv[f][0],:]
                feats_pca = sc.fit_transform(feats_pca)
                pca = PCA(n_components=n_comp,svd_solver='full')
                feats_pca = pca.fit_transform(feats_pca)
                
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

