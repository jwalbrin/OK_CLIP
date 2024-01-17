""" 
lio_component_select.py
For an input DNN layer apply cross-validated RFE
For each iteration (behavioural dimension (y), leave-one-item-out (lio) fold, k components),
get best components (scale, PCA, RFE to training data only) and assign to 
4D matrix (bkc_mat): lio folds * components * best k component sets * behavioural dimensions
"""
import numpy as np
import os
import time

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

from functions.functions import (
    DataObject,
    prep_dim_data,
    custom_cv_split,
    pca_feats,
    repeat_exemplars_y,
    save_data_object,
)

# --- User input

# Dimensions to test
targ_dims = [
    ["V_DL_1", "V_DL_2", "V_DL_3", "V_DL_4", "V_DL_5"],
    ["M_DL_1", "M_DL_2", "M_DL_3", "M_DL_4", "M_DL_5", "M_DL_6"],
    ["F_DL_1", "F_DL_2", "F_DL_3", "F_DL_4"],
]

# Best k features/components to select
best_feat_sizes = np.concatenate((np.array([1, 5]), np.arange(10, 51, 10)))

# Model name
model_name = "clip-vit"

# Features ("eighty_tools","things_images")
feat_name = "eighty_tools"

# Paths
main_path = os.path.dirname(os.path.abspath(__file__))
dnn_feats = os.path.join(main_path, "data", feat_name, model_name, "features.npy")
out_path = os.path.join(main_path, "results/")
dim_data = os.path.join(
    main_path, "data/behavioural_dimensions/", "selected_dimensions.csv"
)

# N PCA components
n_comp = 100

# --- Main

if not os.path.exists(out_path):
    os.makedirs(out_path)

# Prepare dimensions
dim_vals, dim_names = prep_dim_data(dim_data, targ_dims)

# Flattened targ_dims list
targ_dims_flat = sum(targ_dims, [])

# Load dnn_feats, dim_data
feats = np.load(dnn_feats)

# Get cross-validation splits
n_item = len(dim_vals)
n_exemp = int(len(feats) / n_item)
n_fold = n_item
cv_idx = custom_cv_split(n_exemp, n_item, n_fold)

# Make output matrix and headers
bkc_mat = np.zeros((n_fold, n_comp, len(best_feat_sizes), len(targ_dims_flat)))

# RFE (for each best_feat_size, targ_dim, fold)
for best_k_idx, best_k_feats in enumerate(best_feat_sizes):
    tic = time.time()
    for td_idx in np.arange(len(targ_dims_flat)):
        for f in np.arange(n_fold):
            # Scale + PCA
            train_X = pca_feats(feats, n_comp, cv_idx, f)

            # Get training targets
            train_y, _ = repeat_exemplars_y(
                dim_vals[:, td_idx], n_exemp, cv_idx[f][0], cv_idx[f][1]
            )

            # Assign RFE feature rankings
            rfe = RFE(
                estimator=LinearRegression(),
                n_features_to_select=best_k_feats,
                importance_getter="coef_",
            )
            rfe.fit(train_X, train_y)
            bkc_mat[f, :, best_k_idx, td_idx] = rfe.ranking_

    print(
        (
            f"Best {best_k_feats} features run time: "
            + f"{time.time()-tic: .02f} seconds"
        )
    )


# Save output to DataObject data class
data_object = DataObject(
    dim_names=targ_dims,
    model_name=model_name,
    feat_name=feat_name,
    feat_path=dnn_feats,
    n_comp=n_comp,
    n_item=n_item,
    n_exemp=n_exemp,
    n_fold=n_fold,
    cv_idx=cv_idx,
    bkc_mat=bkc_mat,
    bkc_sizes=best_feat_sizes,
    pred_mat=None,
    mod_fit_perm_mat_r2=None,
    mod_fit_perm_mat_adj_r2=None,
)

out_name = "lio_object_%s_%s.pkl" % (model_name, feat_name)
save_data_object(data_object, out_path + out_name)
