""" 
lio_regression_predict_things.py
Generate CV leave-one-item-out (lio) predictions for
63 things images (10 exemplars each) and assign as a matrix 
(pred_mat) to input data_object pred_mat.shape: 
n_exemp * n_fold * best_k_sizes * targ_dims
(e.g. 10 * 63 * 7 * 15)
"""
import numpy as np
import os
import time
from sklearn.linear_model import LinearRegression

from functions.functions import (
    DataObject,
    load_data_object,
    load_things_idx,
    prep_dim_data,
    get_cv_idx_orig_things,
    orig_things_prep_tr_te_feats,
    get_bkc_idx,
    repeat_exemplars_y,
    save_data_object,
)

# --- User input
data_object_name = "lio_object_clip-vit_eighty_tools.pkl"

main_path = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(main_path, "results/")
data_object_path = os.path.join(main_path, "results", data_object_name)
dim_data = os.path.join(
    main_path, "data/behavioural_dimensions/", "selected_dimensions.csv"
)
things_feat_path = os.path.join(main_path, "data/things_images/", "features.npy")
things_idx_path = os.path.join(
    main_path, "data/things_images/", "things_images_idx.csv"
)

# --- Main

# Load data object
data_object = load_data_object(data_object_path)

# Load eighty tools features (train), things features (test)
train_feats = np.load(data_object.feat_path)
test_feats = np.load(things_feat_path)

# Load indices for things data
things_idx = load_things_idx(things_idx_path)

# Prepare dimensions
dim_vals, dim_names = prep_dim_data(dim_data, data_object.dim_names)

# Get variables from data_object, things_idx
targ_dims = data_object.dim_names
targ_dims_flat = sum(targ_dims, [])
best_k_sizes = data_object.bkc_sizes
n_comp = data_object.n_comp
n_exemp = data_object.n_exemp
n_item_tr = data_object.n_fold
n_item_te = len(things_idx)
n_fold = n_item_te

# Create cv_idx
cv_idx = get_cv_idx_orig_things(n_exemp, n_item_te, n_item_tr, things_idx)

# Initialize prediction matrix
pred_mat = np.zeros(
    (n_exemp, n_fold, len(best_k_sizes), len(targ_dims_flat)), dtype=np.float16
)

# Main loop (dimension, best_k_components, cv fold)
for td_idx, td in enumerate(targ_dims_flat):
    tic = time.time()
    for bks_idx, bks in enumerate(best_k_sizes):
        for f in np.arange(n_fold):
            # Get indices of best k components
            bkc_idx = get_bkc_idx(data_object.bkc_mat, f, bks_idx, td_idx)

            # Prepare train and test X
            train_X, test_X = orig_things_prep_tr_te_feats(
                train_feats, test_feats, cv_idx, n_comp, f
            )

            # Slice best k components
            train_X = train_X[:, bkc_idx]
            test_X = test_X[:, bkc_idx]

            # Get train y, repeat across n exemplars
            train_y, _ = repeat_exemplars_y(
                dim_vals[:, td_idx], n_exemp, cv_idx[f][0], cv_idx[f][1]
            )

            # Model fit, collect prediction
            linreg = LinearRegression()
            linreg.fit(train_X, train_y)
            pred_mat[:, f, bks_idx, td_idx] = linreg.predict(test_X)

    print((f"Predictions for {td} run time: " + f"{time.time()-tic: .02f} seconds"))

# Save output to DataObject data class
things_object = DataObject(
    dim_names=targ_dims,
    model_name=data_object.model_name,
    feat_name="things_images",
    feat_path=things_feat_path,
    n_comp=n_comp,
    n_item=n_item_te,
    n_exemp=n_exemp,
    n_fold=n_fold,
    cv_idx=cv_idx,
    bkc_mat=None,
    bkc_sizes=best_k_sizes,
    pred_mat=pred_mat,
    mod_fit_perm_mat_r2=None,
    mod_fit_perm_mat_adj_r2=None,
)
save_data_object(
    things_object, out_path + f"lio_object_{data_object.model_name}_things.pkl"
)
