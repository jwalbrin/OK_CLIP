""" 
oc_predictions_extra.py
Generate cross-validated predictions (for each dimension, component set)
"""

import numpy as np
import os
import time
from sklearn.linear_model import LinearRegression

from functions.functions import (
    DataObject,
    load_data_object,
    load_proxy_data,
    prep_dim_data,
    subset_dims_in_proxy,
    get_cv_idx_tr,
    et_extra_prep_tr_feats,
    get_bkc_idx,
    repeat_exemplars_y,
    save_data_object,
)

# --- User input
data_object_name = "data_object_clip-vit_eighty_tools.pkl"

main_path = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(main_path, "results/")
data_object_path = os.path.join(main_path, "results", data_object_name)
dim_data = os.path.join(
    main_path, "data/behavioural_dimensions/", "selected_dimensions.csv"
)
extra_feat_path = os.path.join(main_path, "data/extra/", "features.npy")
extra_proxy_path = os.path.join(main_path, "data/extra/", "extra_proxy_dimensions.csv")

# --- Main

# Load data object
data_object = load_data_object(data_object_path)

# Load eighty tools features (train), "extra" 20 objects' features (test)
train_feats = np.load(data_object.feat_path)
test_feats = np.load(extra_feat_path)

# Load proxy dimesnion values (for extra data)
proxy_vals, proxy_names = load_proxy_data(extra_proxy_path)

# Prepare dimensions, subset by names that are in proxy_names
dim_vals, dim_names = prep_dim_data(dim_data, data_object.dim_names)
dim_vals, dim_names, dim_orig_idx = subset_dims_in_proxy(
    proxy_names, dim_vals, dim_names
)

# Get variables from data_object, things_idx
targ_dims = dim_names
best_k_sizes = data_object.bkc_sizes
n_comp = data_object.n_comp
n_exemp = data_object.n_exemp
n_item_tr = len(dim_vals)
n_item_te = len(proxy_vals)
n_fold = len(data_object.cv_idx)

# Create cv_idx for training data only
cv_idx_tr = get_cv_idx_tr(n_exemp, n_item_tr)

# Initialize prediction matrix
pred_mat = np.zeros(
    (n_fold, n_exemp * n_item_te, len(best_k_sizes), len(dim_names)), dtype=np.float16
)

# Main loop (dimension, best_k_components, cv fold)
for td_idx, td in enumerate(dim_names):
    tic = time.time()
    for bks_idx, bks in enumerate(best_k_sizes):
        for f in np.arange(n_fold):
            # Get indices of best k components
            bkc_idx = get_bkc_idx(data_object.bkc_mat, f, bks_idx, dim_orig_idx[td_idx])

            # Prepare train and test X
            train_X, test_X = et_extra_prep_tr_feats(
                train_feats, test_feats, cv_idx_tr, n_comp, f
            )

            # Slice best k components
            train_X = train_X[:, bkc_idx]
            test_X = test_X[:, bkc_idx]

            # Get train y, repeat across n exemplars
            train_y, _ = repeat_exemplars_y(
                dim_vals[:, td_idx], n_exemp, cv_idx_tr[f], 0
            )

            # Model fit, collect prediction
            linreg = LinearRegression()
            linreg.fit(train_X, train_y)
            pred_mat[f, :, bks_idx, td_idx] = linreg.predict(test_X)

    print((f"Predictions for {td} run time: " + f"{time.time()-tic: .02f} seconds"))

# Save output to DataObject data class
extra_object = DataObject(
    dim_names=targ_dims,
    model_name=data_object.model_name,
    feat_name="extra",
    feat_path=extra_feat_path,
    n_comp=n_comp,
    n_item=n_item_te,
    n_exemp=n_exemp,
    n_fold=None,
    cv_idx=None,
    bkc_mat=None,
    bkc_sizes=best_k_sizes,
    pred_mat=pred_mat,
    mod_fit_perm_mat_r2=None,
    mod_fit_perm_mat_adj_r2=None,
)
save_data_object(
    extra_object, out_path + f"data_object_{data_object.model_name}_extra.pkl"
)
