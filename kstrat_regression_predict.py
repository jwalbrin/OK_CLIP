""" 
lio_regression_predict.py
Generate CV leave-one-item-out (lio) predictions
and assign as a matrix (pred_mat) to input data_object
pred_mat.shape: n_exemp * n_fold * best_k_sizes * targ_dims
(e.g. 10 * 80 * 7 * 15)
"""
import numpy as np
import os
import time
from sklearn.linear_model import LinearRegression

from functions.functions import (
    load_data_object,
    prep_dim_data,
    tr_te_split,
    pca_tr_te,
    get_bkc_idx,
    repeat_exemplars_y,
    save_data_object,
)

# --- User input
data_object_name = "kstrat_5_object_clip-vit_eighty_tools.pkl"

main_path = os.path.dirname(os.path.abspath(__file__))
data_object_path = os.path.join(main_path, "results", data_object_name)
dim_data = os.path.join(
    main_path, "data/behavioural_dimensions/", "selected_dimensions.csv"
)

# --- Main

# Load data object
data_object = load_data_object(data_object_path)

# Load features
feats = np.squeeze(np.load(data_object.feat_path))

# Prepare dimensions
dim_vals, dim_names = prep_dim_data(dim_data, data_object.dim_names)

# Get variables from data_object
targ_dims_flat = sum(data_object.dim_names, [])
n_exemp = data_object.n_exemp
n_fold = data_object.n_fold
cv_idx = data_object.cv_idx
n_comp = data_object.n_comp
best_k_sizes = data_object.bkc_sizes

# Initialize prediction matrix
n_pred = int(len(dim_vals) * (n_exemp / n_fold))
pred_mat = np.zeros(
    (n_pred, n_fold, len(best_k_sizes), len(targ_dims_flat)), dtype=np.float16
)

# Main loop (dimension, best_k_components, cv fold)
for td_idx, td in enumerate(targ_dims_flat):
    tic = time.time()
    for bks_idx, bks in enumerate(best_k_sizes):
        for f in np.arange(n_fold):
            # Get indices of best k components
            bkc_idx = get_bkc_idx(data_object.bkc_mat, f, bks_idx, td_idx)

            # Split train and test X
            train_X, test_X = tr_te_split(feats, cv_idx[f][0], cv_idx[f][1])

            # PCA fit and transform
            train_X, test_X = pca_tr_te(train_X, test_X, n_comp)

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

# Save pred_mat to data_object
data_object.pred_mat = pred_mat
save_data_object(data_object, data_object_path)
