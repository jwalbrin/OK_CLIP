""" 
lio_regression_permutations.py
Model fit exemplar-set-wise predictions with n_permuted
dimension(s). Returns mod_fit_perm_mat of size:
n exemplar sets * best k sizes * dimensions * ...
(n permutations + 1 un-permuted test score)
"""

import numpy as np
import os

from functions.functions import (
    load_data_object,
    prep_dim_data,
    mod_fit_lio_perm,
    save_data_object,
)

# --- User input
data_object_name = "lio_object_clip-vit_eighty_tools.pkl"
data_object_name = "lio_object_in21k-vit_eighty_tools.pkl"


main_path = os.path.dirname(os.path.abspath(__file__))
data_object_path = os.path.join(main_path, "results", data_object_name)
out_path = os.path.join(main_path, "results/")
dim_data = os.path.join(
    main_path, "data/behavioural_dimensions/", "selected_dimensions.csv"
)

n_perm = 5000
mod_fit_metrics = ["adj_r2", "r2"]

# --- Main

# Load data object
data_object = load_data_object(data_object_path)

# Prepare dimensions
dim_vals, dim_names = prep_dim_data(dim_data, data_object.dim_names)

# Get variables from data_object
targ_dims_flat = sum(data_object.dim_names, [])
n_exemp = data_object.n_exemp
n_fold = data_object.n_fold
cv_idx = data_object.cv_idx
n_comp = data_object.n_comp
best_k_sizes = data_object.bkc_sizes

# Calculate permuted dimension fits
for met_idx, met in enumerate(mod_fit_metrics):
    mod_fit_perm_mat = mod_fit_lio_perm(
        data_object.pred_mat,
        dim_vals,
        data_object.bkc_sizes,
        targ_dims_flat,
        n_perm,
        mod_fit_metrics[met_idx],
    )

    # Assign to data_object
    if met == "r2":
        data_object.mod_fit_perm_mat_r2 = mod_fit_perm_mat
    elif met == "adj_r2":
        data_object.mod_fit_perm_mat_adj_r2 = mod_fit_perm_mat

# Save
save_data_object(data_object, out_path + data_object_name)
