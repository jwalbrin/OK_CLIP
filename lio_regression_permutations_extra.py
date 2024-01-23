""" 
lio_regression_permutations_extra.py
Model fit exemplar-set-wise predictions with n_permuted
dimension(s). Returns mod_fit_perm_mat of size:
n exemp * best k sizes * dimensions * ...
(n permutations + 1 un-permuted test score)
"""

import numpy as np
import os

from functions.functions import (
    load_data_object,
    load_proxy_data,
    mod_fit_lio_extra_perm,
    save_data_object,
)

# --- User input
extra_object_name = "lio_object_clip-vit_extra.pkl"

main_path = os.path.dirname(os.path.abspath(__file__))
extra_object_path = os.path.join(main_path, "results", extra_object_name)
extra_proxy_path = os.path.join(main_path, "data/extra/", "extra_proxy_dimensions.csv")
out_path = os.path.join(main_path, "results/")
dim_data = os.path.join(
    main_path, "data/behavioural_dimensions/", "selected_dimensions.csv"
)

n_perm = 5000
mod_fit_metrics = ["adj_r2", "r2"]

# --- Main

# Load data object
extra_object = load_data_object(extra_object_path)

# Load proxy dimension values (for extra data)
proxy_vals, proxy_names = load_proxy_data(extra_proxy_path)

# Get variables from data_object
n_exemp = extra_object.n_exemp
n_fold = extra_object.n_fold
cv_idx = extra_object.cv_idx
n_comp = extra_object.n_comp
best_k_sizes = extra_object.bkc_sizes

# Calculate permuted dimension fits
for met_idx, met in enumerate(mod_fit_metrics):
    mod_fit_perm_mat = mod_fit_lio_extra_perm(
        extra_object.pred_mat,
        proxy_vals,
        extra_object.bkc_sizes,
        proxy_names,
        n_perm,
        mod_fit_metrics[met_idx],
        n_exemp,
    )

    # Assign to data_object
    if met == "r2":
        extra_object.mod_fit_perm_mat_r2 = mod_fit_perm_mat
    elif met == "adj_r2":
        extra_object.mod_fit_perm_mat_adj_r2 = mod_fit_perm_mat

# Save
save_data_object(extra_object, out_path + extra_object_name)
