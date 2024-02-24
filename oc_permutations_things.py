""" 
oc_permutations_things.py
Permutations (model fits with shuffled dimensions)
"""

import os

from functions.functions import (
    load_data_object,
    load_things_idx,
    prep_dim_data,
    mod_fit_things_perm,
    save_data_object,
)

# --- User input
things_object_name = "data_object_clip-vit_things.pkl"

main_path = os.path.dirname(os.path.abspath(__file__))
things_object_path = os.path.join(main_path, "results", things_object_name)
things_idx_path = os.path.join(main_path, "data/things/", "things_images_idx.csv")
out_path = os.path.join(main_path, "results/")
dim_data = os.path.join(
    main_path, "data/behavioural_dimensions/", "selected_dimensions.csv"
)

n_perm = 5000
mod_fit_metrics = ["adj_r2", "r2"]

# --- Main

# Load data object
things_object = load_data_object(things_object_path)

# Load indices for things data
things_idx = load_things_idx(things_idx_path)

# Prepare dimensions, subset dim_vals by things indices
dim_vals, dim_names = prep_dim_data(dim_data, things_object.dim_names)
dim_vals = dim_vals[things_idx, :]

# Get variables from things_object
targ_dims_flat = sum(things_object.dim_names, [])
n_exemp = things_object.n_exemp
n_fold = things_object.n_fold
cv_idx = things_object.cv_idx
n_comp = things_object.n_comp
best_k_sizes = things_object.bkc_sizes

# Calculate permuted dimension fits
for met_idx, met in enumerate(mod_fit_metrics):
    mod_fit_perm_mat = mod_fit_things_perm(
        things_object.pred_mat,
        dim_vals,
        things_object.bkc_sizes,
        targ_dims_flat,
        n_perm,
        mod_fit_metrics[met_idx],
    )

    # Assign to things_object
    if met == "r2":
        things_object.mod_fit_perm_mat_r2 = mod_fit_perm_mat
    elif met == "adj_r2":
        things_object.mod_fit_perm_mat_adj_r2 = mod_fit_perm_mat

# Save
save_data_object(things_object, out_path + things_object_name)
