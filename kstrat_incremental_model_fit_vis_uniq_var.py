"""lio_incremental_model_fit_vis_uniq_var.py
Load ab_predictions along with each respective
a,b prediction and calculte the unique variance
explained by each a and b
Plots both for a and b as line charts
where x is best k components, 
y is unique variance explained
"""
import os

from functions.functions import (
    PlotObject,
    load_data_object,
    get_ab_pair_idx,
    unpack_ab_variables,
    prep_dim_data,
    kstrat_mod_fit_lio,
    kstrat_incremental_lineplot_unique_variance,
)

# --- User input
pair_object_name = "kstrat_ab_predictions.pkl"
ab_names = ["clip-vit", "in21k-vit"]  # must match order of pair in pair_object

main_path = os.path.dirname(os.path.abspath(__file__))
pair_object_path = os.path.join(main_path, "results", pair_object_name)
dim_data = os.path.join(
    main_path, "data/behavioural_dimensions/", "selected_dimensions.csv"
)
out_path = os.path.join(
    main_path, "results/incremental_model_fit_unique_variance_figs/"
)

mod_fit_metric = "r2"  # "adj_r2"

fig_labels = ["A)", "B)"]

model_name_dict = {"clip-vit": "CLIP-ViT", "in21k-vit": "IN21K-ViT"}

# --- Main

if not os.path.exists(out_path):
    os.makedirs(out_path)

# Load data object, prep dims
pair_object = load_data_object(pair_object_path)

# Unpack variables that are common to a and b
p_idx = get_ab_pair_idx(ab_names, pair_object.object_paths)
targ_dims, n_comp, n_item, n_exemp, n_fold, best_k_sizes = unpack_ab_variables(
    pair_object, p_idx
)

# Prepare dimensions
dim_vals, _ = prep_dim_data(dim_data, targ_dims)

# Calculate model fits for ab pair
mod_fit_mat_ab = kstrat_mod_fit_lio(
    pair_object.pred_mats[p_idx], dim_vals, best_k_sizes, mod_fit_metric
)

# Calculate model fits for a
data_object = load_data_object(pair_object.object_paths[p_idx][0])
mod_fit_mat_a = kstrat_mod_fit_lio(
    data_object.pred_mat, dim_vals, best_k_sizes, mod_fit_metric
)

# Calculate model fits for b
data_object = load_data_object(pair_object.object_paths[p_idx][1])
mod_fit_mat_b = kstrat_mod_fit_lio(
    data_object.pred_mat, dim_vals, best_k_sizes, mod_fit_metric
)

# Delete data_object
del data_object

# Calculate unique variance explained for a, b respectively
uv_mat_a = mod_fit_mat_ab - mod_fit_mat_b
uv_mat_b = mod_fit_mat_ab - mod_fit_mat_a

# Plot unique variance of a
plot_object = PlotObject(
    model_name=ab_names,
    dim_names=targ_dims,
    mod_fit_metric=mod_fit_metric,
    mod_fit_mat=uv_mat_a,
    mod_fit_perm_mat=None,
    bkc_sizes=best_k_sizes,
    out_path=out_path,
    fig_label=fig_labels,
)
kstrat_incremental_lineplot_unique_variance(plot_object, model_name_dict, 0)

# Plot unique variance of b
plot_object = PlotObject(
    model_name=ab_names,
    dim_names=targ_dims,
    mod_fit_metric=mod_fit_metric,
    mod_fit_mat=uv_mat_b,
    mod_fit_perm_mat=None,
    bkc_sizes=best_k_sizes,
    out_path=out_path,
    fig_label=fig_labels,
)
kstrat_incremental_lineplot_unique_variance(plot_object, model_name_dict, 1)
