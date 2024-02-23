"""
oc_plot_lines_uv.py
Plot line charts (x: component set, y: uniq. var. explained, lines: dimensions)
"""

import os

from functions.functions import (
    PlotObject,
    load_data_object,
    get_ab_pair_idx,
    unpack_ab_variables,
    prep_dim_data,
    kstrat_mod_fit_lio,
    kstrat_incremental_lineplot_unique_variance_together,
    kstrat_incremental_lineplot_unique_variance_together_rescale,
)

# --- User input
pair_object_name = "ab_predictions.pkl"
ab_names = ["clip-vit", "in21k-vit"]  # must match order of pair in pair_object

main_path = os.path.dirname(os.path.abspath(__file__))
pair_object_path = os.path.join(main_path, "results", pair_object_name)
dim_data = os.path.join(
    main_path, "data/behavioural_dimensions/", "selected_dimensions.csv"
)
out_path = os.path.join(main_path, "results/line_plots_uv/")

mod_fit_metric = "r2"

fig_label = ""

model_name_dict = {
    "clip-vit": "CLIP-ViT",
    "in21k-vit": "IN-ViT",
    "in1k-resnext101": "IN-ResNeXt101",
    "in1k-vgg16": "IN-VGG16",
    "in1k-alexnet": "IN-AlexNet",
    "ecoset-vgg16": "EcoSet-VGG16",
    "ecoset-alexnet": "EcoSet-AlexNet",
}

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
plot_object_a = PlotObject(
    model_name=ab_names,
    dim_names=targ_dims,
    mod_fit_metric=None,
    mod_fit_mat=uv_mat_a,
    mod_fit_perm_mat=None,
    bkc_sizes=best_k_sizes,
    out_path=out_path,
    fig_label=fig_label,
)

# Plot unique variance of b
plot_object_b = PlotObject(
    model_name=None,
    dim_names=None,
    mod_fit_metric=None,
    mod_fit_mat=uv_mat_b,
    mod_fit_perm_mat=None,
    bkc_sizes=None,
    out_path=None,
    fig_label=None,
)

if any([True if "alexnet" in i else False for i in ab_names]) or any(
    [True if "ecoset-vgg16" in i else False for i in ab_names]
):
    kstrat_incremental_lineplot_unique_variance_together_rescale(
        plot_object_a, plot_object_b, model_name_dict
    )
else:
    kstrat_incremental_lineplot_unique_variance_together(
        plot_object_a, plot_object_b, model_name_dict
    )
