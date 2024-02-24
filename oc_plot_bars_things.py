"""
oc_plot_bars_things.py
Plot bar charts (x/bars: dimensions, y: model fit)
"""

import os
from functions.functions import (
    PlotObject,
    load_data_object,
    load_things_idx,
    prep_dim_data,
    mod_fit_things,
    bar_plot_things,
    bar_plot_things_perm,
)

# --- User input
et_object_name = "data_object_clip-vit_eighty_tools.pkl"
things_object_name = "data_object_clip-vit_things.pkl"

main_path = os.path.dirname(os.path.abspath(__file__))
et_object_path = os.path.join(main_path, "results", et_object_name)
things_object_path = os.path.join(main_path, "results", things_object_name)
things_idx_path = os.path.join(main_path, "data/things/", "things_images_idx.csv")
dim_data = os.path.join(
    main_path, "data/behavioural_dimensions/", "selected_dimensions.csv"
)
out_path = os.path.join(main_path, "results/bar_plots/")

mod_fit_metric = "r2"

fig_label = ""

show_perm = 1

plot_best_k = 10

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
et_object = load_data_object(et_object_path)
things_object = load_data_object(things_object_path)

# Load indices for things data
things_idx = load_things_idx(things_idx_path)

# Prepare dimensions
dim_vals, _ = prep_dim_data(dim_data, et_object.dim_names)
dim_vals = dim_vals[things_idx, :]

# Calculate model fits
mod_fit_mat = mod_fit_things(
    things_object.pred_mat, dim_vals, things_object.bkc_sizes, mod_fit_metric
)

# Run incremental line plots
if show_perm == 1:
    # Get mod_fit_perm_mat
    mfpm_name = f"mod_fit_perm_mat_{mod_fit_metric}"
    mod_fit_perm_mat = getattr(things_object, mfpm_name)

    # Instantiate PlotObject
    plot_object = PlotObject(
        model_name=things_object.model_name,
        dim_names=things_object.dim_names,
        mod_fit_metric=mod_fit_metric,
        mod_fit_mat=mod_fit_mat,
        mod_fit_perm_mat=mod_fit_perm_mat,
        bkc_sizes=things_object.bkc_sizes,
        out_path=out_path,
        fig_label=fig_label,
    )

    bar_plot_things_perm(plot_object, plot_best_k)


else:
    # Instantiate PlotObject
    plot_object = PlotObject(
        model_name=things_object.model_name,
        dim_names=things_object.dim_names,
        mod_fit_metric=mod_fit_metric,
        mod_fit_mat=mod_fit_mat,
        mod_fit_perm_mat=None,
        bkc_sizes=things_object.bkc_sizes,
        out_path=out_path,
        fig_label=fig_label,
    )

    bar_plot_things(plot_object, plot_best_k)
