"""kstrat_best_k_model_fit_vis.py
For each dimension, best k components, 
plot bars
"""
import os
from functions.functions import (
    PlotObject,
    load_data_object,
    prep_dim_data,
    kstrat_mod_fit_lio,
    kstrat_best_k_bar_plot,
    kstrat_best_k_bar_plot_perm,
)

# --- User input
data_object_name = "kstrat_10_object_clip-vit_eighty_tools.pkl"

main_path = os.path.dirname(os.path.abspath(__file__))
data_object_path = os.path.join(main_path, "results", data_object_name)

dim_data = os.path.join(
    main_path, "data/behavioural_dimensions/", "selected_dimensions.csv"
)
out_path = os.path.join(main_path, "results/best_k_model_fit_figs/")

mod_fit_metric = "r2"  # "adj_r2"

fig_label = "A)"

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

# Load data object
data_object = load_data_object(data_object_path)

# Prepare dimensions
dim_vals, _ = prep_dim_data(dim_data, data_object.dim_names)

# Calculate model fits
mod_fit_mat = kstrat_mod_fit_lio(
    data_object.pred_mat, dim_vals, data_object.bkc_sizes, mod_fit_metric
)

# Run incremental line plots
if show_perm == 1:
    # Get mod_fit_perm_mat
    mfpm_name = f"mod_fit_perm_mat_{mod_fit_metric}"
    mod_fit_perm_mat = getattr(data_object, mfpm_name)

    # Instantiate PlotObject
    plot_object = PlotObject(
        model_name=data_object.model_name,
        dim_names=data_object.dim_names,
        mod_fit_metric=mod_fit_metric,
        mod_fit_mat=mod_fit_mat,
        mod_fit_perm_mat=mod_fit_perm_mat,
        bkc_sizes=data_object.bkc_sizes,
        out_path=out_path,
        fig_label=fig_label,
    )

    kstrat_best_k_bar_plot_perm(plot_object, model_name_dict, plot_best_k)

else:
    # Instantiate PlotObject
    plot_object = PlotObject(
        model_name=data_object.model_name,
        dim_names=data_object.dim_names,
        mod_fit_metric=mod_fit_metric,
        mod_fit_mat=mod_fit_mat,
        mod_fit_perm_mat=None,
        bkc_sizes=data_object.bkc_sizes,
        out_path=out_path,
        fig_label=fig_label,
    )

    kstrat_best_k_bar_plot(plot_object, model_name_dict, plot_best_k)