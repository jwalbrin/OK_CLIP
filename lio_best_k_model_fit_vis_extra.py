"""lio_best_k_model_fit_vis_extra.py
For each proxy dimension, plot bars 
"""
import os
from functions.functions import (
    PlotObject,
    load_data_object,
    load_proxy_data,
    mod_fit_lio_extra,
    best_k_bar_plot_extra,
    best_k_bar_plot_extra_perm,
)

# --- User input
extra_object_name = "lio_object_clip-vit_extra.pkl"

main_path = os.path.dirname(os.path.abspath(__file__))
extra_object_path = os.path.join(main_path, "results", extra_object_name)
extra_proxy_path = os.path.join(main_path, "data/extra/", "extra_proxy_dimensions.csv")
dim_data = os.path.join(
    main_path, "data/behavioural_dimensions/", "selected_dimensions.csv"
)
out_path = os.path.join(main_path, "results/best_k_model_fit_figs/")

mod_fit_metric = "r2"  # "adj_r2"

fig_label = "A)"

show_perm = 1

plot_best_k = 10

model_name_dict = {"clip-vit": "CLIP-ViT", "in21k-vit": "IN21K-ViT"}

# --- Main

if not os.path.exists(out_path):
    os.makedirs(out_path)

# Load data object, prep dims
extra_object = load_data_object(extra_object_path)

# Load proxy dimension values (for extra data)
proxy_vals, proxy_names = load_proxy_data(extra_proxy_path)

# Calculate model fits
n_exemp = extra_object.n_exemp
mod_fit_mat = mod_fit_lio_extra(
    extra_object.pred_mat, proxy_vals, extra_object.bkc_sizes, mod_fit_metric, n_exemp
)

# Run incremental line plots
if show_perm == 1:
    # Get mod_fit_perm_mat
    mfpm_name = f"mod_fit_perm_mat_{mod_fit_metric}"
    mod_fit_perm_mat = getattr(extra_object, mfpm_name)

    # Instantiate PlotObject
    plot_object = PlotObject(
        model_name=extra_object.model_name,
        dim_names=extra_object.dim_names,
        mod_fit_metric=mod_fit_metric,
        mod_fit_mat=mod_fit_mat,
        mod_fit_perm_mat=mod_fit_perm_mat,
        bkc_sizes=extra_object.bkc_sizes,
        out_path=out_path,
        fig_label=fig_label,
    )

    best_k_bar_plot_extra_perm(plot_object, model_name_dict, plot_best_k)


else:
    # Instantiate PlotObject
    plot_object = PlotObject(
        model_name=extra_object.model_name,
        dim_names=extra_object.dim_names,
        mod_fit_metric=mod_fit_metric,
        mod_fit_mat=mod_fit_mat,
        mod_fit_perm_mat=None,
        bkc_sizes=extra_object.bkc_sizes,
        out_path=out_path,
        fig_label=fig_label,
    )

    best_k_bar_plot_extra(plot_object, model_name_dict, plot_best_k)
