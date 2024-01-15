"""lio_incremental_model_fit_vis.py
For each dimensions, set of best k components, 
plot model fits as line charts 
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from matplotlib import colormaps
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from functions.functions import (
    load_data_object,
    prep_dim_data,
    mod_fit_lio,
    tr_te_split,
    pca_tr_te,
    get_bkc_idx,
    repeat_exemplars_y,
    save_data_object,
)

# --- User input
data_object_name = "lio_object_clip-vit_eighty_tools.pkl"

main_path = os.path.dirname(os.path.abspath(__file__))
data_object_path = os.path.join(main_path, "results", data_object_name)
dim_data = os.path.join(
    main_path, "data/behavioural_dimensions/", "selected_dimensions.csv"
)
out_path = os.path.join(main_path, "results/incremental_model_fit/")

mod_fit_metric = "r2"
mod_fit_metric = "adj_r2"

fig_label = "A)"

# --- Main

if not os.path.exists(out_path):
    os.makedirs(out_path)

# Load data object, prep dims
data_object = load_data_object(data_object_path)

dim_vals, dim_names = prep_dim_data(dim_data, data_object.dim_names)

# Calculate model fits
mod_fit_mat = mod_fit_lio(
    data_object.pred_mat, dim_vals, data_object.bkc_sizes, mod_fit_metric
)

# Variables
targ_dims_flat = sum(data_object.dim_names, [])
best_k_sizes = data_object.bkc_sizes

# Get knowledge type subtitles
know_type_dict = {"V": "Vision", "M": "Manipulation", "F": "Function"}
know_type_subtitles = [know_type_dict[i[0][0]] for i in data_object.dim_names]

# Get y-axis variables
if mod_fit_metric == "adj_r2":
    y_lims = [-0.2, 1]
    y_label = "Model Fit (adj. $R^2$)"
elif mod_fit_metric == "r2":
    y_lims = [0, 1]
    y_label = "Model Fit ($R^2$)"

# Figure prep
cm = 1 / 2.54
fig, ax = plt.subplots(
    nrows=1,
    ncols=len(know_type_subtitles),
    figsize=(16 * cm, 6 * cm),
    dpi=600,
    sharey=True,
)
fig.set_tight_layout(True)
plt.figtext(0.03, 0.92, fig_label, fontsize=12)

# Sub-plots
targ_dims = data_object.dim_names
for sp in np.arange(len(targ_dims)):
    # Get model fits for current split
    plot_targ_dims = targ_dims[sp]
    plot_td_idx = np.array([targ_dims_flat.index(i) for i in plot_targ_dims])
    plot_mat = mod_fit_mat[:, :, plot_td_idx]

    # Create color maps
    if sp == 0:
        cmap = colormaps["BrBG"]
        cmap_intervals = np.linspace(0.9, 0.6, len(plot_targ_dims))
    elif sp == 1:
        cmap = colormaps["PuOr"]
        cmap_intervals = np.linspace(0.9, 0.6, len(plot_targ_dims))
    elif sp == 2:
        cmap = colormaps["YlOrBr"]
        cmap_intervals = np.linspace(0.6, 0.3, len(plot_targ_dims))

    # Sub-plot set-up
    title_text = f"{know_type_subtitles[sp]}"
    ax[sp].set_ylim(y_lims)
    ax[sp].title.set_text(title_text)
    ax[sp].set_xticks(
        np.arange(1, len(best_k_sizes) + 1), labels=list(np.array(best_k_sizes))
    )
    ax[sp].set_yticks(np.arange(y_lims[0], y_lims[1] + 0.2, 0.2))

    ax[sp].spines["right"].set_visible(False)
    ax[sp].spines["top"].set_visible(False)
    ax[sp].axvline(3, c="grey", linewidth=2, alpha=0.5, linestyle="--")

    if sp == 0:
        ax[sp].set_ylabel(y_label)
        ax[sp].set_xlabel("Best K Components")

    # Plot each dimension
    for d_idx, d_name in enumerate(plot_targ_dims):
        if d_idx == 0:
            ax[sp].errorbar(
                np.arange(1, len(best_k_sizes) + 1),
                np.mean(plot_mat[:, :, d_idx], axis=0),
                yerr=(
                    np.std(plot_mat[:, :, d_idx], axis=0, ddof=1)
                    / np.sqrt(plot_mat.shape[0])
                ),
                label=d_name[-1],
                color=cmap(cmap_intervals[d_idx]),
                linewidth=4,
            )
        else:
            ax[sp].errorbar(
                np.arange(1, len(best_k_sizes) + 1),
                np.mean(plot_mat[:, :, d_idx], axis=0),
                yerr=(
                    np.std(plot_mat[:, :, d_idx], axis=0, ddof=1)
                    / np.sqrt(plot_mat.shape[0])
                ),
                label=d_name[-1],
                color=cmap(cmap_intervals[d_idx]),
                linewidth=2.5,
            )

        ax[sp].legend(
            loc="upper right",
            bbox_to_anchor=(1, 1.1),
            labelspacing=0.1,
            fontsize=5,
            title="Dim.",
            title_fontsize=5,
            frameon=False,
            markerscale=None,
            markerfirst=False,
        )

plt.savefig(
    out_path + f"{data_object.model_name}_{mod_fit_metric}_model_fit_incremental.png"
)
