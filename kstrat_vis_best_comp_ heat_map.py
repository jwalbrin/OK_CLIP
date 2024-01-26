""" 
kstrat_vis_best_comp_ heat_maps.py
For each dimension, visualize: 
1) Binarized best components (based on sum across folds)
2) Best components (scaled by frequency across folds)
"""
import numpy as np
import os
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from functions.functions import load_data_object

# --- User input
data_object_name = "kstrat_10_object_clip-vit_eighty_tools.pkl"

main_path = os.path.dirname(os.path.abspath(__file__))
data_object_path = os.path.join(main_path, "results", data_object_name)
out_path = os.path.join(main_path, "results/component_analyses/")

y_labs = ["1", "2", "3", "4", "5", "1", "2", "3", "4", "5", "6", "1", "2", "3", "4"]

best_k_to_show = 10
cmap_str = "viridis"

fig_text = "B)"

# --- Main

if not os.path.exists(out_path):
    os.makedirs(out_path)


def component_measure_freq(bkc_mat, bks_idx, td_idx, best_k_to_show):
    # Get scaled values for best k components, all else zeros
    cm_mat = np.sum(bkc_mat[:, :, bks_idx, td_idx] == 1, axis=0)
    cm_mat = cm_mat / bkc_mat.shape[0]
    cm_best_idx = np.sort(np.flip(np.argsort(cm_mat))[:best_k_to_show])
    cm_worst_idx = np.sort(np.argsort(cm_mat)[:-best_k_to_show])
    cm_mat[cm_worst_idx] = 0
    return cm_mat


def component_measure_bin(bkc_mat, bks_idx, td_idx, best_k_to_show):
    cm_mat = np.sum(bkc_mat[:, :, bks_idx, td_idx] == 1, axis=0)
    cm_best_idx = np.sort(np.flip(np.argsort(cm_mat))[:best_k_to_show])
    cm_mat = np.zeros(cm_mat.shape, dtype="int_")
    cm_mat[cm_best_idx] = 1
    return cm_mat


# Load data object
data_object = load_data_object(data_object_path)
n_comp = data_object.n_comp
targ_dims_flat = sum(data_object.dim_names, [])
bkc_mat = data_object.bkc_mat
best_k_sizes = data_object.bkc_sizes
bks_idx = np.where(best_k_sizes == [best_k_to_show])[0][0]

# Create component measure matrices (per dimension)
freq_mat = np.zeros((len(targ_dims_flat), n_comp))
bin_mat = np.zeros((len(targ_dims_flat), n_comp))
for td_idx, td in enumerate(targ_dims_flat):
    freq_mat[td_idx, :] = component_measure_freq(
        bkc_mat, bks_idx, td_idx, best_k_to_show
    )
    bin_mat[td_idx, :] = component_measure_bin(bkc_mat, bks_idx, td_idx, best_k_to_show)

# --- Plot best K heat maps
# Label and subplot info
x_labels = list(np.concatenate((np.array([1, 5]), np.arange(10, 101, 10))))
y_labels = np.array(y_labs)
sub_idx = [np.arange(5), np.arange(5, 11), np.arange(11, 15)]
sub_title = ["Vision", "Manipulation", "Function"]

# Component frequency scaled
cm = 1 / 2.54

fig, axes = plt.subplots(
    3,
    1,
    sharex=True,
    figsize=(15 * cm, 12 * cm),
    dpi=600,
    gridspec_kw={"height_ratios": [5 / 6, 1, 4 / 6]},
)

plt.subplots_adjust(hspace=0.5)
plt.figtext(0.04, 0.95, fig_text, fontsize=12)

# X ticks centred on heat map cells
tick_offset = 0.5
for a_idx, a in enumerate(axes):
    im = sns.heatmap(
        freq_mat[sub_idx[a_idx], :],
        yticklabels=y_labels[sub_idx[a_idx]],
        xticklabels=x_labels,
        vmin=0,
        vmax=1,
        cmap=cmap_str,
        ax=a,
        cbar=False,
    )

    a.set_yticklabels(a.get_yticklabels(), rotation=0)
    a.set_title(sub_title[a_idx])
    if a_idx == len(axes) - 1:
        a.set_xticks(
            np.concatenate((np.array([0, 4]), np.arange(9, 100, 10))) + tick_offset
        )
        a.set_xticklabels(x_labels, rotation=0)
        a.set_xlabel("Component", labelpad=10)
    else:
        a.set_xticks([])

    if a_idx == 1:
        a.set_ylabel("Dimension", labelpad=10)

    a.axvline(
        4 + tick_offset, color="magenta", linewidth=1.5, linestyle="--", alpha=0.75
    )
    a.axvline(
        9 + tick_offset, color="magenta", linewidth=1.5, linestyle="--", alpha=0.75
    )

plt.savefig(out_path + f"best_{best_k_to_show}_component_analysis_frequency_scaled.png")

# --- Plot Colorbar only
# Component frequency scaled
fig = plt.figure(figsize=(5 * cm, 5 * cm), dpi=600)
ax = fig.add_axes([0.05, 0.05, 0.05, 0.9])

plt.rc("font", size=12)
cb = mpl.colorbar.ColorbarBase(
    ax, orientation="vertical", cmap=cmap_str, ticks=(0, 1), label="Selection Frequency"
)
plt.savefig(
    out_path + f"best_{best_k_to_show}_component_analysis_frequency_scaled_colorbar.png"
)
