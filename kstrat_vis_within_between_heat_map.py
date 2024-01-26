""" 
kstrat_vis_within_between_heat_map.py
Create within-between best component overlap matrix for 15 dimensions
For each dim, takes the fold x total component matrices, binarizes 
(1s selected component, 0s elsewhere) and then calculates the overlap with
all dimensions
Note for self comparisons, only the lower triangle is taken (only across folds)
but for between dimension comparisons all values of the (non-symetrical) matrix
are taken
"""
import numpy as np
import os
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

from functions.functions import load_data_object

# --- User input
data_object_name = "kstrat_10_object_clip-vit_eighty_tools.pkl"

main_path = os.path.dirname(os.path.abspath(__file__))
data_object_path = os.path.join(main_path, "results", data_object_name)
out_path = os.path.join(main_path, "results/component_analyses/")

y_labs = [
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "F1",
    "F2",
    "F3",
    "F4",
]

comp_meas = "median"
fig_text = "C)"
best_k_to_show = 10
cmap_str = "viridis"


# --- Main

if not os.path.exists(out_path):
    os.makedirs(out_path)


def conj_func(u, v):
    # Conjunction of vectors
    return np.sum(u * v)


if comp_meas == "mean":

    def avg_mat(ovlp_mat):
        overlap = np.nanmean(ovlp_mat)
        return overlap

elif comp_meas == "median":

    def avg_mat(ovlp_mat):
        overlap = np.nanmedian(ovlp_mat)
        return overlap


# Load data object
data_object = load_data_object(data_object_path)
n_comp = data_object.n_comp
targ_dims_flat = sum(data_object.dim_names, [])
bkc_mat = data_object.bkc_mat
best_k_sizes = data_object.bkc_sizes
bks_idx = np.where(best_k_sizes == [best_k_to_show])[0][0]

# Assign binarized best component matrix to best_bin_mat (for each dimension)
best_bin_mat = np.zeros((bkc_mat.shape[0], bkc_mat.shape[1], len(targ_dims_flat)))
for td_idx, td in enumerate(targ_dims_flat):
    comp_mat = bkc_mat[:, :, bks_idx, td_idx]
    best_bin_mat[:, :, td_idx] = 1 * (comp_mat == 1)

"""For non-diagonal comparison pairs, calculate overlap, 
assign to both u and l triangle cells"""
out_mat = np.zeros((len(targ_dims_flat), len(targ_dims_flat)))
tri_idx = np.tril_indices(len(targ_dims_flat), k=-1)
for pwc in np.arange(tri_idx[0].shape[0]):
    # Assign average component overlap to out_mat
    ovlp_mat = cdist(
        best_bin_mat[:, :, tri_idx[0][pwc]],
        best_bin_mat[:, :, tri_idx[1][pwc]],
        conj_func,
    )
    out_mat[tri_idx[0][pwc], tri_idx[1][pwc]] = avg_mat(ovlp_mat)
    out_mat[tri_idx[1][pwc], tri_idx[0][pwc]] = avg_mat(ovlp_mat)

"""For diagonal comparisons, calculate overlap of the 
lower triangle values from the resulting overlap matrix"""
for d_idx in np.arange(len(targ_dims_flat)):
    # Assign average component overlap to out_mat
    ovlp_mat = cdist(best_bin_mat[:, :, d_idx], best_bin_mat[:, :, d_idx], conj_func)
    lt_ovlp_mat = np.tril(ovlp_mat, k=-1)
    lt_ovlp_mat[lt_ovlp_mat == 0] = "nan"
    out_mat[d_idx, d_idx] = avg_mat(lt_ovlp_mat)

# Plot
cm = 1 / 2.54
fig, ax = plt.subplots(figsize=(13 * cm, 10 * cm), dpi=600)
plt.figtext(0.0, 0.95, fig_text, fontsize=12)

sns.heatmap(
    out_mat,
    xticklabels=False,
    cmap=cmap_str,
    annot=False,
    vmax=best_k_to_show,
    cbar=True,
    cbar_kws={"ticks": [0, 10], "label": "Component Overlap"},
)

ax.set_yticklabels(y_labs, rotation=0, fontsize=12)
ax.set_ylabel("Dimension", labelpad=10, fontsize=12)

plt.savefig(
    out_path + f"best_{best_k_to_show}_component_analysis_within_vs_between.png"
)

# Again, just to get color bar for other figure
fig, ax = plt.subplots(figsize=(13 * cm, 10 * cm), dpi=600)
# ax.title.set_text("%s %s: Component overlap (Best %d of %d Components)" %
#                   (cnn_name, layer_name, best_k_to_show, n_comp))
sns.heatmap(
    out_mat / 10,
    xticklabels=False,
    cmap=cmap_str,
    annot=False,
    vmax=1,
    cbar=True,
    cbar_kws={"ticks": [0, 1], "label": "Selection Frequency"},
)
ax.set_yticklabels(y_labs, rotation=0, fontsize=12)
ax.set_ylabel("Dimension", labelpad=10, fontsize=12)

plt.savefig(
    out_path
    + f"best_{best_k_to_show}_component_analysis_selection_freq_colorbar_hack.png"
)
