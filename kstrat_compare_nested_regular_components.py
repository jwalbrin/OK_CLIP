""" 
kstrat_compare_nested_regular_components.py
Compare best k coponent solutions for nested and 
un-nested component selection
For each best k and dimension, get binarized
10 fold x 100 component matrix and calculates an error score:
% of all matrix entries (1000) where the 2 solutions
are not the same (i.e. where a given component was selected
by one solution, but not the other)
"""
import numpy as np
import os
import csv

from functions.functions import (
    load_data_object,
)

# --- User input
data_object_name_regular = "kstrat_10_object_clip-vit_eighty_tools.pkl"
data_object_name_nested = "kstrat_10_nested_object_clip-vit_eighty_tools.pkl"
main_path = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(main_path, "results")
# --- Main

# Load data object
data_object_regular = load_data_object(
    os.path.join(main_path, "results", data_object_name_regular)
)
data_object_nested = load_data_object(
    os.path.join(main_path, "results", data_object_name_nested)
)

# Load bkc_mats
bkc_mat_reg = data_object_regular.bkc_mat
bkc_mat_nest = data_object_nested.bkc_mat

# Calculate errors between 2 bkc_mats
err_mat = np.zeros(bkc_mat_reg.shape[2:])
for bks_idx, bks in enumerate(data_object_regular.bkc_sizes):
    for td_idx, td in enumerate(sum(data_object_regular.dim_names, [])):
        # Binarize nested bkc_mat, sums components for nested folds
        sum_mat = bkc_mat_nest[bkc_mat_nest > 1] = 0
        sum_mat = np.sum(bkc_mat_nest[:, :, :, bks_idx, td_idx], axis=1)

        # Descending order components by highest sum, create binarized matrix
        # (1s for k sum-best components, zeros elsewhere)
        bkc_idx_mat = np.flip(np.argsort(sum_mat, axis=1), axis=1)[:, :bks]
        bkc_bin_mat_nest = np.zeros(sum_mat.shape, dtype=np.int8)
        for r in np.arange(len(bkc_idx_mat)):
            bkc_bin_mat_nest[r, bkc_idx_mat[r, :]] = np.ones(bkc_idx_mat.shape[1])

        # Binarize regular bkc
        bkc_bin_mat_reg = bkc_mat_reg[:, :, bks_idx, td_idx]
        bkc_bin_mat_reg[bkc_bin_mat_reg > 1] = 0

        # Calculate error percentage (i.e. cells where best components don't overlap)
        mat_add = bkc_bin_mat_reg + bkc_bin_mat_nest
        errors = sum(sum(mat_add == 1))
        errors_percent = errors / (mat_add.shape[0] * mat_add.shape[1]) * 100
        err_mat[bks_idx, td_idx] = errors_percent

# Write err_mat to csv
headers = sum(data_object_regular.dim_names, [])
with open(
    os.path.join(out_path, "reg_nest_component_errors.csv"), mode="w", newline=""
) as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(err_mat)
