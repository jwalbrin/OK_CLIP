"""lio_incremental_uniq_var_predict.py
For each pair of data_objects in list of tuples 
e.g [(data_object_a, data_object_b), ...]
calculates the best k components predictions from 
the merging of features of each pair in pred_mats"""

import numpy as np
import os
import time
from sklearn.linear_model import LinearRegression

from functions.functions import (
    PairObject,
    load_data_object,
    prep_dim_data,
    match_ab_get_attrs,
    prep_feats_ab,
    repeat_exemplars_y,
    save_data_object,
)

# --- User input
data_object_pairs = [
    ("lio_object_clip-vit_eighty_tools.pkl", "lio_object_in21k-vit_eighty_tools.pkl")
]

main_path = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(main_path, "results/")
dim_data = os.path.join(
    main_path, "data/behavioural_dimensions/", "selected_dimensions.csv"
)


# --- Main

# Initialize pair_object
init_object_paths = [() for i in np.arange(len(data_object_pairs))]
pair_object = PairObject(
    object_paths=init_object_paths,
    pred_mats=[None] * len(data_object_pairs),
)

for abp_idx, abp in enumerate(data_object_pairs):
    # Names
    data_object_name_a = abp[0]
    data_object_name_b = abp[1]

    # Assign object paths to pair_object
    pair_object.object_paths[abp_idx] = (
        os.path.join(main_path, "results", data_object_name_a),
        os.path.join(main_path, "results", data_object_name_b),
    )

    # Load data_objects
    data_object_a = load_data_object(
        os.path.join(main_path, "results", data_object_name_a)
    )
    data_object_b = load_data_object(
        os.path.join(main_path, "results", data_object_name_b)
    )

    # Check and get common variables
    targ_dims, n_comp, n_item, n_exemp, n_fold, best_k_sizes = match_ab_get_attrs(
        data_object_a, data_object_b
    )

    # Load features
    feats_a = np.load(data_object_a.feat_path)
    feats_b = np.load(data_object_b.feat_path)

    # Best k components matrices
    bkc_mat_a = data_object_a.bkc_mat
    bkc_mat_b = data_object_b.bkc_mat

    # CV indices
    cv_idx = data_object_a.cv_idx

    # Delete objects
    del data_object_a, data_object_b

    # Prepare dimensions
    dim_vals, dim_names = prep_dim_data(dim_data, targ_dims)

    # Initialize prediction matrix
    targ_dims_flat = sum(targ_dims, [])
    pred_mat = np.zeros((n_exemp, n_fold, len(best_k_sizes), len(targ_dims_flat)))

    # Main loop (dimension, best_k_components, cv fold)
    for td_idx, td in enumerate(targ_dims_flat):
        tic = time.time()
        for bks_idx, bks in enumerate(best_k_sizes):
            for f in np.arange(n_fold):
                # Get prepared features (a and b processed then concatenate)
                train_X, test_X = prep_feats_ab(
                    feats_a,
                    feats_b,
                    cv_idx,
                    bkc_mat_a,
                    bkc_mat_b,
                    f,
                    bks_idx,
                    td_idx,
                    n_comp,
                )

                # Get train y, repeat across n exemplars
                train_y, _ = repeat_exemplars_y(
                    dim_vals[:, td_idx], n_exemp, cv_idx[f][0], cv_idx[f][1]
                )

                # Model fit, collect prediction
                linreg = LinearRegression()
                linreg.fit(train_X, train_y)
                pred_mat[:, f, bks_idx, td_idx] = linreg.predict(test_X)

        print((f"Predictions for {td} run time: " + f"{time.time()-tic: .02f} seconds"))

    pair_object.pred_mats[abp_idx] = pred_mat

save_data_object(pair_object, out_path + "ab_predictions.pkl")
