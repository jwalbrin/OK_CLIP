from dataclasses import dataclass

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


@dataclass
class DataObject:
    dim_names: list  # nested list of dimension names
    model_name: str
    feat_name: str
    feat_path: str  # path to input features
    n_comp: int  # number of input components (e.g. 100)
    n_item: int  # number of object identities (e.g. 80)
    n_exemp: int  # exemplars per object (10)
    n_fold: int  # n cv folds
    cv_idx: list  # list containing train/test idx per cv fold
    bkc_mat: np.ndarray  # 4D array of best k components
    bkc_sizes: np.ndarray  # list of tuples for each index best k component pairs


# Save an instance of MyClass
def save_data_object(obj, filename):
    with open(filename, "wb") as file:
        pickle.dump(obj, file)


# Load an instance of MyClass
def load_data_object(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


def load_dim_data(dim_data):
    """Load dimension names and values"""
    df = pd.read_csv(dim_data)
    dim_names = list(df.columns[1:])
    dim_vals = df.iloc[:, 1:].to_numpy()
    return dim_names, dim_vals


def reorder_dim_data(targ_dims, dim_names, dim_vals):
    """Reorder dimension names and values based on target dimension
    ordering"""
    td_idx = [dim_names.index(i) for i in targ_dims]
    dim_vals = np.stack(dim_vals[:, td_idx]).astype(None)  # convert to regular array
    return dim_vals


def custom_cv_split(n_exemp, n_item, n_fold):
    """Create cross-validation splits, specifying
    n_folds, n_items, n_exemplars"""
    te_idx = [
        np.arange(i, i + n_exemp) for i in np.arange(0, n_item * n_exemp, n_exemp)
    ]
    tr_idx = [np.setdiff1d(np.arange(n_item * n_exemp), i) for i in te_idx]
    custom_cv = [i for i in zip(tr_idx, te_idx)]
    return custom_cv


def pca_feats(feats, n_comp, custom_cv, fold):
    """PCA over all samples of the current CV fold"""

    # Subset training items for current fold
    feats_pca = np.copy(feats)
    feats_pca = feats_pca[custom_cv[fold][0], :]

    # Scale + PCA
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("PCA", PCA(n_components=n_comp, svd_solver="full")),
        ]
    )
    feats_pca = pipe.fit_transform(feats_pca)
    return feats_pca


def select_repeat_y(dim_col, n_exemp, train_idx, test_idx):
    """Get y values for given dimension for specified train
    and test values, repeated by n exemplars"""
    train_y = np.repeat(dim_col, n_exemp)[train_idx]
    test_y = np.repeat(dim_col, n_exemp)[test_idx]
    return train_y, test_y


def get_eval_score_function(mod_eval_metric):
    """Get eval_score function and plotting variables for desired metric"""
    if mod_eval_metric == "r2":

        def eval_score(test_y, pred_y, _):
            return r2_score(test_y, pred_y)

        y_label = "R2"
        y_lims = [0, 1]
    elif mod_eval_metric == "r2adj":

        def eval_score(test_y, pred_y, best_k_feats):
            return 1 - (1 - r2_score(test_y, pred_y)) * (len(test_y) - 1) / (
                len(test_y) - best_k_feats
            )

        y_label = "Adj. R2"
        y_lims = [-0.2, 1]
    elif mod_eval_metric == "MSE":

        def eval_score(test_y, pred_y, _):
            return mean_squared_error(test_y, pred_y)

        y_label = "MSE"
        y_lims = [0, 1]
    elif mod_eval_metric == "RMSE":

        def eval_score(test_y, pred_y, _):
            return np.sqrt(mean_squared_error(test_y, pred_y))

        y_label = "RMSE"
        y_lims = [0, 1]
    elif mod_eval_metric == "MAE":

        def eval_score(test_y, pred_y, _):
            return mean_absolute_error(test_y, pred_y)

        y_label = "MAE"
        y_lims = [0, 1]
    return eval_score, y_label, y_lims


# def load_bkf_reordered(bkf_full_path, targ_dims):
#     """Load best k features data:
#     a. data_mat (where dimension ordering is adjusted
#             based on that given by targ_dims)
#     b. Index-value pairs for k feature sets"""
#     bkf = np.load(bkf_full_path)
#     bkf_mat = bkf["out_mat"]
#     bkf_dim_names = list(bkf["out_heads"])
#     bkf_idx_val = bkf["out_bk_idx_val"]

#     # Re-order dimensions by targ_dims
#     td_idx = [bkf_dim_names.index(i) for i in targ_dims]
#     bkf_mat = bkf_mat[:, :, :, td_idx]
#     bkf_dim_names = np.array(bkf_dim_names)[td_idx]
#     return bkf_mat, bkf_idx_val


def tr_te_split(feats, train_idx, test_idx):
    """Copy features and split based on
    train and test indices"""
    train_X = np.copy(feats)
    test_X = np.copy(feats)
    train_X = train_X[train_idx, :]
    test_X = test_X[test_idx, :]
    return train_X, test_X


def pca_tr_te(train_X, test_X, n_comp):
    """Scale and PCA: fit with train_X,
    transform test_X for n components"""
    sc = StandardScaler()
    train_X = sc.fit_transform(train_X)
    test_X = sc.transform(test_X)

    pca = PCA(n_components=n_comp, svd_solver="full")
    train_X = pca.fit_transform(train_X)
    test_X = pca.transform(test_X)
    return train_X, test_X


def mod_fit_bars(data_mat, best_k_feats, targ_dims, y_label, y_lims):
    """Model fit bars (x = dimension, y = mean model fit measure,
    error bars = SEM across exemplar splits)"""
    # Figure, axes set up
    fig, ax = plt.subplots(figsize=(10, 5))
    title_text = f"Model Fits (Best {best_k_feats} Components"
    ax.title.set_text(title_text)
    ax.set_xlabel("Dimension")
    ax.set_xticks(np.arange(1, len(targ_dims) + 1), labels=targ_dims, rotation=90)
    ax.set_ylabel(f"Model Fit ({y_label})")
    ax.set_ylim(y_lims)

    # Plot data
    ax.bar(
        np.arange(1, len(targ_dims) + 1),
        np.mean(data_mat, axis=0),
        yerr=(np.std(data_mat, axis=0, ddof=1) / np.sqrt(data_mat.shape[0])),
    )
