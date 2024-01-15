import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class DataObject:
    """Dataclass object for analysis pipeline"""

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
    bkc_sizes: np.ndarray  # list of best k component sizes


# Save an instance of data_object
def save_data_object(obj, filename):
    with open(filename, "wb") as file:
        pickle.dump(obj, file)


# Load an instance of data_object
def load_data_object(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


def prep_dim_data(dim_data, targ_dims):
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
        dim_vals = np.stack(dim_vals[:, td_idx]).astype(
            None
        )  # convert to regular array
        dim_names = [dim_names[i] for i in td_idx]
        return dim_vals, dim_names

    # Load, reorder, scale dimensions
    dim_names, dim_vals = load_dim_data(dim_data)
    targ_dims_flat = sum(targ_dims, [])
    dim_vals, dim_names = reorder_dim_data(targ_dims_flat, dim_names, dim_vals)
    dim_vals = StandardScaler().fit_transform(dim_vals)
    return dim_vals, dim_names


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


def repeat_exemplars_y(dim_col, n_exemp, train_idx, test_idx):
    """Get y values for given dimension for specified train
    and test values, repeated by n exemplars"""
    train_y = np.repeat(dim_col, n_exemp)[train_idx]
    test_y = np.repeat(dim_col, n_exemp)[test_idx]
    return train_y, test_y


def get_eval_score_func(mod_eval_metric):
    """Get eval_score function and plotting variables for desired metric"""
    if mod_eval_metric == "r2":

        def eval_score(test_y, pred_y, _):
            return r2_score(test_y, pred_y)

    elif mod_eval_metric == "adj_r2":

        def eval_score(test_y, pred_y, best_k_feats):
            return 1 - (1 - r2_score(test_y, pred_y)) * (len(test_y) - 1) / (
                len(test_y) - best_k_feats
            )

    return eval_score


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


# def get_bkc_k_idx(n_best_k_feats, best_k_sizes):
#     """For given k, get corresponding index in
#     best_k_sizes"""
#     bkc_idx = int(np.where(best_k_sizes == n_best_k_feats)[0])
#     return bkc_idx
def get_bkc_idx(bkc_mat, fold, bks_idx, td_idx):
    """Get indices of best k components from bkc_mat
    for a given: fold, k_idx, td_idk"""
    bkc_idx = np.where(bkc_mat[fold, :, bks_idx, td_idx] == 1)[0]
    return bkc_idx


def mod_fit_lio(pred_mat, dim_vals, best_k_sizes, eval_func):
    """Model fit exemplar-set-wise predictions with dimension(s)"""

    def get_eval_score_func(eval_func):
        """Get eval_score function and plotting variables for desired metric"""
        if eval_func == "r2":

            def eval_score(test_y, pred_y, _):
                return r2_score(test_y, pred_y)

        elif eval_func == "adj_r2":

            def eval_score(test_y, pred_y, best_k_feats):
                return 1 - (1 - r2_score(test_y, pred_y)) * (len(test_y) - 1) / (
                    len(test_y) - best_k_feats
                )

        return eval_score

    # Get eval score func
    eval_score = get_eval_score_func(eval_func)

    # Initialize
    n_exemp, _, n_bks, n_targ_dims = pred_mat.shape
    mod_fit_mat = np.zeros((n_exemp, n_bks, n_targ_dims))

    for td in np.arange(n_targ_dims):
        for bks in np.arange(n_bks):
            for e in np.arange(n_exemp):
                mod_fit_mat[e, bks, td] = eval_score(
                    dim_vals[:, td], pred_mat[e, :, bks, td], best_k_sizes[bks]
                )
    return mod_fit_mat


def mod_fit_lio_perm(pred_mat, dim_vals, best_k_sizes, n_perm, eval_func):
    """Model fit exemplar-set-wise predictions with n permuted
    dimension(s)"""

    def get_eval_score_func(eval_func):
        """Get eval_score function and plotting variables for desired metric"""
        if eval_func == "r2":

            def eval_score(test_y, pred_y, _):
                return r2_score(test_y, pred_y)

        elif eval_func == "adj_r2":

            def eval_score(test_y, pred_y, best_k_feats):
                return 1 - (1 - r2_score(test_y, pred_y)) * (len(test_y) - 1) / (
                    len(test_y) - best_k_feats
                )

        return eval_score

    def make_perm_idx(arr, n_perm):
        """n_perm + 1 * dim_vals matrix of shuffled
        indices. Unshuffled indices are first row"""
        perm_idx = np.zeros((n_perm + 1, len(arr)))
        perm_idx[0, :] = arr
        for gp_idx in np.arange(1, n_perm + 1):
            np.random.shuffle(arr)
            perm_idx[gp_idx, :] = arr
        perm_idx = perm_idx.astype("int")
        return perm_idx

    # Get eval score func
    eval_score = get_eval_score_func(eval_func)

    # Initialize
    n_exemp, _, n_bks, n_targ_dims = pred_mat.shape
    mod_fit_perm_mat = np.zeros((n_exemp, n_bks, n_targ_dims, n_perm + 1))

    for td in np.arange(n_targ_dims):
        for bks in np.arange(n_bks):
            for p in np.arange(n_perm):
                # Get permutation indices
                perm_idx = make_perm_idx(np.arange(dim_vals.shape[0]), n_perm)
                for e in np.arange(n_exemp):
                    mod_fit_perm_mat[e, bks, td, p] = eval_score(
                        dim_vals[perm_idx[p, :], td],
                        pred_mat[e, :, bks, td],
                        best_k_sizes[bks],
                    )
    return mod_fit_perm_mat
