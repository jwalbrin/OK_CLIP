import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import time

from dataclasses import dataclass
from matplotlib import colormaps
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


@dataclass
class DataObject:
    """Dataclass object for analysis pipeline"""

    dim_names: list[list]  # nested list of dimension names
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
    pred_mat: np.ndarray
    mod_fit_perm_mat_r2: np.ndarray
    mod_fit_perm_mat_adj_r2: np.ndarray


@dataclass
class PlotObject:
    """Object for plotting"""

    model_name: str
    dim_names: list[list]
    mod_fit_metric: str
    mod_fit_mat: np.ndarray
    mod_fit_perm_mat: np.ndarray
    bkc_sizes: np.ndarray
    out_path: str
    fig_label: str


@dataclass
class PairObject:
    """Paired object data"""

    object_paths: list[tuple]
    pred_mats: list[np.ndarray]
    variables: list[dict]


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
    dim_vals = np.stack(dim_vals[:, td_idx]).astype(None)
    dim_names = [dim_names[i] for i in td_idx]
    return dim_vals, dim_names


def prep_dim_data(dim_data, targ_dims):
    """Load, reorder, scale dimensions"""

    dim_names, dim_vals = load_dim_data(dim_data)
    targ_dims_flat = sum(targ_dims, [])
    dim_vals, dim_names = reorder_dim_data(targ_dims_flat, dim_names, dim_vals)
    dim_vals = StandardScaler().fit_transform(dim_vals)
    return dim_vals, dim_names


def kstrat_cv_split(n_exemp, n_item, n_fold):
    """Create cross-validation splits, specifying
    k_folds and given class labels"""
    class_labels = np.repeat(np.arange(n_item), n_exemp)
    skf = StratifiedKFold(n_splits=n_fold)
    cv_idx = list(skf.split(X=np.zeros(len(class_labels)), y=class_labels))
    return cv_idx


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
    and test splits, repeated by n exemplars"""
    train_y = np.repeat(dim_col, n_exemp)[train_idx]
    test_y = np.repeat(dim_col, n_exemp)[test_idx]
    return train_y, test_y


def save_data_object(obj, filename):
    with open(filename, "wb") as file:
        pickle.dump(obj, file)


def load_data_object(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


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

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_comp, svd_solver="full")),
        ]
    )

    # Fit and transform
    train_X = pipe.fit_transform(train_X)
    test_X = pipe.transform(test_X)
    return train_X, test_X


def get_bkc_idx(bkc_mat, fold, bks_idx, td_idx):
    """Get indices of best k components from bkc_mat
    for a given: fold, k_idx, td_idk"""
    bkc_idx = np.where(bkc_mat[fold, :, bks_idx, td_idx] == 1)[0]
    return bkc_idx


def match_ab_get_attrs(data_object_a, data_object_b):
    """
    Check if the attributes of two data objects match.

    Args:
        data_object_a (object): The first data object.
        data_object_b (object): The second data object.

    Raises:
        ValueError: If any of the key variables don't match in both
        `data_object_a` and `data_object_b`.
    """
    attrs_a = vars(data_object_a)
    attrs_b = vars(data_object_b)

    conditions = (
        attrs_a["dim_names"] != attrs_b["dim_names"],
        attrs_a["n_comp"] != attrs_b["n_comp"],
        attrs_a["n_item"] != attrs_b["n_item"],
        attrs_a["n_exemp"] != attrs_b["n_exemp"],
        attrs_a["n_fold"] != attrs_b["n_fold"],
        attrs_a["bkc_sizes"] != attrs_b["bkc_sizes"],
    )

    if sum(sum(conditions)) > 0:
        raise ValueError("Key variables don't match in both a and b")
    else:
        targ_dims = attrs_a["dim_names"]
        n_comp = attrs_a["n_comp"]
        n_item = attrs_a["n_item"]
        n_exemp = attrs_a["n_exemp"]
        n_fold = attrs_a["n_fold"]
        best_k_sizes = attrs_a["bkc_sizes"]
        return targ_dims, n_comp, n_item, n_exemp, n_fold, best_k_sizes


def prep_feats_ab(
    feats_a, feats_b, cv_idx, bkc_mat_a, bkc_mat_b, fold, bks_idx, td_idx, n_comp
):
    """
    Prepare the features for models a and b.

    Args:
        bkc_mat_a (numpy.ndarray): The matrix of features for dataset A.
        bkc_mat_b (numpy.ndarray): The matrix of features for dataset B.
        fold (int): The fold number for cross-validation.
        bks_idx (int): The index of the best k components.
        td_idx (int): The index of the dataset.
        n_comp (int): The number of components for PCA.

    Returns:
        tuple: A tuple containing the prepared training and testing features.
            train_X (numpy.ndarray): The training features.
            test_X (numpy.ndarray): The testing features.
    """
    # Get indices of best k components
    bkc_idx_a = get_bkc_idx(bkc_mat_a, fold, bks_idx, td_idx)
    bkc_idx_b = get_bkc_idx(bkc_mat_b, fold, bks_idx, td_idx)

    # Split train and test X
    train_X_a, test_X_a = tr_te_split(feats_a, cv_idx[fold][0], cv_idx[fold][1])
    train_X_b, test_X_b = tr_te_split(feats_b, cv_idx[fold][0], cv_idx[fold][1])

    # PCA fit and transform
    train_X_a, test_X_a = pca_tr_te(train_X_a, test_X_a, n_comp)
    train_X_b, test_X_b = pca_tr_te(train_X_b, test_X_b, n_comp)

    # Slice best k components
    train_X_a = train_X_a[:, bkc_idx_a]
    test_X_a = test_X_a[:, bkc_idx_a]
    train_X_b = train_X_b[:, bkc_idx_b]
    test_X_b = test_X_b[:, bkc_idx_b]

    # Combine feats
    train_X = np.concatenate((train_X_a, train_X_b), axis=1)
    test_X = np.concatenate((test_X_a, test_X_b), axis=1)

    return train_X, test_X


def load_things_idx(things_idx_path):
    things_idx = pd.read_csv(things_idx_path)
    things_idx = np.array(things_idx.things_idx)
    return things_idx


def get_cv_idx_things_te(n_exemp, n_item_te):
    """Get cv indices for things testing data only"""
    cv_idx_te = [
        np.arange(i, n_exemp * n_item_te, n_exemp) for i in np.arange(0, n_exemp)
    ]
    return cv_idx_te


def et_things_prep_tr_te_feats(
    train_feats, test_feats, cv_idx_tr, cv_idx_te, fold_tr, fold_te, n_comp
):
    """Prepare features for train (eighty tools) and
    test (things) data"""
    # Subset train and test feats
    feats_tr = np.copy(train_feats)
    feats_te = np.copy(test_feats)
    feats_tr = feats_tr[cv_idx_tr[fold_tr], :]
    feats_te = feats_te[cv_idx_te[fold_te], :]

    # Scale and PCA
    feats_tr, feats_te = pca_tr_te(feats_tr, feats_te, n_comp)
    return feats_tr, feats_te


def load_proxy_data(extra_proxy_path):
    """Load behavioural proxy data, scale it"""
    proxy_vals = pd.read_csv(extra_proxy_path)
    proxy_names = list(proxy_vals.columns[1:])
    proxy_vals = np.array(proxy_vals[["V_DL_1", "M_DL_1", "F_DL_1"]])
    proxy_vals = StandardScaler().fit_transform(proxy_vals)
    return proxy_vals, proxy_names


def subset_dims_in_proxy(proxy_names, dim_vals, dim_names):
    """Subset and reorder dim_vals and dim_names to match order of proxies and
    indices for dimensions in original full set of dimensions"""
    dim_orig_idx = np.array([dim_names.index(i) for i in proxy_names if i in dim_names])
    dim_vals = dim_vals[:, dim_orig_idx]
    dim_names = proxy_names
    return dim_vals, dim_names, dim_orig_idx


def get_cv_idx_tr(n_exemp, n_item_tr):
    """Get cv indices for eighty tools training data only"""
    temp_idx = [
        np.arange(i, i + n_exemp) for i in np.arange(0, n_item_tr * n_exemp, n_exemp)
    ]
    cv_idx_tr = [np.setdiff1d(np.arange(n_item_tr * n_exemp), i) for i in temp_idx]
    return cv_idx_tr


def get_cv_idx_things_te(n_exemp, n_item_te):
    """Get cv indices for things testing data only"""
    cv_idx_te = [
        np.arange(i, n_exemp * n_item_te, n_exemp) for i in np.arange(0, n_exemp)
    ]
    return cv_idx_te


def et_extra_prep_tr_feats(train_feats, test_feats, cv_idx_tr, n_comp, fold):
    """Prepare feats for eighty tools and extra data.
    Fold-wise eighty tool samples are used as training,
    all extra features are used for testing"""
    # Subset train feats, get all test feats
    feats_tr = np.copy(train_feats)
    feats_te = np.copy(test_feats)
    feats_tr = feats_tr[cv_idx_tr[fold], :]

    # Scale (cross-fit)
    sc = StandardScaler()
    feats_tr = sc.fit_transform(feats_tr)
    feats_te = sc.transform(feats_te)

    # PCA
    pca = PCA(n_components=n_comp, svd_solver="full")
    feats_tr = pca.fit_transform(feats_tr)
    feats_te = pca.transform(feats_te)
    return feats_tr, feats_te


def mod_fit_perm(pred_mat, dim_vals, best_k_sizes, targ_dims, n_perm, eval_func):
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
    _, n_fold, n_bks, n_targ_dims = pred_mat.shape
    mod_fit_perm_mat = np.zeros(
        (n_fold, n_bks, n_targ_dims, n_perm + 1), dtype=np.float16
    )

    tic = time.time()
    for td in np.arange(n_targ_dims):
        for bks in np.arange(n_bks):
            # Get permutation indices
            perm_idx = make_perm_idx(np.arange(dim_vals.shape[0]), n_perm)
            for p in np.arange(len(perm_idx)):
                for f in np.arange(n_fold):
                    mod_fit_perm_mat[f, bks, td, p] = eval_score(
                        dim_vals[perm_idx[p, :], td],
                        pred_mat[:, f, bks, td],
                        best_k_sizes[bks],
                    )
                if p % int(n_perm / 10) == 0:
                    print(
                        f"{targ_dims[td]} {best_k_sizes[bks]} components: {p + 1} permutations done.\n"
                        + f"Total run time: {time.time()-tic: .02f} seconds"
                    )
    return mod_fit_perm_mat


def mod_fit_things_perm(pred_mat, dim_vals, best_k_sizes, targ_dims, n_perm, eval_func):
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
    n_fold_tr, n_fold_te, _, n_bks, n_targ_dims = pred_mat.shape
    mod_fit_perm_mat = np.zeros(
        (n_fold_tr, n_fold_te, n_bks, n_targ_dims, n_perm + 1), dtype=np.float16
    )

    tic = time.time()
    for td in np.arange(n_targ_dims):
        for bks in np.arange(n_bks):
            # Get permutation indices
            perm_idx = make_perm_idx(np.arange(dim_vals.shape[0]), n_perm)
            for p in np.arange(len(perm_idx)):
                for tr_f in np.arange(n_fold_tr):
                    for te_f in np.arange(n_fold_te):
                        mod_fit_perm_mat[tr_f, te_f, bks, td, p] = eval_score(
                            dim_vals[perm_idx[p, :], td],
                            pred_mat[tr_f, te_f, :, bks, td],
                            best_k_sizes[bks],
                        )
                if p % int(n_perm / 10) == 0:
                    print(
                        f"{targ_dims[td]} {best_k_sizes[bks]} components: {p + 1} permutations done.\n"
                        + f"Total run time: {time.time()-tic: .02f} seconds"
                    )
    # Mean across test folds
    mod_fit_perm_mat = np.nanmean(mod_fit_perm_mat, axis=1)

    return mod_fit_perm_mat


def mod_fit_extra_perm(
    pred_mat, proxy_vals, best_k_sizes, targ_dims, n_perm, eval_func, n_exemp
):
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

    def make_perm_idx_extra(arr, n_perm, n_exemp):
        """n_perm + 1 * (proxy_vals * n_exemp) matrix of
        shuffled indices. Unshuffled indices are first row.
        Shuffles are stratified (i.e. all 10 consecutive samples
        (exemplars) receive the same suffled index)"""
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
    n_fold, _, n_bks, n_targ_dims = pred_mat.shape
    mod_fit_perm_mat = np.zeros(
        (n_fold, n_bks, n_targ_dims, n_perm + 1), dtype=np.float16
    )

    tic = time.time()
    for td in np.arange(n_targ_dims):
        for bks in np.arange(n_bks):
            # Get permutation indices
            perm_idx = make_perm_idx_extra(
                np.arange(proxy_vals.shape[0]), n_perm, n_exemp
            )
            for p in np.arange(len(perm_idx)):
                for f in np.arange(n_fold):
                    mod_fit_perm_mat[f, bks, td, p] = eval_score(
                        np.repeat(proxy_vals[perm_idx[p, :], td], n_exemp),
                        pred_mat[f, :, bks, td],
                        best_k_sizes[bks],
                    )
                if p % int(n_perm / 10) == 0:
                    print(
                        f"{targ_dims[td]} {best_k_sizes[bks]} components: {p + 1} permutations done.\n"
                        + f"Total run time: {time.time()-tic: .02f} seconds"
                    )
    return mod_fit_perm_mat


def mod_fit(pred_mat, dim_vals, best_k_sizes, eval_func):
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
    n_pred, n_fold, n_bks, n_targ_dims = pred_mat.shape
    mod_fit_mat = np.zeros((n_fold, n_bks, n_targ_dims))

    for td in np.arange(n_targ_dims):
        for bks in np.arange(n_bks):
            for f in np.arange(n_fold):
                mod_fit_mat[f, bks, td] = eval_score(
                    np.repeat(dim_vals[:, td], int((n_pred / len(dim_vals)))),
                    pred_mat[:, f, bks, td],
                    best_k_sizes[bks],
                )
    return mod_fit_mat


def line_plot(plot_object, model_name_dict):
    """Line plot subplots (one per knowledge type)
    where x = best_k_components, y = exemplar-averaged model fits"""

    # Unpack plot_object variables
    (
        model_name,
        targ_dims,
        mod_fit_metric,
        mod_fit_mat,
        _,
        best_k_sizes,
        out_path,
        fig_label,
    ) = plot_object.__dict__.values()

    # Get knowledge type subtitles
    know_type_dict = {"V": "Vision", "M": "Manipulation", "F": "Function"}
    subtitles = [know_type_dict[i[0][0]] for i in targ_dims]

    # Get y-axis variables
    if mod_fit_metric == "adj_r2":
        y_label = "Model Fit (adj. $R^2$)"
    elif mod_fit_metric == "r2":
        y_label = "Model Fit ($R^2$)"

    # Get slice indices to remove "problematic exemplars" and y-variables
    if model_name == "in1k-alexnet":
        slice_idx = np.setdiff1d(np.arange(len(mod_fit_mat)), 7)
        y_lims = [-1, 1.2]
        y_int = 0.5
    elif model_name == "ecoset-alexnet":
        slice_idx = np.setdiff1d(np.arange(len(mod_fit_mat)), 4)
        y_lims = [-1, 1.2]
        y_int = 0.5
    elif model_name == "ecoset-vgg16":
        slice_idx = np.arange(len(mod_fit_mat))
        y_lims = [-1, 1.2]
        y_int = 0.5
    else:
        slice_idx = np.arange(len(mod_fit_mat))
        y_lims = [0, 1.2]
        y_int = 0.2

    # Slice all but the 8th index (i.e. problematic exemplar set)
    mod_fit_mat = mod_fit_mat[slice_idx, :, :]

    # Figure prep
    cm = 1 / 2.54
    fig, ax = plt.subplots(
        nrows=1,
        ncols=len(subtitles),
        figsize=(18 * cm, 7 * cm),
        dpi=600,
        sharey=True,
    )
    fig.suptitle(f"{model_name_dict[model_name]}", fontsize=14)
    plt.subplots_adjust(top=0.8, bottom=0.2, wspace=0.25, left=0.11, right=0.95)
    plt.figtext(0.028, 0.92, fig_label, fontsize=14)

    # Sub-plots
    targ_dims_flat = sum(targ_dims, [])
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
        title_text = f"{subtitles[sp]}"
        ax[sp].set_ylim(y_lims)
        ax[sp].set_title(title_text, fontsize=14)
        ax[sp].set_xticks(
            np.arange(1, len(best_k_sizes) + 1), labels=list(np.array(best_k_sizes))
        )
        ax[sp].set_yticks(np.arange(y_lims[0], y_lims[1] + 0.1, y_int))
        ax[sp].spines["right"].set_visible(False)
        ax[sp].spines["top"].set_visible(False)

        if model_name == "clip-vit":
            ax[sp].axvline(3, 0, 0.59, c="grey", linewidth=2, alpha=0.5, linestyle="--")

        if sp == 0:
            ax[sp].set_ylabel(y_label, fontsize=14)
            ax[sp].set_xlabel("Best K Components", fontsize=14)

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
            bbox_to_anchor=(1.2, 1),
            labelspacing=0.1,
            fontsize=6,
            title="Dim.",
            title_fontsize=5,
            frameon=False,
            markerscale=None,
            markerfirst=False,
        )

    plt.savefig(out_path + f"{model_name}_{mod_fit_metric}_model_fit_lines.png")


def line_plot_perm(plot_object, model_name_dict):
    """Line plot subplots (one per knowledge type)
    where x = best_k_components, y = exemplar-averaged model fits
    along with permutation significance"""

    # Unpack plot_object variables
    (
        model_name,
        targ_dims,
        mod_fit_metric,
        mod_fit_mat,
        mod_fit_perm_mat,
        best_k_sizes,
        out_path,
        fig_label,
    ) = plot_object.__dict__.values()

    # Get knowledge type subtitles
    know_type_dict = {"V": "Vision", "M": "Manipulation", "F": "Function"}
    subtitles = [know_type_dict[i[0][0]] for i in targ_dims]

    # Get y-axis variables
    if mod_fit_metric == "adj_r2":
        y_label = "Model Fit (adj. $R^2$)"
    elif mod_fit_metric == "r2":
        y_label = "Model Fit ($R^2$)"

    # Get slice indices to remove "problematic exemplars" and y-variables
    if model_name == "in1k-alexnet":
        slice_idx = np.setdiff1d(np.arange(len(mod_fit_mat)), 7)
        y_lims = [-1, 1.2]
        y_int = 0.5
    elif model_name == "ecoset-alexnet":
        slice_idx = np.setdiff1d(np.arange(len(mod_fit_mat)), 4)
        y_lims = [-1, 1.2]
        y_int = 0.5
    elif model_name == "ecoset-vgg16":
        slice_idx = np.arange(len(mod_fit_mat))
        y_lims = [-1, 1.2]
        y_int = 0.5
    else:
        slice_idx = np.arange(len(mod_fit_mat))
        y_lims = [0, 1.2]
        y_int = 0.2

    # Slice all but the 8th index (i.e. problematic exemplar set)
    mod_fit_mat = mod_fit_mat[slice_idx, :, :]
    mod_fit_perm_mat = mod_fit_perm_mat[slice_idx, :, :, :]

    # Figure prep
    cm = 1 / 2.54
    fig, ax = plt.subplots(
        nrows=1,
        ncols=len(subtitles),
        figsize=(18 * cm, 7 * cm),
        dpi=600,
        sharey=True,
    )
    fig.suptitle(f"{model_name_dict[model_name]}", fontsize=14)
    plt.subplots_adjust(top=0.8, bottom=0.2, wspace=0.25, left=0.11, right=0.95)
    plt.figtext(0.028, 0.92, fig_label, fontsize=14)

    # Sub-plots
    targ_dims_flat = sum(targ_dims, [])
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
        title_text = f"{subtitles[sp]}"
        ax[sp].set_ylim(y_lims)
        ax[sp].set_title(title_text, fontsize=14)
        ax[sp].set_xticks(
            np.arange(1, len(best_k_sizes) + 1), labels=list(np.array(best_k_sizes))
        )
        ax[sp].set_yticks(np.arange(y_lims[0], y_lims[1], y_int))
        ax[sp].spines["right"].set_visible(False)
        ax[sp].spines["top"].set_visible(False)

        if model_name == "clip-vit":
            ax[sp].axvline(3, 0, 0.59, c="grey", linewidth=2, alpha=0.5, linestyle="--")

        if sp == 0:
            ax[sp].set_ylabel(y_label, fontsize=14)
            ax[sp].set_xlabel("Best K Components", fontsize=14)

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

            # Calculate masks for permutation p-values
            if isinstance(mod_fit_perm_mat, np.ndarray):
                # Get exemplar-averaged permutation fits
                perm_mat = np.nanmean(
                    mod_fit_perm_mat[:, :, plot_td_idx[d_idx], :], axis=0
                )
                p_vals = np.apply_along_axis(
                    perm_p_row_above_zero, axis=1, arr=perm_mat
                )
                p_mask_001, p_mask_05 = perm_p_masks(p_vals)

                # Sig p-values as scatter points
                ax[sp].scatter(
                    np.where(p_mask_001)[0] + 1,
                    (np.ones((len(p_mask_001))) * y_lims[1])[p_mask_001]
                    - (d_idx * 0.07)
                    - 0.05,
                    color="black",
                    marker="o",
                    s=30,
                    facecolors=cmap(cmap_intervals[d_idx]),
                    alpha=1,
                )
                ax[sp].scatter(
                    np.where(p_mask_05)[0] + 1,
                    (np.ones((len(p_mask_05))) * y_lims[1])[p_mask_05]
                    - (d_idx * 0.07)
                    - 0.05,
                    color="black",
                    marker="o",
                    s=30,
                    facecolors=cmap(cmap_intervals[d_idx]),
                    alpha=0.5,
                )
        ax[sp].legend(
            loc="upper right",
            bbox_to_anchor=(1.2, 1),
            labelspacing=0.1,
            fontsize=6,
            title="Dim.",
            title_fontsize=5,
            frameon=False,
            markerscale=None,
            markerfirst=False,
        )

    plt.savefig(out_path + f"{model_name}_{mod_fit_metric}_model_fit_lines_perm.png")


def unpack_ab_variables(pair_object, p_idx):
    targ_dims = pair_object.variables[p_idx]["targ_dims"]
    n_comp = pair_object.variables[p_idx]["n_comp"]
    n_item = pair_object.variables[p_idx]["n_item"]
    n_exemp = pair_object.variables[p_idx]["n_exemp"]
    n_fold = pair_object.variables[p_idx]["n_fold"]
    best_k_sizes = pair_object.variables[p_idx]["best_k_sizes"]
    return targ_dims, n_comp, n_item, n_exemp, n_fold, best_k_sizes


def get_ab_pair_idx(ab_names, pair_object_paths):
    """
    Get the index of the first pair in `pair_object_paths` that
    matches both elements in `ab_names`.

    Parameters:
        ab_names (List[str]): A list of two strings representing
        the names of the `a` and `b` elements.
        pair_object_paths (List[Tuple[str, str]]): A list of tuples,
        where each tuple contains two strings representing the paths
        of the `a` and `b` elements.

    Returns:
        int: The index of the first pair that matches both elements in `ab_names`.

    Raises:
        ValueError: If no pair in `pair_object_paths` matches both elements in `ab_names`.
    """
    orig_object_names = [
        [i[0].split("/")[-1], i[1].split("/")[-1]] for i in pair_object_paths
    ]
    match_count = np.array(
        [
            np.sum(np.array([j in k for j in ab_names for k in i]).reshape(2, 2))
            for i in orig_object_names
        ]
    )

    if np.where(match_count == 2)[0].size > 0:
        p_idx = int(np.where(match_count == 2)[0])
        return p_idx
    else:
        raise ValueError("ab_names are not a valid pair in pair_object")


def line_plot_uv(plot_object_a, plot_object_b, model_name_dict):
    """Line plot subplots (one per knowledge type)
    where x = best_k_components, y = exemplar-averaged unique
    variance explained
    Plots both models on each plot"""

    # Unpack plot_object variables
    (
        model_names,
        targ_dims,
        _,
        mod_fit_mat_a,
        _,
        best_k_sizes,
        out_path,
        fig_label,
    ) = plot_object_a.__dict__.values()

    (
        _,
        _,
        _,
        mod_fit_mat_b,
        _,
        _,
        _,
        _,
    ) = plot_object_b.__dict__.values()

    # Get knowledge type subtitles
    know_type_dict = {"V": "Vision", "M": "Manipulation", "F": "Function"}
    subtitles = [know_type_dict[i[0][0]] for i in targ_dims]

    # Y-axis variables
    y_lims = [0, 0.5]
    y_label = "Unique Var."

    # Figure prep
    cm = 1 / 2.54
    fig, ax = plt.subplots(
        nrows=1,
        ncols=len(subtitles),
        figsize=(18 * cm, 7 * cm),
        dpi=600,
        sharey=True,
    )
    fig.suptitle(
        f"{model_name_dict[model_names[0]]} > {model_name_dict[model_names[1]]}",
        fontsize=14,
    )
    plt.subplots_adjust(top=0.8, bottom=0.2, wspace=0.25, left=0.11, right=0.95)
    plt.figtext(0.028, 0.92, fig_label, fontsize=14)

    # Sub-plots
    targ_dims_flat = sum(targ_dims, [])
    for sp in np.arange(len(targ_dims)):
        # Get model fits for current split
        plot_targ_dims = targ_dims[sp]
        plot_td_idx = np.array([targ_dims_flat.index(i) for i in plot_targ_dims])

        # Get plot mats
        plot_mat_a = mod_fit_mat_a[:, :, plot_td_idx]
        plot_mat_b = mod_fit_mat_b[:, :, plot_td_idx]

        # Create color maps
        gray_cmap = colormaps["gray"]
        if sp == 0:
            cmap = colormaps["BrBG"]
            cmap_intervals = np.linspace(0.9, 0.6, len(plot_targ_dims))
            gray_cmap_intervals = np.linspace(0, 0.65, len(plot_targ_dims))
        elif sp == 1:
            cmap = colormaps["PuOr"]
            cmap_intervals = np.linspace(0.9, 0.6, len(plot_targ_dims))
            gray_cmap_intervals = np.linspace(0, 0.65, len(plot_targ_dims))
        elif sp == 2:
            cmap = colormaps["YlOrBr"]
            cmap_intervals = np.linspace(0.6, 0.3, len(plot_targ_dims))
            gray_cmap_intervals = np.linspace(0, 0.65, len(plot_targ_dims))

        # Sub-plot set-up
        title_text = f"{subtitles[sp]}"
        ax[sp].set_ylim(y_lims)
        ax[sp].set_title(title_text, fontsize=14)
        ax[sp].set_xticks(
            np.arange(1, len(best_k_sizes) + 1), labels=list(np.array(best_k_sizes))
        )
        ax[sp].set_yticks(np.arange(y_lims[0], y_lims[1] + 0.1, y_lims[1] / 5))
        ax[sp].spines["right"].set_visible(False)
        ax[sp].spines["top"].set_visible(False)
        # ax[sp].axhline(0, c="grey", linewidth=2, alpha=0.5, linestyle="--")

        if sp == 0:
            ax[sp].set_ylabel(y_label, fontsize=14)
            ax[sp].set_xlabel("Best K Components", fontsize=14)

            # Plot for object b
        for d_idx, d_name in enumerate(plot_targ_dims):
            if d_idx == 0:
                ax[sp].errorbar(
                    np.arange(1, len(best_k_sizes) + 1),
                    np.mean(plot_mat_b[:, :, d_idx], axis=0),
                    yerr=(
                        np.std(plot_mat_b[:, :, d_idx], axis=0, ddof=1)
                        / np.sqrt(plot_mat_b.shape[0])
                    ),
                    color=gray_cmap(gray_cmap_intervals[d_idx]),
                    linewidth=2,
                    alpha=0.75,
                )
            else:
                ax[sp].errorbar(
                    np.arange(1, len(best_k_sizes) + 1),
                    np.mean(plot_mat_b[:, :, d_idx], axis=0),
                    yerr=(
                        np.std(plot_mat_b[:, :, d_idx], axis=0, ddof=1)
                        / np.sqrt(plot_mat_b.shape[0])
                    ),
                    color=gray_cmap(gray_cmap_intervals[d_idx]),
                    linewidth=1.25,
                    alpha=0.75,
                )

        # Plot each dimension for object a
        for d_idx, d_name in enumerate(plot_targ_dims):
            if d_idx == 0:
                ax[sp].errorbar(
                    np.arange(1, len(best_k_sizes) + 1),
                    np.mean(plot_mat_a[:, :, d_idx], axis=0),
                    yerr=(
                        np.std(plot_mat_a[:, :, d_idx], axis=0, ddof=1)
                        / np.sqrt(plot_mat_a.shape[0])
                    ),
                    label=d_name[-1],
                    color=cmap(cmap_intervals[d_idx]),
                    linewidth=4,
                )
            else:
                ax[sp].errorbar(
                    np.arange(1, len(best_k_sizes) + 1),
                    np.mean(plot_mat_a[:, :, d_idx], axis=0),
                    yerr=(
                        np.std(plot_mat_a[:, :, d_idx], axis=0, ddof=1)
                        / np.sqrt(plot_mat_a.shape[0])
                    ),
                    label=d_name[-1],
                    color=cmap(cmap_intervals[d_idx]),
                    linewidth=2.5,
                )

        ax[sp].legend(
            loc="upper right",
            bbox_to_anchor=(1.2, 1),
            labelspacing=0.1,
            fontsize=6,
            title="Dim.",
            title_fontsize=5,
            frameon=False,
            markerscale=None,
            markerfirst=False,
        )

    plt.savefig(
        out_path
        + f"{model_name_dict[model_names[0]]}_>_{model_name_dict[model_names[1]]}"
        + "_model_fit_lines_uv.png"
    )


def line_plot_uv_rescale(plot_object_a, plot_object_b, model_name_dict):
    """Line plot subplots (one per knowledge type)
    where x = best_k_components, y = exemplar-averaged unique
    variance explained
    Plots both models on each plot"""

    # Unpack plot_object variables
    (
        model_names,
        targ_dims,
        _,
        mod_fit_mat_a,
        _,
        best_k_sizes,
        out_path,
        fig_label,
    ) = plot_object_a.__dict__.values()

    (
        _,
        _,
        _,
        mod_fit_mat_b,
        _,
        _,
        _,
        _,
    ) = plot_object_b.__dict__.values()

    # Get knowledge type subtitles
    know_type_dict = {"V": "Vision", "M": "Manipulation", "F": "Function"}
    subtitles = [know_type_dict[i[0][0]] for i in targ_dims]

    # Get slice indices to remove "problematic exemplars"
    if any([True if "in1k-alexnet" in i else False for i in model_names]):
        slice_idx = np.setdiff1d(np.arange(len(mod_fit_mat_a)), 7)
    elif any([True if "ecoset-alexnet" in i else False for i in model_names]):
        slice_idx = np.setdiff1d(np.arange(len(mod_fit_mat_a)), 4)
    elif any([True if "ecoset-vgg16" in i else False for i in model_names]):
        slice_idx = np.arange(len(mod_fit_mat_a))

    # Slice all but the 8th index (i.e. problematic exemplar set)
    mod_fit_mat_a = mod_fit_mat_a[slice_idx, :, :]
    mod_fit_mat_b = mod_fit_mat_b[slice_idx, :, :]

    # y-variables
    y_lims = [0, 2]
    y_label = "Unique Var."

    # Figure prep
    cm = 1 / 2.54
    fig, ax = plt.subplots(
        nrows=1,
        ncols=len(subtitles),
        figsize=(18 * cm, 7 * cm),
        dpi=600,
        sharey=True,
    )
    fig.suptitle(
        f"{model_name_dict[model_names[0]]} > {model_name_dict[model_names[1]]}",
        fontsize=14,
    )
    plt.subplots_adjust(top=0.8, bottom=0.2, wspace=0.25, left=0.11, right=0.95)
    plt.figtext(0.028, 0.92, fig_label, fontsize=14)

    # Sub-plots
    targ_dims_flat = sum(targ_dims, [])
    for sp in np.arange(len(targ_dims)):
        # Get model fits for current split
        plot_targ_dims = targ_dims[sp]
        plot_td_idx = np.array([targ_dims_flat.index(i) for i in plot_targ_dims])

        # Get plot mats
        plot_mat_a = mod_fit_mat_a[:, :, plot_td_idx]
        plot_mat_b = mod_fit_mat_b[:, :, plot_td_idx]

        # Create color maps
        gray_cmap = colormaps["gray"]
        if sp == 0:
            cmap = colormaps["BrBG"]
            cmap_intervals = np.linspace(0.9, 0.6, len(plot_targ_dims))
            gray_cmap_intervals = np.linspace(0, 0.65, len(plot_targ_dims))
        elif sp == 1:
            cmap = colormaps["PuOr"]
            cmap_intervals = np.linspace(0.9, 0.6, len(plot_targ_dims))
            gray_cmap_intervals = np.linspace(0, 0.65, len(plot_targ_dims))
        elif sp == 2:
            cmap = colormaps["YlOrBr"]
            cmap_intervals = np.linspace(0.6, 0.3, len(plot_targ_dims))
            gray_cmap_intervals = np.linspace(0, 0.65, len(plot_targ_dims))

        # Sub-plot set-up
        title_text = f"{subtitles[sp]}"
        ax[sp].set_ylim(y_lims)
        ax[sp].set_title(title_text, fontsize=14)
        ax[sp].set_xticks(
            np.arange(1, len(best_k_sizes) + 1), labels=list(np.array(best_k_sizes))
        )
        ax[sp].set_yticks(np.arange(y_lims[0], y_lims[1] + 0.1, y_lims[1] / 5))
        ax[sp].spines["right"].set_visible(False)
        ax[sp].spines["top"].set_visible(False)
        # ax[sp].axhline(0, c="grey", linewidth=2, alpha=0.5, linestyle="--")

        if sp == 0:
            ax[sp].set_ylabel(y_label, fontsize=14)
            ax[sp].set_xlabel("Best K Components", fontsize=14)

            # Plot for object b
        for d_idx, d_name in enumerate(plot_targ_dims):
            if d_idx == 0:
                ax[sp].errorbar(
                    np.arange(1, len(best_k_sizes) + 1),
                    np.mean(plot_mat_b[:, :, d_idx], axis=0),
                    yerr=(
                        np.std(plot_mat_b[:, :, d_idx], axis=0, ddof=1)
                        / np.sqrt(plot_mat_b.shape[0])
                    ),
                    color=gray_cmap(gray_cmap_intervals[d_idx]),
                    linewidth=2,
                    alpha=0.75,
                )
            else:
                ax[sp].errorbar(
                    np.arange(1, len(best_k_sizes) + 1),
                    np.mean(plot_mat_b[:, :, d_idx], axis=0),
                    yerr=(
                        np.std(plot_mat_b[:, :, d_idx], axis=0, ddof=1)
                        / np.sqrt(plot_mat_b.shape[0])
                    ),
                    color=gray_cmap(gray_cmap_intervals[d_idx]),
                    linewidth=1.25,
                    alpha=0.75,
                )

        # Plot each dimension for object a
        for d_idx, d_name in enumerate(plot_targ_dims):
            if d_idx == 0:
                ax[sp].errorbar(
                    np.arange(1, len(best_k_sizes) + 1),
                    np.mean(plot_mat_a[:, :, d_idx], axis=0),
                    yerr=(
                        np.std(plot_mat_a[:, :, d_idx], axis=0, ddof=1)
                        / np.sqrt(plot_mat_a.shape[0])
                    ),
                    label=d_name[-1],
                    color=cmap(cmap_intervals[d_idx]),
                    linewidth=4,
                )
            else:
                ax[sp].errorbar(
                    np.arange(1, len(best_k_sizes) + 1),
                    np.mean(plot_mat_a[:, :, d_idx], axis=0),
                    yerr=(
                        np.std(plot_mat_a[:, :, d_idx], axis=0, ddof=1)
                        / np.sqrt(plot_mat_a.shape[0])
                    ),
                    label=d_name[-1],
                    color=cmap(cmap_intervals[d_idx]),
                    linewidth=2.5,
                )

        ax[sp].legend(
            loc="upper right",
            bbox_to_anchor=(1.2, 1),
            labelspacing=0.1,
            fontsize=6,
            title="Dim.",
            title_fontsize=5,
            frameon=False,
            markerscale=None,
            markerfirst=False,
        )

    plt.savefig(
        out_path
        + f"{model_name_dict[model_names[0]]}_>_{model_name_dict[model_names[1]]}"
        + "_model_fit_lines_uv.png"
    )


def bar_plot(plot_object, plot_best_k):
    """Plot bar subplots for each dimension,
    for best k components"""

    # Unpack plot_object variables
    (
        model_name,
        targ_dims,
        mod_fit_metric,
        mod_fit_mat,
        _,
        best_k_sizes,
        out_path,
        fig_label,
    ) = plot_object.__dict__.values()

    # Get index for plot_best_k
    bk_idx = np.where(best_k_sizes == plot_best_k)[0]

    # Get knowledge type subtitles
    know_type_dict = {"V": "Vision", "M": "Manipulation", "F": "Function"}
    subtitles = [know_type_dict[i[0][0]] for i in targ_dims]

    # Get y-axis variables
    if mod_fit_metric == "adj_r2":
        y_lims = [0, 1]
        y_label = "Model Fit (adj. $R^2$)"
    elif mod_fit_metric == "r2":
        y_lims = [0, 1]
        y_label = "Model Fit ($R^2$)"

    # Figure prep
    cm = 1 / 2.54
    fig, ax = plt.subplots(
        nrows=1,
        ncols=len(subtitles),
        figsize=(18 * cm, 7 * cm),
        dpi=600,
        sharey=True,
    )
    # fig.suptitle(f"{model_name_dict[model_name]}", fontsize=14)
    plt.subplots_adjust(top=0.8, bottom=0.2, wspace=0.25, left=0.12, right=0.95)
    plt.figtext(0.028, 0.92, fig_label, fontsize=14)

    # Sub-plots
    targ_dims_flat = sum(targ_dims, [])
    for sp in np.arange(len(targ_dims)):
        # Get model fits for current split
        plot_targ_dims = targ_dims[sp]
        plot_td_idx = np.array([targ_dims_flat.index(i) for i in plot_targ_dims])
        plot_mat = mod_fit_mat[:, bk_idx, plot_td_idx]

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
        title_text = f"{subtitles[sp]}"
        ax[sp].set_ylim(y_lims)
        ax[sp].set_title(title_text, fontsize=14)
        ax[sp].set_xticks(
            np.arange(1, len(plot_targ_dims) + 1),
            labels=list(np.arange(1, len(plot_targ_dims) + 1)),
        )
        ax[sp].set_yticks(np.arange(y_lims[0], y_lims[1] + 0.1, 0.25))
        ax[sp].spines["right"].set_visible(False)
        ax[sp].spines["top"].set_visible(False)
        ax[sp].axhline(0, c="black", linewidth=1)

        if sp == 0:
            ax[sp].set_ylabel(y_label, fontsize=14)
            ax[sp].set_xlabel("Dimension", fontsize=14)

        # Plot bars
        plot_means = np.mean(plot_mat, axis=0)
        plot_sem = np.std(plot_mat, axis=0, ddof=1) / np.sqrt(plot_mat.shape[0])

        ax[sp].bar(
            np.arange(1, len(plot_targ_dims) + 1),
            plot_means,
            yerr=(plot_sem),
            color=cmap(cmap_intervals),
        )

    plt.savefig(
        out_path
        + f"{model_name}_{mod_fit_metric}_model_fit_bars_best_{plot_best_k}_components.png"
    )


def bar_plot_perm(plot_object, plot_best_k):
    """Plot bar subplots for each dimension,
    for best k components, with permutations"""

    # Unpack plot_object variables
    (
        model_name,
        targ_dims,
        mod_fit_metric,
        mod_fit_mat,
        mod_fit_perm_mat,
        best_k_sizes,
        out_path,
        fig_label,
    ) = plot_object.__dict__.values()

    # Get index for plot_best_k
    bk_idx = np.where(best_k_sizes == plot_best_k)[0]

    # Get knowledge type subtitles
    know_type_dict = {"V": "Vision", "M": "Manipulation", "F": "Function"}
    subtitles = [know_type_dict[i[0][0]] for i in targ_dims]

    # Get y-axis variables
    if mod_fit_metric == "adj_r2":
        y_lims = [0, 1]
        y_label = "Model Fit (adj. $R^2$)"
    elif mod_fit_metric == "r2":
        y_lims = [0, 1]
        y_label = "Model Fit ($R^2$)"

    # Figure prep
    cm = 1 / 2.54
    fig, ax = plt.subplots(
        nrows=1,
        ncols=len(subtitles),
        figsize=(18 * cm, 7 * cm),
        dpi=600,
        sharey=True,
    )
    # fig.suptitle(f"{model_name_dict[model_name]}", fontsize=14)
    plt.subplots_adjust(top=0.8, bottom=0.2, wspace=0.25, left=0.12, right=0.95)
    plt.figtext(0.028, 0.92, fig_label, fontsize=14)

    # Sub-plots
    targ_dims_flat = sum(targ_dims, [])
    for sp in np.arange(len(targ_dims)):
        # Get model fits for current split
        plot_targ_dims = targ_dims[sp]
        plot_td_idx = np.array([targ_dims_flat.index(i) for i in plot_targ_dims])
        plot_mat = mod_fit_mat[:, bk_idx, plot_td_idx]

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
        title_text = f"{subtitles[sp]}"
        ax[sp].set_ylim(y_lims)
        ax[sp].set_title(title_text, fontsize=14)
        ax[sp].set_xticks(
            np.arange(1, len(plot_targ_dims) + 1),
            labels=list(np.arange(1, len(plot_targ_dims) + 1)),
        )
        ax[sp].set_yticks(np.arange(y_lims[0], y_lims[1] + 0.1, 0.25))
        ax[sp].spines["right"].set_visible(False)
        ax[sp].spines["top"].set_visible(False)
        ax[sp].axhline(0, c="black", linewidth=1)

        if sp == 0:
            ax[sp].set_ylabel(y_label, fontsize=14)
            ax[sp].set_xlabel("Dimension", fontsize=14)

        # Plot bars
        plot_means = np.mean(plot_mat, axis=0)
        plot_sem = np.std(plot_mat, axis=0, ddof=1) / np.sqrt(plot_mat.shape[0])

        ax[sp].bar(
            np.arange(1, len(plot_targ_dims) + 1),
            plot_means,
            yerr=(plot_sem),
            color=cmap(cmap_intervals),
        )

        # Calculate masks for permutation p-values
        if isinstance(mod_fit_perm_mat, np.ndarray):
            # Get exemplar-averaged permutation fits
            perm_mat = np.nanmean(mod_fit_perm_mat[:, bk_idx, plot_td_idx, :], axis=0)
            p_vals = np.apply_along_axis(perm_p_row_above_zero, axis=1, arr=perm_mat)
            p_mask_001, p_mask_05 = perm_p_masks(p_vals)

            # Sig p-values as scatter points
            ax[sp].scatter(
                np.where(p_mask_001)[0] + 1,
                plot_means[np.where(p_mask_001)[0]]
                + plot_sem[np.where(p_mask_001)[0]]
                + 0.1,
                color="black",
                marker="o",
                s=30,
                facecolors=["black"] * len(np.where(p_mask_001)[0]),
                alpha=1,
            )
            ax[sp].scatter(
                np.where(p_mask_05)[0] + 1,
                plot_means[np.where(p_mask_05)[0]]
                + plot_sem[np.where(p_mask_05)[0]]
                + 0.1,
                color="black",
                marker="o",
                s=30,
                facecolors=["white"] * len(np.where(p_mask_05)[0]),
                alpha=1,
            )

    plt.savefig(
        out_path
        + f"{model_name}_{mod_fit_metric}_model_fit_bars_best_{plot_best_k}_components_perm.png"
    )


def mod_fit_things(pred_mat, dim_vals, best_k_sizes, eval_func):
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
    n_fold_tr, n_fold_te, n_pred, n_bks, n_targ_dims = pred_mat.shape
    mod_fit_mat = np.zeros((n_fold_tr, n_fold_te, n_bks, n_targ_dims))

    for td in np.arange(n_targ_dims):
        for bks in np.arange(n_bks):
            for tr_f in np.arange(n_fold_tr):
                for te_f in np.arange(n_fold_te):
                    mod_fit_mat[tr_f, te_f, bks, td] = eval_score(
                        dim_vals[:, td],
                        pred_mat[tr_f, te_f, :, bks, td],
                        best_k_sizes[bks],
                    )
    mod_fit_mat = np.nanmean(mod_fit_mat, axis=1)
    return mod_fit_mat


def bar_plot_things(plot_object, plot_best_k):
    """Plots bar subplots for each dimension for a
    given best k components"""

    # Unpack plot_object variables
    (
        model_name,
        targ_dims,
        mod_fit_metric,
        mod_fit_mat,
        _,
        best_k_sizes,
        out_path,
        fig_label,
    ) = plot_object.__dict__.values()

    # Get index for plot_best_k
    bk_idx = np.where(best_k_sizes == plot_best_k)[0]

    # Get knowledge type subtitles
    know_type_dict = {"V": "Vision", "M": "Manipulation", "F": "Function"}
    subtitles = [know_type_dict[i[0][0]] for i in targ_dims]

    # Get y-axis variables
    if mod_fit_metric == "adj_r2":
        y_lims = [0, 1]
        y_label = "Model Fit (adj. $R^2$)"
    elif mod_fit_metric == "r2":
        y_lims = [0, 1]
        y_label = "Model Fit ($R^2$)"

    # Figure prep
    cm = 1 / 2.54
    fig, ax = plt.subplots(
        nrows=1,
        ncols=len(subtitles),
        figsize=(18 * cm, 7 * cm),
        dpi=600,
        sharey=True,
    )
    # fig.suptitle(f"{model_name_dict[model_name]}", fontsize=14)
    plt.subplots_adjust(top=0.8, bottom=0.2, wspace=0.25, left=0.12, right=0.95)
    plt.figtext(0.028, 0.92, fig_label, fontsize=14)

    # Sub-plots
    targ_dims_flat = sum(targ_dims, [])
    for sp in np.arange(len(targ_dims)):
        # Get model fits for current split
        plot_targ_dims = targ_dims[sp]
        plot_td_idx = np.array([targ_dims_flat.index(i) for i in plot_targ_dims])
        plot_mat = mod_fit_mat[:, bk_idx, plot_td_idx]

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
        title_text = f"{subtitles[sp]}"
        ax[sp].set_ylim(y_lims)
        ax[sp].set_title(title_text, fontsize=14)
        ax[sp].set_xticks(
            np.arange(1, len(plot_targ_dims) + 1),
            labels=list(np.arange(1, len(plot_targ_dims) + 1)),
        )
        ax[sp].set_yticks(np.arange(y_lims[0], y_lims[1] + 0.1, 0.25))
        ax[sp].spines["right"].set_visible(False)
        ax[sp].spines["top"].set_visible(False)
        ax[sp].axhline(0, c="black", linewidth=1)

        if sp == 0:
            ax[sp].set_ylabel(y_label, fontsize=14)
            ax[sp].set_xlabel("Dimension", fontsize=14)

        # Plot bars
        plot_means = np.mean(plot_mat, axis=0)
        plot_sem = np.std(plot_mat, axis=0, ddof=1) / np.sqrt(plot_mat.shape[0])

        ax[sp].bar(
            np.arange(1, len(plot_targ_dims) + 1),
            plot_means,
            yerr=(plot_sem),
            color=cmap(cmap_intervals),
        )

    plt.savefig(
        out_path
        + f"{model_name}_{mod_fit_metric}_model_fit_bars_best_{plot_best_k}_components_things.png"
    )


def bar_plot_things_perm(plot_object, plot_best_k):
    """Plots bar subplots for each dimension for a
    given best k components"""

    # Unpack plot_object variables
    (
        model_name,
        targ_dims,
        mod_fit_metric,
        mod_fit_mat,
        mod_fit_perm_mat,
        best_k_sizes,
        out_path,
        fig_label,
    ) = plot_object.__dict__.values()

    # Get index for plot_best_k
    bk_idx = np.where(best_k_sizes == plot_best_k)[0]

    # Get knowledge type subtitles
    know_type_dict = {"V": "Vision", "M": "Manipulation", "F": "Function"}
    subtitles = [know_type_dict[i[0][0]] for i in targ_dims]

    # Get y-axis variables
    if mod_fit_metric == "adj_r2":
        y_lims = [0, 1]
        y_label = "Model Fit (adj. $R^2$)"
    elif mod_fit_metric == "r2":
        y_lims = [0, 1]
        y_label = "Model Fit ($R^2$)"

    # Figure prep
    cm = 1 / 2.54
    fig, ax = plt.subplots(
        nrows=1,
        ncols=len(subtitles),
        figsize=(18 * cm, 7 * cm),
        dpi=600,
        sharey=True,
    )
    # fig.suptitle(f"{model_name_dict[model_name]}", fontsize=14)
    plt.subplots_adjust(top=0.8, bottom=0.2, wspace=0.25, left=0.12, right=0.95)
    plt.figtext(0.028, 0.92, fig_label, fontsize=14)

    # Sub-plots
    targ_dims_flat = sum(targ_dims, [])
    for sp in np.arange(len(targ_dims)):
        # Get model fits for current split
        plot_targ_dims = targ_dims[sp]
        plot_td_idx = np.array([targ_dims_flat.index(i) for i in plot_targ_dims])
        plot_mat = mod_fit_mat[:, bk_idx, plot_td_idx]

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
        title_text = f"{subtitles[sp]}"
        ax[sp].set_ylim(y_lims)
        ax[sp].set_title(title_text, fontsize=14)
        ax[sp].set_xticks(
            np.arange(1, len(plot_targ_dims) + 1),
            labels=list(np.arange(1, len(plot_targ_dims) + 1)),
        )
        ax[sp].set_yticks(np.arange(y_lims[0], y_lims[1] + 0.1, 0.25))
        ax[sp].spines["right"].set_visible(False)
        ax[sp].spines["top"].set_visible(False)
        ax[sp].axhline(0, c="black", linewidth=1)

        if sp == 0:
            ax[sp].set_ylabel(y_label, fontsize=14)
            ax[sp].set_xlabel("Dimension", fontsize=14)

        # Plot bars
        plot_means = np.mean(plot_mat, axis=0)
        plot_sem = np.std(plot_mat, axis=0, ddof=1) / np.sqrt(plot_mat.shape[0])

        ax[sp].bar(
            np.arange(1, len(plot_targ_dims) + 1),
            plot_means,
            yerr=(plot_sem),
            color=cmap(cmap_intervals),
        )

        # Calculate masks for permutation p-values
        if isinstance(mod_fit_perm_mat, np.ndarray):
            # Get exemplar-averaged permutation fits
            perm_mat = np.nanmean(mod_fit_perm_mat[:, bk_idx, plot_td_idx, :], axis=0)
            p_vals = np.apply_along_axis(perm_p_row_above_zero, axis=1, arr=perm_mat)
            p_mask_001, p_mask_05 = perm_p_masks(p_vals)

            # Sig p-values as scatter points
            ax[sp].scatter(
                np.where(p_mask_001)[0] + 1,
                plot_means[np.where(p_mask_001)[0]]
                + plot_sem[np.where(p_mask_001)[0]]
                + 0.1,
                color="black",
                marker="o",
                s=30,
                facecolors=["black"] * len(np.where(p_mask_001)[0]),
                alpha=1,
            )
            ax[sp].scatter(
                np.where(p_mask_05)[0] + 1,
                plot_means[np.where(p_mask_05)[0]]
                + plot_sem[np.where(p_mask_05)[0]]
                + 0.1,
                color="black",
                marker="o",
                s=30,
                facecolors=["white"] * len(np.where(p_mask_05)[0]),
                alpha=1,
            )

    plt.savefig(
        out_path
        + f"{model_name}_{mod_fit_metric}_model_fit_bars_best_{plot_best_k}_components_things_perm.png"
    )


def mod_fit_extra(pred_mat, dim_vals, best_k_sizes, eval_func, n_exemp):
    """Model fit fold-wise predictions with dimension(s)"""

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
    _, n_preds, n_bks, n_targ_dims = pred_mat.shape
    n_items = int(n_preds / n_exemp)
    mod_fit_mat = np.zeros((n_exemp, n_bks, n_targ_dims))

    # Reshape to n_exemp * n_item * n_bks * n_dims
    pred_mat_f_avg = np.nanmean(pred_mat, axis=0)
    pred_mat_f_avg = pred_mat_f_avg.reshape(
        (n_exemp, n_items, n_bks, n_targ_dims), order="F"
    )

    for td in np.arange(n_targ_dims):
        for bks in np.arange(n_bks):
            for e in np.arange(n_exemp):
                mod_fit_mat[e, bks, td] = eval_score(
                    dim_vals[:, td],
                    pred_mat_f_avg[e, :, bks, td],
                    best_k_sizes[bks],
                )
    return mod_fit_mat


def bar_plot_extra_perm(plot_object, plot_best_k):
    """Plots bar subplots for each dimension for a
    given best k components"""

    # Unpack plot_object variables
    (
        model_name,
        targ_dims,
        mod_fit_metric,
        mod_fit_mat,
        mod_fit_perm_mat,
        best_k_sizes,
        out_path,
        fig_label,
    ) = plot_object.__dict__.values()

    # Get index for plot_best_k
    bk_idx = np.where(best_k_sizes == plot_best_k)[0]

    # Get targ dims, model fits to plot
    targ_dims_dict = {"V_DL_1": "V1", "M_DL_1": "M1", "F_DL_1": "F1"}
    plot_targ_dims = [targ_dims_dict[i] for i in targ_dims]
    plot_mat = mod_fit_mat[:, bk_idx, :]

    # Get y-axis variables
    if mod_fit_metric == "adj_r2":
        y_lims = [0, 1]
        y_label = "Model Fit (adj. $R^2$)"
    elif mod_fit_metric == "r2":
        y_lims = [0, 1]
        y_label = "Model Fit ($R^2$)"

    # Figure prep
    cm = 1 / 2.54
    fig, ax = plt.subplots(
        figsize=(6 * cm, 7 * cm),
        dpi=600,
        sharey=True,
    )
    # fig.suptitle(f"{model_name_dict[model_name]}", fontsize=14)
    plt.subplots_adjust(top=0.8, bottom=0.2, wspace=0.25, left=0.3, right=0.97)
    plt.figtext(0.028, 0.92, fig_label, fontsize=14)

    # Set colors
    colors = [[1, 101, 93], [83, 38, 135], [240, 120, 24]]
    colors = [np.array(i) / 255 for i in colors]

    # Plot set-up
    ax.set_ylim(y_lims)
    ax.set_xticks(
        np.arange(1, len(plot_targ_dims) + 1),
        labels=plot_targ_dims,
    )
    ax.set_yticks(np.arange(y_lims[0], y_lims[1] + 0.1, 0.25))
    ax.set_ylabel(y_label, fontsize=14)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.axhline(0, c="black", linewidth=1)

    # Plot bars
    plot_means = np.squeeze(np.nanmean(plot_mat, axis=0))
    plot_sem = np.squeeze(np.std(plot_mat, axis=0, ddof=1) / np.sqrt(plot_mat.shape[0]))

    ax.bar(
        np.arange(1, len(plot_targ_dims) + 1),
        plot_means,
        yerr=plot_sem,
        color=colors,
    )

    # Calculate masks for permutation p-values
    if isinstance(mod_fit_perm_mat, np.ndarray):
        # Get exemplar-averaged permutation fits
        perm_mat = np.squeeze(np.nanmean(mod_fit_perm_mat[:, bk_idx, :, :], axis=0))
        p_vals = np.apply_along_axis(perm_p_row_above_zero, axis=1, arr=perm_mat)
        p_mask_001, p_mask_01 = perm_p_masks_extra(p_vals)

        # Sig p-values as scatter points
        ax.scatter(
            np.where(p_mask_001)[0] + 1,
            plot_means[np.where(p_mask_001)[0]]
            + plot_sem[np.where(p_mask_001)[0]]
            + 0.1,
            color="black",
            marker="o",
            s=30,
            facecolors=["black"] * len(np.where(p_mask_001)[0]),
            alpha=1,
        )
        ax.scatter(
            np.where(p_mask_01)[0] + 1,
            plot_means[np.where(p_mask_01)[0]] + plot_sem[np.where(p_mask_01)[0]] + 0.1,
            color="black",
            marker="o",
            s=30,
            facecolors=["grey"] * len(np.where(p_mask_01)[0]),
            alpha=1,
        )

    plt.savefig(
        out_path
        + f"{model_name}_{mod_fit_metric}_model_fit_bars_best_{plot_best_k}_components_extra_perm.png"
    )


def bar_plot_extra(plot_object, plot_best_k):
    """Bar subplots for each dimension,
    for best k components"""

    # Unpack plot_object variables
    (
        model_name,
        targ_dims,
        mod_fit_metric,
        mod_fit_mat,
        _,
        best_k_sizes,
        out_path,
        fig_label,
    ) = plot_object.__dict__.values()

    # Get index for plot_best_k
    bk_idx = np.where(best_k_sizes == plot_best_k)[0]

    # Get targ dims, model fits to plot
    targ_dims_dict = {"V_DL_1": "V1", "M_DL_1": "M1", "F_DL_1": "F1"}
    plot_targ_dims = [targ_dims_dict[i] for i in targ_dims]
    plot_mat = mod_fit_mat[:, bk_idx, :]

    # Get y-axis variables
    if mod_fit_metric == "adj_r2":
        y_lims = [0, 1]
        y_label = "Model Fit (adj. $R^2$)"
    elif mod_fit_metric == "r2":
        y_lims = [0, 1]
        y_label = "Model Fit ($R^2$)"

    # Figure prep
    cm = 1 / 2.54
    fig, ax = plt.subplots(
        figsize=(6 * cm, 7 * cm),
        dpi=600,
        sharey=True,
    )
    # fig.suptitle(f"{model_name_dict[model_name]}", fontsize=14)
    plt.subplots_adjust(top=0.8, bottom=0.2, wspace=0.25, left=0.3, right=0.97)
    plt.figtext(0.028, 0.92, fig_label, fontsize=14)

    # Set colors
    colors = [[1, 101, 93], [83, 38, 135], [240, 120, 24]]
    colors = [np.array(i) / 255 for i in colors]

    # Plot set-up
    ax.set_ylim(y_lims)
    ax.set_xticks(
        np.arange(1, len(plot_targ_dims) + 1),
        labels=plot_targ_dims,
    )
    ax.set_yticks(np.arange(y_lims[0], y_lims[1] + 0.1, 0.25))
    ax.set_ylabel(y_label, fontsize=14)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.axhline(0, c="black", linewidth=1)

    # Plot bars
    plot_means = np.squeeze(np.nanmean(plot_mat, axis=0))
    plot_sem = np.squeeze(np.std(plot_mat, axis=0, ddof=1) / np.sqrt(plot_mat.shape[0]))

    ax.bar(
        np.arange(1, len(plot_targ_dims) + 1),
        plot_means,
        yerr=plot_sem,
        color=colors,
    )

    plt.savefig(
        out_path
        + f"{model_name}_{mod_fit_metric}_model_fit_bars_best_{plot_best_k}_components_extra.png"
    )
