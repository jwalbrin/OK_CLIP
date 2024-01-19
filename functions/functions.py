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
    """Object for incremenetal best k plots"""

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


model_dict = {"clip-vit": "CLIP-VIT", "in21k-vit": "IN21K-ViT"}


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


def mod_fit_lio_perm(pred_mat, dim_vals, best_k_sizes, targ_dims, n_perm, eval_func):
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
    mod_fit_perm_mat = np.zeros(
        (n_exemp, n_bks, n_targ_dims, n_perm + 1), dtype=np.float16
    )

    tic = time.time()
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
                if p % int(n_perm / 10) == 0:
                    print(
                        f"{targ_dims[td]} {best_k_sizes[bks]} components: {p + 1} permutations done.\n"
                        + f"Total run time: {time.time()-tic: .02f} seconds"
                    )
    return mod_fit_perm_mat


def perm_p_row(perm_mat_row):
    """Calculate permutation p per each best_k_component set"""
    p_val = 1 - ((np.sum(perm_mat_row[0] >= perm_mat_row) - 1) / (len(perm_mat_row)))
    return p_val


def perm_p_row_above_zero(perm_mat_row):
    """Calculate permutation p per each best_k_component set,
    set test scores < 0 to p_val = 1"""
    p_val = 1 - ((np.sum(perm_mat_row[0] >= perm_mat_row) - 1) / (len(perm_mat_row)))
    if perm_mat_row[0] <= 0:
        p_val = 1
    return p_val


def perm_p_masks(p_vals):
    """Mask p-vals that are below threshold"""
    p_mask_001 = p_vals <= 0.001
    p_mask_05 = (p_vals <= 0.05) & ~p_mask_001
    return p_mask_001, p_mask_05


def incremental_lineplot(plot_object, model_name_dict):
    """Creates linechart subplots (one per knowledge type)
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
        y_lims = [-0.2, 1]
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
        ax[sp].set_yticks(np.arange(y_lims[0], y_lims[1] + 0.1, 0.2))
        ax[sp].spines["right"].set_visible(False)
        ax[sp].spines["top"].set_visible(False)
        ax[sp].axvline(3, 0, 1, c="grey", linewidth=2, alpha=0.5, linestyle="--")

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

    plt.savefig(out_path + f"{model_name}_{mod_fit_metric}_model_fit_incremental.png")


def incremental_lineplot_unique_variance(plot_object, model_name_dict, ab_idx):
    """Creates linechart subplots (one per knowledge type)
    where x = best_k_components, y = exemplar-averaged unique
    variance explained"""

    # Unpack plot_object variables
    (
        model_names,
        targ_dims,
        mod_fit_metric,
        mod_fit_mat,
        _,
        best_k_sizes,
        out_path,
        fig_label,
    ) = plot_object.__dict__.values()

    # Model a,b names
    model_first = model_names[ab_idx]
    model_second = [i for i in model_names if i != model_first][0]

    # Get knowledge type subtitles
    know_type_dict = {"V": "Vision", "M": "Manipulation", "F": "Function"}
    subtitles = [know_type_dict[i[0][0]] for i in targ_dims]

    # Y-axis variables
    y_lims = [-0.2, 1]
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
    # FIX!!!
    fig.suptitle(
        f"{model_name_dict[model_first]} > {model_name_dict[model_second]}",
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
        ax[sp].set_yticks(np.arange(y_lims[0], y_lims[1] + 0.1, 0.2))
        ax[sp].spines["right"].set_visible(False)
        ax[sp].spines["top"].set_visible(False)
        ax[sp].axhline(0, c="grey", linewidth=2, alpha=0.5, linestyle="--")

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

    plt.savefig(
        out_path
        + f"{model_name_dict[model_first]}_>_{model_name_dict[model_second]}"
        + "_unique_variance_model_fit_incremental.png"
    )


def incremental_lineplot_with_perm(plot_object, model_name_dict):
    """Creates linechart subplots (one per knowledge type)
    where x = best_k_components, y = exemplar-averaged model fits
    along with permutation significance indication"""

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
        y_lims = [-0.2, 1.2]
        y_label = "Model Fit (adj. $R^2$)"
    elif mod_fit_metric == "r2":
        y_lims = [0, 1.2]
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
        ax[sp].set_yticks(np.arange(y_lims[0], y_lims[1], 0.2))
        ax[sp].spines["right"].set_visible(False)
        ax[sp].spines["top"].set_visible(False)
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
                    # np.mean(plot_mat[:, :, d_idx], axis=0)[p_mask_05],
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

    plt.savefig(
        out_path + f"{model_name}_{mod_fit_metric}_model_fit_incremental_perm.png"
    )


def match_ab_get_attrs(data_object_a, data_object_b):
    """
    Check if the attributes of two data objects match.

    Args:
        data_object_a (object): The first data object.
        data_object_b (object): The second data object.

    Raises:
        ValueError: If any of the key variables don't match in both `data_object_a` and `data_object_b`.
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
    Prepare the features a and b models training and testing.

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
    Get the index of the first pair in `pair_object_paths` that matches both elements in `ab_names`.

    Parameters:
        ab_names (List[str]): A list of two strings representing the names of the `a` and `b` elements.
        pair_object_paths (List[Tuple[str, str]]): A list of tuples, where each tuple contains two strings representing the paths of the `a` and `b` elements.

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
