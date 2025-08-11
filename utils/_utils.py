import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np


from utils._python import get_sort_indices_descending
from utils._ebrec._constants import *


rename_cat = {
    "forbrug": "CON",  # consumption
    "incoming": "INC",  # income
    "krimi": "CRM",  # crime
    "musik": "MUS",  # music
    "nationen": "OPN",  # opinion
    "nyheder": "NWS",  # news
    "penge": "PFI",  # private finance
    "sport": "SPT",  # sports
    "underholdning": "ENT",  # entertainment
    # "auto": "auto",
}


def coverage_fraction(R: np.ndarray, C: np.ndarray) -> float:
    """Calculate the fraction of distinct items in the recommendation list compared to a universal set.

    Args:
        R (np.ndarray): An array containing the items in the recommendation list.
        C (np.ndarray): An array representing the universal set of items.
            It should contain all possible items that can be recommended.

    Returns:
        float: The fraction representing the coverage of the recommendation system.
            This is calculated as the size of unique elements in R divided by the size of unique elements in C.

    Examples:
        >>> R1 = np.array([1, 2, 3, 4, 5, 5, 6])
        >>> C1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> print(coverage_fraction(R1, C1))  # Expected output: 0.6
            0.6
    """
    # Distinct items:
    return np.unique(R).size / np.unique(C).size


def plot_genre_probabilities(
    *dicts: dict[str, float],
    labels: list[str] = None,
    fontsize: int = 26,
    title: str = "",
    figsize=(18, 12),
    legend_bbox_to_anchor=(0.5, -0.55),
    legend_ncols=2,
    save_path: str = None,
    show_plot: bool = True,
) -> None:
    if not dicts:
        raise ValueError("At least one dictionary must be provided.")

    num_dicts = len(dicts)
    all_keys = set().union(*[d.keys() for d in dicts])
    categories = sorted(all_keys)

    all_values = [[d.get(cat, 0) for cat in categories] for d in dicts]
    x = np.arange(len(categories))
    width = 0.8 / num_dicts

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    colors = plt.get_cmap("tab10").colors

    for i, values in enumerate(all_values):
        offset = (i - (num_dicts - 1) / 2) * width
        label = labels[i] if labels and i < len(labels) else f"Dataset {i+1}"
        color = colors[i % len(colors)]
        ax.bar(x + offset, values, width, label=label, alpha=0.8, color=color)

    ax.set_title(title, fontsize=fontsize)
    ax.set_ylabel("Genre Probability", fontsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    # ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.set_xticks(x)  # Set ticks at correct positions
    ax.set_xticklabels([""] * len(categories))  # Empty labels
    ax.tick_params(axis="x", which="both", length=3, labelsize=0)  # Small tick marks

    ax.legend(
        fontsize=fontsize,
        loc="lower center",
        bbox_to_anchor=legend_bbox_to_anchor,
        ncols=legend_ncols,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    # plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    if show_plot:
        plt.show()


def plot_kde(
    score_dict,
    xlabel: str = "",
    title: str = "",
    bw_adjust=1,
    fontsize: int = 20,
    figsize=(12, 8),
    save_path: str = None,
    show_plot: bool = True,
):
    """
    Plots a KDE distribution for score values stored in score_dict.
    """
    plt.figure(figsize=figsize)

    for lambda_val, score_dist in score_dict.items():
        sns.kdeplot(
            score_dist,
            label=f"λ={lambda_val:.1f} (μ = {np.mean(score_dist):.4f}$\pm${np.std(score_dist):.4f})",
            fill=True,
            bw_adjust=bw_adjust,
        )

    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel("Density", fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    if show_plot:
        plt.show()


def normalize_matrix_rows(array: np.ndarray) -> np.ndarray:
    return array / array.sum(axis=1, keepdims=True)


def get_index(arr: np.ndarray, topN: int, strategy: str, replace: bool = False):
    if strategy == "der":
        # Take top-N highest.
        idx_desc = get_sort_indices_descending(arr)[:, :topN]
    elif strategy == "stoch":
        # Sample N articles based on article-probablity score.
        article_idx_range = range(arr.shape[1])
        idx_desc = np.stack(
            [
                np.random.choice(article_idx_range, p=p_i, size=topN, replace=replace)
                for p_i in arr
            ]
        )
    else:
        raise ValueError(f"{strategy} is not defined.")
    return idx_desc


def stochastic_rank(y_prob: np.ndarray) -> np.ndarray:
    """
    Assign inverse-rank scores based on stochastic sampling from input probabilities.
    y_prob: Probabilities over items (must sum to 1).

    Example:
    >>> np.random.seed(123)
    >>> runs = 10000
    >>> y_prob = np.array([0.1, 0.2, 0.1, 0.5, 0.1])
    >>> est_prob_scores = []
    >>> rank_counts = np.zeros(len(y_prob))

    >>> for _ in range(runs):
            prob_scores = stochastic_rank(y_prob)
            # Take the highest (min-problem)
            sorted_indices = np.argsort(-prob_scores)
            # Add +1 to one that taken
            rank_counts[sorted_indices[0]] += 1
    >>> rank_counts / rank_counts.sum()
        array([0.0931, 0.2013, 0.103 , 0.5093, 0.0933])
    """
    n = len(y_prob)
    sampled_indices = np.random.choice(n, p=y_prob, size=n, replace=False)
    ranking_scores = np.zeros(n)
    ranking_scores[sampled_indices] = 1 / np.arange(1, n + 1)
    return ranking_scores


def print_output(
    nested_dict_res: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, float]]:
    output = {}
    for lambda_val, eval_metrics in nested_dict_res.items():
        res = {k: np.mean(v) for k, v in eval_metrics.items()}
        output[lambda_val] = res
        print(f"{lambda_val}:")
        for m, v in res.items():
            print(f"{m}: {v*100:.2f}")
    return output
