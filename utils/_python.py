from collections import Counter
from typing import Iterable
import numpy as np


def compute_normalized_distribution(values: Iterable[any]) -> dict[str, float]:
    """
    Compute the normalized distribution of a list of categorical values.

    Parameters:
    values (list): A list of values (e.g., strings or numbers).

    Returns:
    dict: A dictionary where keys are unique values from the input list,
            and values are their normalized frequencies.

    >>> values = ["b", "a", "a", "c"]
    >>> compute_normalized_distribution(values)
        {'a': 0.5, 'b': 0.25, 'c': 0.25}
    """
    count = Counter(values)
    total = sum(count.values())
    normalized_distribution = {key: value / total for key, value in count.items()}
    return dict(sorted(normalized_distribution.items()))


def compute_nails_adjustment_factors(
    p_star: dict[str, float], p_ei: dict[str, float]
) -> dict[str, float]:
    """
    Args:
        p_star (dict): A dictionary where keys represent categories or events and values
            represent their probabilities or weights in the ideal distribution.
        p_ei (dict): A dictionary where keys represent categories or events and values
            represent their probabilities or weights in the model's distribution.

    Returns:
        dict: A dictionary where each key corresponds to a key in the `p_star` and the value
        is the computed scaling factor for that key.

    >>> ideal_dist = {"a": 0.4, "b": 0.5, "c": 0.1}
    >>> model_dist = {"a": 0.3, "b": 0.7}
    >>> compute_subjective_probability_distribution(ideal_dist, model_dist)
        {'a': 1.3333333333333335, 'b': 0.7142857142857143, 'c': 1.0}
    """
    return {
        key: p_star.get(key) / p_ei.get(key, p_star.get(key)) for key in p_star.keys()
    }


def compute_nails(
    p_omega: float | np.ndarray,
    p_star_ei: float | np.ndarray,
    p_ei: float | np.ndarray,
    lambda_: float = 1.0,
) -> np.ndarray:
    """
    Args:
        p_omega (Union[np.ndarray, float]): Probability scores assign by model.
        p_star_ei (Union[np.ndarray, float]): Target distribution.
        p_ei (Union[np.ndarray, float]): Model distribution.
        lambda_ (float, optional): A weighting factor that controls the influence of subjective belief.
            Defaults to 1.0 (full adjustment). Range [0, 1].

    Returns:
        Union[np.ndarray, float]: The adjusted probability or distribution after applying nails transformation.

    Example:
        >>> p_omega = np.array([0.4, 0.3, 0.3])
        >>> p_star_ei = 0.5
        >>> p_ei = 0.4
        >>> compute_nails(p_omega, p_star_ei, p_ei, lambda_=0.8)
            array([0.48, 0.36, 0.36])
    """
    return p_omega * (1 + lambda_ * (p_star_ei / p_ei - 1))


def check_dictionaries_identical(dict1, dict2):
    """
    Checks if two dictionaries are identical.
    """
    return dict1 == dict2


def get_sort_indices_descending(matrix: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Sorts the indices of each row in a 2D array in descending order.
    Examples:
    >>> import numpy as np
    >>> matrix = np.array([[10, 20, 15], [5, 0, 25]])
    >>> sort_indices_descending(matrix)
        array([[1, 2, 0],
                [2, 0, 1]])
    """
    return np.argsort(matrix, axis=axis)[:, ::-1]


def softmax(x: np.array, axis: int = 1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def compute_kl_divergence(p: np.ndarray[float], q: np.ndarray[float]) -> float:
    """
    p: target distribution (true)
    q: estimated distribution (model)

    Assume normalized input:
        - sum(p) = 1
        - sum(q) = 1

    >>> p = np.array([0.4, 0.3, 0.2, 0.1])
    >>> q = np.array([0.1, 0.2, 0.3, 0.4])
    >>> compute_kl_divergence(p, q)
        0.4564348191467835
    """
    assert np.allclose(np.sum(p), 1.0), "p must be normalized"
    assert np.allclose(np.sum(q), 1.0), "q must be normalized"
    return np.sum(p * np.log(p / q))


def smoothed_distribution(p: np.ndarray, q: np.ndarray, alpha=1e-4) -> np.ndarray:
    return (1 - alpha) * q + alpha * p


def compute_smoothed_kl_divergence(
    p: list[float],
    q: list[float],
    alpha=1e-4,
):
    """
    >>> p = [0.4, 0.3, 0.2, 0.1]
    >>> q = [0.1, 0.2, 0.3, 0.4]
    >>> compute_smoothed_kl_divergence(p, q)
        0.4563140045772164
    """
    p = np.asarray(p)
    q = np.asarray(q)

    # Normalize:
    p /= np.sum(p)
    q /= np.sum(q)

    q_tilde = smoothed_distribution(p=p, q=q, alpha=alpha)
    return compute_kl_divergence(p=p, q=q_tilde)


def greedy_steck_rerank(
    ids: np.ndarray,
    scores: np.ndarray,
    lookup_attr: dict[int, str],
    p_target: dict[str, float],
    lambda_: float,
    k: int,
    alpha: float = 1e-4,
):
    """
    Greedy Steck re-ranking using global category-level calibration.
    """
    p_target_keys = list(p_target.keys())
    p_target_dist = np.array([p_target[k] for k in p_target_keys])
    candidate_score_pairs = list(zip(ids, scores))

    selected_ids = []
    selected_id_set = set()
    selected_attr = []
    relevance_sum = 0

    #
    attr2idx = {attr: idx for idx, attr in enumerate(p_target_keys)}
    selected_counts = np.zeros(len(p_target_keys))

    while len(selected_ids) < k and len(selected_ids) < len(ids):
        best_combined_score = -np.inf
        best_id = best_attr = None
        # Precompute KL per attribute
        kl_cache = {}
        total = len(selected_attr) + 1  # +1 simulating adding candidate
        for attr, idx in attr2idx.items():
            temp_counts = selected_counts.copy()
            temp_counts[idx] += 1
            temp_dist = temp_counts / total
            q_smooth = smoothed_distribution(p=p_target_dist, q=temp_dist, alpha=alpha)
            kl = compute_kl_divergence(p=p_target_dist, q=q_smooth)
            kl_cache[attr] = kl

        for id, score in candidate_score_pairs:
            if id in selected_id_set:
                continue

            attr = lookup_attr[id]
            relevance_score = relevance_sum + score
            combined_score = (1 - lambda_) * relevance_score - lambda_ * kl_cache[attr]

            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_score = score
                best_attr = attr
                best_id = id

        # Update
        relevance_sum += best_score
        selected_ids.append(best_id)
        selected_id_set.add(best_id)
        selected_attr.append(best_attr)
        selected_counts[attr2idx[best_attr]] += 1

    return selected_ids


def map_ranked_scores_to_original(
    original_ids: list[int], ranked_ids: list[int]
) -> list[float]:
    """
    Map reciprocal rank scores from ranked_ids to match the order of original_ids.

    Parameters
    original_ids : array-like of int
        Original order of IDs (any unique identifiers).
    ranked_ids : array-like of int
        Same IDs but in ranked order (first = highest rank).

    Returns
    scores_original : np.ndarray
        Scores aligned to the original_ids order.

    Example
    >>> ranked_ids = [9803309, 9803453, 9789509, 9803281, 9803356, 9801639, 9799429]
    >>> original_ids = [9803453, 9789509, 9801639, 9803281, 9803356, 9803309, 9799429]
    >>> map_ranked_scores_to_original(original_ids, ranked_ids)
        array([0.5       , 0.33333333, 0.16666667, 0.25      , 0.2       ,
                1.        , 0.14285714])
    """
    scores_ranked = 1 / np.arange(1, len(ranked_ids) + 1, dtype=float)
    id_to_score = dict(zip(ranked_ids, scores_ranked))

    scores_original = np.array([id_to_score[id_] for id_ in original_ids], dtype=float)

    return scores_original
