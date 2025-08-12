# %%
from pathlib import Path
from tqdm import tqdm
import polars as pl
import numpy as np

from utils._ebrec._constants import *
from utils._ebrec._python import (
    rank_predictions_by_score,
    write_submission_file,
    create_lookup_dict,
    write_json_file,
)

from utils._python import (
    compute_nails_adjustment_factors,
    compute_normalized_distribution,
    compute_smoothed_kl_divergence,
    greedy_steck_rerank,
    compute_nails,
    softmax,
)
from utils._utils import *

from arguments.args_nails_steck import args

# %%
# conda activate ./venv/; python subjective_probability_scripts/nails_steck_combine.py
distribution_type = args.distribution_type  # "Uniform" / "Editorial"
act_func = args.act_func  # "softmax"
# "stoch"  # stochastic (stoch) / deterministic (der)
with_replacement = args.with_replacement  # False
filter_auto = args.filter_auto  # True

alpha = args.alpha  # 1e-4
topN = args.topN  # 10
n_samples = args.n_samples  # 1_000_000
show_plot = args.show_plot

alpha_steck_select = args.alpha_steck_select

lambda_best_editorial_nails_sto = args.lambda_best_editorial_nails_sto
lambda_best_editorial_nails_der = args.lambda_best_editorial_nails_der
lambda_best_editorial_steck = args.lambda_best_editorial_steck
lambda_best_uniform_nails_sto = args.lambda_best_uniform_nails_sto
lambda_best_uniform_nails_der = args.lambda_best_uniform_nails_der
lambda_best_uniform_steck = args.lambda_best_uniform_steck


PATH = Path(args.data_dir)  # Path(f"data")
file = Path(args.score_dir)  # "data/test_scores.parquet"
PATH_EMB = Path(args.emb_dir)

lambda_values = [0]


best_lambda_nails_der = (
    lambda_best_editorial_nails_der
    if distribution_type == "Editorial"
    else lambda_best_uniform_nails_der
)
best_lambda_nails_sto = (
    lambda_best_editorial_nails_sto
    if distribution_type == "Editorial"
    else lambda_best_uniform_nails_sto
)
best_lambda_steck = (
    lambda_best_editorial_steck
    if distribution_type == "Editorial"
    else lambda_best_uniform_steck
)

lambda_values = [0]
if 0.99 not in lambda_values:
    lambda_values.append(0.99)

lambda_values.extend(
    [
        best_lambda_nails_der,
        best_lambda_nails_sto,
        best_lambda_steck,
    ]
)

np.random.seed(123)
#
plot_dir = Path(f"plot/nails_steck_combine_{distribution_type}")
print(f"Running: {plot_dir} ({distribution_type})")


# %%
df_test = pl.scan_parquet(file).filter(pl.col(DEFAULT_IMPRESSION_ID_COL) == 0).collect()
if n_samples:
    df_test = df_test.sample(n=n_samples, seed=123)

if not n_samples:
    n_samples = df_test.height

#
df_articles = (
    pl.read_parquet(PATH.joinpath("articles.parquet"))
    .join(pl.read_parquet(PATH_EMB), on=DEFAULT_ARTICLE_ID_COL)
    .with_columns(pl.col("category_str").replace(rename_cat))
)
lookup_cat_str = create_lookup_dict(
    df_articles, key=DEFAULT_ARTICLE_ID_COL, value="category_str"
)

# %%
INVIEW_ARTICLES_CAT_COL = DEFAULT_INVIEW_ARTICLES_COL + "_cat"
df_ba = (
    df_test.select(DEFAULT_INVIEW_ARTICLES_COL)
    .with_row_index("groupby")
    .explode(DEFAULT_INVIEW_ARTICLES_COL)
    .with_columns(
        pl.col(DEFAULT_INVIEW_ARTICLES_COL)
        .replace_strict(lookup_cat_str)
        .alias(INVIEW_ARTICLES_CAT_COL)
    )
    .group_by("groupby")
    .agg(
        DEFAULT_INVIEW_ARTICLES_COL,
        INVIEW_ARTICLES_CAT_COL,
    )
)


# %%
# Candidate-list ID.
cand_list_ids = np.array(
    df_ba.select(DEFAULT_INVIEW_ARTICLES_COL)[0].to_series().to_list()
).ravel()
# Candidate-list Category.
cand_list_category = np.array(
    df_ba.select(INVIEW_ARTICLES_CAT_COL)[0].to_series().to_list()
).ravel()
# Prediction Scores.
cand_scores = np.array(df_test["scores"].to_list())

# %%
mask_idx = cand_list_category == "auto" if filter_auto else cand_list_category == ""
cand_scores = cand_scores[:, ~mask_idx]
cand_list_category = cand_list_category[~mask_idx]
cand_list_ids = cand_list_ids[~mask_idx]

# ==
if act_func == "softmax":
    cand_scores = softmax(cand_scores)


# %%
# Ideal Distribution
if distribution_type == "Uniform":
    p_star_ei = compute_normalized_distribution(set(cand_list_category.tolist()))
elif distribution_type == "Editorial":
    p_star_ei = compute_normalized_distribution(cand_list_category.tolist())
else:
    raise f"{distribution_type} not defined"

if not np.isclose(sum(p_star_ei.values()), 1):
    raise ValueError(f"P*(Ei) probabilities sum to {sum(p_star_ei.values()):.1f}")

nails_values = list(p_star_ei.values())
nails_keys = list(p_star_ei.keys())

# Model Distribution
p_ei = {}
for cat_i in nails_keys:
    cat_mask = (cand_list_category == cat_i).flatten()
    cat_i_scores = cand_scores[:, cat_mask]
    p_ei[cat_i] = np.sum(cat_i_scores) / n_samples

if not np.isclose(sum(p_ei.values()), 1):
    raise ValueError(f"P(Ei) probabilities sum to {sum(p_ei.values()):.1f}")

print(f"p_star_ei ({sum(nails_values):.1f}): {p_star_ei}")
print(f"p_ei ({sum(p_ei.values()):.1f}): {p_ei}.")
#
p_w = compute_nails_adjustment_factors(p_star_ei, p_ei)
print(f"p_w ({sum(p_w.values()):.1f}): {p_w}")


# %% ################################################################################
p_star_omega_scores_dict = {}
for lambda_val in tqdm(lambda_values, ncols=80):
    cand_scores_copy = cand_scores.copy()
    for cat_i in nails_keys:
        cat_mask = (cand_list_category == cat_i).ravel()
        cand_scores_copy[:, cat_mask] = compute_nails(
            cand_scores_copy[:, cat_mask],
            p_star_ei=p_star_ei.get(cat_i),
            p_ei=p_ei.get(cat_i),
            lambda_=lambda_val,
        )
    cand_scores_copy = normalize_matrix_rows(cand_scores_copy)
    p_star_omega_scores_dict[lambda_val] = cand_scores_copy


# %%
# COMPUTE DISTRIBUTION TOP@ALL
p_star_omega_sum_dist = {k: {} for k in lambda_values}
kl_all_sum = {}
for lambda_val, p_star_omega_scores in tqdm(p_star_omega_scores_dict.items(), ncols=80):
    for cat_i in nails_keys:
        cat_mask = (cand_list_category == cat_i).ravel()
        p_star_omega_sum_dist[lambda_val][cat_i] = (
            np.sum(p_star_omega_scores[:, cat_mask]) / n_samples
        )
    if not np.isclose(sum(p_star_omega_sum_dist[lambda_val].values()), 1):
        raise ValueError(
            f"Probabilies sum to {sum(p_star_omega_sum_dist[lambda_val].values())}"
        )
    kl_all_sum[lambda_val] = compute_smoothed_kl_divergence(
        p=nails_values, q=list(p_star_omega_sum_dist[lambda_val].values())
    )


# %%
def compute_top_dist(_article_selection: bool):
    top_n_category_distribution = {}
    top_n_categories = {}
    top_n_coverage = {}
    top_n_scores = {}
    top_n_ids = {}

    for lambda_val, lambda_scores in tqdm(p_star_omega_scores_dict.items(), ncols=80):

        idx_desc = get_index(
            arr=lambda_scores,
            topN=topN,
            strategy=_article_selection,
            replace=with_replacement,
        )

        # Category Names.
        top_n_cat = cand_list_category[idx_desc]
        top_n_categories[lambda_val] = top_n_cat

        # Scores
        top_n_sco = np.take_along_axis(lambda_scores, idx_desc, axis=1)
        # top_n_sco = normalize_matrix_rows(top_n_sco)
        top_n_scores[lambda_val] = top_n_sco

        # Distribution:
        cat_dist = compute_normalized_distribution(top_n_cat.ravel())
        cat_dist = {k: cat_dist.get(k, 0) for k in nails_keys}
        top_n_category_distribution[lambda_val] = cat_dist

        # Coverage:
        top_n_coverage[lambda_val] = coverage_fraction(
            R=idx_desc.ravel(),
            C=list(range(0, len(cand_list_ids))),
        )

        # Article IDs.
        top_n_ids[lambda_val] = cand_list_ids[idx_desc]

    kl_scores = {k: [] for k in lambda_values}
    for lambda_val, cat_users in tqdm(top_n_categories.items(), ncols=80):
        for user_i in cat_users:
            user_dist = compute_normalized_distribution(user_i)
            user_dist = {key: user_dist.get(key, 0) for key in nails_keys}
            kl_scores[lambda_val].append(
                compute_smoothed_kl_divergence(
                    p=nails_values,
                    q=list(user_dist.values()),
                    alpha=alpha,
                )
            )

    return (
        top_n_category_distribution,
        top_n_categories,
        top_n_coverage,
        top_n_scores,
        top_n_ids,
        kl_scores,
    )


(
    top_n_category_distribution_der,
    top_n_categories_der,
    top_n_coverage_der,
    top_n_scores_der,
    top_n_ids_der,
    kl_scores_der,
) = compute_top_dist("der")
(
    top_n_category_distribution_sto,
    top_n_categories_sto,
    top_n_coverage_sto,
    top_n_scores_sto,
    top_n_ids_sto,
    kl_scores_sto,
) = compute_top_dist("stoch")

# %% CaliRec.

# Initialize results containers
top_n_category_dist_steck = {k: [] for k in lambda_values}
top_n_kl_steck = {k: [] for k in lambda_values}
coverage_steck = {k: [] for k in lambda_values}

target_dist_steck_type = distribution_type

# Prepare candidate distributions
lookup_cand_list = {id: lookup_cat_str[id] for id in cand_list_ids}

# Trim to sample size
temp_cand_scores = cand_scores[:n_samples]

if plot_dir.joinpath(
    f"steck_{topN}_{distribution_type}_{n_samples}_dist.json"
).exists():
    top_n_category_dist_steck[best_lambda_steck] = pl.read_json(
        plot_dir.joinpath(f"steck_{topN}_{distribution_type}_{n_samples}_dist.json")
    ).to_dicts()[0]

    coverage_steck[best_lambda_steck] = list(
        pl.read_json(
            plot_dir.joinpath(
                f"steck_{topN}_{distribution_type}_{n_samples}_best_cov.json"
            )
        )
        .to_dicts()[0]
        .values()
    )[0]
else:
    # Evaluation loop
    for lambda_val in [best_lambda_steck]:

        # Reset
        selected_ids_all = []
        selected_cats_all = []

        for user_scores in tqdm(
            temp_cand_scores, total=len(temp_cand_scores), ncols=80
        ):
            # Get top item IDs using Steck reranking
            top_ids = greedy_steck_rerank(
                ids=cand_list_ids,
                lookup_attr=lookup_cand_list,
                scores=user_scores,
                p_target=p_star_ei,
                lambda_=lambda_val,
                k=topN,
                alpha=alpha_steck_select,
            )
            # Map item IDs to categories.
            top_cats = [lookup_cand_list[id] for id in top_ids]

            # Compute KL user-level.
            cat_dist = compute_normalized_distribution(top_cats)
            cat_dist = {k: cat_dist.get(k, 0.0) for k in nails_keys}
            kl_score = compute_smoothed_kl_divergence(
                p=nails_values,
                q=list(cat_dist.values()),
                alpha=alpha,
            )

            # #
            top_n_kl_steck[lambda_val].append(kl_score)
            selected_cats_all.extend(top_cats)
            selected_ids_all.extend(top_ids)

        # Compute final metrics per lambda
        coverage_steck[lambda_val] = coverage_fraction(
            selected_ids_all, C=cand_list_ids
        )

        cat_dist_steck = compute_normalized_distribution(selected_cats_all)
        top_n_category_dist_steck[lambda_val] = {
            k: cat_dist_steck.get(k, 0.0) for k in nails_keys
        }
    # Store them:
    plot_dir.mkdir(exist_ok=True, parents=True)
    pl.DataFrame(top_n_category_dist_steck[best_lambda_steck]).write_json(
        plot_dir.joinpath(
            f"steck_{topN}_{distribution_type}_{alpha_steck_select}_{n_samples}_dist.json"
        )
    )
    pl.DataFrame([coverage_steck[best_lambda_steck]]).write_json(
        plot_dir.joinpath(
            f"steck_{topN}_{distribution_type}_{alpha_steck_select}_{n_samples}_best_cov.json"
        )
    )


# %%
min_lambda = 0
max_lambda = 0.99

baseline_val = list(top_n_category_distribution_der[min_lambda].values())
sum_scores_val = list(p_star_omega_sum_dist[max_lambda].values())
nails_stoch_val = list(top_n_category_distribution_sto[best_lambda_nails_sto].values())
nails_der_val = list(top_n_category_distribution_der[best_lambda_nails_der].values())
calirec_val = list(top_n_category_dist_steck[best_lambda_steck].values())

baseline_cov = top_n_coverage_der[min_lambda]
sum_scores_cov = 1.0
nails_stoch_cov = top_n_coverage_sto[best_lambda_nails_sto]
nails_der_cov = top_n_coverage_der[best_lambda_nails_der]
calirec_cov = coverage_steck[best_lambda_steck]


kl_baseline = compute_smoothed_kl_divergence(
    p=nails_values,
    q=baseline_val,
    alpha=alpha,
)
kl_sum_scores = compute_smoothed_kl_divergence(
    p=nails_values,
    q=sum_scores_val,
    alpha=alpha,
)
kl_nails_stoch = compute_smoothed_kl_divergence(
    p=nails_values,
    q=nails_stoch_val,
    alpha=alpha,
)
kl_nails_der = compute_smoothed_kl_divergence(
    p=nails_values,
    q=nails_der_val,
    alpha=alpha,
)
kl_calirec = compute_smoothed_kl_divergence(
    p=nails_values,
    q=calirec_val,
    alpha=alpha,
)

# %%
plot_genre_probabilities(
    p_star_ei,  # target
    top_n_category_distribution_der[min_lambda],  # baseline
    p_star_omega_sum_dist[max_lambda],  # prove it works
    top_n_category_distribution_sto[best_lambda_nails_sto],
    top_n_category_distribution_der[best_lambda_nails_der],
    top_n_category_dist_steck[best_lambda_steck],
    # LABELS
    labels=[
        f"Target ({distribution_type.lower()})",
        #
        f"Baseline (@10)\nKL={kl_baseline:.4f}, COV={baseline_cov:.3f}",
        f"NAILS (λ={max_lambda}, @|$\\mathcal{{C}}$|)\nKL={kl_sum_scores:.4f}",
        f"NAILS-stoch (λ={best_lambda_nails_sto}, @10)\nKL={kl_nails_stoch:.4f}, COV={nails_stoch_cov:.3f}",
        #
        f"NAILS-det (λ={best_lambda_nails_der}, @10)\nKL={kl_nails_der:.4f}, COV={nails_der_cov:.3f}",
        #
        f"CaliRec (λ={best_lambda_steck}, @10)\nKL={kl_calirec:.4f}, COV={calirec_cov:.3f}",
    ],
    title=f"",
    figsize=(16, 9),
    fontsize=28,
    save_path=plot_dir.joinpath(
        f"final_{topN}_{distribution_type}_{alpha_steck_select}_{n_samples}.pdf"
    ),
    show_plot=show_plot,
)

# %%
bars = {
    "category": nails_keys + ["kl"] + ["cov"],
    "target": nails_values + [0.0] + [0],
    #
    f"baseline_{min_lambda}": baseline_val + [kl_baseline] + [baseline_cov],
    #
    f"kl_C_all_scores_0.99": sum_scores_val + [kl_sum_scores] + [1.0],
    #
    f"kl_nails_stoch_{best_lambda_nails_sto}": nails_stoch_val
    + [kl_nails_stoch]
    + [nails_stoch_cov],
    #
    f"kl_nails_der_{best_lambda_nails_der}": nails_der_val
    + [kl_nails_der]
    + [nails_der_cov],
    #
    f"kl_calirec_{best_lambda_steck}": calirec_val + [kl_calirec] + [calirec_cov],
}

df_bars = pl.DataFrame(bars, strict=False)
df_bars.write_excel(
    plot_dir.joinpath(
        f"final_{topN}_{distribution_type}_{alpha_steck_select}_{n_samples}.xlsx"
    )
)


# %%
