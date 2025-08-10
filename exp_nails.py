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
from arguments.args_nails import args

# %%
# args:
distribution_type = args.distribution_type  # "Uniform" / "Editorial"
act_func = args.act_func  # "softmax"
article_selection = args.article_selection
# "stoch"  # stochastic (stoch) / deterministic (der)
with_replacement = args.with_replacement  # False
lambda_values = args.lambda_values  # [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
filter_auto = args.filter_auto  # True
make_submission_file = args.make_submission_file  # True

unknown_cat = args.unknown_cat  # 1e-6
alpha = args.alpha  # 1e-4
topN = args.topN  # 10
n_samples = args.n_samples  # 1_000_000
n_samples_test = args.n_samples_test  # 1_000_000
show_plot = args.show_plot

PATH = Path(args.data_dir)  # Path(f"data")
file = Path(args.score_dir)  # "data/test_scores.parquet"
PATH_EMB = Path(args.emb_dir)
# PATH.joinpath(f"Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet")

PATH_DUMP = PATH.joinpath(f"ebnerd_submissions")
PATH_DUMP.mkdir(parents=True, exist_ok=True)
np.random.seed(123)
#
article_selection_name = (
    article_selection if article_selection == "stoch" else article_selection
)
plot_dir = Path(f"plot/nails_{distribution_type}_{article_selection_name}")

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
# @TopN
top_n_category_distribution = {}
top_n_categories = {}
top_n_coverage = {}
top_n_scores = {}
top_n_ids = {}

for lambda_val, lambda_scores in tqdm(p_star_omega_scores_dict.items(), ncols=80):
    idx_desc = get_index(
        arr=lambda_scores,
        topN=topN,
        strategy=article_selection,
        replace=with_replacement,
    )

    # Category Names.
    top_n_cat = cand_list_category[idx_desc]
    top_n_categories[lambda_val] = top_n_cat

    # Scores
    top_n_sco = np.take_along_axis(lambda_scores, idx_desc, axis=1)
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

# %%################################################################################
# USER-LEVEL: Kullback–Leibler (KL) Divergence.
kl_scores_user_level = {k: [] for k in lambda_values}
for lambda_val, cat_users in tqdm(top_n_categories.items(), ncols=80):
    for user_i in cat_users:
        user_dist = compute_normalized_distribution(user_i)
        user_dist = {key: user_dist.get(key, 0) for key in nails_keys}
        kl_scores_user_level[lambda_val].append(
            compute_smoothed_kl_divergence(
                p=nails_values,
                q=list(user_dist.values()),
                alpha=alpha,
            )
        )

# %%
# Table 1: KL & Cov
plot_kde(
    kl_scores_user_level,
    xlabel=f"KL@{topN}",
    bw_adjust=1,
    save_path=plot_dir.joinpath(
        f"user_level_kl_{topN}",
        f"user_level_kl_{topN}_bw1.pdf",
    ),
    show_plot=show_plot,
)
_bw_adjust = 10
plot_kde(
    kl_scores_user_level,
    xlabel=f"KL@{topN}",
    bw_adjust=_bw_adjust,
    save_path=plot_dir.joinpath(
        f"user_level_kl_{topN}",
        f"user_level_kl_{topN}_bw{_bw_adjust}_kde.pdf",
    ),
    show_plot=show_plot,
)

means = [np.mean(val) for val in kl_scores_user_level.values()]
stds = [np.std(val) for val in kl_scores_user_level.values()]
# Table 1:
df_kl_user_level = pl.DataFrame(
    {
        "lambda": list(kl_scores_user_level.keys()),
        f"mean_{topN}": means,
        f"std_{topN}": stds,
        f"cov_{topN}": list(top_n_coverage.values()),
    },
    strict=False,
)
df_kl_user_level.write_excel(
    plot_dir.joinpath(
        f"user_level_kl_{topN}",
        f"user_level_kl_scores_{topN}_aggr.xlsx",
    )
)


# %% ################################################################################
# GENRE PROBABILTY.
# TOP@|C| - Summation of scores:
p_star_omega_sum_dist = {k: {} for k in lambda_values}
kl_all_sum_cat_scores = {}
for lambda_val, p_star_omega_scores in p_star_omega_scores_dict.items():
    for cat_i in nails_keys:
        cat_mask = (cand_list_category == cat_i).ravel()
        p_star_omega_sum_dist[lambda_val][cat_i] = (
            np.sum(p_star_omega_scores[:, cat_mask]) / n_samples
        )
    if not np.isclose(sum(p_star_omega_sum_dist[lambda_val].values()), 1):
        raise ValueError(
            f"Probabilies sum to {sum(p_star_omega_sum_dist[lambda_val].values())}"
        )
    kl_all_sum_cat_scores[lambda_val] = compute_smoothed_kl_divergence(
        p=nails_values,
        q=list(p_star_omega_sum_dist[lambda_val].values()),
        alpha=alpha,
    )

# %%
# TOP@N - KL Distribution:
first_key = list(p_star_omega_sum_dist.keys())[0]
for lambda_val in p_star_omega_sum_dist.keys():
    plot_genre_probabilities(
        p_star_omega_sum_dist[first_key],
        p_star_ei,
        p_star_omega_sum_dist[lambda_val],
        labels=[
            f"nails (λ={first_key})",
            f"{distribution_type}",
            f"nails (λ={lambda_val})",
        ],
        title=f"Normalized sum of article scores per category (@ALL).\nKL={kl_all_sum_cat_scores[lambda_val]:.4f}",
        fontsize=30,
        save_path=plot_dir.joinpath(
            "genre_probability_summation_all",
            f"genre_probability_summation_all_{lambda_val}.pdf",
        ),
        show_plot=show_plot,
    )

df_kl_sum_scores = pl.DataFrame(
    {
        "lambda": list(kl_all_sum_cat_scores.keys()),
        "kl@|C|": list(kl_all_sum_cat_scores.values()),
    },
    strict=False,
)
df_kl_sum_scores.write_excel(
    plot_dir.joinpath(
        "genre_probability_summation_all",
        f"genre_probability_summation_all.xlsx",
    )
)


# %%
# TOP@N - TopN Category Distribution:
first_key = list(top_n_category_distribution.keys())[0]
kl_topN_cat = {}

for lambda_val in lambda_values:
    kl_topN_cat[lambda_val] = compute_smoothed_kl_divergence(
        p=nails_values,
        q=list(top_n_category_distribution[lambda_val].values()),
        alpha=alpha,
    )
    plot_genre_probabilities(
        top_n_category_distribution[first_key],
        p_star_ei,
        top_n_category_distribution[lambda_val],
        labels=[
            f"nails (λ=0).",
            f"{distribution_type}",
            f"nails (λ={lambda_val}).",
        ],
        title=f"""top@{topN} (category).
            λ={first_key}: COV: {top_n_coverage[first_key]*100:.1f}%. KL: {kl_topN_cat[first_key]:.2f}.
            λ={lambda_val}: COV: {top_n_coverage[lambda_val]*100:.1f}%. KL: {kl_topN_cat[lambda_val]:.2f}.""",
        fontsize=30,
        save_path=plot_dir.joinpath(
            f"genre_probability_cat_{topN}",
            f"genre_probability_cat_{topN}_{lambda_val}.pdf",
        ),
        show_plot=show_plot,
    )

kl_topN_cat_aggr = pl.DataFrame(
    {
        "lambda": list(kl_all_sum_cat_scores.keys()),
        f"kl@{topN}": list(kl_topN_cat.values()),
    },
    strict=False,
)
kl_topN_cat_aggr.write_excel(
    plot_dir.joinpath(
        f"genre_probability_cat_{topN}",
        f"genre_probability_kl_aggr.xlsx",
    )
)


# %% ################################################################################
# ############ RANKING ############
p_omega_name = "p_w"
# For unknown categories:
# Articles:
df_articles = df_articles.with_columns(
    pl.col(DEFAULT_CATEGORY_STR_COL)
    .replace_strict(p_w, default=unknown_cat)  #
    .alias(p_omega_name)
)
p_omega_lookup = create_lookup_dict(
    df_articles, key=DEFAULT_ARTICLE_ID_COL, value=p_omega_name
)
# ##
# Behaviors:
df = pl.scan_parquet(file).filter(pl.col(DEFAULT_IMPRESSION_ID_COL) > 0).collect()
if n_samples_test:
    df = df.sample(n=n_samples_test, seed=123)

df = df.with_columns(
    pl.col(DEFAULT_INVIEW_ARTICLES_COL)
    .list.eval(pl.element().replace_strict(p_omega_lookup, default=1.0))
    .alias(p_omega_name)
)

# %%

if make_submission_file:
    runs = 10 if article_selection == "stoch" else 1

    for run in tqdm(range(runs), ncols=80):
        for lambda_val in lambda_values:
            _scores = []
            for row_i in df.iter_rows(named=True):
                pred_scores = np.array(row_i["scores"])
                pred_scores = softmax(pred_scores, axis=0)
                p_omega_val = np.array(row_i[p_omega_name])
                weight_one = np.ones(len(pred_scores))

                p_star_omega = compute_nails(
                    p_omega=pred_scores,
                    p_star_ei=p_omega_val,
                    p_ei=weight_one,
                    lambda_=lambda_val,
                )
                _scores.append(p_star_omega)
            df = df.with_columns(pl.Series(f"scores_{lambda_val}", _scores))

            df = df.with_columns(
                pl.col(f"scores_{lambda_val}")
                .map_elements(lambda x: list(rank_predictions_by_score(x)))
                .alias("ranked_scores")
            )

            write_submission_file(
                impression_ids=df[DEFAULT_IMPRESSION_ID_COL],
                prediction_scores=df["ranked_scores"],
                path=PATH_DUMP.joinpath("predictions.txt"),
                filename_zip=f"nails_predictions_{lambda_val}.zip",
            )
