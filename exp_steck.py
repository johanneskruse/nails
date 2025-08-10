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
    map_ranked_scores_to_original,
    greedy_steck_rerank,
    compute_nails,
    softmax,
)
from utils._utils import *
from arguments.args_steck import args

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

unknown_item_weight = args.unknown_item_weight
alpha_steck_select = args.alpha_steck_select

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
plot_dir = Path(f"plot/steck_{distribution_type}_{article_selection_name}")

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

n_samples = cand_scores.shape[0]

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


################################################################################
# ======= EVERYTHING ABOVE IS JUST COPY PASTE FROM 'subjective_prob.py' =======
################################################################################

# %%################################################################################
# Steck Greedy re-rank.
####################################################################################

# Initialize results containers
top_n_category_dist_steck = {k: [] for k in lambda_values}
top_n_kl_steck = {k: [] for k in lambda_values}
coverage_steck = {k: [] for k in lambda_values}

target_dist_steck_type = distribution_type

# Prepare candidate distributions
lookup_cand_list = {id: lookup_cat_str[id] for id in cand_list_ids}

# Trim to sample size
temp_cand_scores = cand_scores

# Evaluation loop
for lambda_val in lambda_values:

    # Reset
    selected_ids_all = []
    selected_cats_all = []

    for user_scores in tqdm(temp_cand_scores, total=len(temp_cand_scores), ncols=80):
        p_target = p_star_ei
        # Get top item IDs using Steck reranking
        top_ids = greedy_steck_rerank(
            ids=cand_list_ids,
            lookup_attr=lookup_cand_list,
            scores=user_scores,
            p_target=p_target,
            lambda_=lambda_val,
            k=topN,
            alpha=alpha_steck_select,  # 0.5 for more smooth.
        )
        # Map item IDs to categories.
        top_cats = [lookup_cand_list[id] for id in top_ids]

        # Compute KL user-level.
        cat_dist = compute_normalized_distribution(top_cats)
        cat_dist = {k: cat_dist.get(k, 0.0) for k in nails_keys}
        kl_score = compute_smoothed_kl_divergence(
            p=list(p_target.values()),
            q=list(cat_dist.values()),
            alpha=alpha,
        )

        # #
        top_n_kl_steck[lambda_val].append(kl_score)
        selected_cats_all.extend(top_cats)
        selected_ids_all.extend(top_ids)

    # Compute final metrics per lambda
    coverage_steck[lambda_val] = coverage_fraction(selected_ids_all, C=cand_list_ids)

    cat_dist_steck = compute_normalized_distribution(selected_cats_all)
    top_n_category_dist_steck[lambda_val] = {
        k: cat_dist_steck.get(k, 0.0) for k in nails_keys
    }

# %% USER-LEVEL
# Table 1: KL & Cov
plot_kde(
    top_n_kl_steck,
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
    top_n_kl_steck,
    xlabel=f"KL@{topN}",
    bw_adjust=_bw_adjust,
    save_path=plot_dir.joinpath(
        f"user_level_kl_{topN}",
        f"user_level_kl_{topN}_bw{_bw_adjust}_kde.pdf",
    ),
    show_plot=show_plot,
)

means = [np.mean(val) for val in top_n_kl_steck.values()]
stds = [np.std(val) for val in top_n_kl_steck.values()]
# Table 1:
df_kl_user_level = pl.DataFrame(
    {
        "lambda": list(top_n_kl_steck.keys()),
        f"mean_{topN}": means,
        f"std_{topN}": stds,
        f"cov_{topN}": list(coverage_steck.values()),
    },
    strict=False,
)
df_kl_user_level.write_excel(
    plot_dir.joinpath(
        f"user_level_kl_{topN}",
        f"user_level_kl_scores_{topN}_aggr.xlsx",
    )
)


# %%
# Distributions to plot
distributions = [top_n_category_dist_steck[lam] for lam in lambda_values]

kl_system_level = {}
for lambda_val in lambda_values:
    kl_system_level[lambda_val] = compute_smoothed_kl_divergence(
        p=list(p_target.values()),
        q=list(top_n_category_dist_steck[lambda_val].values()),
        alpha=alpha,
    )

# Labels for each plot
labels = [f"{target_dist_steck_type} Distribution"] + [
    f"λ={lam} KL@10 = {round(np.mean(kl_system_level[lam]), 4)}\nCOV@10 = {round(np.mean(coverage_steck[lam]), 3)}"
    for lam in lambda_values
]

# Call the plotting function
plot_genre_probabilities(
    *distributions,
    labels=labels,
    title=f"{distribution_type}",
    figsize=(14, 10),
    fontsize=18,
    save_path=plot_dir.joinpath(
        f"genre_probability_cat_{topN}",
        f"genre_probability_cat_{topN}.pdf",
    ),
    show_plot=show_plot,
)

write_json_file(
    top_n_category_dist_steck,
    path=plot_dir.joinpath(
        f"genre_probability_cat_{topN}",
        f"bars_{topN}_{distribution_type}_{alpha_steck_select}_bars.json",
    ),
)
write_json_file(
    coverage_steck,
    path=plot_dir.joinpath(
        f"genre_probability_cat_{topN}",
        f"bars_{topN}_{distribution_type}_{alpha_steck_select}_cov.json",
    ),
)
write_json_file(
    kl_system_level,
    path=plot_dir.joinpath(
        f"genre_probability_cat_{topN}",
        f"bars_{topN}_{distribution_type}_{alpha_steck_select}_kl_system_level.json",
    ),
)

# %% ============================================
# COMPUTE RANKING METRICS
# ===============================================
# ##
# Behaviors:
df = pl.scan_parquet(file).filter(pl.col(DEFAULT_IMPRESSION_ID_COL) > 0).collect()
if n_samples_test:
    df = df.sample(n=n_samples_test, seed=123)


# %%

lookup_cat_str_fix = {
    aid: lookup_cat_str.get(aid) if lookup_cat_str[aid] in nails_keys else "ukn"
    for aid in lookup_cat_str.keys()
}

p_star_ei["ukn"] = unknown_item_weight

for lambda_val in tqdm(lambda_values, ncols=80):
    _scores = []
    for row_i in df.iter_rows(named=True):
        # dict_keys(['impression_id', 'article_ids_inview', 'scores', 'article_ids_clicked', 'p_w'])
        pred_scores = np.array(row_i["scores"])
        if act_func == "softmax":
            pred_scores = softmax(pred_scores, axis=0)
        impr_ids = np.array(row_i[DEFAULT_INVIEW_ARTICLES_COL])

        ranked_ids = greedy_steck_rerank(
            ids=impr_ids,
            scores=pred_scores,
            lookup_attr=lookup_cat_str_fix,
            p_target=p_star_ei,
            lambda_=lambda_val,
            k=len(impr_ids),
            alpha=alpha,
        )

        scores = map_ranked_scores_to_original(impr_ids, ranked_ids)
        _scores.append(scores)
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
        filename_zip=f"steck_predictions_{lambda_val}.zip",
    )

# %%
