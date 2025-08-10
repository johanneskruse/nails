import argparse
import ast

parser = argparse.ArgumentParser(description="Script with configurable arguments.")

parser.add_argument(
    "--distribution_type",
    type=str,
    default="Editorial",
    choices=["Uniform", "Editorial"],
    help="Type of distribution",
)
parser.add_argument(
    "--act_func", type=str, default="softmax", help="Activation function to use"
)
parser.add_argument(
    "--article_selection",
    type=str,
    default="stoch",
    choices=["stoch", "der"],
    help="Article selection method",
)
parser.add_argument(
    "--with_replacement",
    type=ast.literal_eval,
    default=False,
    help="Sample with replacement (True/False)",
)
parser.add_argument(
    "--lambda_values",
    type=ast.literal_eval,
    default=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
    help="List of lambda values",
)
parser.add_argument(
    "--filter_auto",
    type=ast.literal_eval,
    default=True,
    help="Filter automatically (True/False)",
)
parser.add_argument(
    "--make_submission_file",
    type=ast.literal_eval,
    default=True,
    help="Make submission file (True/False)",
)
parser.add_argument(
    "--unknown_cat",
    type=float,
    default=1e-6,
    help="Unknown category smoothing value",
)
parser.add_argument("--alpha", type=float, default=1e-4, help="Alpha hyperparameter")
parser.add_argument("--topN", type=int, default=10, help="Number of top items")
parser.add_argument(
    "--data_dir", type=str, default="data", help="Path to data directory"
)
parser.add_argument(
    "--score_dir",
    type=str,
    default="data/test_scores.parquet",
    help="Path to score file",
)
parser.add_argument(
    "--emb_dir",
    type=str,
    default="data/Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet",
    help="Path to embeddings file",
)
parser.add_argument(
    "--n_samples",
    type=lambda x: None if x == "None" else int(x),
    default=None,
    help="Number of samples (None for all)",
)
parser.add_argument(
    "--n_samples_test",
    type=lambda x: None if x == "None" else int(x),
    default=None,
    help="Number of test samples (None for all)",
)
parser.add_argument(
    "--show_plot", type=ast.literal_eval, default=False, help="Show plot (True/False)"
)

parser.add_argument(
    "--unknown_item_weight", type=float, default=1e-6, help="Weight for unknown items"
)
parser.add_argument(
    "--alpha_steck_select", type=float, default=0.5, help="Alpha for steck selection"
)

args = parser.parse_args()

print("Arguments received:")
for arg, value in vars(args).items():
    print(f"{arg}: {value}")
