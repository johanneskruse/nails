import argparse
import ast

parser = argparse.ArgumentParser(description="Script with configurable arguments.")

# Main parameters
parser.add_argument(
    "--distribution_type",
    type=str,
    default="Editorial",
    choices=["Uniform", "Editorial"],
    help="Type of distribution",
)
parser.add_argument(
    "--model",
    type=str,
    default="nrms_no_sig",
    choices=["zorro", "nrms_sig", "nrms_no_sig"],
    help="Model name",
)
parser.add_argument(
    "--act_func", type=str, default="softmax", help="Activation function to use"
)
parser.add_argument("--topN", type=int, default=10, help="Number of top items")
parser.add_argument(
    "--with_replacement",
    type=ast.literal_eval,
    default=False,
    help="Sample with replacement (True/False)",
)
parser.add_argument(
    "--filter_auto",
    type=ast.literal_eval,
    default=True,
    help="Filter automatically (True/False)",
)


# Lambda values
parser.add_argument(
    "--lambda_best_editorial_nails_sto",
    type=float,
    default=0.7,
    help="Lambda for editorial nailsse stochastic",
)
parser.add_argument(
    "--lambda_best_editorial_nails_der",
    type=float,
    default=0.7,
    help="Lambda for editorial nailsse deterministic",
)
parser.add_argument(
    "--lambda_best_editorial_steck",
    type=float,
    default=0.5,
    help="Lambda for editorial steck",
)

parser.add_argument(
    "--lambda_best_uniform_nails_sto",
    type=float,
    default=0.99,
    help="Lambda for uniform nailsse stochastic",
)
parser.add_argument(
    "--lambda_best_uniform_nails_der",
    type=float,
    default=0.3,
    help="Lambda for uniform nailsse deterministic",
)
parser.add_argument(
    "--lambda_best_uniform_steck",
    type=float,
    default=0.99,
    help="Lambda for uniform steck",
)

# Sampling & alphas
parser.add_argument("--n_samples", type=int, default=None, help="Number of samples")
parser.add_argument(
    "--alpha_steck_select", type=float, default=0.5, help="Alpha for steck selection"
)
parser.add_argument("--alpha", type=float, default=1e-4, help="Alpha hyperparameter")

# Plotting
parser.add_argument(
    "--show_plot", type=ast.literal_eval, default=False, help="Show plot (True/False)"
)

# Directories
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

args = parser.parse_args()
