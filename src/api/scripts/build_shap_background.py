import argparse
import os

import polars as pl

from utils.preprocessing_handler import PreprocessingHandler
from utils.statics import catboost_model_name, lightgbm_model_name, xgboost_model_name
from utils.utils import split_train_test


def build_shap_background(
    passes_path: str,
    artifacts_root_dir: str,
    model_types: list[str],
    n_samples: int = 100,
    seed: int = 42,
) -> None:
    """Randomly sample training passes used as the SHAP background distribution for each model artifact folder.

    The held-out test match is excluded and the same null handling as training is applied.

    Args:
        passes_path (str): Path to the raw passes parquet file.
        artifacts_root_dir (str): Root folder containing one artifact folder per model type.
        model_types (list[str]): Model types to write the background sample for.
        n_samples (int, optional): Number of background passes. Defaults to 100.
        seed (int, optional): Sampling seed. Defaults to 42.
    """
    train_df, _ = split_train_test(passes_df=pl.read_parquet(passes_path))
    background = (
        PreprocessingHandler(df=train_df, categorical_columns=["height", "body_part"])
        .preprocess_categorical_columns()
        .drop("outcome")
        .sample(n_samples, seed=seed)
    )
    for model_type in model_types:
        background.write_csv(
            os.path.join(artifacts_root_dir, model_type, "shap_background.csv")
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the SHAP background sample into the API artifact folders."
    )
    parser.add_argument(
        "--passes-path", default=os.path.join("data", "02-analysis", "passes.parquet")
    )
    parser.add_argument(
        "--artifacts-root-dir", default=os.path.join("api", "artifacts")
    )
    parser.add_argument("--n-samples", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_shap_background(
        passes_path=args.passes_path,
        artifacts_root_dir=args.artifacts_root_dir,
        model_types=[lightgbm_model_name, xgboost_model_name, catboost_model_name],
        n_samples=args.n_samples,
    )


if __name__ == "__main__":
    main()
