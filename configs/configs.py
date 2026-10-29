from dataclasses import dataclass


@dataclass(init=False, frozen=True)
class PathsConfig:
    raw_data: str = "data/trx_data.csv"  # Raw upstream data.
    model_path: str = "models/"  # Model store


@dataclass(init=False, frozen=True)
class RunConfig:
    # The fraction of the entire raw data to be selected at random.
    # For the sake of speed in development, set this to smaller number. In production, always expected to be
    # set to 1.0.
    sample_rate: float = 1.0
    num_folds: int = 5
    threshold_in_eur_to_consider_large_transactions: float = 123.45


@dataclass(init=False, frozen=True)
class ModelConfig:
    model_name: str = "alpha"
    max_depth: int = 100
    class_weight: str = "balanced"
