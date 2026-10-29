import pickle

import pandas as pd
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

from configs.configs import ModelConfig, PathsConfig, RunConfig


# note: It is useful to use classes for models. It can encapsulate save and load operations along with other
# note: preparations.
class Model:
    def __init__(self):
        self._model = None
        self._cv_scores = None

    # note: Tip: If you use inheritence, you can keep most of the code in ABC, and have concrete classes for trying
    # note: different models.
    @staticmethod
    def _get_model_object() -> RandomForestClassifier:
        return RandomForestClassifier(
            max_depth=ModelConfig.max_depth, class_weight=ModelConfig.class_weight
        )

    @staticmethod
    def _get_x_y(data: pd.DataFrame) -> tuple[DataFrame, Series]:
        return data.drop("fraud", axis=1), data["fraud"]

    def train_model(self, data: pd.DataFrame) -> None:
        x, y = self._get_x_y(data)
        classifier = self._get_model_object()
        self._model = classifier.fit(x, y)

    def get_cv_scores(self, data: pd.DataFrame) -> dict[str, float]:
        # note: #10: Made CV and model training separate
        x, y = self._get_x_y(data)
        classifier = self._get_model_object()
        return cross_validate(
            classifier,
            x,
            y,
            cv=RunConfig.num_folds,
            return_train_score=True,
            n_jobs=-1,
            scoring=["precision", "recall", "f1"],
        )

    def save_model(self):
        with open(f"{PathsConfig.model_path}{ModelConfig.model_name}.pkl", "wb") as f:
            pickle.dump(self._model, f)
