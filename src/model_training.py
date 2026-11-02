import pickle

import numpy as np
import pandas as pd
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
        max_depth=ModelConfig.max_depth,
        class_weight=ModelConfig.class_weight,
        random_state=RunConfig.random_state)

    @staticmethod
    def _get_x_y(data: pd.DataFrame, ) -> tuple[pd.DataFrame, pd.Series]:
        return data.drop(columns="fraud"), data["fraud"]


    # @staticmethod
    # def _get_time_split_x_y(data: pd.DataFrame) -> tuple[DataFrame, Series]:
    #     # splitting the data by time period is crucial against data leakage (preventing training on future data)
    #     # we keep transactions from the same day together so they are in the same split
    #     calendar_dates = pd.date_range(reference_dates.min(), reference_dates.max(), freq="D")

    #     splitter = TimeSeriesSplit(n_splits=RunConfig.num_folds)
    #     splits = []

    #     for train_days, validation_days in splitter.split(calendar_dates):
    #         train_rows_index = np.flatnonzero(reference_dates.isin(calendar_dates[train_days]))    
    #         validation_rows_index = np.flatnonzero(reference_dates.isin(calendar_dates[validation_days])) 
 
    #         if train_rows_index.size == 0 or validation_rows_index.size == 0:
    #             raise ValueError("Split is empty")

    #         splits.append((train_rows_index, validation_rows_index))

    #     return splits

    def train_model(self, data: pd.DataFrame) -> None:
        x, y = self._get_x_y(data)
        classifier = self._get_model_object()
        self._model = classifier.fit(x, y)

    def get_cv_scores(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
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
