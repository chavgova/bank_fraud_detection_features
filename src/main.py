import pickle
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OneHotEncoder

from configs.configs import PathsConfig, RunConfig


def read_data(path: str):
    return pd.read_csv(path)  # note: #1: Too shallow abstraction!  (Remove function)


# note: #2:  N1: Choose descriptive names (data_processing)
def data_crunching() -> pd.DataFrame:
    # note: #3: G30: Functions should do one thing
    data = read_data(PathsConfig.raw_data)
    data = data.sample(frac=RunConfig.sample_rate)
    data = data[["payment_type", "category", "amount", "fraud"]]

    # note: C2: Obsolete Comment (Remove comment)
    # Clean the payment type and category by removing the ' characters.
    data["payment_type"] = data["payment_type"].str.replace("'", "")
    # note: #4: G5: Duplication (Make it a loop)
    data["category"] = data["category"].str.replace("'", "")

    return data


# note: #5: F20: Function names should say what they do (feat_engineering)
def model_preparation(data: pd.DataFrame) -> pd.DataFrame:
    # note: #6: Testability - Too much logic in one function makes testing and debugging difficult
    z_score = lambda x: x["amount"] - x["amount"].mean() / x["amount"].std()
    # note: err: () missing for substraction. This error will be difficult to be caught without its own test cases.
    # note: Even simple math functions may have errors!
    data["z-score"] = z_score(data)

    # note: G19: Use explanatory variables
    z_scores_by_cat = data.groupby("category", group_keys=False).apply(z_score)
    data["z_scores_by_cat"] = z_scores_by_cat

    encoder = OneHotEncoder(sparse_output=False).fit(
        data["category"].to_numpy().reshape(-1, 1)
    )
    # note: Var name differed by 1 char.
    encoded = encoder.transform(data["category"].to_numpy().reshape(-1, 1))
    encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
    data = pd.concat([data, encoded], axis=1)
    data.drop("category", axis=1, inplace=True)

    # note: #7: G5: Duplication (Make it a loop)
    encoder = OneHotEncoder(sparse_output=False).fit(
        data["payment_type"].to_numpy().reshape(-1, 1)
    )
    encoded = encoder.transform(data["payment_type"].to_numpy().reshape(-1, 1))
    encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
    data.drop(columns=["payment_type"], axis=1, inplace=True)
    data = pd.concat([data, encoded], axis=1)

    # note: #8: G25: Replace magic numbers with named constants
    data["is_large_trx"] = (data["amount"] > 123.45).astype(float)

    return data


def model_training(data: pd.DataFrame, is_run_cv: bool = False) -> None:
    y = data["fraud"]
    x = data.drop("fraud", axis=1)
    # note: G21: Understand the algorithm (*) (max_depth)
    classifier = RandomForestClassifier(max_depth=1, class_weight="balanced")
    # note: C5: Commented-out code
    # x = data.drop("amount", axis=1)

    # note: #10: F3: Flag Arguments
    cv_scores = {}
    if is_run_cv:
        cv_scores = cross_validate(
            classifier,
            x,
            y,
            cv=RunConfig.num_folds,
            return_train_score=True,
            n_jobs=-1,
            scoring=["precision", "recall", "f1"],
        )
        print(cv_scores)

    model = classifier.fit(x, y)

    return cv_scores if is_run_cv else model


# note: F1: Too-many arguments
def save_model(
    model: Any,
    directory: str,
    separator: str,
    file_name: str,
    cv_scores: dict = {},
    is_s3: bool = False,
) -> None:
    obj_to_save = model if cv_scores == {} else {"scores": cv_scores, "model": model}

    with open(f"{directory}{separator}{file_name}", "wb") as f:
        pickle.dump(obj_to_save, f)


if __name__ == "__main__":
    # note: #9 this scope is global. The var's in this scope may overshadow others in func's with the same name.
    # note: Good idea to immediately call main() func to prevent that.
    data = data_crunching()
    data = model_preparation(data)
    model = model_training(data, is_run_cv=False)
    save_model(model, "models", "/", "model.pkl", is_s3=True)
