import pandas as pd

from configs.configs import PathsConfig, RunConfig
from src.features import get_encoded_features, get_z_score
from src.model_training import Model


# note: #2: More meaningful name
def data_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    # note: #3: Data read moved out of the function.
    data = data[["payment_type", "category", "amount", "fraud"]]
    data = data.sample(frac=RunConfig.sample_rate)
    # note: #4: Repeating the logic in a loop
    # CSV encoded strings with '. e.g. "'abc'" -> "abc"
    for col in ["payment_type", "category"]:
        data[col] = data[col].str.replace("'", "")
    return data


# note: #5: Changed the name
def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    # note: #6: Made each feature group logic a function.
    data["z_score"] = get_z_score(data)
    data["z_scores_by_category"] = data.groupby("category", group_keys=False).apply(
        get_z_score
    )

    # note: #7: Made encoding per group using a loop.
    for col_name in ["payment_type", "category"]:
        encoded_features = get_encoded_features(data[col_name])
        data.drop(columns=[col_name], axis=1, inplace=True)
        data = pd.concat([data, encoded_features], axis=1)

    # note: #8: Moved hardcoded values to config file.
    data["is_large_transaction"] = (
        data["amount"] > RunConfig.threshold_in_eur_to_consider_large_transactions
    ).astype(float)
    return data


def model_training(data: pd.DataFrame) -> None:
    model = Model()
    print(model.get_cv_scores(data))
    model.train_model(data)
    # note: ?? WDYT?
    model.save_model()


def main():  # pragma: no cover
    # note: #1: Removed the abstraction. Made it part of other code.
    data = pd.read_csv(PathsConfig.raw_data)
    data = data_preprocessing(data)
    data = feature_engineering(data)
    model_training(data)


if __name__ == "__main__":  # pragma: no cover
    # note: #9: Moved code out of the global scope
    main()
