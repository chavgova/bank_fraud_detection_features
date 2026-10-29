import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def get_encoded_features(data: pd.Series) -> pd.DataFrame:
    encoder = OneHotEncoder(sparse_output=False).fit(data.to_numpy().reshape(-1, 1))
    encoded = encoder.transform(data.to_numpy().reshape(-1, 1))
    return pd.DataFrame(encoded, columns=encoder.get_feature_names_out())


def get_z_score(data: pd.DataFrame) -> pd.Series:
    return (data["amount"] - data["amount"].mean()) / data["amount"].std()
