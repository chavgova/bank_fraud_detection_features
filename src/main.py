import numpy as np
import pandas as pd

from configs.configs import ModelConfig, PathsConfig, RunConfig
from src.features import (
    feature_number_transactions_per_category,
    feature_transaction_made_in_phone_registered_country,
    get_encoded_features,
    get_z_score,
)
from src.model_training import Model

# note: #2: More meaningful name
def data_preprocessing(transactions_df: pd.DataFrame) -> pd.DataFrame:
    transactions_df = transactions_df.copy() # removed hardcoded columns 
    transactions_df = transactions_df.sample(frac=RunConfig.sample_rate, random_state=RunConfig.random_state)
    
    # CSV encoded strings with '. e.g. "'abc'" -> "abc"
    for col in ["payment_type", "category"]:
        transactions_df[col] = transactions_df[col].str.replace("'", "")
        
    transactions_df["chck_date"] = pd.to_datetime(transactions_df["chck_date"],
        format="mixed", errors="coerce", utc=True)  
    
    transactions_df = transactions_df.dropna(subset=["chck_date"]).copy()    
    transactions_df["reference_date"] = transactions_df["chck_date"].dt.normalize()  
    
    return transactions_df


# note: #5: Changed the name
def feature_engineering(transactions_df: pd.DataFrame) -> pd.DataFrame:
    # note: #6: Made each feature group logic a function.
    transaction_count_features = (feature_number_transactions_per_category(transactions_df, transactions_df["reference_date"], 
                                 RunConfig.list_of_months_to_consider_for_transaction_count_feature))
    
    features = transactions_df[["amount", "fraud", "reference_date"]].copy()
    features["z_score"] = get_z_score(transactions_df)
    grouped_amounts = transactions_df.groupby("category")["amount"]
    features["z_scores_by_category"] = (transactions_df["amount"] - 
                                        grouped_amounts.transform("mean")) / grouped_amounts.transform("std")  

    # note: #7: Made encoding per group using a loop.
    for col_name in ["payment_type", "category"]:
        encoded_features = get_encoded_features(transactions_df[col_name])
        features = pd.concat([features, encoded_features], axis=1) 

    # note: #8: Moved hardcoded values to config file.
    features["is_large_transaction"] = (
        transactions_df["amount"] > RunConfig.threshold_in_eur_to_consider_large_transactions).astype(float)
    
    features["transaction_made_in_phone_registered_country"] = (
        feature_transaction_made_in_phone_registered_country(transactions_df))
    
    features = features.join(transaction_count_features, on="reference_date", validate="many_to_one") 
    features = features.drop(columns="reference_date")

    return features


def model_training(data: pd.DataFrame) -> None:
    model = Model()
    print(model.get_cv_scores(data))
    model.train_model(data)
    # note: ?? WDYT?
    model.save_model()


def run_experiments(features: pd.DataFrame) -> pd.DataFrame:
    count_columns = [
        column_name
        for column_name in features.columns
        if column_name.startswith("transaction_count_") ]
    
    phone_column = "transaction_made_in_phone_registered_country"

    baseline_columns = [column_name for column_name in features.columns 
                        if column_name not in ["fraud", phone_column, *count_columns] ]

    feature_groups = {"baseline": baseline_columns, "counts": count_columns, "phone": [phone_column]}

    results = []

    for experiment_name, group_names in RunConfig.experiments:
        feature_columns = [
            column_name
            for group_name in group_names
            for column_name in feature_groups[group_name]]

        experiment_data = features[["fraud", *feature_columns]]
        scores = Model().get_cv_scores(experiment_data)

        for fold_number in range(RunConfig.num_folds):
            results.append({ "experiment": experiment_name,
                    "fold": fold_number,
                    "precision": scores["test_precision"][fold_number],
                    "recall": scores["test_recall"][fold_number],
                    "f1": scores["test_f1"][fold_number] } )

    return pd.DataFrame(results)

def main():  # pragma: no cover
    transactions_data = pd.read_csv(PathsConfig.raw_data)
    transactions_data = data_preprocessing(transactions_data) 

    features= feature_engineering(transactions_data)   

    features = (features.loc[features["fraud"].isin([0, 1])])
    features = features.reset_index(drop=True)

    results= run_experiments(features) 

    pd.DataFrame.to_csv(results, PathsConfig.experiment_results, index=False)
    print(results)
    print(results.groupby("experiment")[ ["precision", "recall", "f1"] ].mean() )


if __name__ == "__main__":  # pragma: no cover
    # note: #9: Moved code out of the global scope
    main()
