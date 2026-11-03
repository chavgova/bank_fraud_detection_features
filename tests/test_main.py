import mock
import pandas as pd
import numpy as np

from src.main import data_preprocessing, feature_engineering, model_training, run_experiments_comparing_features
from configs.configs import RunConfig
from src.model_training import Model
from unittest.mock import MagicMock, Mock



# note: If what is to be tested differ significantly, have them as separate tests.
def test_data_preprocessing_sample_1():
    # note: More of a style opinion: You can have multiple test cases that are similar to each other listed together.
    
    dates = ["2025-05-10", "2025-05-11", "2025-05-12"]
    cases = [
        {
            "data": pd.DataFrame(
                {
                    "payment_type": ["'a'", "'b'", "'c'"],
                    "category": ["'cat1'", "'cat2'", "'cat3'"],
                    "foo": [1, 2, 3],
                    "amount": [1, 2, 3],
                    "fraud": [1, 2, 3],
                    "chck_date": dates,
                }
            ),
            "expected": pd.DataFrame(
                {
                    "payment_type": ["a", "b", "c"],
                    "category": ["cat1", "cat2", "cat3"],
                    "amount": [1, 2, 3],
                    "fraud": [1, 2, 3],
                    "chck_date": pd.to_datetime(dates, utc=True),
                    "reference_date": pd.to_datetime(dates, utc=True),
                    "foo": [1, 2, 3],
                }
            ),
        },
        {
            "data": pd.DataFrame(
                {
                    "payment_type": ["a", "b", "c"],
                    "category": ["cat1", "cat2", "cat3'''''''''"],
                    "foo": [1, 2, 3],
                    "amount": [1, 2, 3],
                    "fraud": [1, 2, 3],
                    "chck_date": dates,
                    
                }
            ),
            "expected": pd.DataFrame(
                {
                    "payment_type": ["a", "b", "c"],
                    "category": ["cat1", "cat2", "cat3"],
                    "amount": [1, 2, 3],
                    "fraud": [1, 2, 3],
                    "chck_date": pd.to_datetime(dates, utc=True),
                    "reference_date": pd.to_datetime(dates, utc=True),
                    "foo": [1, 2, 3],
                }
            ),
        },
    ]

    for case in cases:

        class MockRunConfig:
            # note: This test basically blocks the sampling functionality. Therefore this test is not enough.
            # note: That is why, we have a second test below.
            sample_rate = 1.0
            random_state: int = 8

        # note: Important to mock relative import. Relative to the code that is importing the object. In this
        # note: case, the RunConfig is being imported in src/main.py, so we need to mock according to that.
        # note: See: https://docs.python.org/3/library/unittest.mock.html#where-to-patch
        with mock.patch("src.main.RunConfig", MockRunConfig):
            out = data_preprocessing(case["data"])
            pd.testing.assert_frame_equal(out, case["expected"], check_like=True)


def test_data_preprocessing_sample_non_1():
    cases = [
        {
            "data": pd.DataFrame(
                {
                    "payment_type": ["'a'", "'b'", "'c'", "'d'"],
                    "category": ["'cat1'", "'cat2'", "'cat3'", "'cat4'"],
                    "amount": [1, 2, 3, 4],
                    "fraud": [1, 2, 3, 4],
                    "chck_date": ["2025-05-10", "2025-05-11", "2025-05-12", "2025-01-10"]
                }
            ),
            "expected": pd.DataFrame(
                {
                    "payment_type": ["a", "b"],
                    "category": ["cat1", "cat2"],
                    "amount": [1, 2],
                    "fraud": [1, 2],
                    "chck_date": pd.to_datetime(["2025-05-10", "2025-05-11"], utc=True),
                    "reference_date": pd.to_datetime(["2025-05-10", "2025-05-11"], utc=True),
                }
            ),
        }
    ]

    for case in cases:
        with mock.patch(
            "pandas.DataFrame.sample",
            return_value=case["data"][ ["payment_type", "category", "amount", "fraud", "chck_date"] ].iloc[[0, 1]],
        ):
            out = data_preprocessing(case["data"])
            pd.testing.assert_frame_equal(out, case["expected"], check_like=True)


def test_feature_engineering():
    dates = pd.to_datetime( ["2025-04-10", "2025-05-09", "2025-05-10", "2025-05-10"], utc=True)
    data = pd.DataFrame(
        {
            "amount": [12.34, 3.14, 0.01, 45.21],
            "category": ["cat1", "cat1", "cat1", "cat2"],
            "payment_type": ["t1", "t1", "t2", "t2"],
            "chck_date": dates,
            "fraud": [0, 1, 0, 1],
            "reference_date": dates,
            "cust_acc": ["BG000000000000000000", "GB82WEST111111111111", "FR4444444444444444444", "KU88888888888888888888"],
            "phone": [None]*4
        }
    )
    expected = pd.DataFrame(
        {
            "amount": [12.34, 3.14, 0.01, 45.21],
            "fraud": [0, 1, 0, 1],
            "z_score": [
                -0.13698386327001555,
                -0.5815170350810006,
                -0.7327549511427814,
                1.4512558494937975,
            ],
            "z_scores_by_category": [
                1.1197474139236852,
                -0.31569283801749964,
                -0.8040545759061855,
                None,
            ],
            "x0_t1": [1.0, 1.0, 0.0, 0.0],
            "x0_t2": [0.0, 0.0, 1.0, 1.0],
            "x0_cat1": [1.0, 1.0, 1.0, 0.0],
            "x0_cat2": [0.0, 0.0, 0.0, 1.0],
            "is_large_transaction": [0.0, 0.0, 0.0, 1.0],
            "transaction_made_in_phone_registered_country": [ float("nan") ] * 4,
            "transaction_count_cat1_1_months": [0, 1, 2, 2],
            "transaction_count_cat2_1_months": [0, 0, 0, 0],
        }
    )

    class MockRunConfig:
        threshold_in_eur_to_consider_large_transactions = 45.0
        list_of_months_to_consider_for_transaction_count_feature = (1,) # tuple

    with mock.patch("src.main.RunConfig", MockRunConfig):
        out = feature_engineering(data)
        pd.testing.assert_frame_equal(out, expected, check_like=True)


def test_model_training():
    data = pd.DataFrame({ "fraud": [0, 1, 0], "amount": [100.0, 200.0, 150.0]})
    
    # mocking the model class and autospec checks mock vs original
    with mock.patch("src.main.Model",autospec=True) as main_model:
        mock_model = main_model.return_value
        mock_model.get_cv_scores.return_value = {"test_f1": np.array([0.8, 0.9, 0.85])}

        model_training(data)

    main_model.assert_called_once_with()
    
    # we dont check .fit here as its tested in test_model_training.py   
    # we check if each of the methods is called once with the correct data 
    mock_model.get_cv_scores.assert_called_once()  
    assert mock_model.get_cv_scores.call_args.args[0] is data
    mock_model.train_model.assert_called_once()  
    assert mock_model.train_model.call_args.args[0] is data
    mock_model.save_model.assert_called_once_with()
    
def test_run_experiments_comparing_features_1exp():
    features = pd.DataFrame({
        "fraud": [0, 1], 
        "amount": [0.01, 8888.0],
        "z_score": [-1.0, 1.0],  
    }, index=[8, 3])

    original = features.copy(deep=True) 
    experiments =(("baseline", ("baseline",)),)   
    expected_columns = [ ["fraud", "amount", "z_score"]]

    cv_results = [
        {
            "test_precision": np.array([1.0, 0.5]),
            "test_recall": np.array([0.5, 1.0]),
            "test_f1": np.array([0.88, 0.77]),
        },
    ]  

    expected_results = pd.DataFrame({
        "experiment": [ "baseline", "baseline"],  
        "fold": [0, 1],
        "precision": [ 1.0, 0.5], 
        "recall": [0.5, 1.0],
        "f1": [0.88, 0.77],
    })
     
    
    with (  
        mock.patch.object(RunConfig, "num_folds", 2), 
        mock.patch.object(RunConfig, "experiments", experiments),
        mock.patch("src.main.Model", autospec=True) as factory): 
        model = factory.return_value
        model.get_cv_scores.side_effect = cv_results 

        actual = run_experiments_comparing_features(features)  

    assert factory.call_count ==1   
    assert model.get_cv_scores.call_count ==1
 
    for call, columns in zip( model.get_cv_scores.call_args_list, expected_columns, strict=True):
        pd.testing.assert_frame_equal( call.args[0], features[columns])

    model.train_model.assert_not_called()
    model.save_model.assert_not_called()

    pd.testing.assert_frame_equal(actual, expected_results)
    pd.testing.assert_frame_equal(features, original)
    
    
def test_run_experiments_comparing_features_4exp(): 
    features = pd.DataFrame({
        "fraud": [0, 1],   
        "amount": [0.01, 8888.0], 
        "z_score": [-1.0, 1.0],  
        "transaction_count_cyberpunk_1_months":[0, 2],
        "transaction_count_cyberpunk_2_months":[1, 3],   
        "transaction_made_in_phone_registered_country":[ 1.0, float("nan") ],
    }, index=[8, 3])

    original = features.copy(deep=True)

    experiments = (
        ("baseline", ("baseline",)),
        ("baseline_and_counts", ("baseline", "counts")),
        ("baseline_and_phone", ("baseline", "phone")),
        ("all", ("baseline", "counts", "phone")),
    )

    expected_columns = [
        ["fraud", "amount", "z_score"],
        [ "fraud", "amount", "z_score", "transaction_count_cyberpunk_1_months", "transaction_count_cyberpunk_2_months"],
        [ "fraud", "amount", "z_score", "transaction_made_in_phone_registered_country"],
        [ "fraud", "amount", "z_score", "transaction_count_cyberpunk_1_months", "transaction_count_cyberpunk_2_months", 
         "transaction_made_in_phone_registered_country"],
    ]

    cv_results = [
        {
            "test_precision": np.array([1.0, 0.5]),  
            "test_recall": np.array([0.5, 1.0]),
            "test_f1": np.array([0.88, 0.77]),
        },
        {
            "test_precision": np.array([1.0, 1.0]),
            "test_recall": np.array([1.0, 0.5]),  
            "test_f1": np.array([1.0, 0.77]),
        },
        {
            "test_precision": np.array([0.5, 0.5]),
            "test_recall": np.array([0.5, 0.5]),
            "test_f1": np.array([0.5, 0.5]),
        },
        {
            "test_precision": np.array([1.0, 1.0]),  
            "test_recall": np.array([1.0, 1.0]),
            "test_f1": np.array([1.0, 1.0]),  
        },
    ]

    expected_results = pd.DataFrame({
        "experiment": [
            "baseline", "baseline",
            "baseline_and_counts", "baseline_and_counts",
            "baseline_and_phone", "baseline_and_phone",
            "all", "all",  
        ],
        "fold": 4 *[0, 1],
        "precision": [ 1.0, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0],
        "recall": [0.5, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0],
        "f1": [0.88, 0.77, 1.0, 0.77, 0.5, 0.5, 1.0, 1.0],  
    })

    with (  
        mock.patch.object(RunConfig, "num_folds", 2),
        mock.patch.object(RunConfig, "experiments", experiments),
        mock.patch("src.main.Model", autospec=True) as factory): 
        model = factory.return_value
        model.get_cv_scores.side_effect = cv_results 

        actual = run_experiments_comparing_features(features)
  
    assert factory.call_count ==4  
    assert model.get_cv_scores.call_count ==4

    for call, columns in zip( model.get_cv_scores.call_args_list, expected_columns, strict=True):
        pd.testing.assert_frame_equal( call.args[0], features[columns])

    model.train_model.assert_not_called()
    model.save_model.assert_not_called()

    pd.testing.assert_frame_equal(actual, expected_results)
    pd.testing.assert_frame_equal(features, original)