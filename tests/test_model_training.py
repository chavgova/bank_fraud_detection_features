import pickle
import tempfile

import mock
import unittest
from unittest.mock import MagicMock, Mock
import pandas as pd
import numpy as np
from src.model_training import Model
from configs.configs import ModelConfig, PathsConfig, RunConfig
from sklearn.ensemble import RandomForestClassifier

# note: As you see, we do not have unit testing for model training here. I believe testing model training function
# note: is doing more harm than good:
# note: (1) model training is highly stochastic. One need to fix many seeds from many dependencies.
# note: (2) model training is chaotic. A small change in the training data may change the model drastically.
# note: (3) model training is handled by external libraries. These libraries are tested.


def test_model_get_x_y():
    cases = [ {
            "data": pd.DataFrame({"fraud": [1, 0, 0, 1], "amount": [1, 1, 1, 1]}),
            "expected_x": pd.DataFrame({"amount": [1, 1, 1, 1]}),
            "expected_y": pd.Series([1, 0, 0, 1], copy=False, name="fraud"),
        } ]

    for case in cases:
        out_x, out_y = Model._get_x_y(case["data"])
        pd.testing.assert_frame_equal(out_x, case["expected_x"])
        pd.testing.assert_series_equal(out_y, case["expected_y"])


def test_model_save_model():
    class MockModelConfig:
        # note: mock/test variables are good place to have reasonable fun :).
        model_name = "donono"

    # note: Remember the "Isolated" principle. We do not want to change the environment, OS, or anything while
    # note: testing. That is why tempfile is a good tool.
    with tempfile.TemporaryDirectory() as tmpdir:

        class MockPathsConfig:
            model_path = tmpdir + "/"

        with mock.patch("src.model_training.PathsConfig", MockPathsConfig):
            with mock.patch("src.model_training.ModelConfig", MockModelConfig):
                model = Model()
                model._model = "foo"
                model.save_model()

                with open(f"{tmpdir}/donono.pkl", "rb") as f:
                    model = pickle.load(f)
                    assert model == "foo"
                    
def test_get_cv_scores():
    data = pd.DataFrame({"fraud": [0 , 1], "amount": [10.0, 20.0]})
    expected = {
        "test_precision": np.array([0.5, 0.6]),
        "test_recall": np.array([0.7, 0.8]),
        "test_f1": np.array([0.6, 0.7]),
    }

    classifier = mock.Mock()
    
    model = Model()

    with tempfile.TemporaryDirectory() as tmpdir:

        class MockPathsConfig:
            model_path = tmpdir + "/"

        with mock.patch.object(RunConfig, "num_folds", 2):
            with mock.patch.object( model, "_get_model_object", return_value=classifier):
                with mock.patch( "src.model_training.cross_validate", return_value=expected, ) as validate:
                    
                    out = model.get_cv_scores(data)

                    validate.assert_called_once()
                    estimator, actual_x, actual_y = validate.call_args.args

                    assert estimator is classifier 
                    pd.testing.assert_frame_equal( actual_x, data.drop(columns="fraud"))
                    pd.testing.assert_series_equal(actual_y, data["fraud"])

                    assert validate.call_args.kwargs["cv"] ==2  
                    assert validate.call_args.kwargs["scoring"] == [ "precision", "recall", "f1"]
                    assert out is expected

 
def test_train_model():
    test_fraud_data = [1, 0, 0, 1, 1, 2,3,4,5,6, 1, 0, 0, 1, 1, 2,3,4,5,6,1, 0, 0, 1, 1, 2,3,4,5,6,1, 0, 0, 1, 1, 2,3,4,5,6]
    test_amount_data = [1, 1, 1, 1, 5,2,3,4,5,6,1, 1, 1, 1, 5,2,3,4,5,6,1, 1, 1, 1, 5,2,3,4,5,6,1, 1, 1, 1, 5,2,3,4,5,6]
    test_index_data = [8, 3, 5, 1, 2, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,21,22,23,24,25,26,27,28,29,30, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31]
    
    test_input_data = pd.DataFrame({"fraud":test_fraud_data, "amount": test_amount_data} ,index=test_index_data)  
    
    classifier = mock.Mock()
    fitted_model = mock.Mock()  
    classifier.fit.return_value =fitted_model 

    model =Model()  

    with mock.patch.object(model, "_get_model_object", return_value=classifier):
        model.train_model(test_input_data)

    classifier.fit.assert_called_once()
    actual_x, actual_y = classifier.fit.call_args.args

    pd.testing.assert_frame_equal(actual_x, pd.DataFrame({"amount": test_amount_data}, index=test_index_data))
    pd.testing.assert_series_equal( actual_y, pd.Series(test_fraud_data, index=test_index_data, name="fraud"))
    
    assert model._model is fitted_model
    
    
def test_get_model_object():
    model = Model()
    classifier = model._get_model_object()

    assert isinstance(classifier, RandomForestClassifier)
    assert classifier.max_depth == ModelConfig.max_depth
    assert classifier.class_weight == ModelConfig.class_weight
    assert classifier.random_state == RunConfig.random_state