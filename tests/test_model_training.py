import pickle
import tempfile

import mock
import pandas as pd

from src.model_training import Model

# note: As you see, we do not have unit testing for model training here. I believe testing model training function
# note: is doing more harm than good:
# note: (1) model training is highly stochastic. One need to fix many seeds from many dependencies.
# note: (2) model training is chaotic. A small change in the training data may change the model drastically.
# note: (3) model training is handled by external libraries. These libraries are tested.


def test_model_get_x_y():
    cases = [
        {
            "data": pd.DataFrame({"fraud": [1, 0, 0, 1], "amount": [1, 1, 1, 1]}),
            "expected_x": pd.DataFrame({"amount": [1, 1, 1, 1]}),
            "expected_y": pd.Series([1, 0, 0, 1], copy=False, name="fraud"),
        }
    ]

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
