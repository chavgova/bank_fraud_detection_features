import mock
import pandas as pd

from src.main import data_preprocessing, feature_engineering


# note: If what is to be tested differ significantly, have them as separate tests.
def test_data_preprocessing_sample_1():
    # note: More of a style opinion: You can have multiple test cases that are similar to each other listed together.
    cases = [
        {
            "data": pd.DataFrame(
                {
                    "payment_type": ["'a'", "'b'", "'c'"],
                    "category": ["'cat1'", "'cat2'", "'cat3'"],
                    "foo": [1, 2, 3],
                    "amount": [1, 2, 3],
                    "fraud": [1, 2, 3],
                }
            ),
            "expected": pd.DataFrame(
                {
                    "payment_type": ["a", "b", "c"],
                    "category": ["cat1", "cat2", "cat3"],
                    "amount": [1, 2, 3],
                    "fraud": [1, 2, 3],
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
                }
            ),
            "expected": pd.DataFrame(
                {
                    "payment_type": ["a", "b", "c"],
                    "category": ["cat1", "cat2", "cat3"],
                    "amount": [1, 2, 3],
                    "fraud": [1, 2, 3],
                }
            ),
        },
    ]

    for case in cases:

        class MockRunConfig:
            # note: This test basically blocks the sampling functionality. Therefore this test is not enough.
            # note: That is why, we have a second test below.
            sample_rate = 1.0

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
                    "foo": [1, 2, 3, 4],
                    "amount": [1, 2, 3, 4],
                    "fraud": [1, 2, 3, 4],
                }
            ),
            "expected": pd.DataFrame(
                {
                    "payment_type": ["a", "b"],
                    "category": ["cat1", "cat2"],
                    "amount": [1, 2],
                    "fraud": [1, 2],
                }
            ),
        }
    ]

    for case in cases:
        with mock.patch(
            "pandas.DataFrame.sample",
            return_value=case["data"][
                ["payment_type", "category", "amount", "fraud"]
            ].iloc[[0, 1]],
        ):
            out = data_preprocessing(case["data"])
            pd.testing.assert_frame_equal(out, case["expected"], check_like=True)


def test_feature_engineering():
    data = pd.DataFrame(
        {
            "amount": [12.34, 3.14, 0.01, 45.21],
            "category": ["cat1", "cat1", "cat1", "cat2"],
            "payment_type": ["t1", "t1", "t2", "t2"],
        }
    )
    expected = pd.DataFrame(
        {
            "amount": [12.34, 3.14, 0.01, 45.21],
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
        }
    )

    class MockRunConfig:
        threshold_in_eur_to_consider_large_transactions = 45.0

    with mock.patch("src.main.RunConfig", MockRunConfig):
        out = feature_engineering(data)
        pd.testing.assert_frame_equal(out, expected, check_like=True)
