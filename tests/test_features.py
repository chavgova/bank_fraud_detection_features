import pandas as pd

from src.features import get_encoded_features, get_z_score


def test_get_encoded_features():
    cases = [
        {
            "data": pd.Series(["foo", "bar", "foo", "baz", "foo", "foo"], copy=False),
            "expected": pd.DataFrame(
                {
                    "x0_foo": [1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
                    "x0_bar": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    "x0_baz": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                }
            ),
        },
        {
            "data": pd.Series(["foo", "foo", "foo", "foo", "foo", "foo"], copy=False),
            "expected": pd.DataFrame(
                {
                    "x0_foo": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                }
            ),
        },
    ]

    for case in cases:
        out = get_encoded_features(case["data"])
        pd.testing.assert_frame_equal(out, case["expected"], check_like=True)


def test_get_z_score():
    cases = [
        {
            "data": pd.DataFrame({"amount": [1.0, 2.0, 3.0]}),
            "expected": pd.Series([-1.0, 0.0, 1.0], copy=False),
        },
        {
            "data": pd.DataFrame({"amount": [1.1, 0.2, -11.56, 23.43, 67.99]}),
            "expected": pd.Series(
                [
                    -0.47916920013376096,
                    -0.5076685577943733,
                    -0.8800601645597069,
                    0.2279315293789857,
                    1.6389663931088554,
                ],
                copy=False,
            ),
        },
    ]
    for case in cases:
        out = get_z_score(case["data"])
        pd.testing.assert_series_equal(
            out, case["expected"], check_like=True, check_names=False
        )
