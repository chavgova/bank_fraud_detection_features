import pandas as pd
import pytest
from src.features import get_encoded_features, get_z_score, feature_number_transactions_per_category, clean_phone_number, parse_phone_to_country, parse_iban_to_country, feature_transaction_made_in_phone_registered_country


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

def test_clean_phone_number():
    cases = [
        { "data": "+1 888-999-8900",
            "expected": "+18889998900"},
        { "data": "(+44) 1888 999890",
            "expected": "+441888999890"},
        { "data": "0049 1888 4999890",
            "expected": "+4918884999890"},
        { "data": "1888999890", # no country code
            "expected": None},
        { "data": "+13 (0)88999890",
            "expected": "+13088999890"},
    ]

    for case in cases:
        out = clean_phone_number(case["data"])
        assert out == case["expected"]
        
def test_parse_phone_to_country():
    cases = [
        { "data": "+1 888-999-8900",
            "expected": "US"},
        { "data": "(+44) 1888 999890",
            "expected": "GB"},
        { "data": "0049 1888 4999890",
            "expected": "DE"},
        { "data": "+999 (0)88999890", # non existing country code
            "expected": None},
        { "data": 1234, # wrong type
            "expected": None},
        { "data": None,
            "expected": None},
        { "data": "+882 16 000000", # satelite phone number, not a country code
            "expected": None},
    ]

    for case in cases:
        out = parse_phone_to_country(case["data"])
        assert out == case["expected"]
        
def test_parse_iban_to_country():
    cases = [
        { "data": "BG12345432123454322343",
            "expected": "BG"},
        { "data": "GB82WEST12345698765432",
            "expected": "GB"},
        { "data": " FR1420041010050500013M02606", # helped me find a bug as i used to not strip the string before checking if the 3rd and 4th characters are digits
            "expected": "FR"},
        { "data": "KU60X0542811101000000123456", # non existing
            "expected": None},
        { "data": "POL60X0542811101000000123456", # 3-alpha (not supported and not in data) instead of 2-alpha
            "expected": None},
        { "data": " nan", # this test case helped me find a possible problem and fix it
            "expected": None},
        { "data": 1234,
            "expected": None},
        { "data": None,
            "expected": None},
    ]

    for case in cases:
        out = parse_iban_to_country(case["data"])
        assert out == case["expected"]
        
        
def test_feature_transaction_made_in_phone_registered_country():
    cases = [
        {
            # covers known and matching countries
            "data": pd.DataFrame(
                { "cust_acc": ["BG12345432123454322343", "GB82WEST12345698765432", "FR1420041010050500013M02606", "ES1234543212345432234344"],
                    "phone": ["+359 888-999-8900", "(+44) 1888 999890", "+33 1 23 45 67 89", "+34 912 34 56 78"] } ),
            "expected": pd.Series([1.0, 1.0, 1.0, 1.0], copy=False)},
        {
            # covers known, unknown and not matching countris
            "data": pd.DataFrame(
                { "cust_acc": [None, "BG12345432123454322343", "GB82WEST12345698765432", "FR1420041010050500013M02606", "ES1234543212345432234344", "KU60X0542811101000000123456"],
                    "phone": [None, None, "(+33) 1888 999890", "+34 1 23 45 67 89", "+99912 34 56 78", "+359 888-999-8900"] } ),
            "expected": pd.Series([float("nan"), float("nan"), 0.0, 0.0, float("nan"), float("nan")], copy=False) },
    ]

    for case in cases:
        out = feature_transaction_made_in_phone_registered_country(case["data"])
        pd.testing.assert_series_equal(out, case["expected"], check_names=False)
        
        
def test_feature_number_transactions_per_category():
    cases = [
        {
            # normal case
            "data": pd.DataFrame(
                { "chck_date": ["2077-01-01", "2077-01-02", "2077-01-03", "2077-01-04", "2077-01-05"],
                    "category": ["A", "B", "A", "C", "B"] } ),
            "reference_dates": pd.to_datetime(["2077-01-06", "2077-01-07"]),
            "month_windows": (1, 2),
            "expected": pd.DataFrame(
                { "transaction_count_A_1_months": [2, 2],
                    "transaction_count_B_1_months": [2, 2],
                    "transaction_count_C_1_months": [1, 1],
                    "transaction_count_A_2_months": [2, 2],
                    "transaction_count_B_2_months": [2, 2],
                    "transaction_count_C_2_months": [1, 1] },
                index=pd.DatetimeIndex( ["2077-01-06", "2077-01-07"], tz="UTC", name="reference_date")) },
        # case with a category with 0 transactions for the chosen periods
        {
            "data": pd.DataFrame(
                { "chck_date": ["2077-01-01", "2077-01-02", "2077-01-03", "2077-01-04", "2077-01-05", "2077-01-09"],
                    "category": ["A", "B", "A", "C", "B", "D"] } ),
            "reference_dates": pd.to_datetime(["2077-01-06", "2077-01-07"]),
            "month_windows": (1, 2),
            "expected": pd.DataFrame(
                { "transaction_count_A_1_months": [2, 2],
                    "transaction_count_B_1_months": [2, 2],
                    "transaction_count_C_1_months": [1, 1],
                    "transaction_count_D_1_months": [0, 0],  # category D has no transactions
                    "transaction_count_A_2_months": [2, 2],
                    "transaction_count_B_2_months": [2, 2],
                    "transaction_count_C_2_months": [1, 1],
                    "transaction_count_D_2_months": [0, 0] },  # category D has no transactions
                index=pd.DatetimeIndex( ["2077-01-06", "2077-01-07"], tz="UTC", name="reference_date")) },
        {
            "data": pd.DataFrame({
                "chck_date": [
                    "2024-07-09", "2024-07-10",
                    "2024-12-09", "2024-12-10",
                    "2025-03-09", "2025-03-10",
                    "2025-04-09", "2025-04-10", # on border of 1 month window
                    "2025-05-09", "2025-05-10",# on reference date
                    "2025-05-11", None, "blabla", # invalid
                ],
                "category": [
                    "A", "A", "B", "B", "A", "A", "B",
                    "A", "B", "A", "C", "A", "B"]}), # cat C no transactions in period
             
                "reference_dates": pd.to_datetime(["2025-05-10"]),
                    "month_windows": (1, 2, 5, 10),
                    "expected": pd.DataFrame({
                        "transaction_count_A_1_months": [1],
                        "transaction_count_B_1_months": [1],
                        "transaction_count_C_1_months": [0],
                        "transaction_count_A_2_months": [2],
                        "transaction_count_B_2_months": [2],
                        "transaction_count_C_2_months": [0],
                        "transaction_count_A_5_months": [3],
                        "transaction_count_B_5_months": [3],
                        "transaction_count_C_5_months": [0],
                        "transaction_count_A_10_months": [4],
                        "transaction_count_B_10_months": [4],
                        "transaction_count_C_10_months": [0]}, 
                        index=pd.DatetimeIndex(["2025-05-10"], tz="UTC", name="reference_date"))
            }
    ]
    
    for case in cases:
        actual = feature_number_transactions_per_category(
            case["data"],
            case["reference_dates"],
            case["month_windows"])
        pd.testing.assert_frame_equal(actual, case["expected"])