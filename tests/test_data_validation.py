import unittest

import pandas as pd

from app.data_validation import (
    SPENDING_SCORE_BINS,
    SPENDING_SEGMENTS,
    validate_customer_data,
)


def valid_customers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CustomerID": ["001", "002"],
            "Gender": ["Female", "Non-binary"],
            "Age": [25, 40],
            "Annual Income (k$)": [30, 55],
            "Spending Score (1-100)": [39, 70],
        }
    )


class ValidateCustomerDataTests(unittest.TestCase):
    def test_accepts_valid_rows_and_unrestricted_nonblank_gender_labels(self) -> None:
        customers = validate_customer_data(valid_customers())

        self.assertEqual(len(customers), 2)
        self.assertEqual(customers["CustomerID"].tolist(), ["001", "002"])
        self.assertEqual(customers["Gender"].tolist(), ["Female", "Non-binary"])

    def test_rejects_empty_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "dataset is empty"):
            validate_customer_data(pd.DataFrame())

    def test_reports_missing_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing these columns"):
            validate_customer_data(pd.DataFrame({"CustomerID": [1]}))

    def test_rejects_missing_required_values(self) -> None:
        customers = valid_customers()
        customers.loc[0, "Age"] = None

        with self.assertRaisesRegex(ValueError, "missing values"):
            validate_customer_data(customers)

    def test_rejects_duplicate_ids_after_normalization(self) -> None:
        customers = valid_customers()
        customers["CustomerID"] = ["001", " 001 "]

        with self.assertRaisesRegex(ValueError, "repeated identifiers"):
            validate_customer_data(customers)

    def test_rejects_non_numeric_values(self) -> None:
        customers = valid_customers()
        customers["Age"] = customers["Age"].astype(object)
        customers.loc[0, "Age"] = "unknown"

        with self.assertRaisesRegex(ValueError, "not valid numbers"):
            validate_customer_data(customers)

    def test_rejects_out_of_domain_values(self) -> None:
        customers = valid_customers()
        customers.loc[0, "Spending Score (1-100)"] = 101

        with self.assertRaisesRegex(ValueError, "between 1 and 100"):
            validate_customer_data(customers)

    def test_rejects_blank_gender(self) -> None:
        customers = valid_customers()
        customers.loc[0, "Gender"] = "  "

        with self.assertRaisesRegex(ValueError, "Gender has"):
            validate_customer_data(customers)

    def test_spending_segment_boundaries_match_business_rules(self) -> None:
        scores = pd.Series([1, 39, 40, 69, 70, 100])
        segments = pd.cut(
            scores,
            bins=SPENDING_SCORE_BINS,
            labels=SPENDING_SEGMENTS,
            right=False,
            include_lowest=True,
        )

        self.assertEqual(
            segments.astype(str).tolist(),
            ["Low", "Low", "Medium", "Medium", "High", "High"],
        )


if __name__ == "__main__":
    unittest.main()
