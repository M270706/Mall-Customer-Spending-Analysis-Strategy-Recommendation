import math

import pandas as pd

REQUIRED_COLUMNS = (
    "CustomerID",
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
)
SPENDING_SCORE_BINS = (0, 40, 70, 101)
SPENDING_SEGMENTS = ("Low", "Medium", "High")
SEGMENT_STRATEGIES = {
    "Low": "Engage & grow",
    "Medium": "Build loyalty",
    "High": "Reward & retain",
}
NUMERIC_BOUNDS = {
    "Age": (0, None, "must be greater than zero"),
    "Annual Income (k$)": (0, None, "must not be negative"),
    "Spending Score (1-100)": (1, 100, "must be between 1 and 100"),
}


def validate_customer_data(data: pd.DataFrame) -> pd.DataFrame:
    """Check the required customer fields and return a cleaned copy."""
    if data.empty:
        raise ValueError("The customer dataset is empty.")

    missing_columns = set(REQUIRED_COLUMNS).difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"The customer dataset is missing these columns: {', '.join(sorted(missing_columns))}"
        )

    customers = data.copy()
    issues = []

    missing_values = customers[list(REQUIRED_COLUMNS)].isna().sum()
    missing_fields = missing_values[missing_values > 0]
    if not missing_fields.empty:
        details = ", ".join(f"{column}: {count}" for column, count in missing_fields.items())
        issues.append(f"required fields contain missing values ({details})")

    identifiers = customers["CustomerID"].astype("string").str.strip()
    blank_ids = identifiers.isna() | identifiers.eq("")
    if blank_ids.any():
        issues.append(f"CustomerID has {int(blank_ids.sum())} blank values")
    duplicate_ids = identifiers.duplicated(keep=False) & ~blank_ids
    if duplicate_ids.any():
        issues.append(
            f"CustomerID has {int(duplicate_ids.sum())} rows with repeated identifiers"
        )
    customers["CustomerID"] = identifiers

    genders = customers["Gender"].astype("string").str.strip()
    blank_genders = genders.eq("").fillna(False)
    if blank_genders.any():
        issues.append(f"Gender has {int(blank_genders.sum())} blank values")
    customers["Gender"] = genders

    for column, (minimum, maximum, rule) in NUMERIC_BOUNDS.items():
        numeric = pd.to_numeric(customers[column], errors="coerce")
        finite = numeric.map(lambda value: pd.notna(value) and math.isfinite(value))
        invalid_numeric = (customers[column].notna() & numeric.isna()) | (
            numeric.notna() & ~finite
        )
        if invalid_numeric.any():
            issues.append(
                f"{column} has {int(invalid_numeric.sum())} values that are not valid numbers"
            )
        out_of_bounds = finite & numeric.lt(minimum)
        if maximum is not None:
            out_of_bounds |= finite & numeric.gt(maximum)
        if out_of_bounds.any():
            issues.append(f"{column} {rule} ({int(out_of_bounds.sum())} rows)")
        customers[column] = numeric

    if issues:
        raise ValueError("Please check the customer dataset: " + "; ".join(issues) + ".")

    return customers
