# Grahak Gyaan

A small, owner-facing analysis of the Mall Customers dataset. The project explores customer profiles, compares a few simple models, and presents the results in an interactive dashboard.

## What’s here

```text
app/          Streamlit dashboard and input checks
data/         Mall_Customers.csv
notebooks/    Exploratory analysis and model evaluation
deliverables/ Project report and presentation
tests/        Data validation tests
```

## Run the dashboard

From the project folder, install the packages and start the app:

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```

The dashboard includes an overview, segment comparisons, and a searchable customer list. Filters update the charts and table together, and the filtered records can be downloaded as a CSV.

## Analysis at a glance

The dataset contains 200 customer records and five fields: customer ID, gender, age, annual income, and a spending score. It has no missing values or duplicate customer IDs. The score bands used in the analysis are Low (below 40), Medium (40–69), and High (70 or above); these are descriptive labels, not business-defined customer value tiers.

The notebook compares regression and classification approaches using cross-validation and a held-out test set. A shallow decision tree had the lowest cross-validation MAE among the regression models. On the held-out records, its MAE was 13.37 points and its R² was 0.148; the mean-prediction baseline had a test MAE of 18.67. The classifier reproduced the chosen score bands with 85% test accuracy and a macro F1 score of 0.841. Because the bands are derived from the spending score, this result measures how well the model reproduces those labels—not whether it can predict future purchases or business outcomes.

An exploratory K-means analysis selected six clusters from the tested range of two to six. The silhouette score was 0.427, and repeated fits were stable on this dataset. That stability does not establish that the clusters are useful for business decisions.

## Scope and limitations

This is a presentation and analysis project, not a production customer-management system. The dataset does not include transaction history, product preferences, campaign response, customer lifetime value, or revenue. The analysis therefore cannot estimate campaign impact, revenue, or discount profitability. Suggestions shown in the dashboard are simple prompts based on spending-score groups, not tested marketing recommendations.

The dataset is small and observational, so model results may change with a different sample. The notebook keeps preprocessing inside model pipelines and evaluates models on held-out data, but its results should still be treated as exploratory.

## Tests

Run the input-validation tests with:

```bash
python -m unittest discover -s tests
```
