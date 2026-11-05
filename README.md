# Grahak Gyaan

I made Grahak Gyaan to explore the Mall Customers dataset and make its patterns easier to understand. It includes a small analysis notebook and an interactive dashboard where you can look through customer profiles and compare groups.

This is a mini-project for analysis and presentation, not a production tool.

## What’s in the project

```text
app/
  dashboard.py              Streamlit dashboard
  data_validation.py        Checks the dataset before it is used
data/
  Mall_Customers.csv        Dataset used in the analysis
notebooks/
  customer_segmentation_analysis.ipynb
deliverables/
  Grahak_Gyaan_Report.docx
  Grahak_Gyaan_Report.pdf
  Grahak_Gyaan_6_Slides.pptx
tests/
  test_data_validation.py
.streamlit/
  config.toml
requirements.txt
```

## Try the dashboard

Open a terminal in the project folder, install the packages, and start Streamlit:

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```

The dashboard has three views: **At a glance**, **Compare groups**, and **Customer list**. You can filter the data, look for a customer by ID, and download the records that match your filters as a CSV.

## A few things I found

The dataset has 200 customers and includes gender, age, annual income, and a spending score. It has no missing values or repeated customer IDs. For the analysis, I grouped scores into Low (below 40), Medium (40–69), and High (70-100). These cutoffs make the results easier to compare; they are not established customer-value categories.

I compared a few models for estimating the spending score. A shallow decision tree had the lowest cross-validation MAE among the regression models I tested. On the held-out records, its MAE was 13.37 points and its R² was 0.148. The mean-prediction baseline had a test MAE of 18.67, so the tree did better on MAE in this split, but the result is still limited.

I also tried classifying the three score bands. The model reached 85% accuracy and a macro F1 score of 0.841 on the test set. Since the bands come directly from the spending score, this tells us how well the model reproduces those chosen labels—not whether it can predict future purchases.

Finally, I explored K-means clustering. Of the cluster counts I tested (two to six), six had the highest silhouette score (0.427). The groups were consistent across repeated runs, but that alone does not show that they are useful for business decisions.

## What this analysis can’t tell us

The dataset is small and only describes one set of customer profiles. It does not include purchase history, campaign responses, revenue, or customer lifetime value. Because of that, this project cannot show what caused a spending score, predict campaign results, or estimate whether a discount would make money. The dashboard’s group-based suggestions are ideas to explore, not tested recommendations.

## Run the tests

From the project folder, run:

```bash
python -m unittest discover -s tests
```
