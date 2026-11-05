# -*- coding: utf-8 -*-
"""Mini_p

Original file is located at
    https://colab.research.google.com/drive/1JUG3GDCyiMa6aAyj250bW7aNRyHRypie

# Mall Customer Segmentation and Marketing Strategy

## 1. Import Libraries
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import numpy as np

"""## 2. Load Dataset and Display First Rows"""

dataset =  pd.read_csv('Mall_Customers.csv')
dataset.head()

"""## 3. Display Dataset Information"""

dataset.info()
print("\nShape of the DataSet : ", dataset.shape)
print("\nSize of the DataSet : ", dataset.size)

"""## 4. Check for Null Values"""

print("Null Values in DataSet:\n",dataset.isnull().sum())

"""## 5. Outlier Detection and Visualization

Outliers are data points that significantly differ from other observations. They can be due to measurement errors, data entry errors, or genuine but extreme variations. It's important to identify and handle outliers as they can negatively impact model performance.

### Using IQR Method to Detect Outliers
"""

def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# Detect outliers in 'Annual Income (k$)'
outliers_annual_income = detect_outliers_iqr(dataset, 'Annual Income (k$)')
print("Outliers in 'Annual Income (k$)':")
display(outliers_annual_income)

# Detect outliers in 'Spending Score (1-100)'
outliers_spending_score = detect_outliers_iqr(dataset, 'Spending Score (1-100)')
print("\nOutliers in 'Spending Score (1-100)':")
display(outliers_spending_score)

# Detect outliers in 'Age'
outliers_age = detect_outliers_iqr(dataset, 'Age')
print("\nOutliers in 'Age':")
display(outliers_age)

"""From the box plots, 'Annual Income (k$)' seems to have a few potential outliers on the higher end.

The IQR method confirms the presence of outliers in 'Annual Income (k$)' at the higher end. For 'Spending Score (1-100)' and 'Age', no outliers were detected using this method, which is consistent with the box plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Select numerical columns for outlier analysis (excluding CustomerID and one-hot encoded Gender)
numerical_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

plt.figure(figsize=(15, 5))
for i, col in enumerate(numerical_cols):
    plt.subplot(1, len(numerical_cols), i + 1)
    sns.boxplot(y=dataset[col])
    plt.title(f'Box Plot of {col}')
    plt.ylabel(col)
plt.tight_layout()
plt.show()

"""## 6. One-Hot Encode 'Gender' Column"""

dataset = pd.get_dummies(dataset, columns=['Gender'], drop_first=True) #for one-hot encoding
display(dataset.head())

"""## 7. Define Features (X) and Target (y)"""

X = dataset.drop(['CustomerID', 'Spending Score (1-100)'], axis=1)
y = dataset['Spending Score (1-100)']

display(X.head())
display(y.head())

"""In the code above, the `Gender` column has been converted into a numerical column called `Gender_Male`.

*   A value of `1` in `Gender_Male` indicates that the customer is male.
*   A value of `0` in `Gender_Male` indicates that the customer is female.

## 8. Split Data into Training and Testing Sets
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

"""## 9. Train and Predict with Decision Tree Regressor"""

from sklearn.tree import DecisionTreeRegressor

# Initialize the Decision Tree Regressor model
dt_regressor_model = DecisionTreeRegressor(random_state=42)

# Train the model
dt_regressor_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_dt = dt_regressor_model.predict(X_test)

print("Decision Tree Regressor model trained successfully.")

"""## 10. Evaluate Decision Tree Regressor Model"""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calculate regression metrics
mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print(f"Decision Tree Regressor Metrics:")
print(f"  Mean Absolute Error (MAE): {mae_dt:.2f}")
print(f"  Mean Squared Error (MSE): {mse_dt:.2f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_dt:.2f}")
print(f"  R-squared (R2): {r2_dt:.2f}")

"""## 11. Train and Predict with Linear Regression Model"""

from sklearn.linear_model import LinearRegression

# Initialize the Linear Regression model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lr = lr_model.predict(X_test)

print("Linear Regression model trained successfully.")

"""## 12. Evaluate Linear Regression Model"""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calculate regression metrics for Linear Regression
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression Model Metrics:")
print(f"  Mean Absolute Error (MAE): {mae_lr:.2f}")
print(f"  Mean Squared Error (MSE): {mse_lr:.2f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_lr:.2f}")
print(f"  R-squared (R2): {r2_lr:.2f}")

"""## 13. Compare Regression Models Metrics"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create a DataFrame for easy plotting of metrics
metrics_data = {
    'Model': ['Decision Tree Regressor', 'Linear Regression'],
    'MAE': [mae_dt, mae_lr],
    'MSE': [mse_dt, mse_lr],
    'RMSE': [rmse_dt, rmse_lr],
    'R2': [r2_dt, r2_lr]
}
metrics_df = pd.DataFrame(metrics_data)

# Plotting function for each metric
def plot_metric_comparison(df, metric_name, title, ylabel):
    plt.figure(figsize=(7, 5))
    sns.barplot(x='Model', y=metric_name, data=df, palette='viridis', hue='Model', legend=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Model')
    plt.show()

# Generate plots for each metric
plot_metric_comparison(metrics_df, 'MAE', 'Comparison of Mean Absolute Error (MAE)', 'MAE Value')
plot_metric_comparison(metrics_df, 'MSE', 'Comparison of Mean Squared Error (MSE)', 'MSE Value')
plot_metric_comparison(metrics_df, 'RMSE', 'Comparison of Root Mean Squared Error (RMSE)', 'RMSE Value')
plot_metric_comparison(metrics_df, 'R2', 'Comparison of R-squared (R2)', 'R-squared Value')

"""## 14. Create Spending Categories for Classification"""

bins = [0, 40, 70, 101] # Define bins for Low (0-40), Medium (41-70), and High (71-100)
labels = ['Low', 'Medium', 'High'] # Labels for the categories
dataset['Spending_Category'] = pd.cut(dataset['Spending Score (1-100)'], bins=bins, labels=labels, right=False)

print("Distribution of Spending Categories:")
display(dataset['Spending_Category'].value_counts())
display(dataset.head())

"""## 15. Visualize Spending Categories Distribution"""

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.countplot(data=dataset, x='Spending_Category', order=['Low', 'Medium', 'High'], palette='viridis', hue='Spending_Category', legend=False)
plt.title('Distribution of Customer Spending Categories')
plt.xlabel('Spending Category')
plt.ylabel('Number of Customers')
plt.show()

"""## 16. Classification using Decision Tree Classifier"""

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Assuming 'dataset' and 'X' (from regression features) are available from previous cells.
# X is defined as: X = dataset.drop(['CustomerID', 'Spending Score (1-100)'], axis=1)
# dataset['Spending_Category'] is defined.

# Use the existing feature set 'X' for classification features
X_clf = X

# Define the target variable for classification
y_clf = dataset['Spending_Category']

# Split the data into training and testing sets for classification
# Using random_state for reproducibility and stratify to maintain class proportions
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

# Initialize and train a Decision Tree Classifier
dt_classifier_model = DecisionTreeClassifier(random_state=42)
dt_classifier_model.fit(X_train_clf, y_train_clf)

# Make predictions on the test set
y_pred_clf = dt_classifier_model.predict(X_test_clf)

print("\nClassification Report for Decision Tree Classifier:\n")
print(classification_report(y_test_clf, y_pred_clf))

"""## 17. Visualize Classification Metrics"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# Get the classification report as a dictionary
report = classification_report(y_test_clf, y_pred_clf, output_dict=True)

# Extract metrics for 'Low', 'Medium', 'High' classes
metrics_data = []
for label in ['Low', 'Medium', 'High']:
    if label in report:
        metrics_data.append({
            'Class': label,
            'Metric': 'Precision',
            'Value': report[label]['precision']
        })
        metrics_data.append({
            'Class': label,
            'Metric': 'Recall',
            'Value': report[label]['recall']
        })
        metrics_data.append({
            'Class': label,
            'Metric': 'F1-score',
            'Value': report[label]['f1-score']
        })

metrics_df = pd.DataFrame(metrics_data)

plt.figure(figsize=(10, 6))
sns.barplot(x='Class', y='Value', hue='Metric', data=metrics_df, palette='viridis')
plt.title('Classification Report Metrics by Spending Category')
plt.xlabel('Spending Category')
plt.ylabel('Score')
plt.ylim(0, 1) # Metrics are between 0 and 1
plt.legend(title='Metric')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

"""## 18. Define and Apply Marketing Strategies"""

# Payoff matrix: rows=Company strategies, cols=Customer response
payoff = {
    'Discount':    {'Accept': 50,  'Reject': -10},
    'No Discount': {'Accept': 80,  'Reject': 0},
}

def decide_strategy(segment):
    if segment == 1:   # High spender
        return 'No Discount', payoff['No Discount']['Accept']
    else:              # Low spender
        return 'Discount', payoff['Discount']['Accept']

# Map 'Spending_Category' to the 'segment' expected by the decide_strategy function
def map_category_to_segment(category):
    if category == 'High':
        return 1  # High spender
    else:
        return 0  # Low or Medium spender

dataset['Segment'] = dataset['Spending_Category'].apply(map_category_to_segment)

# Apply the decide_strategy function to each customer
dataset['Recommended_Strategy'], dataset['Expected_Payoff'] = zip(*dataset['Segment'].apply(decide_strategy))

print("Customer segments, recommended strategies, and expected payoffs have been added to the dataset.")
display(dataset.head())

"""## 19. Visualize Confusion Matrix"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
cm = confusion_matrix(y_test_clf, y_pred_clf)

# Get the class labels
class_labels = y_clf.cat.categories

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix for Decision Tree Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

"""## 20. Install Streamlit and pyngrok"""

# Install necessary packages
!pip install streamlit pyngrok -q

"""## 21. Streamlit Dashboard Setup

First, I'll generate the Python code for `dashboard.py`. This script will create a simple Streamlit application to visualize the customer data, including the spending categories and recommended strategies.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%writefile dashboard.py
# 
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# 
# st.set_page_config(layout="wide")
# 
# st.title('Mall Customer Segmentation Dashboard')
# 
# # Load and preprocess the dataset
# @st.cache_data
# def load_data():
#     try:
#         df = pd.read_csv('Mall_Customers.csv')
# 
#         # One-hot encode Gender
#         df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
# 
#         # Create Spending_Category
#         bins = [0, 40, 70, 101]  # Define bins for Low (0-40), Medium (41-70), and High (71-100)
#         labels = ['Low', 'Medium', 'High'] # Labels for the categories
#         df['Spending_Category'] = pd.cut(df['Spending Score (1-100)'], bins=bins, labels=labels, right=False)
# 
#         # Create Recommended_Strategy based on Spending_Category (example logic)
#         def assign_strategy(row):
#             if row['Spending_Category'] == 'High':
#                 return 'Reward & Retain Premium Customers'
#             elif row['Spending_Category'] == 'Medium':
#                 return 'Maintain Loyalty & Upsell'
#             else: # Low spending
#                 return 'Encourage Engagement & Growth'
#         df['Recommended_Strategy'] = df.apply(assign_strategy, axis=1)
# 
#         return df
#     except FileNotFoundError:
#         st.error("Error: 'Mall_Customers.csv' not found. Please ensure the data processing cells have been run and the CSV file is saved.")
#         return pd.DataFrame()
# 
# dataset = load_data()
# 
# if not dataset.empty:
#     st.subheader('Raw Data Preview')
#     st.dataframe(dataset.head())
# 
#     st.subheader('Customer Distribution by Spending Category')
#     fig_spending_category = px.bar(
#         dataset['Spending_Category'].value_counts().reset_index(name='Number of Customers'),
#         x='Spending_Category', y='Number of Customers',
#         labels={'Spending_Category': 'Spending Category'},
#         title='Number of Customers in Each Spending Category',
#         color='Spending_Category',
#         color_discrete_sequence=px.colors.qualitative.Pastel
#     )
#     st.plotly_chart(fig_spending_category, use_container_width=True)
# 
#     if 'Recommended_Strategy' in dataset.columns:
#         st.subheader('Recommended Strategies Distribution')
#         fig_strategy = px.pie(
#             dataset, names='Recommended_Strategy',
#             title='Distribution of Recommended Strategies',
#             color_discrete_sequence=px.colors.qualitative.Vivid
#         )
#         st.plotly_chart(fig_strategy, use_container_width=True)
# 
#     st.subheader('Spending Score vs. Annual Income by Spending Category')
#     fig_scatter = px.scatter(
#         dataset, x='Annual Income (k$)', y='Spending Score (1-100)',
#         color='Spending_Category', size='Age', hover_data=['Age', 'Gender_Male', 'Recommended_Strategy'],
#         title='Spending Score vs. Annual Income',
#         color_discrete_map={'Low': 'blue', 'Medium': 'green', 'High': 'red'}
#     )
#     st.plotly_chart(fig_scatter, use_container_width=True)
# 
#     st.subheader('Filtering Options')
#     selected_category = st.selectbox(
#         'Select Spending Category:',
#         options=['All'] + list(dataset['Spending_Category'].unique())
#     )
# 
#     if selected_category != 'All':
#         filtered_df = dataset[dataset['Spending_Category'] == selected_category]
#         st.subheader(f'Customers in {selected_category} Spending Category')
#         st.dataframe(filtered_df)
#     else:
#         st.subheader('All Customers')
#         st.dataframe(dataset)
#

"""## 22. Write Streamlit Dashboard Code"""

# Replace 'YOUR_AUTHTOKEN_HERE' with your actual ngrok authtoken
!ngrok config add-authtoken 3BfEGaOH7F4KncWHuQJOO4HVu3N_4NeF2EMVZEnFLYtvrsVdr

import subprocess
import time
from pyngrok import ngrok

# Kill any running ngrok processes from previous sessions
subprocess.run(['killall', 'ngrok'], capture_output=True, text=True)

# Run Streamlit in the background
get_ipython().system_raw('streamlit run dashboard.py &')

# Give Streamlit a moment to start up
time.sleep(5)

# Connect a new ngrok tunnel to the Streamlit port (8501)
try:
    public_url = ngrok.connect(8501)
    print(f"Streamlit App is live at: {public_url}")
except Exception as e:
    print(f"Could not establish ngrok tunnel: {e}")
    print("Please ensure your ngrok authtoken is configured. Run '!ngrok config add-authtoken YOUR_AUTHTOKEN_HERE'")
