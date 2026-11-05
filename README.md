# 🛍️ Mall Customer Spending Analysis & Strategy Recommendation

## 📖 The Problem We Solved
Malls collect a massive amount of data on us, but they often struggle to actually *use* it to personalize our shopping experience. Instead of guessing, what if a mall knew exactly who to target with a discount? 

This project uses machine learning to figure out how customers spend their money and pairs that with a game-theory model to recommend the smartest marketing move. The ultimate goal? Help the mall maximize its revenue while giving shoppers offers they actually care about.

## ✨ What Makes This Project Stand Out?
* **The Best of Both Worlds:** Most data projects either try to predict an exact number (regression) *or* group people into categories (classification). We ran both pipelines on the exact same dataset to get a richer, 360-degree view of the shoppers.
* **Math Meets Marketing:** We took machine learning predictions and plugged them directly into a formal Game Theory payoff matrix. We didn't just find out *who* the customers were; we defined exactly *what to do* with them to maximize profits.
* **Ready to Use Today:** By deploying a live web dashboard via Streamlit, mall management can jump in, filter the data, and explore these insights in real-time without needing to know a single line of code.

## 🗺️ The Step-by-Step Pipeline

1. **Getting to Know the Data:** Loaded 200 pristine customer records (zero missing values, zero duplicates).
2. **Cleaning Things Up:** Converted text-based gender data into a simple numerical format and dropped non-predictive features like Customer IDs.
3. **Finding the Odd Ones Out:** Used the IQR statistical method to hunt down extreme outliers (specifically spotting "whales" with unusually high annual incomes).
4. **Setting the Stage:** Split the data (80% training / 20% testing) to ensure the models were actually learning, not just memorizing.
5. **Predicting the Exact Score (Regression):** Pitted a Decision Tree Regressor against a Linear Regression model to predict a customer's exact spending score (1-100).
6. **Creating Customer Buckets (Classification):** Trained a Decision Tree Classifier to accurately sort shoppers into three actionable buckets: Low (0–39), Medium (40–69), and High (70–100) spenders.
7. **The Business Strategy (Game Theory Payoff):** Built a payoff matrix. High Spenders were assigned a "No Discount" strategy to protect profit margins, while Low and Medium Spenders were assigned a "Discount" strategy to tempt engagement.
8. **Bringing It to Life:** Built an interactive web dashboard using Streamlit for real-time data exploration and strategy visualization.

## 🏆 Key Takeaways
* **Simple Math Won:** When predicting exact scores on a small dataset, the simplest model (Linear Regression) actually generalized better to new data than the complex AI (Decision Tree Regressor). 
* **80% Accuracy on Segments:** The Decision Tree Classifier successfully bucketed 80% of the shoppers into the right spending category, performing exceptionally well on the core "Medium" spender demographic.
* **Strategy Over Statistics:** The payoff matrix bridged the gap between raw data science and business decision-making, providing a concrete action plan for every single shopper.

## ⚙️ Under the Hood: Tech & Data

**The Dataset**
* **Source:** UCI / Kaggle (Customer Segmentation Dataset).
* **Details:** 200 customer records featuring Gender, Age, Annual Income (k$), and Spending Score (1-100).

**The Tech Stack**
* **Python & Pandas:** Data loading, cleaning, and feature engineering (one-hot encoding, IQR outlier detection).
* **Scikit-learn:** Decision Tree Regressor, Linear Regression, Decision Tree Classifier, evaluation metrics.
* **Matplotlib & Seaborn:** Data visualization (bar charts, scatter plots, confusion matrix heatmaps).
* **Streamlit & Pyngrok:** Interactive web dashboard deployment.

## 🚀 How to Run the Project Locally

### Clone the repository
```bash
git clone [https://github.com/M270706/mall-customer-segmentation.git](https://github.com/M270706/mall-customer-segmentation.git)
cd mall-customer-segmentation

pip install pandas numpy scikit-learn matplotlib seaborn streamlit plotly pyngrok

python mini_p.py

streamlit run dashboard.py
