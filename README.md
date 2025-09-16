# CreditShield-using-ML

**CreditShield — Lending Club Default Predictor**

Predict whether a loan will default using Lending Club (2012) data.
Focus: fast, practical baseline that a risk analyst can run and iterate on.


Primary question: Given application features, will the loan default?
Why it matters: prioritizes underwriting rules, pricing, and portfolio risk.

**Data**
File: loan_data_2012.csv (target column: default)
Each row = a loan; all other columns are candidate features.
You may drop/impute features and sample rows for speed.

**Models used**
Decision Tree (interpretable baseline)
Random Forest (strong tabular baseline)
XGBoost (gradient boosting for best accuracy on tabular data)

Each model is trained on the same split and evaluated on the test set with a confusion matrix.

**Minimal pipeline (what the code does)**

Load CSV → drop obvious high-cardinality IDs (id, title, url if present).
Drop missing rows (dropna() for a simple, fast baseline).
Split into X (features) and y (default).
Encode categoricals with LabelEncoder.
Standardize numerics with StandardScaler.
train_test_split(test_size=0.30, random_state=42).

Train DT, RF, XGB → plot confusion matrices.
