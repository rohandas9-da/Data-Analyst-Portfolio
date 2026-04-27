# furniture_sales_analysis.py

# === 1. IMPORT LIBRARIES ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# === 2. LOAD DATASET ===
df = pd.read_csv("ecommerce_furniture_dataset_2024.csv")

# === 3. CLEANING DATA ===
# Drop column with too many missing values
df.drop(['originalPrice'], axis=1, inplace=True)

# Drop rows with missing values in important columns
df.dropna(subset=['tagText', 'price', 'sold'], inplace=True)

# Clean 'price' column
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)

# Simplify tagText
df['tagText'] = df['tagText'].apply(lambda x: x if x in ['Free shipping', '+Shipping: $5.09'] else 'others')

# Encode tagText
le = LabelEncoder()
df['tagText'] = le.fit_transform(df['tagText'])

# === 4. FEATURE ENGINEERING ===
# TF-IDF Vectorizer on productTitle
tfidf = TfidfVectorizer(max_features=100)
product_title_tfidf = tfidf.fit_transform(df['productTitle'])

# Convert to DataFrame and merge
tfidf_df = pd.DataFrame(product_title_tfidf.toarray(), columns=tfidf.get_feature_names_out())
df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)

# Drop original text column
df.drop(['productTitle'], axis=1, inplace=True)

# Fill any remaining NaNs just in case
df = df.fillna(0)

# === 5. EXPLORATORY DATA ANALYSIS ===
print("\nBasic Stats:")
print(df.describe())

# Uncomment below to plot if you want visuals in VS Code
# sns.histplot(df['price'], kde=True)
# plt.title("Price Distribution")
# plt.show()

# sns.scatterplot(x='price', y='sold', data=df)
# plt.title("Price vs Sold")
# plt.show()

# === 6. MODEL TRAINING ===
X = df.drop('sold', axis=1)
y = df['sold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train models
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Predict
y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Evaluate
print("\nModel Evaluation:")
print(f"Linear Regression - MSE: {mean_squared_error(y_test, y_pred_lr):.2f}, R²: {r2_score(y_test, y_pred_lr):.4f}")
print(f"Random Forest     - MSE: {mean_squared_error(y_test, y_pred_rf):.2f}, R²: {r2_score(y_test, y_pred_rf):.4f}")

# === 7. SAVE CLEANED DATASET ===
df.to_csv("cleaned_furniture_data.csv", index=False)
print("\nCleaned dataset saved to cleaned_furniture_data.csv")
