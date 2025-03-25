import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
file_path = "House_Price_dataset.csv"
df = pd.read_csv(file_path)

# Remove non-numeric columns
non_numeric_columns = df.select_dtypes(exclude=['int64', 'float64']).columns
if 'price' in non_numeric_columns:
    non_numeric_columns = non_numeric_columns.drop('price')
df = df.drop(columns=non_numeric_columns, errors='ignore')

# Display basic information about the dataset
print("\n\nDataset Information:")
print(df.info())

# Check for missing values
missing_values = df.isnull().sum()
print("\n\nMissing Values:")
print(missing_values)

# Summary statistics of numerical variables
summary_stats = df[['Area Size', 'price', 'bedrooms']].describe()
print("\n\nSummary Statistics:")
print(summary_stats)

# Analyzing distributions of numerical variables
print("\n\nDistribution Analysis:")
for column in ['Area Size', 'price']:
    mean_value = df[column].mean()
    median_value = df[column].median()
    std_dev = df[column].std()
    print(f"{column}: Mean = {mean_value}, Median = {median_value}, Std Dev = {std_dev}")

# Identifying and replacing potential outliers using Z-score
threshold = 2.5
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    std = df[col].std()
    mean = df[col].mean()
    zscore = (df[col] - mean) / std
    median = df[col].median()
    df[col] = np.where(abs(zscore) > threshold, median, df[col])

print("\n\nData after replacing outliers with median:")
print(df.head())

# Normalize Numerical Data using Min-Max Scaling
scaler = MinMaxScaler()
df[['Area Size', 'bedrooms']] = scaler.fit_transform(df[['Area Size', 'bedrooms']])

# Encode Categorical Features
if 'Location' in df.columns:
    encoder = OneHotEncoder(drop='first', sparse=False)
    location_encoded = encoder.fit_transform(df[['Location']])
    location_columns = encoder.get_feature_names_out(['Location'])
    df = df.drop(columns=['Location'])
    df[location_columns] = location_encoded

print("\nData after normalization and encoding:")
print(df.head())


# Correlation Analysis (Only Numeric Columns)
numeric_df = df.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix['price'].sort_values(ascending=False))

# Removing Low-Impact Predictors (Threshold: 0.1)
thresh = 0.1
low_impact_features = correlation_matrix['price'][abs(correlation_matrix['price']) < thresh].index.tolist()
df = df.drop(columns=low_impact_features, errors='ignore')

print("\nData after removing low-impact predictors:")
print(df.head())


# Train-Test Split
X = df.drop(columns=['price'])
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2 Score): {r2}")


# Feature Insights (Most Important Predictors)
feature_importance = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:")
print(feature_importance)