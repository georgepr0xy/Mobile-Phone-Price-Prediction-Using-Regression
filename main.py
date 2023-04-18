import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the dataset into a pandas dataframe
df = pd.read_csv('Cellphone.csv')

# Check for missing values
print(df.isnull().sum())

# Replace missing values with the mean
df.fillna(df.mean(), inplace=True)

# Check for outliers
df = df[(df['Price'] - df['Price'].mean()).abs() < 3 * df['Price'].std()]

# Check for correlations
corr_matrix = df.corr()
print(corr_matrix['Price'].sort_values(ascending=False))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Price', 'Product_id'], axis=1), df['Price'], test_size=0.2, random_state=42)

# Normalize the data using standard scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Transform the features to polynomial features
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train_scaled)
X_test_poly = poly_features.transform(X_test_scaled)

# Train a polynomial regression model on the training set
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Evaluate the model on the training set
y_pred = model.predict(X_train_poly)
mse = mean_squared_error(y_train, y_pred)

# Print the MSE and model parameters
print("MSE on training set: ", mse)
print("Model intercept: ", model.intercept_)
print("Model coefficients: ", model.coef_)
