import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load historical stock price data (you can replace this with your own dataset)
# For demonstration purposes, we'll generate some synthetic data
dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
num_days = len(dates)
prices = np.sin(np.linspace(0, 10, num_days)) * 100 + np.random.normal(0, 10, num_days)

# Create a DataFrame with date and price columns
data = pd.DataFrame({'Date': dates, 'Price': prices})

# Feature engineering: Add additional features if desired (e.g., moving averages, technical indicators)

# Define features and target variable
X = data[['Date']]  # Using date as a feature for simplicity (you can add more features)
y = data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Visualize the predictions
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()
