# **House Price Prediction System**


## **1. Introduction**

The objective of this project is to develop a predictive model that can accurately forecast house prices based on various features. This involves several stages including data collection, preprocessing, feature engineering, model selection, and evaluation. The final goal is to create a model that helps in estimating house prices effectively for real estate professionals, buyers, and sellers.

## **2. Data Collection**

### **2.1 Dataset Description**

For this project, the dataset used is `house_prices.csv`, which contains various features related to houses and their sale prices. Key features include:

- **Location**: Neighborhood, ZIP code
- **Size**: Square footage, number of rooms
- **Condition**: Year built, renovations
- **Amenities**: Garage, swimming pool

### **2.2 Data Overview**

```python
import pandas as pd

# Load the dataset
data = pd.read_csv('house_prices.csv')

# Display the first few rows of the dataset
print(data.head())

# Display summary information
print(data.info())

# Display descriptive statistics
print(data.describe())
```

## **3. Data Preprocessing**

### **3.1 Handling Missing Values**

To ensure data quality, missing values are addressed by:

1. Dropping rows where the target variable `SalePrice` is missing.
2. Filling missing values in categorical features with the mode.
3. Filling missing values in numerical features with the mean.

```python
# Drop rows where 'SalePrice' is missing
data = data.dropna(subset=['SalePrice'])

# Fill missing values for categorical features
categorical_columns = data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    data[column].fillna(data[column].mode()[0], inplace=True)

# Fill missing values for numerical features
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
for column in numerical_columns:
    data[column].fillna(data[column].mean(), inplace=True)
```

### **3.2 Encoding Categorical Variables**

Categorical variables are converted into numerical format using one-hot encoding.

```python
# Apply One-Hot Encoding
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
```

### **3.3 Normalizing/Scaling Numerical Features**

Numerical features are standardized to ensure all features contribute equally to the model.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_columns = data_encoded.select_dtypes(include=['int64', 'float64']).columns
data_encoded[numerical_columns] = scaler.fit_transform(data_encoded[numerical_columns])
```

## **4. Feature Engineering**

### **4.1 Creating New Features**

A new feature `HouseAge` is created to represent the age of the house.

```python
# Create a new feature for house age
data_encoded['HouseAge'] = data_encoded['YearBuilt'] - data_encoded['YearRemodAdd']
```

### **4.2 Selecting Relevant Features**

Relevant features are selected for model training.

```python
# Example of selecting relevant features
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'HouseAge']
X = data_encoded[features]
y = data_encoded['SalePrice']
```

## **5. Model Selection**

### **5.1 Splitting Data**

The dataset is divided into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### **5.2 Training Models**

Several models are trained to predict house prices:

- **Linear Regression**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Train Linear Regression model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Train Random Forest model
model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)

# Train Gradient Boosting model
model_gb = GradientBoostingRegressor()
model_gb.fit(X_train, y_train)
```

## **6. Model Evaluation**

### **6.1 Evaluating Model Performance**

The performance of each model is assessed using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²).

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Evaluate Linear Regression
y_pred_lr = model_lr.predict(X_test)
print("Linear Regression - MAE:", mean_absolute_error(y_test, y_pred_lr))
print("Linear Regression - MSE:", mean_squared_error(y_test, y_pred_lr))
print("Linear Regression - R-squared:", r2_score(y_test, y_pred_lr))

# Evaluate Random Forest
y_pred_rf = model_rf.predict(X_test)
print("Random Forest - MAE:", mean_absolute_error(y_test, y_pred_rf))
print("Random Forest - MSE:", mean_squared_error(y_test, y_pred_rf))
print("Random Forest - R-squared:", r2_score(y_test, y_pred_rf))

# Evaluate Gradient Boosting
y_pred_gb = model_gb.predict(X_test)
print("Gradient Boosting - MAE:", mean_absolute_error(y_test, y_pred_gb))
print("Gradient Boosting - MSE:", mean_squared_error(y_test, y_pred_gb))
print("Gradient Boosting - R-squared:", r2_score(y_test, y_pred_gb))
```


## **7. Conclusion**

This report outlines the process of developing a House Price Prediction System. By collecting and preprocessing data, engineering features, selecting and training models, and evaluating their performance, we aim to create a robust predictive model for estimating house prices. Model optimization and saving ensure that the best-performing model is available for future predictions.
