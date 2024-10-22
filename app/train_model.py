import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load the dataset with a specified encoding
data = pd.read_csv('D:/College/DWDM/app/DataCoSupplyChainDataset.csv', encoding='ISO-8859-1')

# Check for NaN values in 'Order Zipcode' and fill them
if data['Order Zipcode'].isnull().sum() > 0:
    mode_zipcode = data['Order Zipcode'].mode()
    if not mode_zipcode.empty:
        data['Order Zipcode'].fillna(mode_zipcode[0], inplace=True)

# Check for NaN values in 'Product Description' and fill them
if data['Product Description'].isnull().sum() > 0:
    mode_description = data['Product Description'].mode()
    if not mode_description.empty:
        data['Product Description'].fillna(mode_description[0], inplace=True)
    else:
        # Fill with a default value if mode is not available
        data['Product Description'].fillna("Unknown", inplace=True)

# Define feature columns and target variable
feature_columns = [
    'Days for shipping (real)', 
    'Days for shipment (scheduled)', 
    'Benefit per order', 
    'Sales per customer', 
    'Category Id', 
    'Customer Zipcode', 
    'Order Item Quantity', 
    'Order Item Product Price'
]
target_column = 'Late_delivery_risk'

# Prepare the data
X = data[feature_columns]
y = data[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline
numeric_features = ['Days for shipping (real)', 'Days for shipment (scheduled)', 
                   'Benefit per order', 'Sales per customer', 'Category Id', 
                   'Customer Zipcode', 'Order Item Quantity', 'Order Item Product Price']

categorical_features = []  # Add any categorical features if applicable

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create the model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Fit the model
pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(pipeline, 'model.joblib')

# Check the model's accuracy
accuracy = pipeline.score(X_test, y_test)
print(f'Model Accuracy: {accuracy:.2f}')

