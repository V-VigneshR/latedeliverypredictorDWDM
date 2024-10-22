from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.joblib')

# Define the feature names for feature importance visualization
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input values from the form
        days_shipping_real = float(request.form['days_shipping_real'])
        days_shipment_scheduled = float(request.form['days_shipment_scheduled'])
        benefit_per_order = float(request.form['benefit_per_order'])
        sales_per_customer = float(request.form['sales_per_customer'])
        category_id = int(request.form['category_id'])
        customer_zipcode = int(request.form['customer_zipcode'])
        order_item_quantity = int(request.form['order_item_quantity'])
        order_item_product_price = float(request.form['order_item_product_price'])

        # Create a DataFrame from the input values
        input_data = pd.DataFrame({
            'Days for shipping (real)': [days_shipping_real],
            'Days for shipment (scheduled)': [days_shipment_scheduled],
            'Benefit per order': [benefit_per_order],
            'Sales per customer': [sales_per_customer],
            'Category Id': [category_id],
            'Customer Zipcode': [customer_zipcode],
            'Order Item Quantity': [order_item_quantity],
            'Order Item Product Price': [order_item_product_price]
        })

        # Make prediction
        prediction = model.predict(input_data)
        
        # Calculate model accuracy
        accuracy = model.score(input_data, prediction)

        # Prepare feature importances
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
        else:
            feature_importances = np.ones(len(feature_columns))  # Equal importance if not available

        # Create a DataFrame for importance
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': feature_importances
        })
        
        # Normalize and sort importances
        importance_df['Importance'] = importance_df['Importance'] / importance_df['Importance'].sum()
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Generate a feature importance graph
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        plt.xlabel('Normalized Feature Importance')
        plt.title('Feature Importances Impacting Prediction')
        plt.grid(axis='x')

        # Save the plot to a BytesIO object and encode it to base64
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()

        # Pass the data to the dashboard template
        return render_template('dashboard.html', 
                               prediction=prediction[0], 
                               accuracy=round(accuracy * 100, 2), 
                               plot_url=plot_url,
                               reasons=importance_df['Importance'].tolist(),
                               feature_names=importance_df['Feature'].tolist())

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
