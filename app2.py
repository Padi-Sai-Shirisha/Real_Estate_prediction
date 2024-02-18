# import numpy as np
# import pickle
import pandas as pd
import streamlit as st 

# Load the trained model
# pickle_in = open("model.pkl","rb")
# model = pickle.load(pickle_in)

# Define function to predict selling price
def predict_selling_price(model, a, b, c, d):
    prediction = model.predict([[a, b, c, d]])
    return prediction


# Define the main function
def main():
    st.title("Selling price prediction")
    st.write("Please enter the item rating, month, and year to predict selling price.")
    
    # Set background color
    st.markdown(
        """
        <style>
        .reportview-container {
            background: #f0f2f6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Input fields with placeholders
    a = st.text_input("Item Category", placeholder="Type Here")
    b = st.text_input("Subcategory-1", placeholder="Type Here")
    c = st.text_input("Subcategory-2", placeholder="Type Here")
    d = st.text_input("Item Rating", placeholder="Type Here")
    
    # Prediction button
    if st.button("Predict"):
        result = predict_selling_price(model, a, b, c, d)
        st.success(f"The predicted selling price is: {result[0]:,.2f}")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pandas as pd

train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

# Combine train and test data for label encoding
combined_data = pd.concat([train, test], ignore_index=True)

label_encoders = {}
for col in combined_data.select_dtypes(include=['object']).columns:
    label_encoders[col] = LabelEncoder()
    # Fit on the combined data
    label_encoders[col].fit(combined_data[col])
    # Transform both train and test data
    train[col] = label_encoders[col].transform(train[col])
    test[col] = label_encoders[col].transform(test[col])

X = train.drop(['Selling_Price', 'Date', 'Product', 'Product_Brand'], axis=1)  
y = train['Selling_Price']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


X_train.head()
       
if __name__ == "__main__":
    main()
