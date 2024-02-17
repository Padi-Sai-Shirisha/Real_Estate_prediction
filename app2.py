import numpy as np
import pickle
import pandas as pd
import streamlit as st 

# Load the trained model
pickle_in = open("model.pkl","rb")
model = pickle.load(pickle_in)

# Define function to predict selling price
def predict_selling_price(a, b, c):
    prediction = model.predict([[a, b, c]])
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
    a = st.text_input("Item Rating", placeholder="Type Here")
    b = st.text_input("Month", placeholder="Type Here")
    c = st.text_input("Year", placeholder="Type Here")
    
    # Prediction button
    if st.button("Predict"):
        result = predict_selling_price(float(a), float(b), float(c))
        st.success(f"The predicted selling price is: {result[0]:,.2f}")
        
if __name__ == "__main__":
    main()
