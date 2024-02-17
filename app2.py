# import numpy as np
# import pickle
import pandas as pd
import streamlit as st 

# Load the trained model
# pickle_in = open("model.pkl","rb")
# model = pickle.load(pickle_in)

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


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')


train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])


train['Month'] = train['Date'].dt.month
test['Month'] = test['Date'].dt.month
train['Year'] = train['Date'].dt.year
test['Year'] = test['Date'].dt.year


X = train[['Item_Rating', 'Month', 'Year']]
y = train['Selling_Price']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



model = LinearRegression()
model.fit(X_train, y_train)


       
if __name__ == "__main__":
    main()
