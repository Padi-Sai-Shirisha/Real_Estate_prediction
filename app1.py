# import pickle
import pandas as pd

import streamlit as st 

# Load the trained model
# pickle_in = open("reg.pkl","rb")
# reg = pickle.load(pickle_in)

# Define the function to predict real estate prices
def Real_Estate_prediction(a, b, c, d, e, f):
    prediction = reg.predict([[a, b, c, d, e, f]])
    return prediction

# Define the main function
def main():
    st.title("Real Estate Price Prediction")
    st.write("This app predicts the price of real estate based on certain features.")
    
    # Set background color
    st.markdown(
        """
        <style>
        .reportview-container {
            background: #f0f2f6
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Input fields with placeholders
    st.write("Enter the details below and click on 'Predict' to get the price prediction.")
    a = st.number_input("X1 Transaction Date", value=0.0, step=1.0, format="%.2f", help="Type here")
    b = st.number_input("X2 House Age", value=0.0, step=1.0, format="%.2f", help="Type here")
    c = st.number_input("X3 Distance to the Nearest MRT Station", value=0.0, step=1.0, format="%.2f", help="Type here")
    d = st.number_input("X4 Number of Convenience Stores", value=0.0, step=1.0, format="%.2f", help="Type here")
    e = st.number_input("X5 Latitude", value=0.0, step=1.0, format="%.2f", help="Type here")
    f = st.number_input("X6 Longitude", value=0.0, step=1.0, format="%.2f", help="Type here")
    
    # Prediction button
    if st.button("Predict"):
        result = Real_Estate_prediction(a, b, c, d, e, f)
        st.success(f"The predicted price is {result[0]:,.2f} units.")


from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('Real estate.csv')

x=df.drop(['Y house price of unit area', 'No'], axis=1)
y=df['Y house price of unit area']

x.head(10)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)

x_train.head(10)


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)


if __name__ == "__main__":
    main()
