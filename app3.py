import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

combined_data = pd.concat([train, test], ignore_index=True)

label_encoders = {}
for col in combined_data.select_dtypes(include=['object']).columns:
    label_encoders[col] = LabelEncoder()
    combined_data[col] = label_encoders[col].fit_transform(combined_data[col])


train_encoded = combined_data[:len(train)]
test_encoded = combined_data[len(train):]


X_train = train_encoded.drop(['Selling_Price', 'Date', 'Product', 'Product_Brand'], axis=1)
y_train = train_encoded['Selling_Price']


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


def predict_selling_price(model, item_category, subcategory_1, subcategory_2, item_rating):
    
    item_category_encoded = label_encoders['Item_Category'].transform([item_category])[0]
    subcategory_1_encoded = label_encoders['Subcategory_1'].transform([subcategory_1])[0]
    subcategory_2_encoded = label_encoders['Subcategory_2'].transform([subcategory_2])[0]
    
    
    prediction = model.predict([[item_category_encoded, subcategory_1_encoded, subcategory_2_encoded, float(item_rating)]])
    return prediction



def main():
    st.title("Selling Price Prediction")
    st.write("Please select item category, subcategory 1, subcategory 2, and enter item rating to predict selling price.")

    item_category_options = sorted(train['Item_Category'].unique())
    item_category = st.selectbox("Item Category", 
                                 options=item_category_options, 
                                 format_func=lambda x: f"Select item category" if x == item_category_options[0] else f"{x} - {label_encoders['Item_Category'].transform([x])[0]}",
                                 index=0)

    subcategory_1_options = sorted(train['Subcategory_1'].unique())
    subcategory_1 = st.selectbox("Subcategory 1", 
                                  options=subcategory_1_options, 
                                  format_func=lambda x: f"Select subcategory 1" if x == subcategory_1_options[0] else f"{x} - {label_encoders['Subcategory_1'].transform([x])[0]}",
                                  index=0)

    subcategory_2_options = sorted(train['Subcategory_2'].unique())
    subcategory_2 = st.selectbox("Subcategory 2", 
                                  options=subcategory_2_options, 
                                  format_func=lambda x: f"Select subcategory 2" if x == subcategory_2_options[0] else f"{x} - {label_encoders['Subcategory_2'].transform([x])[0]}",
                                  index=0)

    item_rating = st.text_input("Item Rating", placeholder="Type Here")

    if st.button("Predict"):
        result = predict_selling_price(model, item_category, subcategory_1, subcategory_2, item_rating)
        st.success(f"The predicted selling price is: {result[0]:,.2f}")



if __name__ == "__main__":
    main()
