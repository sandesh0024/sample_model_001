import streamlit as st
import numpy as np
import pickle
import pandas as pd

with open(r'Model_deployment_sample_project001_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open(r'Model_deployment_sample_project001_sc.pkl', 'rb') as file:
    sc = pickle.load(file)

def main():
    st.title("Ad Budget Prediction Model")
    st.write("Enter the budgets for TV, Radio, and Newspaper ads to get predictions.")

    tv_budget = st.number_input("TV Ad Budget ($)", min_value=0.0, format="%.2f")
    radio_budget = st.number_input("Radio Ad Budget ($)", min_value=0.0, format="%.2f")
    newspaper_budget = st.number_input("Newspaper Ad Budget ($)", min_value=0.0, format="%.2f")

    if st.button("Predict"):
        # data_to_predict = [[tv_budget, radio_budget, newspaper_budget]]

        data_to_predict = pd.DataFrame(
            {'TV Ad Budget ($)': tv_budget,
             'Radio Ad Budget ($)': radio_budget,
             'Newspaper Ad Budget ($)': newspaper_budget},
            index=[0]
        )

        transformed_data = sc.transform(data_to_predict)
        predictions = loaded_model.predict(transformed_data)
        st.success(f"Predicted Value: {predictions[0]:.2f}")

if __name__ == "__main__":
    main()

# st.title("Ad Budget Prediction App without defining a function")
#
# st.write("Enter the budgets for TV, Radio, and Newspaper ads to get predictions.")
#
# tv_budget = st.number_input("TV Ad Budget ($)", min_value=0.0, format="%.2f")
# radio_budget = st.number_input("Radio Ad Budget ($)", min_value=0.0, format="%.2f")
# newspaper_budget = st.number_input("Newspaper Ad Budget ($)", min_value=0.0, format="%.2f")
#
# if st.button("Predict"):
#     data_to_predict = [[tv_budget, radio_budget, newspaper_budget]]
#     transformed_data = sc.transform(data_to_predict)
#     predictions = loaded_model.predict(transformed_data)
#     st.success(f"Predicted Value: {predictions[0]:.2f}")