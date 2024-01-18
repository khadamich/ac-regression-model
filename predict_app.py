import pandas as pd
import numpy as np
import joblib
import streamlit as st

model, ref_cols, target = joblib.load("model.pkl")

# df = pd.read_pickle("../../data/interim/data_iterim_01.pkl")
# df = df[2:][ref_cols + [target]].reset_index(drop=True)


def enconder(df):
    df = df.copy()
    df.loc[df["Subtechnology"] == "Body Mist", "Subtechnology"] = 0
    df.loc[df["Subtechnology"] == "EDP", "Subtechnology"] = 1

    return df


def predictions(df):
    df = enconder(df)

    features = df[ref_cols]

    predictions = model.predict(features)

    return predictions


def modeling(df):
    df["Predictions"] = predictions(df)

    values = df[["Price", "Predictions"]]

    condition = []

    for i in range(0, len(values)):
        y = values.iloc[i][0]
        y_hat = values.iloc[i][1]

        percentage = 100 * abs(y - y_hat) / y_hat

        if 0 <= percentage < 5:
            condition.append("Good")
        elif 5 <= percentage <= 10:
            condition.append("Normal")
        else:
            condition.append("Bad")

    df["Recomendation"] = condition

    return df


# Streamlit app
st.title("Price Prediction App")

# Upload a CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Make predictions and add a new column
    df = modeling(df)

    # Show a preview of the output
    st.write("Preview of the Output:")
    st.write(df.head())

    # Allow the user to download the output CSV file
    output_file = "output_predictions.csv"
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Output CSV",
        data=csv.encode(),
        file_name=output_file,
        key="download_button",
    )
