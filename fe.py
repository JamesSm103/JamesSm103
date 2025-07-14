import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib

df = pd.read_csv("NL_Batting_2024.csv")

model = joblib.load("xgb_main_model.pkl")
st.title("MLB Player Performance Prediction")

st.write("This app predicts player performance based on historical data.")
st.subheader("Player Statistics")

