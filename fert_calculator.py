import streamlit as st
import numpy as np
import pandas as pd

st.title("Fertilizer Nutrient Calculator")

st.write("""
Enter the desired PPM for each nutrient, and the app will calculate how many kilograms of each fertilizer to dissolve for the tank volume.
""")

# User input
N_ppm = st.number_input("Nitrogen (N) PPM", 0.0)
K_ppm = st.number_input("Potassium (K) PPM", 0.0)
P_ppm = st.number_input("Phosphorus (P2O5) PPM", 0.0)
Mg_ppm = st.number_input("Magnesium (Mg) PPM", 0.0)
Ca_ppm = st.number_input("Calcium (Ca) PPM", 0.0)
S_ppm = st.number_input("Sulfur (S) PPM", 0.0)
volume = st.number_input("Tank volume (liters)", 500.0)

target_ppm = np.array([N_ppm, K_ppm, P_ppm, Mg_ppm, Ca_ppm, S_ppm])

# Fertilizers and nutrient percentages (%)
fertilizers = pd.DataFrame({
    "N": [13.7, 0, 0, 13.7, 15, 0, 0],
    "K": [38.7, 50, 0, 0, 0, 0, 34],
    "P": [0, 0, 0, 0, 0, 48, 52],
    "Mg": [0, 0, 9.7, 9.7, 0, 0, 0],
    "Ca": [0, 0, 0, 0, 19, 0, 0],
    "S": [0, 18, 13, 0, 0, 0, 0]
}, index=["K GG","SOP","MgSO4","Mg(NO3)2","Ca(NO3)2","MAP","MKP"])

percent_matrix = fertilizers.values / 100  # convert percentages to decimal

# Calculate kg per fertilizer
kg_solution = np.linalg.lstsq(percent_matrix.T, target_ppm * volume / 1000, rcond=None)[0]

result = pd.Series(kg_solution, index=fertilizers.index)
result = result.apply(lambda x: round(x, 2))

if st.button("Calculate fertilizer kg"):
    st.write("Recommended fertilizer amounts (kg) for the entered tank volume and PPM:")
    st.table(result)
