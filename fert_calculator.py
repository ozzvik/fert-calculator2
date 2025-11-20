import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog

st.title("Fertilizer Nutrient Calculator (LP)")

st.write("""
Enter desired PPM for each nutrient, and the app will calculate realistic kg amounts for each fertilizer for the tank volume.
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
target_total_kg = target_ppm * volume / 1000  # convert ppm*L to kg

# Fertilizers and nutrient percentages (%)
fertilizers = pd.DataFrame({
    "N": [13.7, 0, 0, 13.7, 15, 0, 0],
    "K": [38.7, 50, 0, 0, 0, 0, 34],
    "P": [0, 0, 0, 0, 0, 48, 52],
    "Mg": [0, 0, 9.7, 9.7, 0, 0, 0],
    "Ca": [0, 0, 0, 0, 19, 0, 0],
    "S": [0, 18, 13, 0, 0, 0, 0]
}, index=["K GG","SOP","MgSO4","Mg(NO3)2","Ca(NO3)2","MAP","MKP"])

# Convert percentages to fractions
A = fertilizers.values.T / 100  # shape: nutrients x fertilizers

# Objective: minimize total kg used (arbitrary choice)
c = np.ones(len(fertilizers))  # minimize total weight

# Bounds: all fertilizers ≥ 0
bounds = [(0, None) for _ in range(len(fertilizers))]

# Solve LP
res = linprog(c, A_eq=A, b_eq=target_total_kg, bounds=bounds, method='highs')

if res.success:
    result = pd.Series(res.x, index=fertilizers.index)
    result = result.apply(lambda x: round(x, 2))
    if st.button("Calculate fertilizer kg"):
        st.write("Recommended fertilizer amounts (kg) for the entered tank volume and PPM:")
        st.table(result)
else:
    st.error("No feasible solution found with the given PPM values and fertilizer composition.")
