import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog

st.title("Fertilizer Nutrient Calculator with Tolerance")

st.write("""
Enter desired PPM for each nutrient, and a tolerance value.
The app will calculate realistic kg amounts for each fertilizer for the tank volume within the given tolerance.
""")

# User input
N_ppm = st.number_input("Nitrogen (N) PPM", 0.0)
K_ppm = st.number_input("Potassium (K) PPM", 0.0)
P_ppm = st.number_input("Phosphorus (P2O5) PPM", 0.0)
Mg_ppm = st.number_input("Magnesium (Mg) PPM", 0.0)
Ca_ppm = st.number_input("Calcium (Ca) PPM", 0.0)
S_ppm = st.number_input("Sulfur (S) PPM", 0.0)
tolerance = st.number_input("Tolerance (PPM)", 20.0)
volume = st.number_input("Tank volume (liters)", 500.0)

target_ppm = np.array([N_ppm, K_ppm, P_ppm, Mg_ppm, Ca_ppm, S_ppm])
target_total_kg = target_ppm * volume / 1000  # ppm*L to kg
tolerance_kg = tolerance * volume / 1000

# Fertilizers and nutrient percentages (%)
fertilizers = pd.DataFrame({
    "N": [13.7, 0, 0, 13.7, 15, 0, 0],
    "K": [38.7, 50, 0, 0, 0, 0, 34],
    "P": [0, 0, 0, 0, 0, 48, 52],
    "Mg": [0, 0, 9.7, 9.7, 0, 0, 0],
    "Ca": [0, 0, 0, 0, 19, 0, 0],
    "S": [0, 18, 13, 0, 0, 0, 0]
}, index=["K GG","SOP","MgSO4","Mg(NO3)2","Ca(NO3)2","MAP","MKP"])

A = fertilizers.values.T / 100  # nutrients x fertilizers

# Create inequalities for tolerance
# A*x <= b_max and -A*x <= -b_min
b_max = target_total_kg + tolerance_kg
b_min = target_total_kg - tolerance_kg

A_ub = np.vstack([A, -A])
b_ub = np.concatenate([b_max, -b_min])

# Objective: minimize total fertilizer weight
c = np.ones(len(fertilizers))
bounds = [(0, None) for _ in range(len(fertilizers))]

res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

if res.success:
    result = pd.Series(res.x, index=fertilizers.index)
    result = result.apply(lambda x: round(x,2))
    if st.button("Calculate fertilizer kg"):
        st.write("Recommended fertilizer amounts (kg) within tolerance:")
        st.table(result)
else:
    st.error("No feasible solution found within the given tolerance and fertilizer composition.")
