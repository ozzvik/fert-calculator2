import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog

st.title("Fertilizer Stock Calculator with Tolerance and Reuse")

st.write("""
Step 1: Enter desired PPM for your stock solution and tolerance.  
Step 2: Get kg of each fertilizer to dissolve.  
Step 3: The app calculates PPM per liter of stock for later use in other fertilizer plans.
""")

# Step 1: User inputs
st.header("Stock Solution Setup")
N_ppm = st.number_input("Nitrogen (N) PPM", 0.0)
K_ppm = st.number_input("Potassium (K) PPM", 0.0)
P_ppm = st.number_input("Phosphorus (P2O5) PPM", 0.0)
Mg_ppm = st.number_input("Magnesium (Mg) PPM", 0.0)
Ca_ppm = st.number_input("Calcium (Ca) PPM", 0.0)
S_ppm = st.number_input("Sulfur (S) PPM", 0.0)
tolerance = st.number_input("Tolerance (PPM)", 20.0, min_value=0.0)
stock_volume = st.number_input("Stock tank volume (liters)", 500.0)

target_ppm = np.array([N_ppm, K_ppm, P_ppm, Mg_ppm, Ca_ppm, S_ppm])
target_total_kg = target_ppm * stock_volume / 1000
tolerance_kg = tolerance * stock_volume / 1000

# Fertilizer composition (%)
fertilizers = pd.DataFrame({
    "N": [13.7, 0, 0, 13.7, 15, 0, 0],
    "K": [38.7, 50, 0, 0, 0, 0, 34],
    "P": [0, 0, 0, 0, 0, 48, 52],
    "Mg": [0, 0, 9.7, 9.7, 0, 0, 0],
    "Ca": [0, 0, 0, 0, 19, 0, 0],
    "S": [0, 18, 13, 0, 0, 0, 0]
}, index=["K GG","SOP","MgSO4","Mg(NO3)2","Ca(NO3)2","MAP","MKP"])

A = fertilizers.values.T / 100

# Inequalities for tolerance
b_max = target_total_kg + tolerance_kg
b_min = target_total_kg - tolerance_kg

A_ub = np.vstack([A, -A])
b_ub = np.concatenate([b_max, -b_min])

# Objective: minimize total fertilizer kg
c = np.ones(len(fertilizers))
bounds = [(0, None) for _ in range(len(fertilizers))]

res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

if res.success:
    stock_kg = pd.Series(res.x, index=fertilizers.index).apply(lambda x: round(x,2))
    ppm_per_liter = (res.x * 1000 / stock_volume * (fertilizers.values.T)).round(2)
    ppm_per_liter_df = pd.DataFrame(ppm_per_liter, index=fertilizers.index, columns=["N","K","P","Mg","Ca","S"])
    
    if st.button("Calculate Stock Solution"):
        st.subheader("1️⃣ Fertilizer kg to dissolve in stock tank:")
        st.table(stock_kg)
        
        st.subheader("2️⃣ PPM per liter of stock solution:")
        st.table(ppm_per_liter_df)
        
        st.header("Step 2: Use Stock for Other Fertilizer Plans")
        st.write("Enter desired PPM for a different plan to calculate liters of stock to use per tank:")
        
        N2 = st.number_input("New Plan N PPM", 0.0, key="new_N")
        K2 = st.number_input("New Plan K PPM", 0.0, key="new_K")
        P2 = st.number_input("New Plan P2O5 PPM", 0.0, key="new_P")
        Mg2 = st.number_input("New Plan Mg PPM", 0.0, key="new_Mg")
        Ca2 = st.number_input("New Plan Ca PPM", 0.0, key="new_Ca")
        S2 = st.number_input("New Plan S PPM", 0.0, key="new_S")
        
        desired_ppm = np.array([N2,K2,P2,Mg2,Ca2,S2])
        
        # Compute liters of stock needed per tank
        # Avoid division by zero
        liters_needed = np.where(ppm_per_liter_df.values.sum(axis=0)>0,
                                 desired_ppm / ppm_per_liter_df.sum(axis=0).values,
                                 np.nan)
        liters_needed_series = pd.Series(liters_needed, index=["N","K","P","Mg","Ca","S"]).round(2)
        st.subheader("3️⃣ Liters of stock to add per tank for new plan:")
        st.table(liters_needed_series)
        
else:
    st.error("No feasible solution found within the given tolerance and fertilizer composition.")
