import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog

st.title("Fertilizer Calculator with Preset Programs")

# Fertilizer composition (%)
fertilizers = pd.DataFrame({
    "N": [13.7, 0, 0, 13.7, 15, 0, 0],
    "K": [38.7, 50, 0, 0, 0, 0, 34],
    "P": [0, 0, 0, 0, 0, 48, 52],
    "Mg": [0, 0, 9.7, 9.7, 0, 0, 0],
    "Ca": [0, 0, 0, 0, 19, 0, 0],
    "S": [0, 18, 13, 0, 0, 0, 0]
}, index=["K GG","SOP","MgSO4","Mg(NO3)2","Ca(NO3)2","MAP","MKP"])

A = fertilizers.values.T / 100  # nutrients per kg fertilizer

# Preset programs
programs = {
    "Veg Early": {"N":200, "K":50, "P":30, "Mg":50, "Ca":80, "S":20},
    "Veg Late":  {"N":180, "K":70, "P":40, "Mg":50, "Ca":80, "S":20},
    "Flower Stage 1": {"N":150, "K":100, "P":50, "Mg":50, "Ca":80, "S":20},
    "Flower Stage 2": {"N":120, "K":130, "P":60, "Mg":50, "Ca":80, "S":20},
    "Flower Stage 3": {"N":100, "K":160, "P":70, "Mg":50, "Ca":80, "S":20},
}

st.header("1️⃣ Stock and Irrigation Setup")
stock_volume = st.number_input("Stock tank volume (liters)", 500, step=1)
irrigation_volume = st.number_input("Irrigation tank volume (liters)", 1000, step=1)
tolerance = st.number_input("Tolerance (PPM)", 5, step=1)

st.header("2️⃣ Choose Program")
program_name = st.selectbox("Select a preset program", list(programs.keys()))
target_ppm_dict = programs[program_name]
target_ppm = np.array([target_ppm_dict["N"], target_ppm_dict["K"], target_ppm_dict["P"],
                       target_ppm_dict["Mg"], target_ppm_dict["Ca"], target_ppm_dict["S"]])

# Solve for stock kg
target_total_kg = target_ppm * stock_volume / 1000
b_max = target_total_kg + tolerance*stock_volume/1000
b_min = target_total_kg - tolerance*stock_volume/1000
A_ub = np.vstack([A, -A])
b_ub = np.concatenate([b_max, -b_min])

c = np.ones(len(fertilizers))
bounds = [(0, None) for _ in range(len(fertilizers))]

res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

if res.success:
    stock_kg = pd.Series(res.x, index=fertilizers.index).apply(lambda x: int(round(x)))
    
    if st.button("Calculate Stock and Irrigation"):
        st.subheader("1️⃣ Fertilizer kg to dissolve in stock tank:")
        st.table(stock_kg)
        
        st.header("2️⃣ PPM contribution per liter of irrigation tank")
        ppm_per_liter_irrigation = (res.x[:, None] * A * stock_volume / irrigation_volume * 1000)
        ppm_irrigation_df = pd.DataFrame(ppm_per_liter_irrigation, index=fertilizers.index,
                                        columns=["N","K","P","Mg","Ca","S"])
        ppm_irrigation_df = ppm_irrigation_df.astype(int)
        st.table(ppm_irrigation_df)
        
        st.header("3️⃣ Stock liters to add per irrigation tank")
        total_ppm_per_liter = ppm_per_liter_irrigation.sum(axis=0)
        stock_liters_needed = np.linalg.lstsq(ppm_per_liter_irrigation.values.T, target_ppm, rcond=None)[0]
        stock_liters_series = pd.Series(stock_liters_needed, index=fertilizers.index).apply(lambda x: int(round(x)))
        st.table(stock_liters_series)
        
else:
    st.error("No feasible solution found within the given tolerance and fertilizer composition.")
