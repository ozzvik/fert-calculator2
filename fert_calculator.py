import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog

st.title("Dual Program Fertilizer Calculator with Presets and Manual Input")

# Fertilizer composition (%)
fertilizers = pd.DataFrame({
    "N": [13.7, 0, 0, 13.7, 15, 0, 0],
    "K": [38.7, 50, 0, 0, 0, 0, 34],
    "P": [0, 0, 0, 0, 0, 48, 52],
    "Mg": [0, 0, 9.7, 9.7, 0, 0, 0],
    "Ca": [0, 0, 0, 0, 19, 0, 0],
    "S": [0, 18, 13, 0, 0, 0, 0]
}, index=["K GG","SOP","MgSO4","Mg(NO3)2","Ca(NO3)2","MAP","MKP"])

A = fertilizers.values.T / 100  # 6x7 matrix (N,K,P,Mg,Ca,S) per kg fertilizer

# Preset programs
veg_programs = {
    "Veg Early": {"N":200, "K":50, "P":30, "Mg":50, "Ca":80, "S":20},
    "Veg Late":  {"N":180, "K":70, "P":40, "Mg":50, "Ca":80, "S":20}
}

flower_programs = {
    "Flower Stage 1": {"N":150, "K":100, "P":50, "Mg":50, "Ca":80, "S":20},
    "Flower Stage 2": {"N":120, "K":130, "P":60, "Mg":50, "Ca":80, "S":20},
    "Flower Stage 3": {"N":100, "K":160, "P":70, "Mg":50, "Ca":80, "S":20}
}

st.header("1️⃣ Stock and Irrigation Setup")
stock_volume = st.number_input("Stock tank volume (liters)", 500, step=1)
irrigation_volume = st.number_input("Irrigation tank volume (liters)", 1000, step=1)
tolerance = st.number_input("Tolerance (PPM)", 5, step=1)

st.header("2️⃣ Program Selection (Optional)")
veg_choice = st.selectbox("Select Veg Program (or 'Manual')", ["Manual"] + list(veg_programs.keys()))
flower_choice = st.selectbox("Select Flower Program (or 'Manual')", ["Manual"] + list(flower_programs.keys()))

st.header("3️⃣ Manual PPM Input (overrides preset if filled)")
def manual_input(label, default=0):
    return st.number_input(label, value=default, step=1)

def get_program_ppm(choice, programs):
    if choice=="Manual":
        return np.array([
            manual_input("Nitrogen (N) PPM"),
            manual_input("Potassium (K) PPM"),
            manual_input("Phosphorus (P2O5) PPM"),
            manual_input("Magnesium (Mg) PPM"),
            manual_input("Calcium (Ca) PPM"),
            manual_input("Sulfur (S) PPM")
        ])
    else:
        prog = programs[choice]
        return np.array([prog["N"], prog["K"], prog["P"], prog["Mg"], prog["Ca"], prog["S"]])

target_veg = get_program_ppm(veg_choice, veg_programs)
target_flower = get_program_ppm(flower_choice, flower_programs)

# Solve for stock kg to satisfy both programs within tolerance
target_min = np.minimum(target_veg, target_flower) * stock_volume/1000 - tolerance*stock_volume/1000
target_max = np.maximum(target_veg, target_flower) * stock_volume/1000 + tolerance*stock_volume/1000

A_ub = np.vstack([A, -A])
b_ub = np.concatenate([target_max, -target_min])
c = np.ones(len(fertilizers))
bounds = [(0, None) for _ in range(len(fertilizers))]

res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

if res.success:
    stock_kg = pd.Series(res.x, index=fertilizers.index).apply(lambda x: int(round(x)))
    
    if st.button("Calculate Stock and Irrigation"):
        st.subheader("1️⃣ Fertilizer kg to dissolve in stock tank:")
        st.table(stock_kg)
        
        st.header("2️⃣ PPM contribution per liter of irrigation tank")
        # Correct multiplication to match dimensions: each fertilizer (kg) times A / irrigation_volume
        ppm_per_liter_irrigation = (res.x.reshape(-1,1) * A) * stock_volume / irrigation_volume * 1000
        ppm_irrigation_df = pd.DataFrame(ppm_per_liter_irrigation, index=fertilizers.index,
                                         columns=["N","K","P","Mg","Ca","S"]).astype(int)
        st.table(ppm_irrigation_df)
        
        st.header("3️⃣ Stock liters to add per irrigation tank")
        def calc_stock_liters(target_ppm):
            total_ppm_matrix = ppm_per_liter_irrigation.T
            stock_liters, residuals, rank, s = np.linalg.lstsq(total_ppm_matrix, target_ppm, rcond=None)
            return np.clip(stock_liters, 0, None)
        
        liters_veg = pd.Series(calc_stock_liters(target_veg), index=fertilizers.index).apply(lambda x: int(round(x)))
        liters_flower = pd.Series(calc_stock_liters(target_flower), index=fertilizers.index).apply(lambda x: int(round(x)))
        
        st.subheader("Veg Program - liters of stock per irrigation tank:")
        st.table(liters_veg)
        
        st.subheader("Flower Program - liters of stock per irrigation tank:")
        st.table(liters_flower)
        
else:
    st.error("No feasible solution found within the given tolerance and fertilizer composition.")
