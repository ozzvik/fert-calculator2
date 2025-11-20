import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog

st.title("Dual Program Fertilizer Calculator")

st.write("""
Enter desired PPM for two programs (A = Veg, B = Flower), stock tank volume, and irrigation tank volume.  
The app calculates kg of fertilizers for the stock solution and liters/mL to add per irrigation tank for both programs.
""")

# Inputs
st.header("1️⃣ Stock and Irrigation Setup")
stock_volume = st.number_input("Stock tank volume (liters)", 500, step=1)
irrigation_volume = st.number_input("Irrigation tank volume (liters)", 1000, step=1)

st.header("2️⃣ Program A PPM (Veg)")
N_A = st.number_input("Program A Nitrogen (N) PPM", 0, step=1, key="NA")
K_A = st.number_input("Program A Potassium (K) PPM", 0, step=1, key="KA")
P_A = st.number_input("Program A Phosphorus (P2O5) PPM", 0, step=1, key="PA")
Mg_A = st.number_input("Program A Magnesium (Mg) PPM", 0, step=1, key="MgA")
Ca_A = st.number_input("Program A Calcium (Ca) PPM", 0, step=1, key="CaA")
S_A = st.number_input("Program A Sulfur (S) PPM", 0, step=1, key="SA")

st.header("3️⃣ Program B PPM (Flower)")
N_B = st.number_input("Program B Nitrogen (N) PPM", 0, step=1, key="NB")
K_B = st.number_input("Program B Potassium (K) PPM", 0, step=1, key="KB")
P_B = st.number_input("Program B Phosphorus (P2O5) PPM", 0, step=1, key="PB")
Mg_B = st.number_input("Program B Magnesium (Mg) PPM", 0, step=1, key="MgB")
Ca_B = st.number_input("Program B Calcium (Ca) PPM", 0, step=1, key="CaB")
S_B = st.number_input("Program B Sulfur (S) PPM", 0, step=1, key="SB")

tolerance = st.number_input("Tolerance (PPM)", 20, step=1)

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

target_ppm_A = np.array([N_A,K_A,P_A,Mg_A,Ca_A,S_A])
target_ppm_B = np.array([N_B,K_B,P_B,Mg_B,Ca_B,S_B])

target_total_kg_A = target_ppm_A * stock_volume / 1000
target_total_kg_B = target_ppm_B * stock_volume / 1000

b_max = np.maximum(target_total_kg_A, target_total_kg_B) + tolerance*stock_volume/1000
b_min = np.minimum(target_total_kg_A, target_total_kg_B) - tolerance*stock_volume/1000
A_ub = np.vstack([A, -A])
b_ub = np.concatenate([b_max, -b_min])

c = np.ones(len(fertilizers))
bounds = [(0, None) for _ in range(len(fertilizers))]

res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

if res.success:
    stock_kg = pd.Series(res.x, index=fertilizers.index).apply(lambda x: int(round(x)))
    ppm_per_liter_stock = (res.x[:, None] * fertilizers.values / stock_volume * 1000)
    
    if st.button("Calculate Stock and Irrigation"):
        st.subheader("1️⃣ Fertilizer kg to dissolve in stock tank:")
        st.table(stock_kg)
        
        st.subheader("2️⃣ PPM per liter in irrigation tank")
        # PPM in irrigation tank = ppm_per_liter_stock * stock_liters / irrigation_volume
        ppm_irrigation_per_fertilizer = (ppm_per_liter_stock * stock_volume / irrigation_volume)
        ppm_irrigation_df = pd.DataFrame(ppm_irrigation_per_fertilizer, index=fertilizers.index, columns=["N","K","P","Mg","Ca","S"])
        ppm_irrigation_df = ppm_irrigation_df.astype(int)
        st.table(ppm_irrigation_df)
        
        st.header("3️⃣ Stock liters to add per irrigation tank for each program")
        # Total PPM per liter of irrigation from the stock
        total_ppm_per_liter = ppm_irrigation_df.sum(axis=0)
        
        liters_needed_A = np.where(total_ppm_per_liter>0, target_ppm_A / total_ppm_per_liter, np.nan)
        liters_needed_B = np.where(total_ppm_per_liter>0, target_ppm_B / total_ppm_per_liter, np.nan)
        
        liters_needed_A_series = pd.Series(liters_needed_A, index=["N","K","P","Mg","Ca","S"]).apply(lambda x: int(round(x*1000)))  # mL
        liters_needed_B_series = pd.Series(liters_needed_B, index=["N","K","P","Mg","Ca","S"]).apply(lambda x: int(round(x*1000)))  # mL
        
        st.subheader("Program A (Veg) - stock to add per irrigation tank (mL):")
        st.table(liters_needed_A_series)
        
        st.subheader("Program B (Flower) - stock to add per irrigation tank (mL):")
        st.table(liters_needed_B_series)
        
else:
    st.error("No feasible solution found within the given tolerance and fertilizer composition.")
