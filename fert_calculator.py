import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog

st.title("Dual Program Fertilizer Calculator (Stable Clean Version)")

# ---------------------------------------------------------
# Fertilizer composition
# ---------------------------------------------------------
fertilizers = pd.DataFrame({
    "N":  [13.7, 0,   0,   13.7, 15,  0,   0 ],
    "K":  [38.7, 50,  0,   0,    0,   0,  34 ],
    "P":  [0,    0,   0,   0,    0,  48,  52 ],
    "Mg": [0,    0,   9.7, 9.7,  0,   0,   0 ],
    "Ca": [0,    0,   0,   0,   19,  0,   0 ],
    "S":  [0,   18,  13,   0,    0,   0,   0 ]
}, index=["K GG","SOP","MgSO4","Mg(NO3)2","Ca(NO3)2","MAP","MKP"])

A = fertilizers.values.T / 100   # 6×7 matrix

# ---------------------------------------------------------
# Preset Programs
# ---------------------------------------------------------
veg_programs = {
    "Veg Early": {"N":200, "K":50, "P":30, "Mg":50, "Ca":80, "S":20},
    "Veg Late":  {"N":180, "K":70, "P":40, "Mg":50, "Ca":80, "S":20}
}

flower_programs = {
    "Flower Stage 1": {"N":150, "K":100, "P":50, "Mg":50, "Ca":80, "S":20},
    "Flower Stage 2": {"N":120, "K":130, "P":60, "Mg":50, "Ca":80, "S":20},
    "Flower Stage 3": {"N":100, "K":160, "P":70, "Mg":50, "Ca":80, "S":20}
}

# ---------------------------------------------------------
# Inputs
# ---------------------------------------------------------
st.header("1️⃣ Tank Setup")
stock_volume = st.number_input("Stock tank volume (L)", value=500, step=1)
irrigation_volume = st.number_input("Irrigation tank volume (L)", value=1000, step=1)
tolerance = st.number_input("PPM Tolerance", value=5, step=1)

st.header("2️⃣ Select Veg & Flower Programs (or Manual)")
veg_choice = st.selectbox("Veg Program", ["Manual"] + list(veg_programs.keys()))
flower_choice = st.selectbox("Flower Program", ["Manual"] + list(flower_programs.keys()))

# ---------------------------------------------------------
# Manual inputs (with unique IDs)
# ---------------------------------------------------------
def manual_ppm(prefix):
    return np.array([
        st.number_input(f"{prefix} Nitrogen (N)", 0, step=1),
        st.number_input(f"{prefix} Potassium (K)", 0, step=1),
        st.number_input(f"{prefix} Phosphorus (P)", 0, step=1),
        st.number_input(f"{prefix} Magnesium (Mg)", 0, step=1),
        st.number_input(f"{prefix} Calcium (Ca)", 0, step=1),
        st.number_input(f"{prefix} Sulfur (S)", 0, step=1)
    ])

def get_program(choice, programs, prefix):
    if choice == "Manual":
        return manual_ppm(prefix)
    p = programs[choice]
    return np.array([p["N"], p["K"], p["P"], p["Mg"], p["Ca"], p["S"]])

target_veg = get_program(veg_choice, veg_programs, "Veg")
target_flower = get_program(flower_choice, flower_programs, "Flower")

# ---------------------------------------------------------
# Solve for stock tank fertilizer KG
# ---------------------------------------------------------
target_min = (np.minimum(target_veg, target_flower) - tolerance) * (stock_volume / 1000)
target_max = (np.maximum(target_veg, target_flower) + tolerance) * (stock_volume / 1000)

A_ub = np.vstack([A, -A])
b_ub = np.concatenate([target_max, -target_min])

c = np.ones(len(fertilizers))
bounds = [(0, None)] * len(fertilizers)

res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# ---------------------------------------------------------
# If success → Calculate
# ---------------------------------------------------------
if res.success and st.button("Calculate"):
    stock_kg = pd.Series(np.round(res.x).astype(int), index=fertilizers.index)

    st.subheader("1️⃣ Fertilizer (kg) to dissolve in Stock Tank")
    st.table(stock_kg)

    # ---------------------------------------------------------
    # PPM produced per 1 liter of irrigation from each stock component
    # ---------------------------------------------------------
    ppm_per_liter = (A.T * res.x[:, None]) * (stock_volume / irrigation_volume) * 1000
    ppm_df = pd.DataFrame(ppm_per_liter.astype(int),
                          index=fertilizers.index,
                          columns=["N","K","P","Mg","Ca","S"])
    
    st.subheader("2️⃣ PPM Contribution per Liter in Irrigation Tank")
    st.table(ppm_df)

    # ---------------------------------------------------------
    # Stock liters needed to reach target PPM
    # ---------------------------------------------------------
    A_ppm = ppm_per_liter.T  # 6×7

    def calc_stock_liters(target_ppm):
        x, *_ = np.linalg.lstsq(A_ppm, target_ppm, rcond=None)
        return np.clip(np.round(x).astype(int), 0, None)

    liters_veg = pd.Series(calc_stock_liters(target_veg), index=fertilizers.index)
    liters_flower = pd.Series(calc_stock_liters(target_flower), index=fertilizers.index)

    st.subheader("3️⃣ Veg Program → Liters of Stock to Add per Irrigation Tank")
    st.table(liters_veg)

    st.subheader("4️⃣ Flower Program → Liters of Stock to Add per Irrigation Tank")
    st.table(liters_flower)

elif not res.success:
    st.error("❌ No feasible solution within tolerance using given fertilizer set.")
