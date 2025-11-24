import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog

st.title("Dual Program Fertilizer Calculator — Separate Stocks (Stable)")

# ------------------------------
# Fertilizer composition (%) per fertilizer (rows) x nutrients (cols)
# index = fertilizer names, columns = N,K,P,Mg,Ca,S
# ------------------------------
fertilizers = pd.DataFrame({
    "N":  [13.7, 0,   0,   13.7, 15,  0,   0 ],
    "K":  [38.7, 50,  0,   0,    0,   0,  34 ],
    "P":  [0,    0,   0,   0,    0,  48,  52 ],
    "Mg": [0,    0,   9.7, 9.7,  0,   0,   0 ],
    "Ca": [0,    0,   0,   0,   19,  0,   0 ],
    "S":  [0,   18,  13,   0,    0,   0,   0 ]
}, index=["K GG","SOP","MgSO4","Mg(NO3)2","Ca(NO3)2","MAP","MKP"])

percent_fert = fertilizers.values / 100.0   # shape (7,6) fertilizer x nutrient
A = percent_fert.T                           # shape (6,7) nutrient x fertilizer

# ------------------------------
# Preset programs (examples)
# ------------------------------
veg_programs = {
    "Veg Early": {"N":200, "K":50, "P":30, "Mg":50, "Ca":80, "S":20},
    "Veg Late":  {"N":180, "K":70, "P":40, "Mg":50, "Ca":80, "S":20}
}

flower_programs = {
    "Flower Stage 1": {"N":150, "K":100, "P":50, "Mg":50, "Ca":80, "S":20},
    "Flower Stage 2": {"N":120, "K":130, "P":60, "Mg":50, "Ca":80, "S":20},
    "Flower Stage 3": {"N":100, "K":160, "P":70, "Mg":50, "Ca":80, "S":20}
}

# ------------------------------
# Inputs
# ------------------------------
st.header("1) Tanks & tolerance")
stock_vol_default = 500
irr_vol_default = 1000
stock_volume = st.number_input("Stock tank volume (L)", value=stock_vol_default, step=1)
irrigation_volume = st.number_input("Irrigation tank volume (L)", value=irr_vol_default, step=1)
tolerance_ppm = st.number_input("Tolerance (±PPM)", value=5, step=1, min_value=0)

st.header("2) Choose programs (or Manual)")
veg_choice = st.selectbox("Veg program", ["Manual"] + list(veg_programs.keys()))
flower_choice = st.selectbox("Flower program", ["Manual"] + list(flower_programs.keys()))

# manual inputs must have unique labels
def manual_ppm(prefix):
    return np.array([
        st.number_input(f"{prefix} Nitrogen (N) PPM", value=0, step=1),
        st.number_input(f"{prefix} Potassium (K) PPM", value=0, step=1),
        st.number_input(f"{prefix} Phosphorus (P2O5) PPM", value=0, step=1),
        st.number_input(f"{prefix} Magnesium (Mg) PPM", value=0, step=1),
        st.number_input(f"{prefix} Calcium (Ca) PPM", value=0, step=1),
        st.number_input(f"{prefix} Sulfur (S) PPM", value=0, step=1)
    ])

def get_target(choice, presets, prefix):
    if choice == "Manual":
        return manual_ppm(prefix)
    p = presets[choice]
    return np.array([p["N"], p["K"], p["P"], p["Mg"], p["Ca"], p["S"]])

target_veg_ppm = get_target(veg_choice, veg_programs, "Veg")
target_flower_ppm = get_target(flower_choice, flower_programs, "Flower")

# ------------------------------
# Helper: solve stock composition for one program
# ------------------------------
def solve_stock_for_program(target_ppm, stock_volume, tolerance_ppm):
    # Convert ppm (mg/L) to total kg required in stock tank:
    # total_kg = (ppm mg/L * tank_L) / 1e6  (because 1 kg = 1e6 mg)
    target_total_kg = target_ppm * stock_volume / 1e6
    tol_kg = tolerance_ppm * stock_volume / 1e6
    b_max = target_total_kg + tol_kg
    b_min = target_total_kg - tol_kg
    # build inequalities A_ub x <= b_ub
    A_ub = np.vstack([A, -A])      # (12x7)
    b_ub = np.concatenate([b_max, -b_min])
    c = np.ones(A.shape[1])        # minimize total kg
    bounds = [(0, None)] * A.shape[1]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    return res

# ------------------------------
# Solve separately for Veg and Flower
# ------------------------------
res_veg = solve_stock_for_program(target_veg_ppm, stock_volume, tolerance_ppm)
res_flower = solve_stock_for_program(target_flower_ppm, stock_volume, tolerance_ppm)

# ------------------------------
# Display results for each program separately
# ------------------------------
def present_result(res, program_name, target_ppm):
    st.subheader(f"{program_name} results")
    if not res.success:
        st.error(f"No feasible solution for {program_name} within tolerance.")
        return
    # kg per fertilizer to dissolve in stock
    kg_per_fert = pd.Series(np.round(res.x, 2), index=fertilizers.index)
    st.write("A) Fertilizer to dissolve in STOCK (kg):")
    st.table(kg_per_fert.astype(float))

    # ppm contribution per liter of stock (mg/L in stock) per fertilizer per nutrient:
    # mg/L in stock for nutrient j from fert i = (kg_i * percent_ij * 1e6) / stock_volume
    # ppm in irrigation per liter of stock added = (mg/L in stock) / irrigation_volume  (mg/L)
    percent = percent_fert  # (7,6)
    mg_per_L_stock = (res.x[:, None] * percent) * 1e6 / stock_volume    # shape (7,6) mg/L in stock
    ppm_per_Lstock_in_irrig = mg_per_L_stock / irrigation_volume       # mg/L per 1 L stock added -> ppm per L stock
    ppm_per_Lstock_df = pd.DataFrame(ppm_per_Lstock_in_irrig,
                                     index=fertilizers.index,
                                     columns=["N","K","P","Mg","Ca","S"])
    st.write("B) Contribution per 1 L of STOCK added to irrigation (PPM per nutrient):")
    st.table(ppm_per_Lstock_df.round(2))

    # Sum over fertilizers -> total ppm increase in irrigation per 1 L stock added (per nutrient)
    total_ppm_per_Lstock = ppm_per_Lstock_df.sum(axis=0).values   # shape (6,)

    # Compute optimal single liters of stock to add (scalar) to best match desired PPM in irrigation:
    # solve scalar L minimizing || total_ppm_per_Lstock * L - target_ppm ||^2
    denom = np.dot(total_ppm_per_Lstock, total_ppm_per_Lstock)
    if denom == 0:
        st.warning("Total ppm contribution per L stock is zero (no nutrients). Can't compute liters.")
        return
    L_opt = np.dot(total_ppm_per_Lstock, target_ppm) / denom
    L_opt = max(0.0, L_opt)   # non-negative
    # resulting ppm achieved
    ppm_achieved = total_ppm_per_Lstock * L_opt
    residual = target_ppm - ppm_achieved

    st.write("C) Suggested stock to add per irrigation (single scalar):")
    st.write(f"- Liters of STOCK to add per irrigation tank: {round(L_opt,2)} L  ({int(round(L_opt*1000))} mL)")
    st.write("- Resulting PPM in irrigation (per nutrient):")
    df_res = pd.DataFrame({
        "target_ppm": target_ppm.astype(int),
        "achieved_ppm": np.round(ppm_achieved, 2),
        "residual_ppm": np.round(residual, 2),
        "ppm_per_Lstock": np.round(total_ppm_per_Lstock, 4)
    }, index=["N","K","P","Mg","Ca","S"])
    st.table(df_res)

# Button to compute and show both
if st.button("Calculate Veg & Flower (separate stocks)"):
    present_result(res_veg, "VEG", target_veg_ppm)
    st.markdown("---")
    present_result(res_flower, "FLOWER", target_flower_ppm)
