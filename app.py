import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Sleep Duration Predictor", layout="wide")

st.title("😴 Sleep Duration Prediction App")
st.markdown("Predict your sleep duration based on daily activity, location, and phone usage patterns.")

# --- Load models ---
@st.cache_resource
def load_models():
    models = {
        "Linear Regression": {"model": joblib.load("model_linear.pkl"), "accuracy": 0.82},
        "Random Forest": {"model": joblib.load("model_rf.pkl"), "accuracy": 0.85},
        "XGBoost": {"model": joblib.load("model_xgb.pkl"), "accuracy": 0.87},
        "MLP": {"model": joblib.load("model_mlp.pkl"), "accuracy": 0.86},
    }
    return models

models = load_models()

# --- Feature groups ---
sleep_cols = [
    "f_slp:fitbit_sleep_summary_rapids_avgefficiencymain_norm:allday",
    "f_slp:fitbit_sleep_summary_rapids_sumdurationasleepmain_norm:allday",
]

bluetooth_cols = [
    "f_blue:phone_bluetooth_doryab_uniquedevicesall_norm:allday",
    "f_blue:phone_bluetooth_doryab_uniquedevicesall_norm:afternoon",
    "f_blue:phone_bluetooth_doryab_uniquedevicesall_norm:evening",
    "f_blue:phone_bluetooth_doryab_uniquedevicesall_norm:morning",
]

location_cols = [
    "f_loc:phone_locations_doryab_movingtostaticratio_norm:afternoon",
    "f_loc:phone_locations_doryab_timeathome_norm:afternoon",
    "f_loc:phone_locations_doryab_totaldistance_norm:afternoon",
    "f_loc:phone_locations_doryab_movingtostaticratio_norm:allday",
    "f_loc:phone_locations_doryab_timeathome_norm:allday",
    "f_loc:phone_locations_doryab_totaldistance_norm:allday",
]

screen_cols = [
    "f_screen:phone_screen_rapids_countepisodeunlock_norm:allday",
    "f_screen:phone_screen_rapids_sumdurationunlock_norm:allday",
    "f_screen:phone_screen_rapids_sumdurationunlock_locmap_home_norm:allday",
    "f_screen:phone_screen_rapids_sumdurationunlock_locmap_study_norm:allday",
    "f_screen:phone_screen_rapids_sumdurationunlock_locmap_greens_norm:allday",
]

steps_cols = [
    "f_steps:fitbit_steps_intraday_rapids_sumsteps_norm:allday",
    "f_steps:fitbit_steps_intraday_rapids_countepisodeactivebout_norm:allday",
    "f_steps:fitbit_steps_intraday_rapids_sumdurationactivebout_norm:allday",
]

# --- Input widgets ---
def input_section(label, cols):
    st.subheader(label)
    inputs = {}
    for c in cols:
        inputs[c] = st.number_input(c, min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    return inputs

st.sidebar.header("🧮 Controls")
randomize = st.sidebar.button("🎲 Randomize All")
predict = st.sidebar.button("🔮 Predict Sleep Duration")

st.markdown("### Input Your Daily Data")

col1, col2, col3 = st.columns(3)

with col1:
    sleep_inputs = input_section("Sleep", sleep_cols)
    bluetooth_inputs = input_section("Bluetooth", bluetooth_cols)

with col2:
    location_inputs = input_section("Location", location_cols)

with col3:
    screen_inputs = input_section("Screen", screen_cols)
    steps_inputs = input_section("Steps", steps_cols)

# --- Combine all inputs ---
all_inputs = {**sleep_inputs, **bluetooth_inputs, **location_inputs, **screen_inputs, **steps_inputs}

# --- Randomize values ---
if randomize:
    for key in all_inputs:
        st.session_state[key] = np.round(np.random.uniform(0.0, 1.0), 2)
    st.experimental_rerun()

# --- Prediction ---
if predict:
    input_df = pd.DataFrame([all_inputs])
    st.markdown("### 🧾 Model Predictions")
    results = []
    for name, info in models.items():
        pred = info["model"].predict(input_df)[0]
        results.append((name, round(pred, 2), info["accuracy"]))

    result_df = pd.DataFrame(results, columns=["Model", "Predicted Sleep Duration (hrs)", "Accuracy"])
    st.table(result_df)
    st.success("Prediction complete!")

