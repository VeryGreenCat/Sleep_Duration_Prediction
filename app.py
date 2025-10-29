import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Sleep Duration Predictor", layout="wide")

st.title("Sleep Duration Prediction App")
st.markdown("Predict your sleep duration based on daily activity, location, and phone usage patterns.")

# --- Load models ---
@st.cache_resource
def load_models():
    models = {
        "Linear Regression": {"model": joblib.load("model_lin.pkl"), "accuracy": 0.82},
        "Random Forest": {"model": joblib.load("model_tree.pkl"), "accuracy": 0.85},
        "XGBoost": {"model": joblib.load("model_xgb.pkl"), "accuracy": 0.87},
        "MLP": {"model": joblib.load("model_mlp.pkl"), "accuracy": 0.86},
    }
    return models

models = load_models()

# --- Feature groups ---
sleep_cols = [
    'f_slp:fitbit_sleep_summary_rapids_avgefficiencymain:allday',
    'f_slp:fitbit_sleep_summary_rapids_avgefficiencymain_norm:allday',
]

bluetooth_cols = [
    'f_blue:phone_bluetooth_doryab_uniquedevicesall:morning',
    'f_blue:phone_bluetooth_doryab_uniquedevicesall:afternoon',
    'f_blue:phone_bluetooth_doryab_uniquedevicesall:evening',
    'f_blue:phone_bluetooth_doryab_uniquedevicesall:night',
]

call_cols = ['f_call:phone_calls_rapids_incoming_count:allday',
    'f_call:phone_calls_rapids_outgoing_count:allday',
    'f_call:phone_calls_rapids_outgoing_sumduration:allday',]

location_cols = [
    'f_loc:phone_locations_doryab_movingtostaticratio:afternoon',
    'f_loc:phone_locations_doryab_timeathome:afternoon',
    'f_loc:phone_locations_locmap_duration_in_locmap_study:afternoon',
    'f_loc:phone_locations_locmap_duration_in_locmap_exercise:afternoon',
    'f_loc:phone_locations_locmap_duration_in_locmap_greens:afternoon',
    'f_loc:phone_locations_barnett_hometime:allday',
    'f_loc:phone_locations_barnett_rog:allday',
    'f_loc:phone_locations_barnett_siglocsvisited:allday',
    'f_loc:phone_locations_barnett_wkenddayrtn:allday',
    'f_loc:phone_locations_doryab_movingtostaticratio:allday',
    'f_loc:phone_locations_doryab_timeathome:allday',
    'f_loc:phone_locations_doryab_totaldistance:allday',
    'f_loc:phone_locations_locmap_duration_in_locmap_study:allday',
    'f_loc:phone_locations_locmap_duration_in_locmap_exercise:allday',
    'f_loc:phone_locations_locmap_duration_in_locmap_greens:allday',
    'f_loc:phone_locations_doryab_movingtostaticratio:evening',
    'f_loc:phone_locations_doryab_totaldistance:evening',
    'f_loc:phone_locations_locmap_duration_in_locmap_greens:evening',
    'f_loc:phone_locations_doryab_timeathome:morning',
    'f_loc:phone_locations_doryab_totaldistance:morning',
    'f_loc:phone_locations_locmap_duration_in_locmap_study:morning',
    'f_loc:phone_locations_doryab_movingtostaticratio:night',
    'f_loc:phone_locations_doryab_timeathome:night',
    'f_loc:phone_locations_doryab_totaldistance:night',
    'f_loc:phone_locations_locmap_duration_in_locmap_exercise:night',
]

screen_cols = [
    'f_screen:phone_screen_rapids_countepisodeunlock:afternoon',
    'f_screen:phone_screen_rapids_sumdurationunlock_locmap_greens:afternoon',
    'f_screen:phone_screen_rapids_sumdurationunlock_locmap_living:afternoon',
    'f_screen:phone_screen_rapids_sumdurationunlock_locmap_study:afternoon',
    'f_screen:phone_screen_rapids_sumdurationunlock_locmap_home:afternoon',
    'f_screen:phone_screen_rapids_countepisodeunlock:allday',
    'f_screen:phone_screen_rapids_sumdurationunlock:allday',
    'f_screen:phone_screen_rapids_sumdurationunlock_locmap_greens:allday',
    'f_screen:phone_screen_rapids_sumdurationunlock_locmap_living:allday',
    'f_screen:phone_screen_rapids_sumdurationunlock_locmap_study:allday',
    'f_screen:phone_screen_rapids_sumdurationunlock_locmap_home:allday',
    'f_screen:phone_screen_rapids_countepisodeunlock:evening',
    'f_screen:phone_screen_rapids_sumdurationunlock:evening',
    'f_screen:phone_screen_rapids_sumdurationunlock_locmap_living:evening',
    'f_screen:phone_screen_rapids_sumdurationunlock_locmap_home:evening',
    'f_screen:phone_screen_rapids_sumdurationunlock:morning',
    'f_screen:phone_screen_rapids_sumdurationunlock_locmap_home:morning',
    'f_screen:phone_screen_rapids_countepisodeunlock:night',
    'f_screen:phone_screen_rapids_sumdurationunlock:night',
    'f_screen:phone_screen_rapids_sumdurationunlock_locmap_living:night',
    'f_screen:phone_screen_rapids_sumdurationunlock_locmap_home:night',
]

steps_cols = [
    'f_steps:fitbit_steps_intraday_rapids_sumsteps:afternoon',
    'f_steps:fitbit_steps_intraday_rapids_countepisodeactivebout:afternoon',
    'f_steps:fitbit_steps_intraday_rapids_countepisodeactivebout:allday',
    'f_steps:fitbit_steps_intraday_rapids_sumdurationactivebout:allday',
    'f_steps:fitbit_steps_intraday_rapids_sumsteps:evening',
    'f_steps:fitbit_steps_intraday_rapids_sumdurationactivebout:evening',
    'f_steps:fitbit_steps_intraday_rapids_countepisodeactivebout:morning',
    'f_steps:fitbit_steps_intraday_rapids_sumsteps:night',
    'f_steps:fitbit_steps_intraday_rapids_countepisodeactivebout:night',
    'f_steps:fitbit_steps_intraday_rapids_sumdurationactivebout:night',
]

# time features should be in range 1-31 for day and 1-12 for month with no floats values
time = ['day', 'month']


#--------------------- start fix here ---------------------
# ['pid'] not included as input feature but always set as 0
# --- Input widgets (two per row, inline) ---
def input_section(label, cols):
    st.markdown(f"### {label}")
    inputs = {}
    num_cols = 3  # inputs per row
    rows = [cols[i:i + num_cols] for i in range(0, len(cols), num_cols)]

    for row in rows:
        c = st.columns(num_cols)
        for idx, col_name in enumerate(row):
            with c[idx]:
                val = st.session_state.get(col_name, 0.5)
                inputs[col_name] = st.number_input(
                    col_name, min_value=0.0, max_value=100.0, value=val, step=0.01, key=col_name
                )
    return inputs

def time_inputs():
    st.markdown("### Time Features")
    inputs = {}
    c1, c2 = st.columns(2)
    with c1:
        val = st.session_state.get('day', 1)
        inputs['day'] = st.number_input("Day", min_value=1, max_value=31, value=val, step=1, key='day')
    with c2:
        val = st.session_state.get('month', 1)
        inputs['month'] = st.number_input("Month", min_value=1, max_value=12, value=val, step=1, key='month')
    return inputs

time_input = time_inputs()

# Always set pid to 0
pid_input = {'pid': 0}

# --- Sidebar controls ---
st.sidebar.header("Controls")
randomize = st.sidebar.button("Randomize All")
predict = st.sidebar.button("Predict Sleep Duration")

# --- Input sections ---
col1, = st.columns(1)

with col1:
    sleep_inputs = input_section("Sleep", sleep_cols)
    bluetooth_inputs = input_section("Bluetooth", bluetooth_cols)
    call_inputs = input_section("Call", call_cols)
    location_inputs = input_section("Location", location_cols)
    screen_inputs = input_section("Screen", screen_cols)
    steps_inputs = input_section("Steps", steps_cols)

# --- Combine all inputs ---
all_inputs = {**pid_input, **time_input, **sleep_inputs, **bluetooth_inputs, **call_inputs,
              **location_inputs, **screen_inputs, **steps_inputs}

# --- Randomize values ---
if randomize:
    for key in all_inputs:
        if key not in ['pid']:  # pid stays 0
            if key in ['day']:
                st.session_state[key] = np.random.randint(1, 32)
            elif key in ['month']:
                st.session_state[key] = np.random.randint(1, 13)
            else:
                st.session_state[key] = np.round(np.random.uniform(0.0, 100.0), 2)
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