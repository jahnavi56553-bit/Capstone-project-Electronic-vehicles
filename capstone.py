import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import joblib
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart EV AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS — SAME STYLE
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(0,230,118,0.12), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(0,188,212,0.12), transparent 25%),
        linear-gradient(135deg, #06111f 0%, #0a1828 50%, #06111f 100%);
    color: white;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.topbar {
    background: linear-gradient(135deg, rgba(10,27,45,0.98), rgba(6,20,34,0.98));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 16px 22px;
    margin-bottom: 15px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.25);
}

.logo {
    display: flex;
    align-items: center;
    gap: 12px;
}

.logo-icon {
    font-size: 38px;
}

.logo-title {
    font-size: 24px;
    font-weight: 800;
    color: white;
}

.logo-subtitle {
    font-size: 11px;
    color: #8fa4b8;
}

div.stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(15,34,53,0.9);
    color: #d9e4ee;
    font-weight: 600;
    height: 46px;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, rgba(0,230,118,0.20), rgba(0,188,212,0.20));
    border-color: rgba(0,230,118,0.35);
    color: white;
    transform: translateY(-2px);
}

.hero {
    background: linear-gradient(135deg, rgba(0,230,118,0.12), rgba(0,188,212,0.08));
    border: 1px solid rgba(0,230,118,0.18);
    border-radius: 24px;
    padding: 40px;
    margin-bottom: 30px;
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(90deg, #00e676, #00d9ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #aebfd0;
    font-size: 17px;
    line-height: 1.6;
    margin-top: 12px;
}

.section-title {
    font-size: 30px;
    font-weight: 800;
    margin-top: 25px;
    margin-bottom: 20px;
}

.card {
    background: rgba(13,31,49,0.88);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}

.feature-card {
    background: rgba(13,31,49,0.90);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 25px;
    min-height: 200px;
}

.feature-icon {
    font-size: 40px;
}

.feature-title {
    font-size: 21px;
    font-weight: 700;
    margin-top: 12px;
}

.feature-text {
    color: #9eb0c2;
    line-height: 1.5;
    font-size: 14px;
    margin-top: 8px;
}

.metric-card {
    background: linear-gradient(145deg, rgba(15,43,62,0.95), rgba(8,25,41,0.95));
    border: 1px solid rgba(0,230,118,0.15);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    min-height: 135px;
}

.metric-title {
    color: #8ea4b8;
    font-size: 13px;
}

.metric-value {
    font-size: 29px;
    font-weight: 800;
    color: #00e676;
    margin-top: 8px;
}

.metric-small {
    color: #8196aa;
    font-size: 12px;
    margin-top: 5px;
}

.result-good {
    background: rgba(0,230,118,0.08);
    border: 1px solid rgba(0,230,118,0.30);
    border-radius: 16px;
    padding: 18px;
}

.result-warning {
    background: rgba(255,193,7,0.08);
    border: 1px solid rgba(255,193,7,0.30);
    border-radius: 16px;
    padding: 18px;
}

.result-danger {
    background: rgba(244,67,54,0.08);
    border: 1px solid rgba(244,67,54,0.30);
    border-radius: 16px;
    padding: 18px;
}

.info-box {
    background: rgba(0,188,212,0.07);
    border-left: 4px solid #00bcd4;
    border-radius: 10px;
    padding: 16px;
    margin: 20px 0;
}

.footer {
    text-align: center;
    color: #71879b;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 60px;
    padding-top: 25px;
}

@media (max-width: 768px) {
    .hero-title { font-size: 34px; }
    .hero { padding: 25px; }
    .logo-title { font-size: 20px; }
}
</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# GENERIC MODEL LOADER
# ============================================================

def load_model(files):
    for file in files:
        if os.path.exists(file):
            try:
                return joblib.load(file), file
            except Exception:
                try:
                    with open(file, "rb") as f:
                        return pickle.load(f), file
                except Exception:
                    pass
    return None, None


# ============================================================
# LOAD OPTIONAL PRE-TRAINED TEMPERATURE MODEL
# ============================================================

battery_temperature_model, battery_temperature_model_file = load_model(
    [
        "battery_temperature_model.pkl",
        "battery_temperature_model.joblib",
        "temperature_model.pkl",
        "temperature_model.joblib"
    ]
)


# ============================================================
# DATASET / ML MODEL CONFIGURATION
#
# The GitHub project contains:
# - XGBoost regression for estimated_range_km
# - KNN classification for Low/High Range
# - Decision Tree classification for Low/High Range
# - Logistic Regression classification for Low/High Range
#
# The original notebooks use these 7 features.
# ============================================================

ML_FEATURES = [
    "battery_capacity_kwh",
    "state_of_charge",
    "battery_health",
    "energy_consumption_kwh_per_100km",
    "vehicle_age",
    "average_speed",
    "vehicle_weight_kg"
]


@st.cache_resource(show_spinner="Training Smart EV machine-learning models...")
def train_ev_models():
    dataset_candidates = [
        "Electric_vehicle.csv",
        "preprocessed_data.csv"
    ]

    dataset_path = None
    for candidate in dataset_candidates:
        if os.path.exists(candidate):
            dataset_path = candidate
            break

    if dataset_path is None:
        return {
            "ready": False,
            "message": (
                "Dataset not found. Put Electric_vehicle.csv or "
                "preprocessed_data.csv beside app.py."
            )
        }

    df = pd.read_csv(dataset_path)

    required = ML_FEATURES + ["estimated_range_km"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        return {
            "ready": False,
            "message": f"Missing dataset columns: {missing}"
        }

    df = df.dropna(subset=required).copy()

    # The GitHub KNN / Decision Tree notebooks create the binary
    # range class from the median estimated range.
    median_range = float(df["estimated_range_km"].median())

    X = df[ML_FEATURES].astype(float)
    y_range = df["estimated_range_km"].astype(float)
    y_class = (y_range >= median_range).astype(int)

    X_train, X_test, y_train, y_test, yc_train, yc_test = train_test_split(
        X,
        y_range,
        y_class,
        test_size=0.20,
        random_state=42,
        stratify=y_class
    )

    # --------------------------------------------------------
    # XGBOOST REGRESSOR
    # Same main hyperparameters used in the GitHub notebook.
    # --------------------------------------------------------

    xgb_model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    # --------------------------------------------------------
    # STANDARD SCALER FOR KNN + LOGISTIC REGRESSION
    # --------------------------------------------------------

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------------
    # KNN CLASSIFIER
    # --------------------------------------------------------

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, yc_train)
    knn_pred = knn_model.predict(X_test_scaled)

    # --------------------------------------------------------
    # DECISION TREE CLASSIFIER
    # --------------------------------------------------------

    dt_model = DecisionTreeClassifier(
        random_state=42,
        max_depth=8
    )
    dt_model.fit(X_train, yc_train)
    dt_pred = dt_model.predict(X_test)

    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    logistic_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    logistic_model.fit(X_train_scaled, yc_train)
    logistic_pred = logistic_model.predict(X_test_scaled)

    return {
        "ready": True,
        "dataset": dataset_path,
        "rows": len(df),
        "median_range": median_range,
        "xgb": xgb_model,
        "knn": knn_model,
        "decision_tree": dt_model,
        "logistic": logistic_model,
        "scaler": scaler,
        "xgb_accuracy": None,
        "knn_accuracy": accuracy_score(yc_test, knn_pred),
        "decision_tree_accuracy": accuracy_score(yc_test, dt_pred),
        "logistic_accuracy": accuracy_score(yc_test, logistic_pred),
        "xgb_mae": float(np.mean(np.abs(y_test.to_numpy() - xgb_pred)))
    }


ev_ml = train_ev_models()


# ============================================================
# SESSION STATE
# ============================================================

defaults_session = {
    "page": "Home",
    "range": None,
    "consumption": None,
    "battery_capacity": None,
    "soc": None,
    "vehicle_type": None,
    "vehicle_model": None,
    "vehicle_weight": None,
    "vehicle_age": 5.0,
    "estimated_battery_temperature": None,
    "estimated_current_load": None,
    "estimated_power": None,
    "battery_voltage": None,
    "temperature_model_status": None,
    "battery_health": None,
    "battery_degradation": None,
    "battery_health_status": None,
    "battery_model_status": None,
    "range_class": None,
    "range_votes": None,
    "ml_range_prediction": None,
    "traffic_score": None,
    "traffic_level": None,
    "free_flow_speed": None
}

for key, value in defaults_session.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# TOP BAR
# ============================================================

st.markdown(
    """
<div class="topbar">
<div class="logo">
<div class="logo-icon">⚡</div>
<div>
<div class="logo-title">
Smart EV Range and Dynamic Battery Degradation Estimator
</div>
<div class="logo-subtitle">
Intelligent EV Range & Battery Analytics
</div>
</div>
</div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# NAVIGATION
# ============================================================

nav1, nav2, nav3, nav4, nav5, nav6 = st.columns(6)

with nav1:
    if st.button("🏠 Home", key="nav_home", use_container_width=True):
        st.session_state.page = "Home"

with nav2:
    if st.button("⚡ Range", key="nav_range", use_container_width=True):
        st.session_state.page = "Range Estimator"

with nav3:
    if st.button("🔋 Battery", key="nav_battery", use_container_width=True):
        st.session_state.page = "Battery Health"

with nav4:
    if st.button("💰 Cost", key="nav_cost", use_container_width=True):
        st.session_state.page = "EV vs Petrol"

with nav5:
    if st.button("📊 Dashboard", key="nav_dashboard", use_container_width=True):
        st.session_state.page = "Dashboard"

with nav6:
    if st.button("ℹ️ About", key="nav_about", use_container_width=True):
        st.session_state.page = "About"

page = st.session_state.page
st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# VEHICLE DEFAULTS
# ============================================================

def get_vehicle_defaults(vehicle_type):
    if vehicle_type == "Electric Scooter":
        return {
            "weight": 120,
            "battery": 5.0,
            "speed": 40,
            "passengers": 1,
            "max_passengers": 2,
            "base_power": 4.0,
            "voltage": 72,
            "age": 5.0
        }

    if vehicle_type == "Electric Bus":
        return {
            "weight": 12000,
            "battery": 155.0,
            "speed": 45,
            "passengers": 30,
            "max_passengers": 60,
            "base_power": 120.0,
            "voltage": 600,
            "age": 5.0
        }

    return {
        "weight": 1400,
        "battery": 40.0,
        "speed": 45,
        "passengers": 1,
        "max_passengers": 5,
        "base_power": 40.0,
        "voltage": 400,
        "age": 5.0
    }


def dataset_vehicle_name(vehicle_type):
    return {
        "Electric Car": "Car",
        "Electric Scooter": "Scooty",
        "Electric Bus": "Bus"
    }.get(vehicle_type, "Car")


# ============================================================
# AUTOMATIC CURRENT / LOAD ESTIMATION
# ============================================================

def estimate_current_load(
    vehicle_type,
    weight,
    speed,
    soc,
    driving,
    terrain,
    passengers,
    ac,
    ambient_temperature
):
    defaults = get_vehicle_defaults(vehicle_type)
    base_power = defaults["base_power"]
    voltage = defaults["voltage"]

    if vehicle_type == "Electric Scooter":
        weight_factor = max(weight - 100, 0) / 1000
    elif vehicle_type == "Electric Bus":
        weight_factor = max(weight - 10000, 0) / 10000 * 15
    else:
        weight_factor = max(weight - 1000, 0) / 1000 * 4

    speed_factor = max(speed - 30, 0) * 0.35

    driving_factor = {
        "Eco": 0.75,
        "Normal": 1.00,
        "Sport": 1.20,
        "Aggressive": 1.40
    }[driving]

    terrain_factor = {
        "Flat": 1.00,
        "Hilly": 1.20,
        "Mountain": 1.40
    }[terrain]

    if vehicle_type == "Electric Bus":
        passenger_factor = 1 + passengers * 0.008
    else:
        passenger_factor = 1 + passengers * 0.03

    ac_factor = 1.15 if ac else 1.00

    temperature_factor = (
        1 + max(abs(ambient_temperature - 25) - 5, 0) * 0.01
    )

    soc_factor = 1.0
    if soc < 20:
        soc_factor = 1.08
    elif soc > 90:
        soc_factor = 1.03

    estimated_power = base_power + speed_factor + weight_factor
    estimated_power *= driving_factor
    estimated_power *= terrain_factor
    estimated_power *= passenger_factor
    estimated_power *= ac_factor
    estimated_power *= temperature_factor
    estimated_power *= soc_factor

    estimated_current = (estimated_power * 1000) / voltage

    if vehicle_type == "Electric Scooter":
        estimated_current = np.clip(estimated_current, 5, 150)
    elif vehicle_type == "Electric Bus":
        estimated_current = np.clip(estimated_current, 50, 500)
    else:
        estimated_current = np.clip(estimated_current, 20, 300)

    return float(estimated_current), float(estimated_power)


# ============================================================
# TRAFFIC
# ============================================================

def calculate_traffic_level(current_speed, free_flow_speed):
    if free_flow_speed <= 0:
        return 0.0, "Unknown"

    congestion = 1 - (current_speed / free_flow_speed)
    traffic_score = np.clip(congestion * 100, 0, 100)

    if traffic_score < 25:
        traffic_level = "Low"
    elif traffic_score < 50:
        traffic_level = "Medium"
    elif traffic_score < 75:
        traffic_level = "High"
    else:
        traffic_level = "Very High"

    return float(traffic_score), traffic_level


# ============================================================
# CONSUMPTION — USED AS AN INPUT TO THE XGBOOST MODEL
# ============================================================

def calculate_consumption(
    vehicle_type,
    weight,
    speed,
    temperature,
    traffic,
    terrain,
    ac,
    passengers,
    driving
):
    if vehicle_type == "Electric Scooter":
        base = 5.5
        weight_factor = max(weight - 100, 0) * 0.008
        speed_factor = max(speed - 30, 0) * 0.055
        passenger_factor = passengers * 0.25

    elif vehicle_type == "Electric Bus":
        base = 100
        weight_factor = max(weight - 10000, 0) * 0.005
        speed_factor = max(speed - 35, 0) * 0.25
        passenger_factor = passengers * 0.35

    else:
        base = 15.5
        weight_factor = max(weight - 1000, 0) * 0.002
        speed_factor = max(speed - 40, 0) * 0.045
        passenger_factor = passengers * 0.45

    temperature_factor = abs(temperature - 25) * 0.08
    traffic_factor = traffic * 0.025

    terrain_factor = {
        "Flat": 0,
        "Hilly": 3.5,
        "Mountain": 5.5
    }[terrain]

    ac_factor = 1.8 if ac else 0

    driving_factor = {
        "Eco": -2,
        "Normal": 0,
        "Sport": 3.5,
        "Aggressive": 6
    }[driving]

    consumption = (
        base + weight_factor + speed_factor +
        temperature_factor + traffic_factor +
        terrain_factor + ac_factor +
        passenger_factor + driving_factor
    )

    if vehicle_type == "Electric Scooter":
        return max(consumption, 3)
    if vehicle_type == "Electric Bus":
        return max(consumption, 80)
    return max(consumption, 8)


# ============================================================
# ORIGINAL FORMULA RANGE — FALLBACK ONLY
# ============================================================

def calculate_range(capacity, soc, consumption):
    energy = capacity * soc / 100
    return (energy / consumption) * 100


# ============================================================
# ML RANGE PREDICTION
# ============================================================

def predict_range_with_ml(
    battery_capacity,
    soc,
    battery_health,
    consumption,
    vehicle_age,
    speed,
    weight
):
    if not ev_ml.get("ready", False):
        return None, None, None

    sample = pd.DataFrame([{
        "battery_capacity_kwh": battery_capacity,
        "state_of_charge": soc,
        "battery_health": battery_health,
        "energy_consumption_kwh_per_100km": consumption,
        "vehicle_age": vehicle_age,
        "average_speed": speed,
        "vehicle_weight_kg": weight
    }])[ML_FEATURES]

    # XGBoost gives the numerical range prediction.
    xgb_prediction = float(ev_ml["xgb"].predict(sample)[0])
    xgb_prediction = max(xgb_prediction, 1.0)

    # KNN + Decision Tree + Logistic Regression classify
    # the predicted operating point as Low or High range.
    scaled_sample = ev_ml["scaler"].transform(sample)

    knn_class = int(ev_ml["knn"].predict(scaled_sample)[0])
    dt_class = int(ev_ml["decision_tree"].predict(sample)[0])
    logistic_class = int(ev_ml["logistic"].predict(scaled_sample)[0])

    votes = [knn_class, dt_class, logistic_class]
    high_votes = sum(votes)
    range_class = "High Range" if high_votes >= 2 else "Low Range"

    return xgb_prediction, range_class, {
        "KNN": "High Range" if knn_class else "Low Range",
        "Decision Tree": "High Range" if dt_class else "Low Range",
        "Logistic Regression": "High Range" if logistic_class else "Low Range",
        "XGBoost": f"{xgb_prediction:.1f} km"
    }


# ============================================================
# BATTERY HEALTH
# ============================================================

def calculate_battery_health(initial, cycles, temperature, charging):
    cycle_loss = cycles * 0.005

    temp_loss = (
        max(abs(temperature - 25) - 5, 0) * 0.12
    )

    charging_loss = {
        "Slow AC": 0,
        "Fast AC": 0.7,
        "DC Fast": 1.5,
        "Not Charging": 0,
        "AC Charging": 0,
        "DC Fast Charging": 1.5,
        "Fully Charged": 0,
        "Charging Fault": 1.0
    }.get(charging, 0)

    return max(
        min(initial - cycle_loss - temp_loss - charging_loss, 100),
        50
    )


# ============================================================
# BATTERY TEMPERATURE
# ============================================================

def estimate_battery_temperature(
    vehicle_type,
    ambient_temperature,
    speed,
    current_load,
    soc,
    charging,
    driving_style
):
    if battery_temperature_model is not None:
        charging_encoded = {
            "Not Charging": 0,
            "Slow AC": 1,
            "Fast AC": 2,
            "DC Fast": 3,
            "AC Charging": 1,
            "DC Fast Charging": 3,
            "Fully Charged": 0,
            "Charging Fault": 3
        }.get(charging, 0)

        driving_encoded = {
            "Eco": 0,
            "Normal": 1,
            "Sport": 2,
            "Aggressive": 3
        }.get(driving_style, 1)

        vehicle_encoded = {
            "Electric Scooter": 0,
            "Electric Car": 1,
            "Electric Bus": 2
        }.get(vehicle_type, 1)

        # Try DataFrame-style models first.
        features = pd.DataFrame([{
            "vehicle_type": vehicle_type,
            "outside_temperature": ambient_temperature,
            "ambient_temperature": ambient_temperature,
            "average_speed": speed,
            "speed": speed,
            "battery_current": current_load,
            "current_load": current_load,
            "state_of_charge": soc,
            "soc": soc,
            "charging_status": charging,
            "charging": charging,
            "driving_style": driving_style
        }])

        try:
            prediction = battery_temperature_model.predict(features)
            value = float(np.asarray(prediction).ravel()[0])
            return value, "ML model"
        except Exception:
            pass

        # Try the numeric version.
        x = np.array([[
            vehicle_encoded,
            ambient_temperature,
            speed,
            current_load,
            soc,
            charging_encoded,
            driving_encoded
        ]])

        try:
            prediction = battery_temperature_model.predict(x)
            value = float(np.asarray(prediction).ravel()[0])
            return value, "ML model"
        except Exception:
            pass

    # Smart fallback.
    vehicle_heat = {
        "Electric Scooter": 0.5,
        "Electric Car": 2.0,
        "Electric Bus": 4.0
    }[vehicle_type]

    style_heat = {
        "Eco": -0.8,
        "Normal": 0.0,
        "Sport": 1.5,
        "Aggressive": 2.8
    }[driving_style]

    charging_heat = {
        "Not Charging": 0.0,
        "Slow AC": 1.0,
        "Fast AC": 2.5,
        "DC Fast": 4.0,
        "AC Charging": 1.0,
        "DC Fast Charging": 4.0,
        "Fully Charged": 0.0,
        "Charging Fault": 2.0
    }.get(charging, 0)

    ambient_heat = max(ambient_temperature - 25, 0) * 0.28
    speed_heat = max(speed - 30, 0) * 0.045
    load_heat = max(current_load, 0) * 0.035
    soc_heat = max(soc - 80, 0) * 0.025

    estimated = (
        ambient_temperature + 2.5 + vehicle_heat +
        ambient_heat + speed_heat + load_heat +
        soc_heat + style_heat + charging_heat
    )

    estimated = float(np.clip(estimated, 15, 65))
    return estimated, "Smart fallback estimator"


def temperature_status(temperature):
    if temperature < 30:
        return "Cool"
    if temperature < 40:
        return "Normal"
    if temperature < 50:
        return "Warm"
    return "High"


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.markdown(
        """
<div class="hero">
<div class="hero-title">Drive Smarter with AI ⚡</div>
<div class="hero-subtitle">
Smart EV Intelligence helps you understand your electric
vehicle's real-world range, battery health, running cost
and long-term performance.
</div>
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Explore Smart EV Features</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
<div class="feature-card">
<div class="feature-icon">⚡</div>
<div class="feature-title">Smart Range Estimator</div>
<div class="feature-text">
Estimate practical driving range using battery, speed,
traffic, terrain, temperature and driving style.
</div>
</div>
""",
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
<div class="feature-card">
<div class="feature-icon">🔋</div>
<div class="feature-title">Dynamic Battery Health</div>
<div class="feature-text">
Analyse battery degradation using charging cycles,
estimated temperature and charging behaviour.
</div>
</div>
""",
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
<div class="feature-card">
<div class="feature-icon">💰</div>
<div class="feature-title">EV vs Petrol</div>
<div class="feature-text">
Compare daily, monthly, yearly and five-year
running costs.
</div>
</div>
""",
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">Supported Vehicles</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    home_items = [
        ("🚗", "Cars", "Electric Cars"),
        ("🛵", "Scooters", "Electric Scooters"),
        ("🚌", "Buses", "Electric Buses")
    ]

    for column, item in zip([c1, c2, c3], home_items):
        with column:
            st.markdown(
                f"""
<div class="metric-card">
<div class="metric-title">{item[0]} Vehicle</div>
<div class="metric-value">{item[1]}</div>
<div class="metric-small">{item[2]}</div>
</div>
""",
                unsafe_allow_html=True
            )

    # ML status is kept unobtrusive.
    if ev_ml.get("ready", False):
        st.markdown(
            """
<div class="info-box">
🤖 <b>ML Engine Ready:</b>
XGBoost + KNN + Decision Tree + Logistic Regression
are available for EV range analysis.
</div>
""",
            unsafe_allow_html=True
        )
    else:
        st.warning(ev_ml.get("message", "ML engine is unavailable."))


# ============================================================
# RANGE ESTIMATOR
# ============================================================

elif page == "Range Estimator":

    st.markdown(
        '<div class="hero-title">⚡ Smart Range Estimator</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter normal driving conditions. Electrical current "
        "and battery temperature are estimated automatically. "
        "The GitHub-trained range models are then used for prediction."
    )

    vehicle_type = st.selectbox(
        "🚗 Vehicle Type",
        ["Electric Car", "Electric Scooter", "Electric Bus"],
        key="range_vehicle_type"
    )

    defaults = get_vehicle_defaults(vehicle_type)

    st.markdown(
        f"""
<div class="info-box">
🤖 <b>Machine Learning Prediction</b>
<br><br>
Current vehicle: <b>{vehicle_type}</b>
<br>
Range model: <b>XGBoost Regression</b>
<br>
Range classification: <b>KNN + Decision Tree + Logistic Regression</b>
</div>
""",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        vehicle_model = st.text_input(
            "Vehicle Model",
            "My EV",
            key="range_vehicle_model"
        )

        battery_capacity = st.number_input(
            "Battery Capacity (kWh)",
            1.0,
            500.0,
            float(defaults["battery"]),
            0.5,
            key="range_battery_capacity"
        )

        soc = st.slider(
            "State of Charge (%)",
            0,
            100,
            80,
            key="range_soc"
        )

        weight = st.number_input(
            "Vehicle Weight (kg)",
            50,
            50000,
            int(defaults["weight"])
        )

        vehicle_age = st.number_input(
            "Vehicle Age (years)",
            0.0,
            30.0,
            float(defaults["age"]),
            0.1,
            key="range_vehicle_age"
        )

    with col2:
        speed = st.number_input(
            "Average Speed (km/h)",
            1.0,
            180.0,
            float(defaults["speed"]),
            key="range_speed"
        )

        ambient_temperature = st.number_input(
            "Ambient Temperature (°C)",
            -10.0,
            60.0,
            32.0,
            0.5,
            key="range_ambient_temperature"
        )

        free_flow_speed = st.number_input(
            "Normal Road Speed (km/h)",
            10.0,
            180.0,
            float(defaults["speed"]),
            1.0,
            key="range_free_flow_speed"
        )

        traffic_score, traffic_level = calculate_traffic_level(
            speed, free_flow_speed
        )

        driving = st.selectbox(
            "Driving Style",
            ["Eco", "Normal", "Sport", "Aggressive"],
            key="range_driving"
        )

        st.markdown(
            f"""
<div class="info-box">
🚦 <b>Automatic Traffic Detection</b><br>
Current Speed: <b>{speed:.1f} km/h</b><br>
Normal Road Speed: <b>{free_flow_speed:.1f} km/h</b><br>
Traffic Score: <b>{traffic_score:.1f}%</b><br>
Traffic Level: <b>{traffic_level}</b>
</div>
""",
            unsafe_allow_html=True
        )

        traffic = traffic_score

    with col3:
        terrain = st.selectbox(
            "Terrain",
            ["Flat", "Hilly", "Mountain"],
            key="range_terrain"
        )

        passengers = st.slider(
            "Passenger Load",
            0,
            int(defaults["max_passengers"]),
            min(
                int(defaults["passengers"]),
                int(defaults["max_passengers"])
            ),
            key="range_passengers"
        )

        ac = st.checkbox(
            "AC / Climate Control",
            key="range_ac"
        )

        charging_status = st.selectbox(
            "Charging Status",
            [
                "Not Charging",
                "AC Charging",
                "DC Fast Charging",
                "Fully Charged",
                "Charging Fault"
            ],
            key="range_charging"
        )

    if st.button(
        "🚀 Estimate My Range",
        key="estimate_range",
        use_container_width=True
    ):

        estimated_current, estimated_power = estimate_current_load(
            vehicle_type,
            weight,
            speed,
            soc,
            driving,
            terrain,
            passengers,
            ac,
            ambient_temperature
        )

        estimated_temperature, temperature_model_status = (
            estimate_battery_temperature(
                vehicle_type,
                ambient_temperature,
                speed,
                estimated_current,
                soc,
                charging_status,
                driving
            )
        )

        # Initial health is estimated from operating data and is used
        # as an input to the GitHub range models.
        initial_health_for_range = calculate_battery_health(
            100,
            500,
            estimated_temperature,
            charging_status
        )

        consumption = calculate_consumption(
            vehicle_type,
            weight,
            speed,
            estimated_temperature,
            traffic,
            terrain,
            ac,
            passengers,
            driving
        )

        # --------------------------------------------------------
        # USE ALL FOUR REQUIRED ALGORITHMS
        # --------------------------------------------------------

        ml_range, range_class, range_votes = predict_range_with_ml(
            battery_capacity=battery_capacity,
            soc=soc,
            battery_health=initial_health_for_range,
            consumption=consumption,
            vehicle_age=vehicle_age,
            speed=speed,
            weight=weight
        )

        # Fallback if the dataset/model is unavailable.
        if ml_range is None:
            estimated_range = calculate_range(
                battery_capacity,
                soc,
                consumption
            )
            range_class = (
                "High Range"
                if estimated_range >= 100
                else "Low Range"
            )
            range_votes = {
                "XGBoost": "Unavailable",
                "KNN": "Unavailable",
                "Decision Tree": "Unavailable",
                "Logistic Regression": "Unavailable"
            }
            ml_source = "Formula fallback"
        else:
            estimated_range = ml_range
            ml_source = "XGBoost + classification ensemble"

        st.session_state.range = estimated_range
        st.session_state.consumption = consumption
        st.session_state.battery_capacity = battery_capacity
        st.session_state.soc = soc
        st.session_state.vehicle_type = vehicle_type
        st.session_state.vehicle_model = vehicle_model
        st.session_state.vehicle_weight = weight
        st.session_state.vehicle_age = vehicle_age
        st.session_state.estimated_battery_temperature = estimated_temperature
        st.session_state.estimated_current_load = estimated_current
        st.session_state.estimated_power = estimated_power
        st.session_state.battery_voltage = defaults["voltage"]
        st.session_state.temperature_model_status = temperature_model_status
        st.session_state.range_class = range_class
        st.session_state.range_votes = range_votes
        st.session_state.ml_range_prediction = estimated_range
        st.session_state.traffic_score = traffic_score
        st.session_state.traffic_level = traffic_level
        st.session_state.free_flow_speed = free_flow_speed

        st.markdown(
            '<div class="section-title">Your EV Results</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        metrics = [
            ("Estimated Range", f"{estimated_range:.1f}", "kilometres"),
            ("Consumption", f"{consumption:.1f}", "kWh / 100 km"),
            ("Battery SOC", f"{soc}%", "Current charge"),
            ("🚦 Traffic", traffic_level, f"{traffic_score:.1f}% congestion"),
            ("⚡ Est. Load", f"{estimated_current:.1f} A", "Automatically estimated")
        ]

        for column, metric in zip([c1, c2, c3, c4, c5], metrics):
            with column:
                st.markdown(
                    f"""
<div class="metric-card">
<div class="metric-title">{metric[0]}</div>
<div class="metric-value">{metric[1]}</div>
<div class="metric-small">{metric[2]}</div>
</div>
""",
                    unsafe_allow_html=True
                )

        st.markdown(
            f"""
<div class="info-box">
🤖 <b>ML Prediction Source:</b> {ml_source}<br>
📈 <b>XGBoost Range:</b> {estimated_range:.1f} km<br>
🏁 <b>Final Range Class:</b> {range_class}
</div>
""",
            unsafe_allow_html=True
        )

        if range_votes:
            st.dataframe(
                pd.DataFrame(
                    list(range_votes.items()),
                    columns=["Algorithm", "Prediction"]
                ),
                use_container_width=True,
                hide_index=True
            )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=estimated_range,
                title={"text": "Estimated Driving Range (km)"},
                gauge={
                    "axis": {
                        "range": [
                            0,
                            max(300, estimated_range * 1.2)
                        ]
                    },
                    "bar": {"color": "#00e676"}
                }
            )
        )

        fig.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"}
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="section-title">⚡ Automatic Electrical Load Analysis</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
<div class="info-box">
<b>Vehicle:</b> {vehicle_type}<br>
<b>Estimated Electrical Power:</b> {estimated_power:.1f} kW<br>
<b>Estimated Battery Current:</b> {estimated_current:.1f} A<br>
<b>Battery Voltage:</b> {defaults["voltage"]} V
</div>
""",
            unsafe_allow_html=True
        )

        temp_status = temperature_status(estimated_temperature)

        st.markdown(
            '<div class="section-title">🌡️ Smart Battery Temperature</div>',
            unsafe_allow_html=True
        )

        if estimated_temperature < 40:
            st.markdown(
                f"""
<div class="result-good">
🟢 <b>Normal Battery Temperature</b><br>
Estimated battery temperature: <b>{estimated_temperature:.1f}°C</b><br>
Status: <b>{temp_status}</b>
</div>
""",
                unsafe_allow_html=True
            )
        elif estimated_temperature < 50:
            st.markdown(
                f"""
<div class="result-warning">
🟡 <b>Elevated Battery Temperature</b><br>
Estimated battery temperature: <b>{estimated_temperature:.1f}°C</b><br>
Consider reducing aggressive driving or high electrical load.
</div>
""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
<div class="result-danger">
🔴 <b>High Battery Temperature</b><br>
Estimated battery temperature: <b>{estimated_temperature:.1f}°C</b><br>
High thermal load may increase battery degradation.
</div>
""",
                unsafe_allow_html=True
            )

        st.markdown(
            '<div class="section-title">📊 Range Impact Analysis</div>',
            unsafe_allow_html=True
        )

        impact = pd.DataFrame({
            "Factor": [
                "Speed", "Traffic", "Temperature",
                "Terrain", "AC", "Driving Style"
            ],
            "Impact": [
                max(speed - 40, 0) * 0.5,
                traffic * 0.3,
                abs(estimated_temperature - 25) * 0.4,
                {"Flat": 0, "Hilly": 20, "Mountain": 35}[terrain],
                15 if ac else 0,
                {"Eco": 0, "Normal": 8, "Sport": 20, "Aggressive": 30}[driving]
            ]
        })

        fig = px.bar(
            impact,
            x="Factor",
            y="Impact",
            title="Factors Affecting Range"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )

        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# BATTERY HEALTH
# ============================================================

elif page == "Battery Health":

    st.markdown(
        '<div class="hero-title">🔋 Dynamic Battery Health</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Battery temperature and electrical load are estimated "
        "automatically from vehicle operating conditions."
    )

    st.markdown(
        """
<div class="info-box">
<h2>🤖 Smart Battery Analysis</h2>
<br>
The system automatically estimates:<br>
⚡ Electrical Current / Load<br>
🌡️ Battery Temperature<br>
🔋 Battery Health<br>
📉 Battery Degradation
</div>
""",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        vehicle_type = st.selectbox(
            "Vehicle Type",
            ["Electric Car", "Electric Scooter", "Electric Bus"],
            key="battery_vehicle_type"
        )

        defaults = get_vehicle_defaults(vehicle_type)

        initial_health = st.slider(
            "Initial Battery Health (%)",
            70,
            100,
            100,
            key="battery_initial_health"
        )

        cycles = st.number_input(
            "Battery Cycles",
            0,
            5000,
            500,
            key="battery_cycles"
        )

        ambient_temperature = st.number_input(
            "🌤️ Ambient Temperature (°C)",
            -10.0,
            60.0,
            32.0,
            0.5,
            key="ambient_temperature"
        )

        speed = st.number_input(
            "Average Speed (km/h)",
            0.0,
            180.0,
            float(defaults["speed"]),
            1.0,
            key="battery_speed"
        )

        soc_temperature = st.slider(
            "🔋 State of Charge (%)",
            0,
            100,
            65,
            key="battery_soc"
        )

        driving_temperature = st.selectbox(
            "🏎️ Driving Style",
            ["Eco", "Normal", "Sport", "Aggressive"],
            key="battery_driving_style"
        )

        charging_status = st.selectbox(
            "🔌 Charging Status",
            ["Not Charging", "Slow AC", "Fast AC", "DC Fast"],
            key="battery_charging_status"
        )

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
<div class="card">
<h3>🤖 Automatic Estimation</h3>
<p>
The system estimates electrical load automatically
instead of asking you to enter battery current.
</p>
<br>
<p>🚗 Vehicle Type</p>
<p>⚖️ Vehicle Weight</p>
<p>🏎️ Vehicle Speed</p>
<p>🔋 State of Charge</p>
<p>🛣️ Terrain</p>
<p>👥 Passenger Load</p>
<p>❄️ Climate Control</p>
<p>🌤️ Ambient Temperature</p>
<p>🏁 Driving Style</p>
<br>
<h3>🌡️ Battery Temperature</h3>
<p>
The estimated electrical current is used with the
other operating conditions to estimate battery temperature.
</p>
</div>
""",
            unsafe_allow_html=True
        )

    if st.button(
        "🌡️ Estimate Temperature & Analyze Battery",
        key="battery_button",
        use_container_width=True
    ):

        estimated_current, estimated_power = estimate_current_load(
            vehicle_type,
            defaults["weight"],
            speed,
            soc_temperature,
            driving_temperature,
            "Flat",
            defaults["passengers"],
            False,
            ambient_temperature
        )

        estimated_temperature, model_status = (
            estimate_battery_temperature(
                vehicle_type,
                ambient_temperature,
                speed,
                estimated_current,
                soc_temperature,
                charging_status,
                driving_temperature
            )
        )

        health = calculate_battery_health(
            initial_health,
            cycles,
            estimated_temperature,
            charging_status
        )

        degradation = 100 - health
        temp_status = temperature_status(estimated_temperature)

        if health >= 90:
            health_status = "Excellent"
        elif health >= 80:
            health_status = "Healthy"
        elif health >= 70:
            health_status = "Moderate"
        else:
            health_status = "Needs Attention"

        st.session_state.estimated_battery_temperature = estimated_temperature
        st.session_state.estimated_current_load = estimated_current
        st.session_state.estimated_power = estimated_power
        st.session_state.battery_voltage = defaults["voltage"]
        st.session_state.temperature_model_status = model_status
        st.session_state.battery_health = health
        st.session_state.battery_degradation = degradation
        st.session_state.battery_health_status = health_status
        st.session_state.battery_model_status = "Calculated"

        st.markdown(
            '<div class="section-title">Smart Battery Results</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)

        battery_metrics = [
            ("🌡️ Battery Temperature", f"{estimated_temperature:.1f}°C", temp_status),
            ("⚡ Estimated Current", f"{estimated_current:.1f} A", "Automatic estimate"),
            ("🔋 Battery Health", f"{health:.1f}%", health_status),
            ("📉 Degradation", f"{degradation:.1f}%", "Estimated loss")
        ]

        for column, metric in zip([c1, c2, c3, c4], battery_metrics):
            with column:
                st.markdown(
                    f"""
<div class="metric-card">
<div class="metric-title">{metric[0]}</div>
<div class="metric-value">{metric[1]}</div>
<div class="metric-small">{metric[2]}</div>
</div>
""",
                    unsafe_allow_html=True
                )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=estimated_temperature,
                title={"text": "Estimated Battery Temperature (°C)"},
                gauge={
                    "axis": {"range": [15, 65]},
                    "bar": {"color": "#00e676"}
                }
            )
        )

        fig.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"}
        )

        st.plotly_chart(fig, use_container_width=True)

        temperature_data = pd.DataFrame({
            "Parameter": [
                "Vehicle Type",
                "Ambient Temperature",
                "Average Speed",
                "Estimated Electrical Load",
                "Estimated Electrical Power",
                "Battery Voltage",
                "State of Charge",
                "Driving Style",
                "Charging Status",
                "Estimated Battery Temperature",
                "Battery Health",
                "Battery Degradation"
            ],
            "Value": [
                vehicle_type,
                f"{ambient_temperature:.1f} °C",
                f"{speed:.1f} km/h",
                f"{estimated_current:.1f} A",
                f"{estimated_power:.1f} kW",
                f"{defaults['voltage']} V",
                f"{soc_temperature}%",
                driving_temperature,
                charging_status,
                f"{estimated_temperature:.1f} °C",
                f"{health:.1f}%",
                f"{degradation:.1f}%"
            ]
        })

        st.markdown(
            '<div class="section-title">Battery Analysis Inputs</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            temperature_data,
            use_container_width=True,
            hide_index=True
        )

        if health >= 90:
            st.markdown(
                f"""
<div class="result-good">
🟢 <b>Excellent Battery Health</b><br>
Estimated battery health: <b>{health:.1f}%</b><br>
Battery degradation: <b>{degradation:.1f}%</b>
</div>
""",
                unsafe_allow_html=True
            )
        elif health >= 70:
            st.markdown(
                f"""
<div class="result-warning">
🟡 <b>Battery Health Requires Monitoring</b><br>
Estimated battery health: <b>{health:.1f}%</b><br>
Battery degradation: <b>{degradation:.1f}%</b>
</div>
""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
<div class="result-danger">
🔴 <b>Battery Health Needs Attention</b><br>
Estimated battery health: <b>{health:.1f}%</b><br>
Battery degradation: <b>{degradation:.1f}%</b>
</div>
""",
                unsafe_allow_html=True
            )


# ============================================================
# EV VS PETROL
# ============================================================

elif page == "EV vs Petrol":

    st.markdown(
        '<div class="hero-title">💰 EV vs Petrol Calculator</div>',
        unsafe_allow_html=True
    )

    st.write("Compare the running cost of an EV with a petrol vehicle.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="card"><h2>⚡ Electric Vehicle</h2></div>',
            unsafe_allow_html=True
        )

        electricity_rate = st.number_input(
            "Electricity Rate (₹/kWh)",
            1.0, 50.0, 8.0,
            key="electricity_rate"
        )

        default_consumption = (
            st.session_state.consumption
            if st.session_state.consumption is not None
            else 15.0
        )

        ev_consumption = st.number_input(
            "EV Consumption (kWh/100 km)",
            1.0, 150.0,
            float(default_consumption),
            key="ev_consumption"
        )

    with col2:
        st.markdown(
            '<div class="card"><h2>⛽ Petrol Vehicle</h2></div>',
            unsafe_allow_html=True
        )

        petrol_price = st.number_input(
            "Petrol Price (₹/L)",
            50.0, 500.0, 105.0,
            key="petrol_price"
        )

        petrol_mileage = st.number_input(
            "Petrol Mileage (km/L)",
            1.0, 60.0, 18.0,
            key="petrol_mileage"
        )

    distance = st.number_input(
        "Daily Distance (km)",
        1.0, 1000.0, 40.0,
        key="daily_distance"
    )

    if st.button(
        "💰 Calculate Savings",
        key="cost_button",
        use_container_width=True
    ):

        ev_cost_km = (ev_consumption / 100) * electricity_rate
        petrol_cost_km = petrol_price / petrol_mileage

        ev_daily = ev_cost_km * distance
        petrol_daily = petrol_cost_km * distance

        daily_savings = petrol_daily - ev_daily
        monthly_savings = daily_savings * 30
        yearly_savings = daily_savings * 365
        five_year_savings = yearly_savings * 5

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("EV Cost / Day", f"₹{ev_daily:.2f}")

        with c2:
            st.metric("Petrol Cost / Day", f"₹{petrol_daily:.2f}")

        with c3:
            st.metric("Daily Savings", f"₹{daily_savings:.2f}")

        with c4:
            st.metric("Yearly Savings", f"₹{yearly_savings:,.0f}")

        savings = pd.DataFrame({
            "Period": ["Daily", "Monthly", "Yearly", "5 Years"],
            "Savings": [
                daily_savings,
                monthly_savings,
                yearly_savings,
                five_year_savings
            ]
        })

        fig = px.bar(
            savings,
            x="Period",
            y="Savings",
            title="EV Savings Compared with Petrol"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )

        st.plotly_chart(fig, use_container_width=True)

        if daily_savings > 0:
            st.markdown(
                f"""
<div class="result-good">
⚡ <b>EV Cost Advantage</b><br><br>
Your EV could save approximately
<b>₹{yearly_savings:,.0f}</b>
per year under these assumptions.
</div>
""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
<div class="result-warning">
The current assumptions do not show an EV running-cost advantage.
</div>
""",
                unsafe_allow_html=True
            )


# ============================================================
# DASHBOARD
# ============================================================

elif page == "Dashboard":

    st.markdown(
        '<div class="hero-title">📊 Smart EV Intelligence Dashboard</div>',
        unsafe_allow_html=True
    )

    st.write(
        "A complete overview of your latest EV range, "
        "battery, temperature and electrical-load analysis."
    )

    has_range_data = st.session_state.get("range") is not None

    if not has_range_data:
        st.markdown(
            """
<div class="info-box">
⚡ <b>No EV analysis available yet.</b><br><br>
Go to <b>⚡ Range</b> and click <b>Estimate My Range</b>.
<br><br>
The dashboard will automatically display the ML range prediction,
classification results, energy consumption, battery SOC,
battery temperature and electrical load.
</div>
""",
            unsafe_allow_html=True
        )
    else:

        estimated_range = float(st.session_state.range)
        consumption = float(st.session_state.consumption)
        capacity = float(st.session_state.battery_capacity)
        soc = float(st.session_state.soc)

        vehicle_type = st.session_state.get("vehicle_type") or "Electric Car"
        vehicle_model = st.session_state.get("vehicle_model") or "My EV"
        vehicle_weight = st.session_state.get("vehicle_weight")
        vehicle_age = st.session_state.get("vehicle_age", 5.0)

        estimated_temperature = st.session_state.get(
            "estimated_battery_temperature"
        )
        estimated_current = st.session_state.get(
            "estimated_current_load"
        )
        estimated_power = st.session_state.get("estimated_power")
        battery_voltage = st.session_state.get("battery_voltage")
        temperature_model_status = st.session_state.get(
            "temperature_model_status"
        )

        battery_health = st.session_state.get("battery_health")
        battery_degradation = st.session_state.get("battery_degradation")
        battery_health_status = st.session_state.get(
            "battery_health_status"
        )

        range_class = st.session_state.get("range_class") or "Not available"
        range_votes = st.session_state.get("range_votes") or {}

        remaining_energy = capacity * soc / 100
        used_energy = capacity - remaining_energy

        if estimated_range >= 250:
            range_status = "Excellent"
            range_class_css = "result-good"
            range_icon = "🟢"
        elif estimated_range >= 150:
            range_status = "Good"
            range_class_css = "result-good"
            range_icon = "🟢"
        elif estimated_range >= 80:
            range_status = "Moderate"
            range_class_css = "result-warning"
            range_icon = "🟡"
        else:
            range_status = "Low"
            range_class_css = "result-danger"
            range_icon = "🔴"

        temp_status = (
            temperature_status(estimated_temperature)
            if estimated_temperature is not None
            else "Not Available"
        )

        st.markdown(
            f"""
<div class="card">
<h2>{vehicle_model}</h2>
<p>
<b>Vehicle Type:</b> {vehicle_type}
&nbsp;&nbsp; | &nbsp;&nbsp;
<b>Battery Capacity:</b> {capacity:.1f} kWh
&nbsp;&nbsp; | &nbsp;&nbsp;
<b>Vehicle Age:</b> {vehicle_age:.1f} years
</p>
</div>
""",
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">⚡ Latest EV Analysis</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)

        dashboard_metrics = [
            ("⚡ Estimated Range", f"{estimated_range:.1f} km", range_class),
            ("🔋 Battery SOC", f"{soc:.0f}%", "Current charge"),
            ("🔋 Remaining Energy", f"{remaining_energy:.1f} kWh", "Available battery energy"),
            ("⚡ Consumption", f"{consumption:.1f}", "kWh / 100 km")
        ]

        for column, metric in zip([c1, c2, c3, c4], dashboard_metrics):
            with column:
                st.markdown(
                    f"""
<div class="metric-card">
<div class="metric-title">{metric[0]}</div>
<div class="metric-value">{metric[1]}</div>
<div class="metric-small">{metric[2]}</div>
</div>
""",
                    unsafe_allow_html=True
                )

        st.markdown(
            '<div class="section-title">🤖 Machine Learning Results</div>',
            unsafe_allow_html=True
        )

        ml_data = pd.DataFrame({
            "Algorithm": [
                "XGBoost",
                "KNN",
                "Decision Tree",
                "Logistic Regression"
            ],
            "Prediction": [
                range_votes.get("XGBoost", f"{estimated_range:.1f} km"),
                range_votes.get("KNN", "N/A"),
                range_votes.get("Decision Tree", "N/A"),
                range_votes.get("Logistic Regression", "N/A")
            ]
        })

        st.dataframe(
            ml_data,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            f"""
<div class="info-box">
🏁 <b>Final Range Classification:</b> {range_class}<br>
📈 <b>Numerical Range Model:</b> XGBoost Regressor<br>
🧠 <b>Classification Models:</b> KNN + Decision Tree + Logistic Regression
</div>
""",
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">🔋 Electrical & Thermal Intelligence</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)

        current_text = (
            f"{estimated_current:.1f} A"
            if estimated_current is not None
            else "N/A"
        )

        power_text = (
            f"{estimated_power:.1f} kW"
            if estimated_power is not None
            else "N/A"
        )

        temperature_text = (
            f"{estimated_temperature:.1f}°C"
            if estimated_temperature is not None
            else "N/A"
        )

        electrical_metrics = [
            ("⚡ Estimated Current", current_text, "Automatic load estimation"),
            ("🔌 Estimated Power", power_text, "Battery electrical power"),
            ("🌡️ Battery Temperature", temperature_text, temp_status),
            (
                "🔋 Battery Health",
                f"{battery_health:.1f}%" if battery_health is not None else "Run Battery",
                battery_health_status if battery_health_status else "Battery analysis"
            )
        ]

        for column, metric in zip([c1, c2, c3, c4], electrical_metrics):
            with column:
                st.markdown(
                    f"""
<div class="metric-card">
<div class="metric-title">{metric[0]}</div>
<div class="metric-value">{metric[1]}</div>
<div class="metric-small">{metric[2]}</div>
</div>
""",
                    unsafe_allow_html=True
                )

        st.markdown(
            '<div class="section-title">🚗 Range Performance</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
<div class="{range_class_css}">
{range_icon} <b>{range_status} Range Performance</b><br>
Estimated driving range: <b>{estimated_range:.1f} km</b><br>
Energy consumption: <b>{consumption:.1f} kWh/100 km</b>
</div>
""",
            unsafe_allow_html=True
        )

        left, right = st.columns(2)

        with left:
            st.markdown(
                '<div class="section-title">🔋 Battery Charge Level</div>',
                unsafe_allow_html=True
            )

            fig_soc = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=soc,
                    title={"text": "Current Battery SOC (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#00e676"},
                        "steps": [
                            {"range": [0, 20], "color": "#4a1515"},
                            {"range": [20, 50], "color": "#4a4015"},
                            {"range": [50, 100], "color": "#153d2c"}
                        ]
                    }
                )
            )

            fig_soc.update_layout(
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"}
            )

            st.plotly_chart(fig_soc, use_container_width=True)

        with right:
            st.markdown(
                '<div class="section-title">🌡️ Battery Thermal Status</div>',
                unsafe_allow_html=True
            )

            if estimated_temperature is not None:
                fig_temp = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=estimated_temperature,
                        title={"text": "Battery Temperature (°C)"},
                        gauge={
                            "axis": {"range": [15, 65]},
                            "bar": {"color": "#00e676"},
                            "steps": [
                                {"range": [15, 30], "color": "#153b4a"},
                                {"range": [30, 40], "color": "#153d2c"},
                                {"range": [40, 50], "color": "#4a4015"},
                                {"range": [50, 65], "color": "#4a1515"}
                            ]
                        }
                    )
                )

                fig_temp.update_layout(
                    height=350,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "white"}
                )

                st.plotly_chart(fig_temp, use_container_width=True)
            else:
                st.info("Battery temperature is not available yet.")

        st.markdown(
            '<div class="section-title">⚡ Battery Energy Overview</div>',
            unsafe_allow_html=True
        )

        energy_data = pd.DataFrame({
            "Energy Type": ["Remaining Energy", "Used Energy"],
            "Energy (kWh)": [remaining_energy, max(used_energy, 0)]
        })

        fig_energy = px.bar(
            energy_data,
            x="Energy Type",
            y="Energy (kWh)",
            text="Energy (kWh)",
            title="Battery Energy Distribution"
        )

        fig_energy.update_traces(
            texttemplate="%{text:.1f} kWh",
            textposition="outside"
        )

        fig_energy.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )

        st.plotly_chart(fig_energy, use_container_width=True)

        st.markdown(
            '<div class="section-title">📈 EV Performance Analysis</div>',
            unsafe_allow_html=True
        )

        performance_data = pd.DataFrame({
            "Metric": ["Range", "Consumption"],
            "Value": [estimated_range, consumption]
        })

        fig_performance = px.bar(
            performance_data,
            x="Metric",
            y="Value",
            text="Value",
            title="Latest EV Performance Metrics"
        )

        fig_performance.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside"
        )

        fig_performance.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )

        st.plotly_chart(fig_performance, use_container_width=True)

        st.markdown(
            '<div class="section-title">🔋 Battery Health Overview</div>',
            unsafe_allow_html=True
        )

        if battery_health is None:
            st.markdown(
                """
<div class="info-box">
🔋 <b>Battery health has not been analysed yet.</b><br><br>
Go to <b>🔋 Battery</b> and run
<b>Estimate Temperature & Analyze Battery</b>.
</div>
""",
                unsafe_allow_html=True
            )
        else:
            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Battery Health", f"{battery_health:.1f}%")

            with c2:
                st.metric("Degradation", f"{battery_degradation:.1f}%")

            with c3:
                st.metric("Health Status", battery_health_status)

        st.markdown(
            '<div class="section-title">📋 Complete EV Analysis</div>',
            unsafe_allow_html=True
        )

        dashboard_data = pd.DataFrame({
            "Parameter": [
                "Vehicle Model",
                "Vehicle Type",
                "Vehicle Age",
                "Vehicle Weight",
                "Battery Capacity",
                "Current SOC",
                "Remaining Energy",
                "Energy Consumption",
                "Estimated Range",
                "Range Classification",
                "Estimated Battery Current",
                "Estimated Electrical Power",
                "Battery Voltage",
                "Battery Temperature",
                "Temperature Status",
                "Range Status",
                "Battery Health",
                "Battery Degradation",
                "Temperature Model"
            ],
            "Value": [
                vehicle_model,
                vehicle_type,
                f"{vehicle_age:.1f} years",
                f"{vehicle_weight} kg" if vehicle_weight is not None else "N/A",
                f"{capacity:.2f} kWh",
                f"{soc:.0f}%",
                f"{remaining_energy:.2f} kWh",
                f"{consumption:.2f} kWh/100 km",
                f"{estimated_range:.2f} km",
                range_class,
                f"{estimated_current:.2f} A" if estimated_current is not None else "N/A",
                f"{estimated_power:.2f} kW" if estimated_power is not None else "N/A",
                f"{battery_voltage} V" if battery_voltage is not None else "N/A",
                f"{estimated_temperature:.2f} °C" if estimated_temperature is not None else "N/A",
                temp_status,
                range_status,
                f"{battery_health:.2f}%" if battery_health is not None else "Not analysed",
                f"{battery_degradation:.2f}%" if battery_degradation is not None else "Not analysed",
                temperature_model_status if temperature_model_status else "Not available"
            ]
        })

        st.dataframe(
            dashboard_data,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            '<div class="section-title">🤖 Smart Recommendations</div>',
            unsafe_allow_html=True
        )

        recommendations = []

        if soc < 20:
            recommendations.append(
                "🔴 Battery SOC is very low. Consider charging before continuing a long trip."
            )
        elif soc < 40:
            recommendations.append(
                "🟡 Battery charge is moderate. Plan your next charging session."
            )
        else:
            recommendations.append(
                "🟢 Battery charge level is currently sufficient."
            )

        if consumption > 25:
            recommendations.append(
                "⚠️ Energy consumption is relatively high. Eco driving and lower speeds may improve range."
            )
        else:
            recommendations.append(
                "🟢 Energy consumption is within a reasonable range."
            )

        if estimated_temperature is not None:
            if estimated_temperature >= 50:
                recommendations.append(
                    "🔴 Battery temperature is high. Reduce aggressive driving and electrical load."
                )
            elif estimated_temperature >= 40:
                recommendations.append(
                    "🟡 Battery temperature is elevated. Monitor thermal conditions."
                )
            else:
                recommendations.append(
                    "🟢 Battery temperature is in a comfortable range."
                )

        if battery_health is not None:
            if battery_health < 70:
                recommendations.append(
                    "🔴 Battery health requires attention. Consider professional battery inspection."
                )
            elif battery_health < 80:
                recommendations.append(
                    "🟡 Battery health is moderate. Monitor charging and thermal behaviour."
                )
            else:
                recommendations.append(
                    "🟢 Battery health is currently healthy."
                )

        for recommendation in recommendations:
            st.markdown(
                f'<div class="info-box">{recommendation}</div>',
                unsafe_allow_html=True
            )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.markdown(
        '<div class="hero-title">ℹ️ About Smart EV AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="card">
<h2>⚡ Smart EV Range & Dynamic Battery Degradation Estimator</h2>

<p>
Smart EV AI is an intelligent electric vehicle analytics platform
designed to estimate real-world range, analyse battery health and
compare EV running costs with petrol vehicles.
</p>

<h3>🎯 Project Objectives</h3>
<ul>
<li>Smart EV range estimation</li>
<li>Automatic electrical load estimation</li>
<li>Dynamic battery temperature estimation</li>
<li>Dynamic battery degradation estimation</li>
<li>Battery health analysis</li>
<li>EV versus petrol cost comparison</li>
<li>Interactive analytics dashboard</li>
</ul>

<h3>🤖 Machine Learning Models</h3>
<p>
The range prediction pipeline uses the four algorithms implemented
in the project's GitHub notebooks:
</p>
<ul>
<li><b>XGBoost Regressor</b> — numerical EV range prediction</li>
<li><b>K-Nearest Neighbors (KNN)</b> — Low/High range classification</li>
<li><b>Decision Tree</b> — Low/High range classification</li>
<li><b>Logistic Regression</b> — Low/High range classification</li>
</ul>

<h3>🚗 Supported Vehicles</h3>
<ul>
<li>Electric Cars</li>
<li>Electric Scooters</li>
<li>Electric Buses</li>
</ul>

<h3>⚡ Smart Electrical Load Estimation</h3>
<p>
Users do not need to know the battery current. The application
automatically estimates electrical load using vehicle type,
vehicle weight, speed, SOC, driving style, terrain, passenger load,
climate control and ambient temperature.
</p>

<h3>🌡️ Smart Battery Temperature Estimation</h3>
<p>
Battery temperature is estimated from operating conditions.
If a compatible temperature model is supplied, the application
uses it; otherwise the built-in smart estimator is used.
</p>

<h3>📊 ML Training Features</h3>
<ul>
<li>Battery Capacity</li>
<li>State of Charge</li>
<li>Battery Health</li>
<li>Energy Consumption</li>
<li>Vehicle Age</li>
<li>Average Speed</li>
<li>Vehicle Weight</li>
</ul>

</div>
""",
        unsafe_allow_html=True
    )

    if ev_ml.get("ready", False):
        st.markdown(
            '<div class="section-title">📈 Model Status</div>',
            unsafe_allow_html=True
        )

        model_status = pd.DataFrame({
            "Algorithm": [
                "XGBoost Regressor",
                "KNN",
                "Decision Tree",
                "Logistic Regression"
            ],
            "Purpose": [
                "Numerical range prediction",
                "Low / High range classification",
                "Low / High range classification",
                "Low / High range classification"
            ],
            "Status": [
                "Ready",
                "Ready",
                "Ready",
                "Ready"
            ]
        })

        st.dataframe(
            model_status,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            f"""
<div class="info-box">
📁 <b>Dataset:</b> {ev_ml.get("dataset")}<br>
📊 <b>Training Rows:</b> {ev_ml.get("rows")}<br>
📌 <b>Median Range Threshold:</b> {ev_ml.get("median_range", 0):.2f} km<br>
🎯 <b>KNN Accuracy:</b> {ev_ml.get("knn_accuracy", 0) * 100:.2f}%<br>
🎯 <b>Decision Tree Accuracy:</b> {ev_ml.get("decision_tree_accuracy", 0) * 100:.2f}%<br>
🎯 <b>Logistic Regression Accuracy:</b> {ev_ml.get("logistic_accuracy", 0) * 100:.2f}%<br>
📉 <b>XGBoost Test MAE:</b> {ev_ml.get("xgb_mae", 0):.2f} km
</div>
""",
            unsafe_allow_html=True
        )
    else:
        st.warning(ev_ml.get("message", "ML models are not available."))


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
⚡ <b>Smart EV AI</b><br>
Intelligent EV Range • Battery Health • Automatic Load • Temperature Analytics
<br><br>
Machine Learning Powered EV Intelligence
</div>
""",
    unsafe_allow_html=True
)