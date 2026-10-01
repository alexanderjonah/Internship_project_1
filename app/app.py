
from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "lead_time_model.joblib"

st.set_page_config(
    page_title="Nassau Candy Factory Optimization",
    page_icon="🍬",
    layout="wide"
)

@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)

artifact = load_artifact()
pipe = artifact["pipeline"]
model_name = artifact["model_name"]
df = artifact["historical_data"].copy()
FACTORIES = artifact["factories"]
PRODUCT_FACTORY = artifact["product_factory"]
ALPHA = artifact["scenario_alpha"]

REGION_CENTROIDS = {
    "Interior": (39.0, -98.0),
    "Atlantic": (39.0, -77.0),
    "Gulf": (32.0, -90.0),
    "Pacific": (37.0, -119.0),
}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2-lat1)
    dl = np.radians(lon2-lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def distance(region, factory):
    lat, lon = REGION_CENTROIDS[region]
    flat, flon = FACTORIES[factory]
    return float(haversine_km(lat, lon, flat, flon))

def make_input(subset):
    # Representative transaction for the selected scenario.
    row = subset.iloc[0].copy()
    return pd.DataFrame([{
        "Ship Mode": row["Ship Mode"],
        "Country/Region": row["Country/Region"],
        "City": row["City"],
        "State/Province": row["State/Province"],
        "Division": row["Division"],
        "Region": row["Region"],
        "Product ID": row["Product ID"],
        "Product Name": row["Product Name"],
        "Sales": float(subset["Sales"].median()),
        "Units": float(subset["Units"].median()),
        "Gross Profit": float(subset["Gross Profit"].median()),
        "Cost": float(subset["Cost"].median()),
        "Profit_Margin": float(subset["Profit_Margin"].median()),
        "Order_Year": int(subset["Order_Year"].median()),
        "Order_Month": int(subset["Order_Month"].median()),
        "Order_DayOfWeek": int(subset["Order_DayOfWeek"].median()),
        "Order_Quarter": int(subset["Order_Quarter"].median()),
    }])

st.title("🍬 Nassau Candy — Factory Reallocation & Shipping Optimization")
st.caption(
    "Decision-support dashboard for lead-time prediction, profitability context, "
    "and factory what-if analysis."
)

with st.sidebar:
    st.header("Scenario Controls")
    products = sorted(df["Product Name"].dropna().unique())
    product = st.selectbox("Product", products)

    regions = sorted(df["Region"].dropna().unique())
    region = st.selectbox("Region", regions)

    modes = sorted(df["Ship Mode"].dropna().unique())
    ship_mode = st.selectbox("Ship Mode", modes)

    priority = st.slider(
        "Optimization priority: Speed ↔ Profit",
        0, 100, 70,
        help="100 prioritizes speed; 0 prioritizes historical gross margin."
    )
    speed_weight = priority / 100

subset = df[
    (df["Product Name"] == product) &
    (df["Region"] == region) &
    (df["Ship Mode"] == ship_mode)
].copy()

# Fall back progressively if the exact combination has no historical rows.
if len(subset) == 0:
    subset = df[
        (df["Product Name"] == product) &
        (df["Region"] == region)
    ].copy()
if len(subset) == 0:
    subset = df[df["Product Name"] == product].copy()

current_factory = PRODUCT_FACTORY[product]
input_row = make_input(subset)
baseline_pred = float(pipe.predict(input_row)[0])

margin = float(subset["Profit_Margin"].mean())
orders = int(subset["Order ID"].nunique())

if orders < 10:
    risk = "High"
    risk_reason = "Very limited historical observations for this product/scenario."
elif orders < 30:
    risk = "Medium"
    risk_reason = "Moderate historical support; scenario should be validated before operational rollout."
else:
    risk = "Lower"
    risk_reason = "More historical observations support the scenario estimate."

st.subheader("Current Situation")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Factory", current_factory)
c2.metric("Predicted Lead Time", f"{baseline_pred:.1f} days")
c3.metric("Historical Gross Margin", f"{margin*100:.1f}%")
c4.metric("Scenario Risk", risk)

st.info(
    "Factory reassignment is a scenario estimate, not a causal factory prediction. "
    "The dataset does not contain historical factory assignments, transport costs, "
    "or factory capacity."
)

rows = []
current_distance = distance(region, current_factory)

for factory in FACTORIES:
    d = distance(region, factory)
    relative_change = (d-current_distance) / max(current_distance, 1e-9)
    scenario_lead = baseline_pred * (1 + ALPHA * relative_change)
    improvement = baseline_pred - scenario_lead
    improvement_pct = improvement / baseline_pred * 100

    speed_score = max(improvement_pct, 0)
    rows.append({
        "Factory": factory,
        "Current": "Yes" if factory == current_factory else "No",
        "Approx. Distance (km)": d,
        "Scenario Lead Time (days)": scenario_lead,
        "Lead-Time Improvement (%)": improvement_pct,
        "Speed Score": speed_score
    })

scenarios = pd.DataFrame(rows)

max_speed = max(scenarios["Speed Score"].max(), 1e-9)
scenarios["Speed Score"] = scenarios["Speed Score"] / max_speed
scenarios["Profit Score"] = margin
scenarios["Decision Score"] = (
    speed_weight * scenarios["Speed Score"] +
    (1-speed_weight) * scenarios["Profit Score"]
)

# Do not recommend a switch when it doesn't improve lead time.
best = scenarios.sort_values(
    ["Decision Score", "Lead-Time Improvement (%)"],
    ascending=[False, False]
).iloc[0]

if best["Factory"] == current_factory or best["Lead-Time Improvement (%)"] <= 0:
    recommendation = "Keep current factory"
    recommendation_detail = "No alternative has a positive estimated lead-time improvement under this scenario."
else:
    recommendation = f"Consider {best['Factory']}"
    recommendation_detail = (
        f"Scenario estimate: {best['Lead-Time Improvement (%)']:.1f}% "
        "lead-time improvement versus the current assignment."
    )

st.subheader("Recommendation")
if recommendation == "Keep current factory":
    st.warning(f"**{recommendation}**")
else:
    st.success(f"**{recommendation}**")
st.write(recommendation_detail)
st.caption(
    f"Risk: {risk}. {risk_reason} "
    "Historical margin is used as profitability context because factory-specific "
    "profitability and transport costs are unavailable."
)

st.subheader("Factory What-If Comparison")
display_cols = [
    "Factory", "Current", "Approx. Distance (km)",
    "Scenario Lead Time (days)", "Lead-Time Improvement (%)", "Decision Score"
]
display_df = scenarios[display_cols].copy()
display_df["Approx. Distance (km)"] = display_df["Approx. Distance (km)"].round(0)
display_df["Scenario Lead Time (days)"] = display_df["Scenario Lead Time (days)"].round(1)
display_df["Lead-Time Improvement (%)"] = display_df["Lead-Time Improvement (%)"].round(1)
display_df["Decision Score"] = display_df["Decision Score"].round(3)

st.dataframe(display_df, use_container_width=True, hide_index=True)

chart = px.bar(
    scenarios.sort_values("Scenario Lead Time (days)"),
    x="Factory",
    y="Scenario Lead Time (days)",
    title="Scenario Lead Time by Factory",
    text_auto=".1f"
)
st.plotly_chart(chart, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.metric("Historical Orders Used", orders)
with c2:
    st.metric("Average Historical Margin", f"{margin*100:.1f}%")

with st.expander("How this system works"):
    st.markdown(f"""
**ML model:** {model_name}

The ML model predicts baseline lead time using historical order, product,
location, shipping-mode, sales, cost, profit, and date features.

**Factory layer:** The historical dataset does not identify which factory
fulfilled each order. The dashboard therefore uses a transparent distance-based
scenario proxy rather than claiming a learned factory effect.

**Profitability:** Gross margin is calculated as `Gross Profit / Sales`.

**Missing inputs:** Actual transportation cost, factory capacity, production
capacity, and historical factory assignments are not available, so the dashboard
does not invent them.
""")

st.caption("Nassau Candy internship project — decision-support prototype")
