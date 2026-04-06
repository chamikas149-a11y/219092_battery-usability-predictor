import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Battery Health Dashboard",
    page_icon="🔋",
    layout="wide"
)

# Title
st.title("🔋 Battery Health Monitoring Dashboard")
st.markdown("### Real-time State of Health (SoH) & Usability Prediction")
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("📊 Battery Parameters")

# Create input sliders
voltage = st.sidebar.slider(
    "Voltage (V)",
    min_value=6.0,
    max_value=8.4,
    value=7.2,
    step=0.01,
    help="Battery terminal voltage"
)

current = st.sidebar.slider(
    "Current (A)",
    min_value=0.0,
    max_value=5.0,
    value=1.5,
    step=0.01,
    help="Discharging current (positive value)"
)

power = st.sidebar.slider(
    "Power (W)",
    min_value=0.0,
    max_value=40.0,
    value=12.0,
    step=0.5,
    help="Power output"
)

temperature = st.sidebar.slider(
    "Temperature (°C)",
    min_value=10.0,
    max_value=55.0,
    value=32.0,
    step=0.5,
    help="Battery temperature"
)

cycle_count = st.sidebar.number_input(
    "Cycle Count",
    min_value=0,
    max_value=500,
    value=50,
    step=5,
    help="Number of charge/discharge cycles"
)

state = st.sidebar.selectbox(
    "Operating State",
    options=["DISCHARGING", "CHARGING"],
    help="Current battery state"
)

# State encoding
state_encoded = 1 if state == "DISCHARGING" else 0

# Model prediction function
def predict_soh(voltage, current, power, temperature, state_encoded, cycle_count):
    """
    Calculate SoH based on voltage with cycle degradation factor
    """
    # Base SoH from voltage (6.0V = 0%, 8.4V = 100%)
    V_MIN = 6.0
    V_MAX = 8.4
    soh_voltage = ((voltage - V_MIN) / (V_MAX - V_MIN)) * 100
    soh_voltage = max(0, min(100, soh_voltage))
    
    # Cycle degradation (each cycle reduces SoH by ~0.05% after 100 cycles)
    cycle_degradation = max(0, cycle_count - 100) * 0.05
    soh_cycle = soh_voltage - cycle_degradation
    
    # Temperature effect (optimal around 25°C)
    temp_effect = -0.1 * abs(temperature - 25)
    
    # Current effect (high current reduces efficiency)
    current_effect = -0.5 * max(0, current - 1.0)
    
    # Final SoH
    soh = soh_cycle + temp_effect + current_effect
    soh = max(0, min(100, soh))
    
    return soh

def predict_usability(soh):
    """Classify usability based on SoH"""
    if soh >= 70:
        return "🟢 Good", "Good"
    elif soh >= 50:
        return "🟡 Fair", "Fair"
    else:
        return "🔴 Poor", "Poor"

# Make prediction
soh = predict_soh(voltage, current, power, temperature, state_encoded, cycle_count)
usability_display, usability_class = predict_usability(soh)

# Main display area
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="📈 Predicted SoH",
        value=f"{soh:.1f}%",
        delta=None
    )

with col2:
    st.metric(
        label="🔋 Usability Status",
        value=usability_display,
        delta=None
    )

with col3:
    st.metric(
        label="🔄 Cycle Count",
        value=f"{cycle_count}",
        delta=None
    )

# Gauge chart for SoH
st.markdown("---")
st.subheader("📊 State of Health Visualization")

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=soh,
    title={"text": "Battery State of Health (%)"},
    gauge={
        "axis": {"range": [0, 100], "tickwidth": 1},
        "bar": {"color": "darkgreen" if soh >= 70 else "orange" if soh >= 50 else "red"},
        "steps": [
            {"range": [0, 50], "color": "#ffcccc"},
            {"range": [50, 70], "color": "#ffffcc"},
            {"range": [70, 100], "color": "#ccffcc"}
        ],
        "threshold": {
            "line": {"color": "black", "width": 4},
            "thickness": 0.75,
            "value": soh
        }
    }
))

fig_gauge.update_layout(height=400, margin=dict(l=50, r=50, t=50, b=50))
st.plotly_chart(fig_gauge, use_container_width=True)

# Parameter influence chart
st.subheader("📉 Parameter Influence Analysis")

# Create data for influence chart
params = ['Voltage', 'Temperature', 'Cycle Count', 'Current']
base_soh = predict_soh(7.2, 1.5, 12.0, 32.0, 1, 50)
influences = []

# Calculate influence for each parameter
for param, low_val, high_val in [
    ('Voltage', 6.0, 8.4),
    ('Temperature', 15, 45),
    ('Cycle Count', 0, 300),
    ('Current', 0.5, 3.0)
]:
    if param == 'Voltage':
        low_soh = predict_soh(low_val, 1.5, 12.0, 32.0, 1, 50)
        high_soh = predict_soh(high_val, 1.5, 12.0, 32.0, 1, 50)
    elif param == 'Temperature':
        low_soh = predict_soh(7.2, 1.5, 12.0, low_val, 1, 50)
        high_soh = predict_soh(7.2, 1.5, 12.0, high_val, 1, 50)
    elif param == 'Cycle Count':
        low_soh = predict_soh(7.2, 1.5, 12.0, 32.0, 1, low_val)
        high_soh = predict_soh(7.2, 1.5, 12.0, 32.0, 1, high_val)
    else:  # Current
        low_soh = predict_soh(7.2, low_val, low_val*8, 32.0, 1, 50)
        high_soh = predict_soh(7.2, high_val, high_val*8, 32.0, 1, 50)
    
    influence = (high_soh - low_soh) / low_soh * 100
    influences.append(influence)

fig_bar = go.Figure(data=[
    go.Bar(
        x=params,
        y=influences,
        marker_color=['steelblue', 'tomato', 'orange', 'coral'],
        text=[f"{v:.1f}%" for v in influences],
        textposition='outside'
    )
])
fig_bar.update_layout(
    title="Parameter Impact on SoH (%)",
    xaxis_title="Parameter",
    yaxis_title="Influence (%)",
    height=400
)
st.plotly_chart(fig_bar, use_container_width=True)

# Recommendations
st.markdown("---")
st.subheader("💡 Recommendations")

col_rec1, col_rec2, col_rec3 = st.columns(3)

with col_rec1:
    if soh >= 70:
        st.success("✅ **Battery Health: Excellent**\n\n- Continue normal operation\n- Regular monitoring recommended")
    elif soh >= 50:
        st.warning("⚠️ **Battery Health: Degrading**\n\n- Consider reducing load\n- Monitor temperature closely\n- Plan for replacement within 6-12 months")
    else:
        st.error("❌ **Battery Health: Critical**\n\n- Immediate replacement recommended\n- Reduce usage to minimum\n- Safety check required")

with col_rec2:
    if temperature > 45:
        st.warning("🌡️ **High Temperature Detected**\n\n- Improve ventilation\n- Reduce charging current\n- Check cooling system")
    elif temperature < 15:
        st.info("❄️ **Low Temperature**\n\n- Performance may be reduced\n- Allow warm-up before heavy use")
    else:
        st.success("✅ **Temperature: Normal**")

with col_rec3:
    if cycle_count > 300:
        st.warning("🔁 **High Cycle Count**\n\n- Battery approaching end-of-life\n- Consider capacity testing\n- Plan replacement")
    elif cycle_count > 150:
        st.info("📊 **Moderate Cycle Count**\n\n- Regular health checks recommended")
    else:
        st.success("✅ **Cycle Count: Good**")

# Footer
st.markdown("---")
st.caption("🔋 Battery Health Monitoring System | R.M.C.S.L Jayathilaka | 219092")
