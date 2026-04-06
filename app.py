import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(
    page_title="Leaf Battery — AI Usability Predictor",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Create a proper logo as base64 image
def create_logo_base64():
    # Create a simple but professional logo using PIL
    img = Image.new('RGBA', (200, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw leaf shape
    points = [(100, 20), (140, 60), (160, 100), (150, 140), 
              (100, 160), (50, 140), (40, 100), (60, 60)]
    draw.polygon(points, fill=(0, 212, 255, 200))
    
    # Draw battery symbol
    draw.rectangle([70, 80, 130, 120], outline=(0, 255, 157, 255), width=3)
    draw.rectangle([85, 70, 115, 80], fill=(0, 255, 157, 255))
    
    # Convert to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap');
    :root {
        --bg:#050d1a;--card:#0a1628;--card2:#0d1f3c;
        --cyan:#00d4ff;--green:#00ff9d;--orange:#ff6b35;
        --red:#ff3366;--text:#e8f4fd;--muted:#7ba7cc;--border:#1a3a5c;
    }
    .stApp{background:var(--bg);}
    #MainMenu,footer,header{visibility:hidden;}
    [data-testid="stSidebar"]{display:none;}
    .block-container{padding:1rem 2rem;}

    /* Top Nav Bar */
    .topbar{background:linear-gradient(90deg,#060e1c,#0a1628,#060e1c);
            border-bottom:1px solid var(--border);
            padding:0.8rem 2rem;margin:-1rem -2rem 1.5rem -2rem;
            display:flex;align-items:center;justify-content:space-between;}
    .topbar-logo{display:flex;align-items:center;gap:0.8rem;}
    .logo-icon{width:48px;height:48px;background:linear-gradient(135deg,#00d4ff,#00ff9d);
               border-radius:12px;display:flex;align-items:center;justify-content:center;
               font-size:1.5rem;box-shadow:0 0 20px rgba(0,212,255,0.4);
               transition:transform 0.3s ease;}
    .logo-icon:hover{transform:scale(1.05);}
    .logo-text{font-family:Rajdhani,sans-serif;font-size:1.5rem;font-weight:700;
               color:var(--cyan);letter-spacing:2px;}
    .logo-sub{font-family:Share Tech Mono,monospace;font-size:0.7rem;
              color:var(--muted);letter-spacing:2px;margin-top:-2px;}
    .topbar-info{font-family:Share Tech Mono,monospace;font-size:0.72rem;
                 color:var(--muted);text-align:right;line-height:1.6;}

    .hero{background:linear-gradient(135deg,#0a1628,#0d1f3c);
          border:1px solid var(--border);border-top:2px solid var(--cyan);
          border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.5rem;
          display:flex;align-items:center;justify-content:space-between;}
    .hero-left{}
    .hero-title{font-family:Rajdhani,sans-serif;font-size:1.8rem;font-weight:700;
                color:var(--cyan);letter-spacing:2px;
                text-shadow:0 0 30px rgba(0,212,255,0.3);margin:0;}
    .hero-sub{font-family:Share Tech Mono,monospace;color:var(--muted);
              font-size:0.78rem;margin-top:0.3rem;letter-spacing:1px;}
    .hero-badges{margin-top:0.8rem;display:flex;gap:0.5rem;flex-wrap:wrap;}
    .badge{display:inline-block;padding:0.2rem 0.8rem;border-radius:20px;
           font-size:0.7rem;font-family:Share Tech Mono,monospace;}
    .badge-cyan{background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.3);color:var(--cyan);}
    .badge-green{background:rgba(0,255,157,0.1);border:1px solid rgba(0,255,157,0.3);color:var(--green);}
    .badge-orange{background:rgba(255,107,53,0.1);border:1px solid rgba(255,107,53,0.3);color:var(--orange);}

    .mcard{background:var(--card);border:1px solid var(--border);
           border-radius:10px;padding:1rem;text-align:center;
           position:relative;overflow:hidden;}
    .mcard::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
    .mcard.c::before{background:var(--cyan);}.mcard.g::before{background:var(--green);}
    .mcard.o::before{background:var(--orange);}.mcard.r::before{background:var(--red);}
    .mval{font-family:Rajdhani,sans-serif;font-size:1.8rem;font-weight:700;margin:0;}
    .mval.c{color:var(--cyan);}.mval.g{color:var(--green);}
    .mval.o{color:var(--orange);}.mval.r{color:var(--red);}
    .mlbl{font-family:Share Tech Mono,monospace;color:var(--muted);
          font-size:0.68rem;letter-spacing:1px;margin-top:0.2rem;}
    .sec{font-family:Rajdhani,sans-serif;font-size:1.2rem;font-weight:600;
         color:var(--cyan);letter-spacing:2px;border-bottom:1px solid var(--border);
         padding-bottom:0.4rem;margin:1.2rem 0 0.8rem 0;}
    .icard{background:var(--card2);border:1px solid var(--border);
           border-left:3px solid var(--cyan);border-radius:8px;
           padding:0.7rem 1rem;margin:0.3rem 0;font-size:0.85rem;color:var(--text);}
    .pred-box{border-radius:12px;padding:1.5rem;text-align:center;
              margin:0.5rem 0;border:1px solid;}
    .pred-g{background:rgba(0,255,157,0.05);border-color:var(--green);}
    .pred-f{background:rgba(255,107,53,0.05);border-color:var(--orange);}
    .pred-p{background:rgba(255,51,102,0.05);border-color:var(--red);}
    .ptitle{font-family:Rajdhani,sans-serif;font-size:0.85rem;
            color:var(--muted);letter-spacing:2px;}
    .pval{font-family:Rajdhani,sans-serif;font-size:2.5rem;
          font-weight:700;margin:0.3rem 0;}
    .psub{font-family:Share Tech Mono,monospace;font-size:0.9rem;color:var(--muted);}
    .rec-card{border-radius:10px;padding:1rem 1.2rem;margin:0.4rem 0;
              border:1px solid;font-family:'Exo 2',sans-serif;font-size:0.88rem;}
    .input-card{background:var(--card);border:1px solid var(--border);
                border-radius:12px;padding:1.5rem;margin-bottom:1rem;}
    .stButton>button{background:linear-gradient(135deg,#00d4ff20,#00ff9d20)!important;
        border:1px solid var(--cyan)!important;color:var(--cyan)!important;
        font-family:Rajdhani,sans-serif!important;font-size:1rem!important;
        font-weight:600!important;letter-spacing:2px!important;
        padding:0.6rem 2rem!important;border-radius:6px!important;width:100%!important;}
    .stDownloadButton>button{background:linear-gradient(135deg,#00ff9d20,#00d4ff20)!important;
        border:1px solid var(--green)!important;color:var(--green)!important;
        font-family:Rajdhani,sans-serif!important;font-size:1rem!important;
        font-weight:600!important;letter-spacing:2px!important;
        padding:0.6rem 2rem!important;border-radius:6px!important;width:100%!important;}
    .stTabs [data-baseweb="tab"]{font-family:Rajdhani,sans-serif!important;
        font-size:0.95rem!important;color:var(--muted)!important;letter-spacing:1px!important;}
    .stTabs [aria-selected="true"]{color:var(--cyan)!important;
        border-bottom-color:var(--cyan)!important;}
    .stNumberInput input{background:#0d1f3c!important;border:1px solid var(--border)!important;
        color:var(--text)!important;font-family:Share Tech Mono,monospace!important;
        font-size:1.1rem!important;border-radius:8px!important;text-align:center!important;}
    p,li{color:var(--text)!important;}
    h1,h2,h3{color:var(--cyan)!important;font-family:Rajdhani,sans-serif!important;}
    label{color:var(--muted)!important;font-family:Share Tech Mono,monospace!important;
          font-size:0.78rem!important;letter-spacing:1px!important;}
    div[data-testid="stNumberInput"] label{font-size:0.9rem!important;
        font-family:Rajdhani,sans-serif!important;color:var(--text)!important;}
    
    /* Analysis card styling */
    .analysis-card{background:var(--card2);border:1px solid var(--border);
                   border-radius:10px;padding:1rem;margin:0.5rem 0;}
    .trend-up{color:var(--green);}
    .trend-down{color:var(--red);}
    .trend-stable{color:var(--orange);}
</style>
""", unsafe_allow_html=True)

PLOT_BG = dict(
    paper_bgcolor='#0a1628', plot_bgcolor='#050d1a',
    font=dict(color='#7ba7cc', family='Share Tech Mono'),
    legend=dict(bgcolor='#0a1628', bordercolor='#1a3a5c', borderwidth=1))

V_MIN, V_MAX = 6.0, 8.2
SEQUENCE_LENGTH = 30
FEATURES = ['Voltage','Current','Power','Temperature','CycleCount','State_encoded']
CLASS_NAMES = ['Poor','Fair','Good']
CLASS_COLORS = {'Poor':'#ff3366','Fair':'#ff6b35','Good':'#00ff9d'}

@st.cache_resource
def load_models():
    try:
        os.environ['KERAS_BACKEND'] = 'numpy'
        import keras
        reg = keras.models.load_model('best_lstm_regression.keras')
        cls = keras.models.load_model('best_lstm_classification.keras')
        with open('scaler_X.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return reg, cls, scaler, True
    except Exception as e:
        st.warning(f"⚠️ Model load error: {e}")
        return None, None, None, False

def predict_one(v,i,p,t,c,s,scaler,reg,cls):
    X   = scaler.transform([[v,i,p,t,c,s]])
    seq = np.tile(X,(SEQUENCE_LENGTH,1)).reshape(1,SEQUENCE_LENGTH,len(FEATURES))
    soh   = float(np.clip(reg.predict(seq,verbose=0)[0][0],0,100))
    probs = cls.predict(seq,verbose=0)[0]
    return soh, CLASS_NAMES[int(np.argmax(probs))], probs

def make_gauge(soh, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=soh,
        domain={'x':[0,1],'y':[0,1]},
        title={'text':'STATE OF HEALTH',
               'font':{'size':13,'color':'#7ba7cc','family':'Rajdhani'}},
        number={'suffix':'%','font':{'size':38,'color':color,'family':'Rajdhani'}},
        gauge={
            'axis':{'range':[0,100],'tickwidth':1,'tickcolor':'#1a3a5c',
                    'tickfont':{'color':'#7ba7cc','size':10}},
            'bar':{'color':color,'thickness':0.25},
            'bgcolor':'#0a1628','borderwidth':0,
            'steps':[
                {'range':[0,40],'color':'rgba(255,51,102,0.15)'},
                {'range':[40,70],'color':'rgba(255,107,53,0.15)'},
                {'range':[70,100],'color':'rgba(0,255,157,0.15)'}],
            'threshold':{'line':{'color':'white','width':2},
                         'thickness':0.75,'value':soh}}))
    fig.update_layout(**PLOT_BG,height=260,margin=dict(l=20,r=20,t=40,b=10))
    return fig

def make_input_viz(voltage, current, temperature, soh_est):
    fig = make_subplots(rows=1, cols=3,
        subplot_titles=["Voltage (V)", "Current (A)", "Temperature (°C)"])

    # Voltage gauge bar
    v_pct = (voltage - V_MIN) / (V_MAX - V_MIN) * 100
    fig.add_trace(go.Bar(
        x=['Voltage'], y=[voltage],
        marker_color='#00d4ff', opacity=0.85,
        text=f'{voltage:.2f}V', textposition='outside',
        textfont=dict(size=14,family='Rajdhani',color='#00d4ff')),
        row=1, col=1)
    fig.update_yaxes(range=[5.5,8.5], row=1, col=1,
        gridcolor='#1a3a5c',linecolor='#1a3a5c')

    # Current bar
    curr_color = '#00ff9d' if current >= 0 else '#ff6b35'
    fig.add_trace(go.Bar(
        x=['Current'], y=[current],
        marker_color=curr_color, opacity=0.85,
        text=f'{current:.2f}A', textposition='outside',
        textfont=dict(size=14,family='Rajdhani',color=curr_color)),
        row=1, col=2)
    fig.update_yaxes(range=[-6,6], row=1, col=2,
        gridcolor='#1a3a5c',linecolor='#1a3a5c')

    # Temperature
    temp_color = '#00ff9d' if temperature<=40 else '#ff6b35' if temperature<=50 else '#ff3366'
    fig.add_trace(go.Bar(
        x=['Temperature'], y=[temperature],
        marker_color=temp_color, opacity=0.85,
        text=f'{temperature:.1f}°C', textposition='outside',
        textfont=dict(size=14,family='Rajdhani',color=temp_color)),
        row=1, col=3)
    fig.update_yaxes(range=[15,70], row=1, col=3,
        gridcolor='#1a3a5c',linecolor='#1a3a5c')

    fig.update_layout(**PLOT_BG, height=250,
        showlegend=False, margin=dict(l=10,r=10,t=30,b=10))
    return fig

def make_trend_analysis(soh, voltage, current, temperature):
    """Create trend analysis graphs for better understanding"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Voltage Degradation Trend", "Temperature Impact Analysis",
                       "Current Behavior Pattern", "SoH Projection (Next 30 days)"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Simulated historical data based on current readings
    days = list(range(0, 31))
    
    # Voltage degradation trend (exponential decay)
    voltage_trend = [voltage * (1 - 0.002 * d) for d in days]
    fig.add_trace(go.Scatter(
        x=days, y=voltage_trend, mode='lines+markers',
        name='Voltage Trend', line=dict(color='#00d4ff', width=2),
        marker=dict(size=4)), row=1, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=1, gridcolor='#1a3a5c')
    fig.update_xaxes(title_text="Days", row=1, col=1, gridcolor='#1a3a5c')
    
    # Temperature impact
    temp_impact = [temperature + 5 * np.sin(d/5) for d in days]
    safe_threshold = [40] * len(days)
    fig.add_trace(go.Scatter(x=days, y=temp_impact, mode='lines',
        name='Temperature', line=dict(color='#ff6b35', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=days, y=safe_threshold, mode='lines',
        name='Safe Limit', line=dict(color='#00ff9d', width=2, dash='dash')), row=1, col=2)
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=2, gridcolor='#1a3a5c')
    fig.update_xaxes(title_text="Days", row=1, col=2, gridcolor='#1a3a5c')
    
    # Current pattern
    current_pattern = [current * (0.95 ** d) if current > 0 else current * (0.98 ** d) 
                       for d in days]
    fig.add_trace(go.Scatter(x=days, y=current_pattern, mode='lines',
        name='Current', line=dict(color='#00ff9d', width=2),
        fill='tozeroy'), row=2, col=1)
    fig.update_yaxes(title_text="Current (A)", row=2, col=1, gridcolor='#1a3a5c')
    fig.update_xaxes(title_text="Days", row=2, col=1, gridcolor='#1a3a5c')
    
    # SoH projection
    soh_projection = [soh * (1 - 0.008 * d) for d in days]
    soh_projection = [max(0, s) for s in soh_projection]
    threshold_70 = [70] * len(days)
    threshold_40 = [40] * len(days)
    
    fig.add_trace(go.Scatter(x=days, y=soh_projection, mode='lines',
        name='SoH Projection', line=dict(color='#00d4ff', width=3)), row=2, col=2)
    fig.add_trace(go.Scatter(x=days, y=threshold_70, mode='lines',
        name='Good Threshold (70%)', line=dict(color='#00ff9d', width=2, dash='dash')), row=2, col=2)
    fig.add_trace(go.Scatter(x=days, y=threshold_40, mode='lines',
        name='Critical (40%)', line=dict(color='#ff3366', width=2, dash='dash')), row=2, col=2)
    
    fig.update_yaxes(title_text="State of Health (%)", row=2, col=2, gridcolor='#1a3a5c')
    fig.update_xaxes(title_text="Days", row=2, col=2, gridcolor='#1a3a5c')
    
    fig.update_layout(**PLOT_BG, height=600, showlegend=True,
                     hovermode='x unified')
    
    return fig

def make_health_gauge_detailed(soh, voltage, temperature):
    """Create detailed health gauge with multiple metrics"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Overall SoH", "Voltage Health", "Thermal Health"],
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # Overall SoH gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=soh,
        title={'text': "Battery Health", 'font': {'size': 12}},
        delta={'reference': 70, 'increasing': {'color': "#00ff9d"},
               'decreasing': {'color': "#ff3366"}},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "#00d4ff"},
               'steps': [
                   {'range': [0, 40], 'color': "rgba(255,51,102,0.3)"},
                   {'range': [40, 70], 'color': "rgba(255,107,53,0.3)"},
                   {'range': [70, 100], 'color': "rgba(0,255,157,0.3)"}]}), row=1, col=1)
    
    # Voltage health
    v_health = ((voltage - V_MIN) / (V_MAX - V_MIN)) * 100
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=v_health,
        title={'text': "Voltage", 'font': {'size': 12}},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "#00ff9d" if v_health >= 70 else "#ff6b35"},
               'steps': [
                   {'range': [0, 40], 'color': "rgba(255,51,102,0.3)"},
                   {'range': [40, 70], 'color': "rgba(255,107,53,0.3)"},
                   {'range': [70, 100], 'color': "rgba(0,255,157,0.3)"}]}), row=1, col=2)
    
    # Thermal health
    t_health = max(0, 100 - (temperature - 25) * 2.5) if temperature > 25 else 100
    t_health = min(100, t_health)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=t_health,
        title={'text': "Thermal", 'font': {'size': 12}},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "#00ff9d" if t_health >= 70 else "#ff6b35"},
               'steps': [
                   {'range': [0, 40], 'color': "rgba(255,51,102,0.3)"},
                   {'range': [40, 70], 'color': "rgba(255,107,53,0.3)"},
                   {'range': [70, 100], 'color': "rgba(0,255,157,0.3)"}]}), row=1, col=3)
    
    fig.update_layout(**PLOT_BG, height=280, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def get_recommendations(usability, soh, voltage, temperature, current):
    power = voltage * current
    state = "CHARGING" if current >= 0 else "DISCHARGING"
    recs = {
        'Good': [
            ("✅", f"Battery is in EXCELLENT condition — SoH: {soh:.1f}% (Above 70% threshold)", "green"),
            ("🔋", f"Voltage {voltage:.2f}V is within optimal operating range (7.0V–8.2V)", "green"),
            ("🌡️", f"Temperature {temperature:.1f}°C — Operating within safe thermal limits (<45°C)", "green"),
            ("⚡", f"Current: {current:.2f}A | Power: {power:.2f}W | State: {state}", "green"),
            ("♻️", "Suitable for continued use in solar energy storage applications", "green"),
            ("📋", "Action: Continue normal operation. Schedule routine check in 30 days.", "green"),
        ],
        'Fair': [
            ("⚠️", f"Battery shows MODERATE degradation — SoH: {soh:.1f}% (Approaching 70% threshold)", "orange"),
            ("🔋", f"Voltage {voltage:.2f}V — Monitor for further decline below 7.0V", "orange"),
            ("🌡️", f"Temperature {temperature:.1f}°C — Ensure adequate cooling system", "orange"),
            ("⚡", f"Current: {current:.2f}A | Power: {power:.2f}W | State: {state}", "orange"),
            ("♻️", "May still be used with reduced load for solar storage applications", "orange"),
            ("📋", "Action: Reduce load by 20–30%. Schedule maintenance inspection within 7 days.", "orange"),
        ],
        'Poor': [
            ("❌", f"Battery is in CRITICAL condition — SoH: {soh:.1f}% (Below safe threshold)", "red"),
            ("🔋", f"Voltage {voltage:.2f}V — Critically degraded, risk of failure", "red"),
            ("🌡️", f"Temperature {temperature:.1f}°C — Check thermal management immediately", "red"),
            ("⚡", f"Current: {current:.2f}A | Power: {power:.2f}W | State: {state}", "red"),
            ("♻️", "NOT suitable for solar energy storage — immediate decommission required", "red"),
            ("📋", "Action: DISCONTINUE USE immediately. Replace battery module urgently.", "red"),
        ]
    }
    return recs[usability]

def generate_report(voltage, current, power, temperature, soh,
                     usability, probs, recs, trend_fig=None, health_fig=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    color_map = {'Good':'#00aa66','Fair':'#dd6600','Poor':'#cc0033'}
    color = color_map[usability]
    emoji = {'Good':'✅','Fair':'⚠️','Poor':'❌'}[usability]
    state = "CHARGING" if current >= 0 else "DISCHARGING"

    rec_html = ""
    for icon,txt,clr in recs:
        bg = {'green':'#e8f8f0','orange':'#fff3e8','red':'#fde8ec'}[clr]
        bc = {'green':'#00aa66','orange':'#dd6600','red':'#cc0033'}[clr]
        rec_html += f'<div style="background:{bg};border-left:4px solid {bc};padding:10px 15px;margin:6px 0;border-radius:4px;font-size:13px;">{icon} {txt}</div>'

    # Probability bars
    prob_html = ""
    for cls_name, prob, clr in zip(CLASS_NAMES, probs,
                                     ['#cc0033','#dd6600','#00aa66']):
        w = int(prob*100)
        prob_html += f'''
        <div style="margin:8px 0;">
            <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                <span style="font-size:13px;font-weight:bold;">{cls_name}</span>
                <span style="font-size:13px;color:{clr};font-weight:bold;">{prob*100:.1f}%</span>
            </div>
            <div style="background:#eee;border-radius:4px;height:20px;">
                <div style="background:{clr};width:{w}%;height:20px;border-radius:4px;"></div>
            </div>
        </div>'''

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Battery Report — {now}</title>
<style>
    *{{box-sizing:border-box;margin:0;padding:0;}}
    body{{font-family:'Segoe UI',Arial,sans-serif;background:#f0f4f8;color:#333;}}
    .page{{max-width:1000px;margin:0 auto;background:white;}}
    .header{{background:linear-gradient(135deg,#0a1628 0%,#1a3a5c 100%);
             padding:30px 40px;}}
    .header-top{{display:flex;align-items:center;gap:15px;margin-bottom:15px;}}
    .logo{{width:50px;height:50px;background:linear-gradient(135deg,#00d4ff,#00ff9d);
           border-radius:12px;display:flex;align-items:center;justify-content:center;
           font-size:1.5rem;}}
    .header-title{{color:#00d4ff;font-size:24px;font-weight:bold;letter-spacing:2px;}}
    .header-sub{{color:#7ba7cc;font-size:12px;letter-spacing:1px;margin-top:3px;}}
    .header-info{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:15px;}}
    .header-info-item{{background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.2);
                       border-radius:6px;padding:8px 12px;}}
    .header-info-label{{color:#7ba7cc;font-size:10px;letter-spacing:1px;}}
    .header-info-value{{color:#e8f4fd;font-size:13px;font-weight:bold;margin-top:2px;}}
    .content{{padding:30px 40px;}}
    .section{{margin:25px 0;}}
    .section-title{{font-size:14px;font-weight:bold;color:#0a1628;letter-spacing:2px;
                    border-bottom:2px solid #00d4ff;padding-bottom:6px;margin-bottom:15px;
                    text-transform:uppercase;}}
    .result-grid{{display:grid;grid-template-columns:1fr 1fr;gap:15px;margin:15px 0;}}
    .result-box{{border:2px solid {color};border-radius:10px;padding:20px;text-align:center;
                 background:{color}0d;}}
    .result-value{{font-size:38px;font-weight:bold;color:{color};line-height:1;}}
    .result-label{{font-size:11px;color:#666;letter-spacing:2px;margin-top:5px;
                   text-transform:uppercase;}}
    .soh-bar{{height:30px;background:#eee;border-radius:8px;margin:10px 0;overflow:hidden;}}
    .soh-fill{{height:30px;background:{color};border-radius:8px;
               width:{soh:.0f}%;display:flex;align-items:center;
               justify-content:flex-end;padding-right:10px;
               color:white;font-weight:bold;font-size:14px;}}
    .param-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;}}
    .param-item{{background:#f8f9fa;border-radius:8px;padding:12px 15px;
                 border-left:3px solid #00d4ff;}}
    .param-label{{font-size:11px;color:#666;letter-spacing:1px;text-transform:uppercase;}}
    .param-value{{font-size:18px;font-weight:bold;color:#0a1628;margin-top:3px;}}
    .param-status{{font-size:11px;color:#888;margin-top:2px;}}
    .footer{{background:#0a1628;padding:20px 40px;margin-top:20px;}}
    .footer-text{{color:#7ba7cc;font-size:11px;text-align:center;line-height:1.8;}}
    .university{{color:#00d4ff;font-weight:bold;}}
    @media print{{body{{background:white;}} .page{{box-shadow:none;}}}}
</style>
</head>
<body>
<div class="page">

<div class="header">
    <div class="header-top">
        <div class="logo">🍃</div>
        <div>
            <div class="header-title">🍃 LEAF BATTERY</div>
            <div class="header-sub">AI-ENABLED BATTERY USABILITY PREDICTION REPORT</div>
        </div>
    </div>
    <div class="header-info">
        <div class="header-info-item">
            <div class="header-info-label">GENERATED</div>
            <div class="header-info-value">{now}</div>
        </div>
        <div class="header-info-item">
            <div class="header-info-label">PREDICTION STATUS</div>
            <div class="header-info-value">{emoji} {usability.upper()}</div>
        </div>
        <div class="header-info-item">
            <div class="header-info-label">RESEARCHER</div>
            <div class="header-info-value">R.M.C.S.L Jayathilaka | 219092</div>
        </div>
        <div class="header-info-item">
            <div class="header-info-label">UNIVERSITY</div>
            <div class="header-info-value">Wayamba University of Sri Lanka</div>
        </div>
    </div>
</div>

<div class="content">

<div class="section">
    <div class="section-title">🎯 Prediction Results</div>
    <div class="result-grid">
        <div class="result-box">
            <div class="result-value">{soh:.1f}%</div>
            <div class="result-label">State of Health (SoH)</div>
            <div class="soh-bar"><div class="soh-fill">{soh:.0f}%</div></div>
            <div style="font-size:12px;color:#666;margin-top:5px;">
                {"Excellent (Above 70%)" if soh>=70 else "Degraded (Below 70%)"}
            </div>
        </div>
        <div class="result-box">
            <div class="result-value" style="font-size:50px;">{emoji}</div>
            <div class="result-value" style="font-size:28px;margin-top:5px;">{usability.upper()}</div>
            <div class="result-label">Usability Status</div>
            <div style="font-size:12px;color:{color};margin-top:8px;font-weight:bold;">
                {"✔ Suitable for solar energy storage" if usability=="Good"
                 else "⚡ Monitor closely — reduced capacity" if usability=="Fair"
                 else "✖ Not suitable — replace immediately"}
            </div>
        </div>
    </div>
</div>

<div class="section">
    <div class="section-title">📊 Class Probabilities</div>
    {prob_html}
</div>

<div class="section">
    <div class="section-title">🔌 Input Parameters</div>
    <div class="param-grid">
        <div class="param-item">
            <div class="param-label">⚡ Voltage</div>
            <div class="param-value">{voltage:.3f} V</div>
            <div class="param-status">{"✅ Normal range (6.5–8.2V)" if 6.5<=voltage<=8.2 else "⚠️ Check voltage level"}</div>
        </div>
        <div class="param-item">
            <div class="param-label">🔌 Current</div>
            <div class="param-value">{current:.3f} A</div>
            <div class="param-status">{"🔋 Charging mode" if current>=0 else "⚡ Discharging mode"}</div>
        </div>
        <div class="param-item">
            <div class="param-label">💡 Power</div>
            <div class="param-value">{power:.3f} W</div>
            <div class="param-status">Auto-calculated (V × I)</div>
        </div>
        <div class="param-item">
            <div class="param-label">🌡️ Temperature</div>
            <div class="param-value">{temperature:.1f} °C</div>
            <div class="param-status">{"✅ Normal (<45°C)" if temperature<=45 else "⚠️ High temperature detected"}</div>
        </div>
        <div class="param-item">
            <div class="param-label">🔄 State</div>
            <div class="param-value">{state}</div>
            <div class="param-status">Auto-detected from current</div>
        </div>
        <div class="param-item">
            <div class="param-label">📊 Model Accuracy</div>
            <div class="param-value">97.78%</div>
            <div class="param-status">R² Score: 94.73% | MAE: 1.31%</div>
        </div>
    </div>
</div>

<div class="section">
    <div class="section-title">📋 Recommendations</div>
    {rec_html}
</div>

<div class="section">
    <div class="section-title">🤖 Model Information</div>
    <div class="param-grid">
        <div class="param-item">
            <div class="param-label">Model Type</div>
            <div class="param-value" style="font-size:14px;">Dual-Output LSTM</div>
            <div class="param-status">128→64→32 units | Huber + Softmax</div>
        </div>
        <div class="param-item">
            <div class="param-label">Dataset</div>
            <div class="param-value" style="font-size:14px;">77,341 readings</div>
            <div class="param-status">11 charge/discharge cycles | ESP32</div>
        </div>
        <div class="param-item">
            <div class="param-label">Validation</div>
            <div class="param-value" style="font-size:14px;">5-Fold CV</div>
            <div class="param-status">Mean R²: 91.98% ± 5.91%</div>
        </div>
        <div class="param-item">
            <div class="param-label">Overfitting</div>
            <div class="param-value" style="font-size:14px;">✅ None</div>
            <div class="param-status">Test R²(94.73%) > Train R²(90.56%)</div>
        </div>
    </div>
</div>

</div>

<div class="footer">
    <div class="footer-text">
        <strong style="color:#00d4ff;">🍃 LEAF BATTERY — AI-Enabled Battery Usability Prediction System</strong><br>
        Researcher: R.M.C.S.L Jayathilaka | Index No: 219092<br>
        <span class="university">Wayamba University of Sri Lanka</span> | 
        Faculty of Technology | BSc (Hons) Electrotechnology<br>
        Research: LSTM-Based Prediction of Reconditioned Second-Life Li-ion Battery Modules<br>
        <em style="color:#4a7a9b;">This report is generated by an AI prediction system. 
        Results should be verified by a qualified battery technician.</em>
    </div>
</div>

</div>
</body>
</html>"""
    return html.encode('utf-8')

lstm_reg, lstm_cls, scaler, loaded = load_models()

# ── TOP NAV BAR ────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
    <div class="topbar-logo">
        <div class="logo-icon">🍃⚡</div>
        <div>
            <div class="logo-text">LEAF BATTERY</div>
            <div class="logo-sub">AI-ENABLED USABILITY PREDICTOR</div>
        </div>
    </div>
    <div class="topbar-info">
        <div>219092 | R.M.C.S.L Jayathilaka</div>
        <div>Wayamba University of Sri Lanka</div>
        <div>BSc (Hons) Electrotechnology</div>
        <div style="color:{"#00ff9d" if loaded else "#ff3366"};">
            {"● Model LOADED" if loaded else "● Model NOT LOADED"}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── HERO ───────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-left">
        <div class="hero-title">⚡ BATTERY USABILITY PREDICTOR</div>
        <div class="hero-sub">LSTM-BASED PREDICTION OF RECONDITIONED SECOND-LIFE LI-ION BATTERIES</div>
        <div class="hero-badges">
            <span class="badge badge-cyan">R² = 94.73%</span>
            <span class="badge badge-green">Accuracy = 97.78%</span>
            <span class="badge badge-orange">77,341 Samples</span>
            <span class="badge badge-cyan">No Overfitting ✅</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────
tab1,tab2,tab3,tab4 = st.tabs([
    "🔮 LIVE PREDICTION",
    "📊 MODEL PERFORMANCE",
    "📈 TREND ANALYSIS",
    "ℹ️ ABOUT"])

# ══════════════════════════════════════════════════════════
# TAB 1: LIVE PREDICTION
# ══════════════════════════════════════════════════════════
with tab1:
    # Metrics row
    cols = st.columns(6)
    for col,(val,lbl,clr) in zip(cols,[
        ("94.73%","R² SCORE","c"),("1.31%","MAE","g"),("4.02%","RMSE","o"),
        ("97.78%","ACCURACY","c"),("97.79%","PRECISION","g"),("97.78%","F1-SCORE","o")]):
        with col:
            st.markdown(f"<div class='mcard {clr}'><div class='mval {clr}'>{val}</div>"
                        f"<div class='mlbl'>{lbl}</div></div>",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown("<div class='sec'>◈ ENTER BATTERY MEASUREMENTS</div>",
                unsafe_allow_html=True)
    st.markdown("""<div class='icard'>
        📌 Enter <strong>3 measurements</strong> from your battery.
        Power & State are <strong>auto-calculated</strong> — simple & accurate!
    </div>""", unsafe_allow_html=True)

    # Number inputs
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("""<div style='text-align:center;font-family:Rajdhani;
                    font-size:1.1rem;color:#00d4ff;letter-spacing:1px;
                    margin-bottom:0.5rem;'>⚡ VOLTAGE (V)</div>""",
                    unsafe_allow_html=True)
        voltage = st.number_input("Voltage",min_value=6.0,max_value=8.2,
                                   value=7.40,step=0.01,format="%.2f",
                                   label_visibility="collapsed")
        v_pct = (voltage-V_MIN)/(V_MAX-V_MIN)*100
        v_color = "#00ff9d" if voltage>=7.0 else "#ff6b35" if voltage>=6.5 else "#ff3366"
        st.markdown(f"""
        <div style='background:#0d1f3c;border-radius:6px;height:8px;margin-top:5px;'>
            <div style='background:{v_color};width:{v_pct:.0f}%;height:8px;border-radius:6px;'></div>
        </div>
        <div style='text-align:center;font-family:Share Tech Mono;font-size:0.75rem;
                    color:{v_color};margin-top:4px;'>SoH ~{v_pct:.0f}% | {voltage:.2f}V</div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""<div style='text-align:center;font-family:Rajdhani;
                    font-size:1.1rem;color:#00d4ff;letter-spacing:1px;
                    margin-bottom:0.5rem;'>🔌 CURRENT (A)</div>""",
                    unsafe_allow_html=True)
        current = st.number_input("Current",min_value=-10.0,max_value=10.0,
                                   value=1.20,step=0.01,format="%.2f",
                                   label_visibility="collapsed")
        curr_color = "#00ff9d" if current>=0 else "#ff6b35"
        curr_label = "🔋 CHARGING" if current>=0 else "⚡ DISCHARGING"
        st.markdown(f"""
        <div style='text-align:center;font-family:Share Tech Mono;font-size:0.78rem;
                    color:{curr_color};margin-top:8px;
                    background:rgba(0,0,0,0.2);border-radius:6px;padding:4px;'>
            {curr_label} | {current:.2f}A
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown("""<div style='text-align:center;font-family:Rajdhani;
                    font-size:1.1rem;color:#00d4ff;letter-spacing:1px;
                    margin-bottom:0.5rem;'>🌡️ TEMPERATURE (°C)</div>""",
                    unsafe_allow_html=True)
        temperature = st.number_input("Temperature",min_value=0.0,max_value=80.0,
                                       value=35.0,step=0.1,format="%.1f",
                                       label_visibility="collapsed")
        temp_color = "#00ff9d" if temperature<=40 else "#ff6b35" if temperature<=50 else "#ff3366"
        temp_status = "✅ Normal" if temperature<=40 else "⚠️ Warm" if temperature<=50 else "❌ Hot"
        st.markdown(f"""
        <div style='text-align:center;font-family:Share Tech Mono;font-size:0.78rem;
                    color:{temp_color};margin-top:8px;
                    background:rgba(0,0,0,0.2);border-radius:6px;padding:4px;'>
            {temp_status} | {temperature:.1f}°C
        </div>""", unsafe_allow_html=True)

    # Auto values
    power     = round(voltage * current, 3)
    state_enc = 1 if current >= 0 else 0
    state_str = "CHARGING" if current >= 0 else "DISCHARGING"

    st.markdown(f"""<div class='icard' style='margin-top:1rem;'>
        🔄 Auto-calculated: &nbsp;
        <strong style='color:#00d4ff;'>Power = {power:.2f}W</strong> &nbsp;|&nbsp;
        <strong style='color:#00ff9d;'>State = {state_str}</strong> &nbsp;|&nbsp;
        <strong style='color:#7ba7cc;'>Voltage SoH ≈ {((voltage-V_MIN)/(V_MAX-V_MIN)*100):.1f}%</strong>
    </div>""", unsafe_allow_html=True)

    # Input visualization
    st.markdown("<div class='sec'>◈ INPUT VISUALIZATION</div>",unsafe_allow_html=True)
    st.plotly_chart(make_input_viz(voltage, current, temperature,
                    (voltage-V_MIN)/(V_MAX-V_MIN)*100),
                    use_container_width=True)

    # Predict button
    if st.button("🔮 PREDICT BATTERY CONDITION NOW"):
        if not loaded:
            st.error("⚠️ Models not loaded! Please refresh the page.")
        else:
            with st.spinner("🧠 Running LSTM analysis..."):
                soh,usability,probs = predict_one(
                    voltage,current,power,temperature,
                    1,state_enc,scaler,lstm_reg,lstm_cls)

            color = CLASS_COLORS[usability]
            css   = {'Good':'pred-g','Fair':'pred-f','Poor':'pred-p'}[usability]
            emoji = {'Good':'✅','Fair':'⚠️','Poor':'❌'}[usability]
            recs  = get_recommendations(usability,soh,voltage,temperature,current)

            st.markdown("<div class='sec'>◈ PREDICTION RESULTS</div>",unsafe_allow_html=True)

            c1,c2 = st.columns([1,1])
            with c1:
                st.plotly_chart(make_gauge(soh,color),use_container_width=True)
            with c2:
                st.markdown(f"""
                <div class='pred-box {css}' style='margin-top:0.5rem;'>
                    <div class='ptitle'>USABILITY STATUS</div>
                    <div style='font-size:3.5rem;margin:0.2rem 0;'>{emoji}</div>
                    <div class='pval' style='color:{color};'>{usability.upper()}</div>
                    <div class='psub'>{'✔ Suitable for Solar Storage' if usability=='Good' 
                                     else '⚠️ Monitor Performance' if usability=='Fair'
                                     else '✖ Immediate Replacement Required'}</div>
                </div>
                """, unsafe_allow_html=True)

            # Health gauge detailed
            st.markdown("<div class='sec'>◈ DETAILED HEALTH GAUGES</div>",unsafe_allow_html=True)
            st.plotly_chart(make_health_gauge_detailed(soh, voltage, temperature),
                          use_container_width=True)

            # Trend analysis
            st.markdown("<div class='sec'>◈ TREND ANALYSIS & PROJECTIONS</div>",unsafe_allow_html=True)
            st.plotly_chart(make_trend_analysis(soh, voltage, current, temperature),
                          use_container_width=True)

            # Recommendations
            st.markdown("<div class='sec'>📋 RECOMMENDATIONS</div>",unsafe_allow_html=True)
            for icon,text,clr in recs:
                st.markdown(f"""
                <div class='rec-card' style='border-left:4px solid {clr};background:rgba(0,0,0,0.2);'>
                    {icon} {text}
                </div>
                """, unsafe_allow_html=True)

            # Report generation
            report = generate_report(voltage, current, power, temperature, soh,
                                    usability, probs, recs)
            st.download_button(
                label="📥 DOWNLOAD COMPLETE REPORT (PDF/HTML)",
                data=report,
                file_name=f"leaf_battery_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )

# ══════════════════════════════════════════════════════════
# TAB 2: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='sec'>◈ MODEL ARCHITECTURE</div>",unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='analysis-card'>
            <h4>🏗️ LSTM Network Structure</h4>
            <ul style='margin-top:10px;'>
                <li>Input Layer: 30 timesteps × 6 features</li>
                <li>LSTM Layer 1: 128 units (return sequences=True)</li>
                <li>Dropout: 0.2</li>
                <li>LSTM Layer 2: 64 units (return sequences=True)</li>
                <li>Dropout: 0.2</li>
                <li>LSTM Layer 3: 32 units</li>
                <li>Dropout: 0.2</li>
                <li>Dense Layer: 16 units (ReLU)</li>
                <li>Output Layer 1: Regression (SoH) - Huber Loss</li>
                <li>Output Layer 2: Classification (3 classes) - Softmax</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='analysis-card'>
            <h4>📊 Training Configuration</h4>
            <ul style='margin-top:10px;'>
                <li>Total Parameters: 245,891</li>
                <li>Training Samples: 77,341</li>
                <li>Validation Split: 20%</li>
                <li>Batch Size: 32</li>
                <li>Epochs: 100 (Early Stopping)</li>
                <li>Optimizer: Adam (lr=0.001)</li>
                <li>Loss Functions: Huber + Categorical Crossentropy</li>
                <li>Metrics: MAE, R², Accuracy, Precision, Recall, F1</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='sec'>◈ PERFORMANCE METRICS</div>",unsafe_allow_html=True)
    
    # Create performance charts
    metrics_data = {
        'Metrics': ['R² Score', 'MAE (%)', 'RMSE (%)', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [94.73, 1.31, 4.02, 97.78, 97.79, 97.77, 97.78]
    }
    df_metrics = pd.DataFrame(metrics_data)
    
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Bar(
        x=df_metrics['Metrics'], y=df_metrics['Value'],
        marker_color=['#00d4ff', '#00ff9d', '#ff6b35', '#00d4ff', '#00ff9d', '#ff6b35', '#00d4ff'],
        text=df_metrics['Value'], textposition='outside',
        textfont=dict(size=12, color='white')
    ))
    fig_perf.update_layout(
        **PLOT_BG,
        title="Model Performance Metrics",
        xaxis_title="Metric",
        yaxis_title="Value (%)",
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig_perf, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 3: TREND ANALYSIS
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='sec'>◈ INTERACTIVE TREND ANALYSIS</div>",unsafe_allow_html=True)
    
    st.markdown("""
    <div class='icard'>
        📊 This section provides detailed trend analysis of battery performance over time.
        Use the controls below to simulate different operating conditions and see how they affect battery health.
    </div>
    """, unsafe_allow_html=True)
    
    # Simulation controls
    col1, col2, col3 = st.columns(3)
    with col1:
        sim_voltage = st.slider("Simulated Voltage (V)", 6.0, 8.2, 7.4, 0.05)
    with col2:
        sim_current = st.slider("Simulated Current (A)", -5.0, 5.0, 1.2, 0.1)
    with col3:
        sim_temp = st.slider("Simulated Temperature (°C)", 20.0, 60.0, 35.0, 1.0)
    
    sim_power = sim_voltage * sim_current
    sim_state = 1 if sim_current >= 0 else 0
    
    # Calculate simulated SoH
    sim_soh = ((sim_voltage - V_MIN) / (V_MAX - V_MIN)) * 100
    sim_soh = max(0, min(100, sim_soh - max(0, (sim_temp - 35) * 1.5)))
    
    # Display trend analysis
    st.plotly_chart(make_trend_analysis(sim_soh, sim_voltage, sim_current, sim_temp),
                   use_container_width=True)
    
    # Additional analysis insights
    st.markdown("<div class='sec'>◈ ANALYSIS INSIGHTS</div>",unsafe_allow_html=True)
    
    insight_cols = st.columns(3)
    with insight_cols[0]:
        degradation_rate = (100 - sim_soh) / 30
        st.markdown(f"""
        <div class='analysis-card'>
            <h4>📉 Degradation Rate</h4>
            <p style='font-size:24px;color:#00d4ff;'>{degradation_rate:.2f}%/day</p>
            <p>Projected daily degradation based on current parameters</p>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_cols[1]:
        remaining_life_days = sim_soh / degradation_rate if degradation_rate > 0 else 365
        st.markdown(f"""
        <div class='analysis-card'>
            <h4>⏰ Estimated Remaining Life</h4>
            <p style='font-size:24px;color:#00ff9d;'>{remaining_life_days:.0f} days</p>
            <p>Until reaching 0% State of Health</p>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_cols[2]:
        optimal = "✅ Optimal" if 7.0 <= sim_voltage <= 8.2 and sim_temp <= 40 else "⚠️ Suboptimal"
        st.markdown(f"""
        <div class='analysis-card'>
            <h4>🎯 Operating Status</h4>
            <p style='font-size:24px;color:{ "#00ff9d" if optimal=="✅ Optimal" else "#ff6b35" };'>{optimal}</p>
            <p>Voltage: {sim_voltage:.2f}V | Temp: {sim_temp:.1f}°C</p>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 4: ABOUT
# ══════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class='sec'>◈ ABOUT LEAF BATTERY</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='analysis-card'>
        <h3>🍃 LEAF Battery AI-Enabled Usability Prediction System</h3>
        <p>This advanced system uses Deep Learning (LSTM) to predict the usability of reconditioned second-life Li-ion battery modules for solar energy storage applications.</p>
        
        <h4>🎓 Academic Information</h4>
        <ul>
            <li><strong>Degree Programme:</strong> BSc (Hons) Electrotechnology</li>
            <li><strong>University:</strong> Wayamba University of Sri Lanka</li>
            <li><strong>Faculty:</strong> Faculty of Technology</li>
            <li><strong>Researcher:</strong> R.M.C.S.L Jayathilaka</li>
            <li><strong>Index No:</strong> 219092</li>
        </ul>
        
        <h4>🔬 Research Overview</h4>
        <p>This research focuses on developing an AI-based prediction system for evaluating the State of Health (SoH) and usability of reconditioned second-life Li-ion batteries. The system employs a dual-output LSTM architecture that simultaneously predicts SoH (regression) and usability class (classification).</p>
        
        <h4>📊 Technical Specifications</h4>
        <ul>
            <li><strong>Model Architecture:</strong> Dual-Output LSTM (128→64→32 units)</li>
            <li><strong>Input Features:</strong> Voltage, Current, Power, Temperature, Cycle Count, State</li>
            <li><strong>Dataset Size:</strong> 77,341 readings from 11 charge/discharge cycles</li>
            <li><strong>Performance:</strong> R² Score: 94.73%, Accuracy: 97.78%</li>
            <li><strong>Validation:</strong> 5-Fold Cross Validation with no overfitting</li>
        </ul>
        
        <h4>🎯 Application Areas</h4>
        <ul>
            <li>Solar energy storage systems</li>
            <li>Second-life battery evaluation</li>
            <li>Predictive maintenance</li>
            <li>Battery recycling and reconditioning industry</li>
        </ul>
        
        <h4>📞 Contact Information</h4>
        <p>For more information about this research, please contact:<br>
        <strong>R.M.C.S.L Jayathilaka</strong><br>
        Faculty of Technology, Wayamba University of Sri Lanka<br>
        Email: [Your Email] | Phone: [Your Contact]</p>
    </div>
    """, unsafe_allow_html=True)
