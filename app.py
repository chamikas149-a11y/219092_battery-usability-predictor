# මෙම කොටස Colab එකේ Run කරන්න. එවිට app.py ෆයිල් එක download වේවි.
from google.colab import files

app_code = '''import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime

st.set_page_config(
    page_title="Solar-Batt — AI Usability Predictor",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url(\'https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap\');
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
    .logo-icon{width:42px;height:42px;background:linear-gradient(135deg,#ffcc00,#ff6b35);
               border-radius:10px;display:flex;align-items:center;justify-content:center;
               font-size:1.3rem;box-shadow:0 0 15px rgba(255,204,0,0.4);}
    .logo-text{font-family:Rajdhani,sans-serif;font-size:1.4rem;font-weight:700;
               color:var(--cyan);letter-spacing:2px;}
    .logo-sub{font-family:Share Tech Mono,monospace;font-size:0.65rem;
              color:var(--muted);letter-spacing:1px;margin-top:-2px;}
    .topbar-info{font-family:Share Tech Mono,monospace;font-size:0.72rem;
                 color:var(--muted);text-align:right;line-height:1.6;}

    .hero{background:linear-gradient(135deg,#0a1628,#0d1f3c);
          border:1px solid var(--border);border-top:2px solid var(--cyan);
          border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.5rem;
          display:flex;align-items:center;justify-content:space-between;}
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
    .mcard::before{content:\'\';position:absolute;top:0;left:0;right:0;height:2px;}
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
    .stButton>button{background:linear-gradient(135deg,#00d4ff20,#00ff9d20)!important;
        border:1px solid var(--cyan)!important;color:var(--cyan)!important;
        font-family:Rajdhani,sans-serif!important;font-size:1rem!important;
        font-weight:600!important;letter-spacing:2px!important;
        padding:0.6rem 2rem!important;border-radius:6px!important;width:100%!important;}
    .stNumberInput input{background:#0d1f3c!important;border:1px solid var(--border)!important;
        color:var(--text)!important;font-family:Share Tech Mono,monospace!important;
        font-size:1.1rem!important;border-radius:8px!important;text-align:center!important;}
</style>
""", unsafe_allow_html=True)

PLOT_BG = dict(
    paper_bgcolor=\'#0a1628\', plot_bgcolor=\'#050d1a\',
    font=dict(color=\'#7ba7cc\', family=\'Share Tech Mono\'),
    legend=dict(bgcolor=\'#0a1628\', bordercolor=\'#1a3a5c\', borderwidth=1))

V_MIN, V_MAX = 6.0, 8.2
SEQUENCE_LENGTH = 30
FEATURES = [\'Voltage\',\'Current\',\'Power\',\'Temperature\',\'CycleCount\',\'State_encoded\']
CLASS_NAMES = [\'Poor\',\'Fair\',\'Good\']
CLASS_COLORS = {\'Poor\':\'#ff3366\',\'Fair\':\'#ff6b35\',\'Good\':\'#00ff9d\'}

@st.cache_resource
def load_models():
    try:
        os.environ[\'KERAS_BACKEND\'] = \'numpy\'
        import keras
        reg = keras.models.load_model(\'best_lstm_regression.keras\')
        cls = keras.models.load_model(\'best_lstm_classification.keras\')
        with open(\'scaler_X.pkl\', \'rb\') as f:
            scaler = pickle.load(f)
        return reg, cls, scaler, True
    except Exception as e:
        return None, None, None, False

def predict_one(v,i,p,t,c,s,scaler,reg,cls):
    X   = scaler.transform([[v,i,p,t,c,s]])
    seq = np.tile(X,(SEQUENCE_LENGTH,1)).reshape(1,SEQUENCE_LENGTH,len(FEATURES))
    soh   = float(np.clip(reg.predict(seq,verbose=0)[0][0],0,100))
    probs = cls.predict(seq,verbose=0)[0]
    return soh, CLASS_NAMES[int(np.argmax(probs))], probs

def make_prob_chart(probs):
    fig = go.Figure(go.Bar(
        x=CLASS_NAMES,
        y=probs * 100,
        marker_color=[CLASS_COLORS[c] for c in CLASS_NAMES],
        text=[f"{p*100:.1f}%" for p in probs],
        textposition=\'auto\',
    ))
    fig.update_layout(**PLOT_BG, height=300, title="Confidence Levels (%)", 
                      margin=dict(l=20,r=20,t=40,b=20))
    return fig

def make_input_viz(voltage, current, temperature):
    fig = make_subplots(rows=1, cols=3,
        subplot_titles=["Voltage (V)", "Current (A)", "Temp (°C)"])
    fig.add_trace(go.Bar(x=[\'V\'], y=[voltage], marker_color=\'#00d4ff\'), row=1, col=1)
    fig.add_trace(go.Bar(x=[\'A\'], y=[current], marker_color=\'#00ff9d\'), row=1, col=2)
    fig.add_trace(go.Bar(x=[\'T\'], y=[temperature], marker_color=\'#ff6b35\'), row=1, col=3)
    fig.update_layout(**PLOT_BG, height=250, showlegend=False, margin=dict(l=10,r=10,t=30,b=10))
    return fig

lstm_reg, lstm_cls, scaler, loaded = load_models()

# TOP NAV BAR
st.markdown(f"""
<div class="topbar">
    <div class="topbar-logo">
        <div class="logo-icon">☀️🔋</div>
        <div>
            <div class="logo-text">SOLAR-BATT AI</div>
            <div class="logo-sub">PREDICTOR | RESEARCH PROJECT</div>
        </div>
    </div>
    <div class="topbar-info">
        <div>219092 | R.M.C.S.L Jayathilaka</div>
        <div>Wayamba University of Sri Lanka</div>
    </div>
</div>
""", unsafe_allow_html=True)

# HERO
st.markdown("""
<div class="hero">
    <div class="hero-left">
        <div class="hero-title">⚡ BATTERY USABILITY PREDICTOR</div>
        <div class="hero-sub">LSTM-BASED PREDICTION OF RECONDITIONED SECOND-LIFE LI-ION BATTERIES</div>
        <div class="hero-badges">
            <span class="badge badge-cyan">R² = 94.73%</span>
            <span class="badge badge-green">Accuracy = 97.78%</span>
            <span class="badge badge-orange">Electrotechnology Research</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮 LIVE PREDICTION", "📊 PERFORMANCE", "ℹ️ ABOUT"])

with tab1:
    st.markdown("<div class=\'sec\'>◈ ENTER BATTERY MEASUREMENTS</div>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1: voltage = st.number_input("Voltage", 6.0, 8.2, 7.40, step=0.01)
    with c2: current = st.number_input("Current", -10.0, 10.0, 1.20, step=0.01)
    with c3: temperature = st.number_input("Temp (°C)", 0.0, 80.0, 35.0, step=0.1)

    if st.button("🔮 PREDICT BATTERY CONDITION NOW"):
        if loaded:
            power = voltage * current
            soh, usability, probs = predict_one(voltage, current, power, temperature, 1, 1 if current >=0 else 0, scaler, lstm_reg, lstm_cls)
            
            st.markdown(f"<div class=\'icard\'>Result: <b>{usability.upper()}</b> | SoH Estimate: <b>{soh:.2f}%</b></div>", unsafe_allow_html=True)
            
            # Graphs
            gc1, gc2 = st.columns(2)
            with gc1:
                st.plotly_chart(make_input_viz(voltage, current, temperature), use_container_width=True)
            with gc2:
                st.plotly_chart(make_prob_chart(probs), use_container_width=True)
        else:
            st.error("Model Error: Please check if model files are in the drive folder.")

# FOOTER
st.markdown(f"""
<div style="background:#0a1628; padding:20px; border-top:1px solid #1a3a5c; margin-top:50px; text-align:center; font-family:Share Tech Mono; font-size:0.75rem; color:var(--muted);">
    ☀️ <b>SOLAR-BATT AI</b> — RECONDITIONED BATTERY USABILITY ANALYSIS<br>
    Researcher: R.M.C.S.L Jayathilaka | Index No: 219092<br>
    BSc (Hons) Electrotechnology | Wayamba University of Sri Lanka
</div>
""", unsafe_allow_html=True)
'''

# Write to file
with open('app.py', 'w') as f:
    f.write(app_code)

# Download to local PC
files.download('app.py')
print("✅ Done! 'app.py' has been modified and download triggered.")
