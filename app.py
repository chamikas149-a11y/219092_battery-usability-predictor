import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime

st.set_page_config(
    page_title="Leaf Battery — AI Usability Predictor",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

    /* Modern Tech Logo Style */
    .logo-icon {
        width: 50px; height: 50px;
        background: rgba(0, 212, 255, 0.1);
        border: 2px solid var(--cyan);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.8rem;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3), inset 0 0 10px rgba(0, 212, 255, 0.2);
        position: relative;
    }
    .logo-icon::after {
        content: '⚡'; position: absolute; font-size: 0.8rem; bottom: -5px; right: -5px;
        background: var(--bg); border: 1px solid var(--green); border-radius: 50%; padding: 2px;
    }

    .topbar{background:linear-gradient(90deg,#060e1c,#0a1628,#060e1c);
            border-bottom:1px solid var(--border);
            padding:0.8rem 2rem;margin:-1rem -2rem 1.5rem -2rem;
            display:flex;align-items:center;justify-content:space-between;}
    .topbar-logo{display:flex;align-items:center;gap:1.2rem;}
    .logo-text{font-family:Rajdhani,sans-serif;font-size:1.6rem;font-weight:700;
               color:var(--cyan);letter-spacing:3px; text-transform: uppercase;}
    .logo-sub{font-family:Share Tech Mono,monospace;font-size:0.7rem;
              color:var(--muted);letter-spacing:1.5px;margin-top:-2px;}
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
    .stNumberInput input{background:#0d1f3c!important;border:1px solid var(--border)!important;
        color:var(--text)!important;font-family:Share Tech Mono,monospace!important;
        font-size:1.1rem!important;border-radius:8px!important;text-align:center!important;}
    p,li{color:var(--text)!important;}
    h1,h2,h3{color:var(--cyan)!important;font-family:Rajdhani,sans-serif!important;}
    label{color:var(--muted)!important;font-family:Share Tech Mono,monospace!important;
          font-size:0.78rem!important;letter-spacing:1px!important;}
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

def make_prob_chart(probs):
    fig = go.Figure([go.Bar(
        x=CLASS_NAMES,
        y=[p*100 for p in probs],
        marker_color=['#ff3366', '#ff6b35', '#00ff9d'],
        text=[f'{p*100:.1f}%' for p in probs],
        textposition='auto',
    )])
    fig.update_layout(**PLOT_BG, title={'text':'MODEL CONFIDENCE ANALYSIS','font':{'size':14,'family':'Rajdhani'}},
                      height=260, margin=dict(l=20,r=20,t=50,b=20), xaxis_title="Battery Grade", yaxis_title="Probability (%)")
    return fig

def make_input_viz(voltage, current, temperature, soh_est):
    fig = make_subplots(rows=1, cols=3,
        subplot_titles=["Voltage (V)", "Current (A)", "Temperature (°C)"])

    fig.add_trace(go.Bar(
        x=['Voltage'], y=[voltage],
        marker_color='#00d4ff', opacity=0.85,
        text=f'{voltage:.2f}V', textposition='outside',
        textfont=dict(size=14,family='Rajdhani',color='#00d4ff')),
        row=1, col=1)
    fig.update_yaxes(range=[5.5,8.5], row=1, col=1, gridcolor='#1a3a5c',linecolor='#1a3a5c')

    curr_color = '#00ff9d' if current >= 0 else '#ff6b35'
    fig.add_trace(go.Bar(
        x=['Current'], y=[current],
        marker_color=curr_color, opacity=0.85,
        text=f'{current:.2f}A', textposition='outside',
        textfont=dict(size=14,family='Rajdhani',color=curr_color)),
        row=1, col=2)
    fig.update_yaxes(range=[-6,6], row=1, col=2, gridcolor='#1a3a5c',linecolor='#1a3a5c')

    temp_color = '#00ff9d' if temperature<=40 else '#ff6b35' if temperature<=50 else '#ff3366'
    fig.add_trace(go.Bar(
        x=['Temperature'], y=[temperature],
        marker_color=temp_color, opacity=0.85,
        text=f'{temperature:.1f}°C', textposition='outside',
        textfont=dict(size=14,family='Rajdhani',color=temp_color)),
        row=1, col=3)
    fig.update_yaxes(range=[15,70], row=1, col=3, gridcolor='#1a3a5c',linecolor='#1a3a5c')

    fig.update_layout(**PLOT_BG, height=250, showlegend=False, margin=dict(l=10,r=10,t=30,b=10))
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
                     usability, probs, recs, gauge_img=None):
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

    prob_html = ""
    for cls_name, prob, clr in zip(CLASS_NAMES, probs, ['#cc0033','#dd6600','#00aa66']):
        w = int(prob*100)
        prob_html += f'''
        <div style="margin:8px 0;">
            <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                <span style="font-size:13px;font-weight:bold;">{cls_name} Grade Confidence</span>
                <span style="font-size:13px;color:{clr};font-weight:bold;">{prob*100:.1f}%</span>
            </div>
            <div style="background:#eee;border-radius:4px;height:12px;">
                <div style="background:{clr};width:{w}%;height:12px;border-radius:4px;"></div>
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
    .page{{max-width:800px;margin:20px auto;background:white;box-shadow:0 0 10px rgba(0,0,0,0.1);}}
    .header{{background:linear-gradient(135deg,#0a1628 0%,#1a3a5c 100%);padding:30px 40px;}}
    .header-top{{display:flex;align-items:center;gap:15px;margin-bottom:15px;}}
    .logo{{width:50px;height:50px;background:white;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.5rem;border:2px solid #00d4ff;}}
    .header-title{{color:#00d4ff;font-size:24px;font-weight:bold;letter-spacing:2px;}}
    .header-sub{{color:#7ba7cc;font-size:12px;letter-spacing:1px;margin-top:3px;}}
    .header-info{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:15px;}}
    .header-info-item{{background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.2);border-radius:6px;padding:8px 12px;}}
    .header-info-label{{color:#7ba7cc;font-size:10px;letter-spacing:1px;}}
    .header-info-value{{color:#e8f4fd;font-size:13px;font-weight:bold;margin-top:2px;}}
    .content{{padding:30px 40px;}}
    .section{{margin:25px 0;}}
    .section-title{{font-size:14px;font-weight:bold;color:#0a1628;letter-spacing:2px;border-bottom:2px solid #00d4ff;padding-bottom:6px;margin-bottom:15px;text-transform:uppercase;}}
    .result-grid{{display:grid;grid-template-columns:1fr 1fr;gap:15px;margin:15px 0;}}
    .result-box{{border:2px solid {color};border-radius:10px;padding:20px;text-align:center;background:{color}0d;}}
    .result-value{{font-size:38px;font-weight:bold;color:{color};line-height:1;}}
    .result-label{{font-size:11px;color:#666;letter-spacing:2px;margin-top:5px;text-transform:uppercase;}}
    .soh-bar{{height:30px;background:#eee;border-radius:8px;margin:10px 0;overflow:hidden;}}
    .soh-fill{{height:30px;background:{color};border-radius:8px;width:{soh:.0f}%;display:flex;align-items:center;justify-content:flex-end;padding-right:10px;color:white;font-weight:bold;font-size:14px;}}
    .param-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;}}
    .param-item{{background:#f8f9fa;border-radius:8px;padding:12px 15px;border-left:3px solid #00d4ff;}}
    .param-label{{font-size:11px;color:#666;letter-spacing:1px;text-transform:uppercase;}}
    .param-value{{font-size:18px;font-weight:bold;color:#0a1628;margin-top:3px;}}
    .footer{{background:#0a1628;padding:20px 40px;margin-top:20px;}}
    .footer-text{{color:#7ba7cc;font-size:11px;text-align:center;line-height:1.8;}}
    .university{{color:#00d4ff;font-weight:bold;}}
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
        <div class="header-info-item"><div class="header-info-label">GENERATED</div><div class="header-info-value">{now}</div></div>
        <div class="header-info-item"><div class="header-info-label">PREDICTION STATUS</div><div class="header-info-value">{emoji} {usability.upper()}</div></div>
        <div class="header-info-item"><div class="header-info-label">RESEARCHER</div><div class="header-info-value">R.M.C.S.L Jayathilaka | 219092</div></div>
        <div class="header-info-item"><div class="header-info-label">UNIVERSITY</div><div class="header-info-value">Wayamba University of Sri Lanka</div></div>
    </div>
</div>
<div class="content">
<div class="section">
    <div class="section-title">🎯 Prediction & Analysis Summary</div>
    <div class="result-grid">
        <div class="result-box">
            <div class="result-value">{soh:.1f}%</div>
            <div class="result-label">State of Health (SoH)</div>
            <div class="soh-bar"><div class="soh-fill">{soh:.0f}%</div></div>
        </div>
        <div class="result-box">
            <div class="result-value" style="font-size:50px;">{emoji}</div>
            <div class="result-value" style="font-size:28px;margin-top:5px;">{usability.upper()}</div>
            <div class="result-label">Usability Grade</div>
        </div>
    </div>
</div>
<div class="section">
    <div class="section-title">📊 Confidence Analysis</div>
    {prob_html}
</div>
<div class="section">
    <div class="section-title">🔌 Technical Parameters</div>
    <div class="param-grid">
        <div class="param-item"><div class="param-label">⚡ Voltage</div><div class="param-value">{voltage:.3f} V</div></div>
        <div class="param-item"><div class="param-label">🔌 Current</div><div class="param-value">{current:.3f} A</div></div>
        <div class="param-item"><div class="param-label">💡 Power</div><div class="param-value">{power:.3f} W</div></div>
        <div class="param-item"><div class="param-label">🌡️ Temperature</div><div class="param-value">{temperature:.1f} °C</div></div>
    </div>
</div>
<div class="section"><div class="section-title">📋 Recommendations</div>{rec_html}</div>
</div>
<div class="footer">
    <div class="footer-text">
        <strong style="color:#00d4ff;">🍃 LEAF BATTERY — AI-Enabled Battery Usability Prediction System</strong><br>
        Researcher: R.M.C.S.L Jayathilaka | Index No: 219092<br>
        <span class="university">Wayamba University of Sri Lanka</span> | Faculty of Technology | BSc (Hons) Electrotechnology<br>
        Research: LSTM-Based Prediction of Reconditioned Second-Life Li-ion Battery Modules
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
        <div class="logo-icon">🍃</div>
        <div>
            <div class="logo-text">LEAF BATTERY</div>
            <div class="logo-sub">AI-ENABLED USABILITY PREDICTOR</div>
        </div>
    </div>
    <div class="topbar-info">
        <div>219092 | R.M.C.S.L Jayathilaka</div>
        <div>Wayamba University of Sri Lanka</div>
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
            <span class="badge badge-cyan">Degree: Electrotechnology</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

tab1,tab2,tab3,tab4 = st.tabs(["🔮 LIVE PREDICTION","📊 PERFORMANCE","🔍 ANALYSIS","ℹ️ ABOUT"])

with tab1:
    cols = st.columns(6)
    for col,(val,lbl,clr) in zip(cols,[
        ("94.73%","R² SCORE","c"),("1.31%","MAE","g"),("4.02%","RMSE","o"),
        ("97.78%","ACCURACY","c"),("97.79%","PRECISION","g"),("97.78%","F1-SCORE","o")]):
        with col:
            st.markdown(f"<div class='mcard {clr}'><div class='mval {clr}'>{val}</div>"
                        f"<div class='mlbl'>{lbl}</div></div>",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown("<div class='sec'>◈ ENTER BATTERY MEASUREMENTS</div>", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        voltage = st.number_input("Voltage (V)",min_value=6.0,max_value=8.2,value=7.40,step=0.01)
    with c2:
        current = st.number_input("Current (A)",min_value=-10.0,max_value=10.0,value=1.20,step=0.01)
    with c3:
        temperature = st.number_input("Temperature (°C)",min_value=0.0,max_value=80.0,value=35.0,step=0.1)

    power = round(voltage * current, 3)
    state_enc = 1 if current >= 0 else 0
    state_str = "CHARGING" if current >= 0 else "DISCHARGING"

    st.markdown("<div class='sec'>◈ INPUT VISUALIZATION</div>",unsafe_allow_html=True)
    st.plotly_chart(make_input_viz(voltage, current, temperature, 0), use_container_width=True)

    if st.button("🔮 PREDICT BATTERY CONDITION NOW"):
        if not loaded:
            st.error("⚠️ Models not loaded!")
        else:
            with st.spinner("🧠 Analyzing data..."):
                soh,usability,probs = predict_one(voltage,current,power,temperature,1,state_enc,scaler,lstm_reg,lstm_cls)

            color = CLASS_COLORS[usability]
            css   = {'Good':'pred-g','Fair':'pred-f','Poor':'pred-p'}[usability]
            emoji = {'Good':'✅','Fair':'⚠️','Poor':'❌'}[usability]
            recs  = get_recommendations(usability,soh,voltage,temperature,current)

            st.markdown("<div class='sec'>◈ PREDICTION & ANALYTICS</div>",unsafe_allow_html=True)

            c1,c2 = st.columns(2)
            with c1:
                st.plotly_chart(make_gauge(soh,color),use_container_width=True)
            with c2:
                st.plotly_chart(make_prob_chart(probs),use_container_width=True)

            st.markdown(f"""<div class='pred-box {css}'><div class='ptitle'>FINAL USABILITY STATUS</div>
            <div class='pval' style='color:{color}'>{emoji} {usability.upper()}</div>
            <div class='psub'>The battery module is graded as {usability} based on LSTM sequence analysis.</div></div>""", unsafe_allow_html=True)

            st.markdown("<div class='sec'>◈ SMART RECOMMENDATIONS</div>",unsafe_allow_html=True)
            for icon,txt,clr in recs:
                st.markdown(f"<div class='rec-card' style='border-color:{clr}; background:rgba(0,0,0,0.2);'>{icon} {txt}</div>",unsafe_allow_html=True)

            report_data = generate_report(voltage,current,power,temperature,soh,usability,probs,recs)
            st.download_button("📥 DOWNLOAD FULL ANALYSIS REPORT", data=report_data, file_name=f"Battery_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.html", mime="text/html")
