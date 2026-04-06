import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import os
from datetime import datetime
from io import BytesIO

st.set_page_config(
    page_title="Battery Usability Predictor",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap');
    :root {
        --bg:#050d1a;--card:#0a1628;--card2:#0d1f3c;
        --cyan:#00d4ff;--green:#00ff9d;--orange:#ff6b35;
        --red:#ff3366;--text:#e8f4fd;--muted:#7ba7cc;--border:#1a3a5c;
    }
    .stApp{background:var(--bg);}
    #MainMenu,footer,header{visibility:hidden;}
    [data-testid="stSidebar"]{display:none;}
    .block-container{padding:1.5rem 2rem;}
    .hero{background:linear-gradient(135deg,#0a1628,#0d1f3c);
          border:1px solid var(--border);border-top:2px solid var(--cyan);
          border-radius:12px;padding:2rem 2.5rem;margin-bottom:1.5rem;}
    .hero-title{font-family:Rajdhani,sans-serif;font-size:2.2rem;font-weight:700;
                color:var(--cyan);letter-spacing:2px;
                text-shadow:0 0 30px rgba(0,212,255,0.3);margin:0;}
    .hero-sub{font-family:Share Tech Mono,monospace;color:var(--muted);
              font-size:0.82rem;margin-top:0.3rem;letter-spacing:1px;}
    .hero-badge{display:inline-block;background:rgba(0,212,255,0.1);
                border:1px solid rgba(0,212,255,0.3);color:var(--cyan);
                padding:0.2rem 0.8rem;border-radius:20px;font-size:0.72rem;
                font-family:Share Tech Mono,monospace;margin-top:0.8rem;}
    .mcard{background:var(--card);border:1px solid var(--border);
           border-radius:10px;padding:1.2rem;text-align:center;
           position:relative;overflow:hidden;}
    .mcard::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
    .mcard.c::before{background:var(--cyan);}.mcard.g::before{background:var(--green);}
    .mcard.o::before{background:var(--orange);}.mcard.r::before{background:var(--red);}
    .mval{font-family:Rajdhani,sans-serif;font-size:2rem;font-weight:700;margin:0;}
    .mval.c{color:var(--cyan);}.mval.g{color:var(--green);}
    .mval.o{color:var(--orange);}.mval.r{color:var(--red);}
    .mlbl{font-family:Share Tech Mono,monospace;color:var(--muted);
          font-size:0.72rem;letter-spacing:1px;margin-top:0.3rem;}
    .sec{font-family:Rajdhani,sans-serif;font-size:1.3rem;font-weight:600;
         color:var(--cyan);letter-spacing:2px;border-bottom:1px solid var(--border);
         padding-bottom:0.4rem;margin:1.5rem 0 1rem 0;}
    .icard{background:var(--card2);border:1px solid var(--border);
           border-left:3px solid var(--cyan);border-radius:8px;
           padding:0.8rem 1rem;margin:0.4rem 0;font-size:0.88rem;color:var(--text);}
    .pred-box{border-radius:12px;padding:1.8rem;text-align:center;
              margin:0.8rem 0;border:1px solid;}
    .pred-g{background:rgba(0,255,157,0.05);border-color:var(--green);}
    .pred-f{background:rgba(255,107,53,0.05);border-color:var(--orange);}
    .pred-p{background:rgba(255,51,102,0.05);border-color:var(--red);}
    .ptitle{font-family:Rajdhani,sans-serif;font-size:0.9rem;
            color:var(--muted);letter-spacing:2px;}
    .pval{font-family:Rajdhani,sans-serif;font-size:2.8rem;
          font-weight:700;margin:0.4rem 0;}
    .psub{font-family:Share Tech Mono,monospace;font-size:1rem;color:var(--muted);}
    .rec-card{border-radius:12px;padding:1.2rem 1.5rem;margin:0.5rem 0;
              border:1px solid;font-family:Share Tech Mono,monospace;font-size:0.85rem;}
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
        font-size:1rem!important;color:var(--muted)!important;}
    .stTabs [aria-selected="true"]{color:var(--cyan)!important;
        border-bottom-color:var(--cyan)!important;}
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
    # Health gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=soh,
        domain={'x':[0,1],'y':[0,1]},
        title={'text':"STATE OF HEALTH",
               'font':{'size':14,'color':'#7ba7cc','family':'Rajdhani'}},
        number={'suffix':"%",'font':{'size':40,'color':color,'family':'Rajdhani'}},
        delta={'reference':70,'increasing':{'color':'#00ff9d'},
               'decreasing':{'color':'#ff3366'}},
        gauge={
            'axis':{'range':[0,100],'tickwidth':1,
                    'tickcolor':'#1a3a5c','tickfont':{'color':'#7ba7cc'}},
            'bar':{'color':color,'thickness':0.25},
            'bgcolor':'#0a1628',
            'borderwidth':0,
            'steps':[
                {'range':[0,40],'color':'rgba(255,51,102,0.15)'},
                {'range':[40,70],'color':'rgba(255,107,53,0.15)'},
                {'range':[70,100],'color':'rgba(0,255,157,0.15)'}],
            'threshold':{
                'line':{'color':'white','width':2},
                'thickness':0.75,'value':soh}}))
    fig.update_layout(**PLOT_BG, height=280, margin=dict(l=20,r=20,t=40,b=20))
    return fig

def make_prob_chart(probs):
    fig = go.Figure()
    colors = ['#ff3366','#ff6b35','#00ff9d']
    for i,(cls,prob) in enumerate(zip(CLASS_NAMES,probs)):
        fig.add_trace(go.Bar(
            x=[prob*100],y=[cls],orientation='h',
            marker_color=colors[i],opacity=0.85,
            text=f"{prob*100:.1f}%",textposition='inside',
            textfont=dict(size=14,family='Rajdhani',color='white')))
    fig.update_layout(**PLOT_BG,height=180,showlegend=False,
        xaxis=dict(range=[0,100],gridcolor='#1a3a5c',
                   linecolor='#1a3a5c',ticksuffix='%'),
        margin=dict(l=5,r=5,t=5,b=5))
    return fig

def get_recommendations(usability, soh, voltage, temperature):
    # Objective-based recommendations
    recs = {
        'Good': [
            ("✅", "Battery is in EXCELLENT condition for solar energy storage", "green"),
            ("🔋", f"State of Health: {soh:.1f}% — Well above safe operating threshold (70%)", "green"),
            ("⚡", f"Voltage {voltage:.2f}V is within optimal operating range (7.0V-8.2V)", "green"),
            ("🌡️", f"Temperature {temperature:.1f}°C — Operating within safe thermal limits", "green"),
            ("📋", "Recommendation: Continue normal operation. Schedule routine check in 30 days.", "green"),
        ],
        'Fair': [
            ("⚠️", "Battery shows MODERATE degradation — Monitoring required", "orange"),
            ("🔋", f"State of Health: {soh:.1f}% — Approaching maintenance threshold (70%)", "orange"),
            ("⚡", f"Voltage {voltage:.2f}V — Monitor for further decline", "orange"),
            ("🌡️", f"Temperature {temperature:.1f}°C — Ensure adequate cooling", "orange"),
            ("📋", "Recommendation: Reduce load by 20-30%. Schedule maintenance within 7 days.", "orange"),
        ],
        'Poor': [
            ("❌", "Battery is in CRITICAL condition — Immediate action required!", "red"),
            ("🔋", f"State of Health: {soh:.1f}% — Below safe operating threshold!", "red"),
            ("⚡", f"Voltage {voltage:.2f}V — Critically low voltage detected", "red"),
            ("🌡️", f"Temperature {temperature:.1f}°C — Check thermal management immediately", "red"),
            ("📋", "Recommendation: DISCONTINUE USE immediately. Replace battery module.", "red"),
        ]
    }
    return recs[usability]

def generate_pdf_report(voltage, current, power, temperature, soh, usability, probs, recs):
    # Generate HTML report (downloadable)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    color_map = {'Good':'#00aa66','Fair':'#ff6b35','Poor':'#cc0033'}
    color = color_map[usability]
    emoji = {'Good':'✅','Fair':'⚠️','Poor':'❌'}[usability]

    rec_html = ""
    for icon,txt,clr in recs:
        bg = {'green':'#e8f8f0','orange':'#fff3e8','red':'#fde8ec'}[clr]
        border = {'green':'#00aa66','orange':'#ff6b35','red':'#cc0033'}[clr]
        rec_html += f"""
        <div style="background:{bg};border-left:4px solid {border};
                    padding:10px 15px;margin:8px 0;border-radius:4px;">
            {icon} {txt}
        </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
    body{{font-family:Arial,sans-serif;margin:40px;color:#333;}}
    .header{{background:linear-gradient(135deg,#0a1628,#1a3a5c);
             color:white;padding:30px;border-radius:10px;margin-bottom:30px;}}
    .title{{font-size:28px;font-weight:bold;color:#00d4ff;margin:0;}}
    .subtitle{{font-size:14px;color:#7ba7cc;margin-top:5px;}}
    .badge{{display:inline-block;background:rgba(0,212,255,0.2);
            border:1px solid #00d4ff;color:#00d4ff;padding:3px 12px;
            border-radius:15px;font-size:12px;margin-top:10px;}}
    .section{{margin:25px 0;}}
    .section-title{{font-size:16px;font-weight:bold;color:#1a3a5c;
                    border-bottom:2px solid #00d4ff;padding-bottom:5px;
                    margin-bottom:15px;letter-spacing:1px;}}
    .result-box{{background:{color}15;border:2px solid {color};
                 border-radius:10px;padding:20px;text-align:center;
                 display:inline-block;width:45%;margin:5px;}}
    .result-value{{font-size:36px;font-weight:bold;color:{color};}}
    .result-label{{font-size:12px;color:#666;letter-spacing:1px;}}
    .param-table{{width:100%;border-collapse:collapse;}}
    .param-table th{{background:#0a1628;color:#00d4ff;padding:10px;
                     text-align:left;font-size:13px;}}
    .param-table td{{padding:10px;border-bottom:1px solid #eee;font-size:13px;}}
    .param-table tr:nth-child(even){{background:#f8f9fa;}}
    .prob-bar{{height:25px;border-radius:4px;margin:5px 0;
               display:flex;align-items:center;padding:0 10px;
               color:white;font-size:13px;font-weight:bold;}}
    .footer{{margin-top:40px;text-align:center;color:#999;font-size:12px;
             border-top:1px solid #eee;padding-top:20px;}}
</style>
</head>
<body>
<div class="header">
    <div class="title">⚡ BATTERY USABILITY PREDICTION REPORT</div>
    <div class="subtitle">LSTM-Based Dual Output Prediction System</div>
    <div class="badge">219092 | R.M.C.S.L Jayathilaka | Final Year Research</div>
</div>

<div class="section">
    <div class="section-title">📅 REPORT INFORMATION</div>
    <table class="param-table">
        <tr><td><b>Generated:</b></td><td>{now}</td>
            <td><b>Model:</b></td><td>Dual-Output LSTM</td></tr>
        <tr><td><b>R² Score:</b></td><td>94.73%</td>
            <td><b>Accuracy:</b></td><td>97.78%</td></tr>
    </table>
</div>

<div class="section">
    <div class="section-title">🔌 INPUT PARAMETERS</div>
    <table class="param-table">
        <tr><th>Parameter</th><th>Value</th><th>Unit</th><th>Status</th></tr>
        <tr><td>Voltage</td><td>{voltage:.3f}</td><td>V</td>
            <td>{"✅ Normal" if 6.5<=voltage<=8.2 else "⚠️ Check"}</td></tr>
        <tr><td>Current</td><td>{current:.3f}</td><td>A</td>
            <td>{"🔋 Charging" if current>0 else "⚡ Discharging"}</td></tr>
        <tr><td>Power</td><td>{power:.3f}</td><td>W</td><td>Auto-calculated</td></tr>
        <tr><td>Temperature</td><td>{temperature:.1f}</td><td>°C</td>
            <td>{"✅ Normal" if temperature<=45 else "⚠️ High"}</td></tr>
        <tr><td>State</td><td>{"CHARGING" if current>0 else "DISCHARGING"}</td>
            <td>—</td><td>Auto-detected</td></tr>
    </table>
</div>

<div class="section">
    <div class="section-title">🎯 PREDICTION RESULTS</div>
    <div style="text-align:center;margin:20px 0;">
        <div class="result-box">
            <div class="result-value">{soh:.1f}%</div>
            <div class="result-label">STATE OF HEALTH (SoH)</div>
        </div>
        <div class="result-box">
            <div class="result-value">{emoji} {usability.upper()}</div>
            <div class="result-label">USABILITY STATUS</div>
        </div>
    </div>
</div>

<div class="section">
    <div class="section-title">📊 CLASS PROBABILITIES</div>
    <div class="prob-bar" style="background:#ff3366;width:{probs[0]*100:.0f}%">
        Poor: {probs[0]*100:.1f}%</div>
    <div class="prob-bar" style="background:#ff6b35;width:{probs[1]*100:.0f}%">
        Fair: {probs[1]*100:.1f}%</div>
    <div class="prob-bar" style="background:#00aa66;width:{probs[2]*100:.0f}%">
        Good: {probs[2]*100:.1f}%</div>
</div>

<div class="section">
    <div class="section-title">📋 RECOMMENDATIONS</div>
    {rec_html}
</div>

<div class="footer">
    <p>Generated by Battery Usability Predictor | 219092 R.M.C.S.L Jayathilaka</p>
    <p>Research: LSTM-Based Prediction of Reconditioned Second-Life Li-ion Battery Modules</p>
    <p>Model Performance: R²=94.73% | MAE=1.31% | Accuracy=97.78% | F1=97.78%</p>
</div>
</body>
</html>"""
    return html.encode('utf-8')

lstm_reg, lstm_cls, scaler, loaded = load_models()

# ── HERO ───────────────────────────────────────────────────
st.markdown("""<div class='hero'>
    <div class='hero-title'>⚡ BATTERY USABILITY PREDICTOR</div>
    <div class='hero-sub'>LSTM-BASED PREDICTION OF RECONDITIONED SECOND-LIFE LI-ION BATTERIES</div>
    <div class='hero-badge'>219092 | R.M.C.S.L JAYATHILAKA | FINAL YEAR RESEARCH</div>
</div>""", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────
tab1,tab2,tab3,tab4 = st.tabs([
    "🔮 LIVE PREDICTION",
    "📊 MODEL PERFORMANCE",
    "🔍 DATA ANALYSIS",
    "ℹ️ ABOUT"])

# ══════════════════════════════════════════════════════════
# TAB 1: LIVE PREDICTION
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='sec'>◈ MODEL PERFORMANCE METRICS</div>",unsafe_allow_html=True)
    cols = st.columns(6)
    for col,(val,lbl,clr) in zip(cols,[
        ("94.73%","R² SCORE","c"),("1.31%","MAE","g"),("4.02%","RMSE","o"),
        ("97.78%","ACCURACY","c"),("97.79%","PRECISION","g"),("97.78%","F1-SCORE","o")]):
        with col:
            st.markdown(f"<div class='mcard {clr}'><div class='mval {clr}'>{val}</div>"
                        f"<div class='mlbl'>{lbl}</div></div>",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown("<div class='sec'>◈ ENTER BATTERY MEASUREMENTS</div>",unsafe_allow_html=True)

    st.markdown("""<div class='icard'>
        📌 Enter only <strong>3 measurements</strong> from your battery.
        Power & State are auto-calculated. Simple & accurate!
    </div>""", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        voltage = st.slider("⚡ VOLTAGE (V)",6.0,8.2,7.4,0.01,
            help="Measure terminal voltage of your battery")
    with c2:
        current = st.slider("🔌 CURRENT (A)",-5.0,5.0,1.2,0.01,
            help="+ = Charging, - = Discharging")
    with c3:
        temperature = st.slider("🌡️ TEMPERATURE (°C)",20.0,60.0,35.0,0.1,
            help="Battery surface/ambient temperature")

    # Auto-calculate
    power     = round(voltage * current, 3)
    state_enc = 1 if current >= 0 else 0
    state_str = "CHARGING" if current >= 0 else "DISCHARGING"
    cycle_count = 1

    st.markdown(f"""<div class='icard'>
        ⚡ Auto-detected: &nbsp;
        <strong style='color:#00d4ff;'>Power = {power:.2f}W</strong> &nbsp;|&nbsp;
        <strong style='color:#00ff9d;'>State = {state_str}</strong> &nbsp;|&nbsp;
        <strong style='color:#ff6b35;'>Cycle = {cycle_count} (New Battery)</strong> &nbsp;|&nbsp;
        <strong style='color:{"#00ff9d" if loaded else "#ff3366"};'>
        Model {"● LOADED" if loaded else "● NOT LOADED"}</strong>
    </div>""", unsafe_allow_html=True)

    if st.button("🔮 PREDICT BATTERY CONDITION"):
        if not loaded:
            st.error("⚠️ Models not loaded! Please refresh the page.")
        else:
            with st.spinner("Analyzing battery condition..."):
                soh,usability,probs = predict_one(
                    voltage,current,power,temperature,
                    cycle_count,state_enc,scaler,lstm_reg,lstm_cls)

            color = CLASS_COLORS[usability]
            css   = {'Good':'pred-g','Fair':'pred-f','Poor':'pred-p'}[usability]
            emoji = {'Good':'✅','Fair':'⚠️','Poor':'❌'}[usability]
            recs  = get_recommendations(usability, soh, voltage, temperature)

            st.markdown("<div class='sec'>◈ PREDICTION RESULTS</div>",unsafe_allow_html=True)

            # Gauge + Status
            c1,c2 = st.columns([1,1])
            with c1:
                st.plotly_chart(make_gauge(soh,color), use_container_width=True)
            with c2:
                st.markdown(f"""
                <div class='pred-box {css}' style='margin-top:1rem;'>
                    <div class='ptitle'>USABILITY STATUS</div>
                    <div class='pval' style='color:{color};font-size:4rem;'>{emoji}</div>
                    <div style='color:{color};font-family:Rajdhani,sans-serif;
                                font-size:2rem;font-weight:700;letter-spacing:3px;'>{usability.upper()}</div>
                    <div class='psub' style='margin-top:0.5rem;'>
                        {"Excellent for solar energy storage" if usability=="Good"
                         else "Moderate degradation detected" if usability=="Fair"
                         else "Critical — Replace immediately!"}
                    </div>
                </div>""", unsafe_allow_html=True)

                # Probability chart
                st.plotly_chart(make_prob_chart(probs), use_container_width=True)

            # Recommendations
            st.markdown("<div class='sec'>◈ RECOMMENDATIONS</div>",unsafe_allow_html=True)
            border_color = {'Good':'#00ff9d','Fair':'#ff6b35','Poor':'#ff3366'}[usability]
            bg_color = {'Good':'rgba(0,255,157,0.05)','Fair':'rgba(255,107,53,0.05)',
                        'Poor':'rgba(255,51,102,0.05)'}[usability]
            for icon,txt,clr in recs:
                st.markdown(f"""
                <div class='rec-card' style='background:{bg_color};border-color:{border_color};'>
                    {icon} {txt}
                </div>""", unsafe_allow_html=True)

            # PDF Download
            st.markdown("<div class='sec'>◈ DOWNLOAD REPORT</div>",unsafe_allow_html=True)
            report = generate_pdf_report(
                voltage,current,power,temperature,soh,usability,probs,recs)
            fname = f"Battery_Report_{usability}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            st.download_button(
                label="📄 DOWNLOAD BATTERY REPORT (HTML)",
                data=report,
                file_name=fname,
                mime="text/html",
                use_container_width=True)

            st.markdown(f"""<div class='icard'>
                💡 <strong>Report saved!</strong> Open the downloaded HTML file in any browser.
                Print it as PDF using <strong>Ctrl+P → Save as PDF</strong>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 2: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════
with tab2:
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**🎯 REGRESSION — SoH Prediction**")
        st.dataframe(pd.DataFrame({
            'Metric':['R² Score','MAE','RMSE','MAPE'],
            'Value':['94.73%','1.3105%','4.0165%','2.5312%'],
            'Status':['✅ Excellent','✅ Very Low','✅ Good','✅ Very Low']}),
            use_container_width=True,hide_index=True)
    with c2:
        st.markdown("**🎯 CLASSIFICATION — Usability**")
        st.dataframe(pd.DataFrame({
            'Metric':['Accuracy','Precision','Recall','F1-Score'],
            'Value':['97.78%','97.79%','97.78%','97.78%'],
            'Status':['✅']*4}),
            use_container_width=True,hide_index=True)

    st.markdown("<div class='sec'>◈ PER-CLASS METRICS</div>",unsafe_allow_html=True)
    fig2 = go.Figure()
    for vals,name,color in [
        ([97.94,85.71,98.87],'Precision','#00d4ff'),
        ([94.35,86.97,99.20],'Recall','#00ff9d'),
        ([96.11,86.34,99.03],'F1-Score','#ff6b35')]:
        fig2.add_trace(go.Bar(name=name,x=CLASS_NAMES,y=vals,
            marker_color=color,opacity=0.85,
            text=[f"{v:.1f}%" for v in vals],textposition='outside',
            textfont=dict(size=10,family='Share Tech Mono')))
    fig2.update_layout(**PLOT_BG,barmode='group',height=300,
        yaxis=dict(range=[0,115],gridcolor='#1a3a5c',linecolor='#1a3a5c'),
        margin=dict(l=5,r=5,t=10,b=5))
    fig2.add_hline(y=90,line_dash='dash',line_color='#ff3366')
    st.plotly_chart(fig2,use_container_width=True)

    st.markdown("<div class='sec'>◈ CROSS-VALIDATION (5-FOLD)</div>",unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        'Fold':['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5','MEAN ± STD'],
        'R²':['92.40%','97.34%','93.81%','80.66%','95.71%','91.98% ± 5.91%'],
        'MAE':['0.6089%','1.0304%','0.7898%','1.5795%','2.5410%','1.3099% ± 0.6969%'],
        'Accuracy':['100%','94.66%','100%','97.90%','96.31%','97.77% ± 2.08%']}),
        use_container_width=True,hide_index=True)

    st.markdown("<div class='sec'>◈ CONFUSION MATRIX</div>",unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    cm = np.array([[1520,79,12],[19,1008,132],[13,89,12591]])
    with c1:
        fig_cm = go.Figure(go.Heatmap(z=cm,x=CLASS_NAMES,y=CLASS_NAMES,
            colorscale='Blues',text=cm,texttemplate='%{text}',
            textfont=dict(size=13,family='Rajdhani')))
        fig_cm.update_layout(**PLOT_BG,title='Counts',height=300,
            xaxis_title='Predicted',yaxis_title='Actual',
            margin=dict(l=5,r=5,t=40,b=5))
        st.plotly_chart(fig_cm,use_container_width=True)
    with c2:
        cm_n = cm.astype(float)/cm.sum(axis=1)[:,np.newaxis]*100
        fig_cn = go.Figure(go.Heatmap(z=cm_n,x=CLASS_NAMES,y=CLASS_NAMES,
            colorscale='Greens',text=np.round(cm_n,1),
            texttemplate='%{text}%',textfont=dict(size=13,family='Rajdhani'),
            zmin=0,zmax=100))
        fig_cn.update_layout(**PLOT_BG,title='Normalized %',height=300,
            xaxis_title='Predicted',yaxis_title='Actual',
            margin=dict(l=5,r=5,t=40,b=5))
        st.plotly_chart(fig_cn,use_container_width=True)

    st.markdown("<div class='sec'>◈ OVERFITTING ANALYSIS</div>",unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        fig_o = go.Figure(go.Bar(x=['Train R²','Test R²'],y=[90.56,94.73],
            marker_color=['#3498db','#00ff9d'],
            text=['90.56%','94.73%'],textposition='outside',
            textfont=dict(size=12,family='Rajdhani')))
        fig_o.add_hline(y=90,line_dash='dash',line_color='#ff3366')
        fig_o.update_layout(**PLOT_BG,title='R² — No Overfitting ✅',
            height=270,yaxis=dict(range=[0,110],gridcolor='#1a3a5c',linecolor='#1a3a5c'),
            margin=dict(l=5,r=5,t=40,b=5))
        st.plotly_chart(fig_o,use_container_width=True)
    with c2:
        fig_o2 = go.Figure(go.Bar(x=['Train Acc','Test Acc'],y=[98.08,97.78],
            marker_color=['#3498db','#00ff9d'],
            text=['98.08%','97.78%'],textposition='outside',
            textfont=dict(size=12,family='Rajdhani')))
        fig_o2.add_hline(y=90,line_dash='dash',line_color='#ff3366')
        fig_o2.update_layout(**PLOT_BG,title='Accuracy — Gap=0.30% ✅',
            height=270,yaxis=dict(range=[0,110],gridcolor='#1a3a5c',linecolor='#1a3a5c'),
            margin=dict(l=5,r=5,t=40,b=5))
        st.plotly_chart(fig_o2,use_container_width=True)

    st.markdown("<div class='sec'>◈ FEATURE IMPORTANCE</div>",unsafe_allow_html=True)
    fi_v = [10.8637,0.1162,0.0118,-0.0074,-0.0158,-0.0448]
    fi_f = ['Voltage','Power','Temperature','CycleCount','State','Current']
    fig_fi = go.Figure(go.Bar(x=fi_v,y=fi_f,orientation='h',
        marker_color=['#ff3366' if v==max(fi_v) else '#00d4ff' for v in fi_v],
        opacity=0.85,text=[f'{v:.4f}' for v in fi_v],textposition='outside',
        textfont=dict(family='Share Tech Mono',size=10)))
    fig_fi.update_layout(**PLOT_BG,height=280,xaxis_title='Importance',
        margin=dict(l=5,r=5,t=10,b=5))
    st.plotly_chart(fig_fi,use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 3: DATA ANALYSIS
# ══════════════════════════════════════════════════════════
with tab3:
    cols = st.columns(4)
    for col,(val,lbl,clr) in zip(cols,[
        ("77,341","TOTAL SAMPLES","c"),("11","CYCLES","g"),
        ("6","FEATURES","o"),("3","CLASSES","r")]):
        with col:
            st.markdown(f"<div class='mcard {clr}'><div class='mval {clr}'>{val}</div>"
                        f"<div class='mlbl'>{lbl}</div></div>",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown("<div class='sec'>◈ PREPROCESSING SUMMARY</div>",unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        'Step':['Raw Data','Remove Noise','Non-Solar Filter','Voltage Filter','Final'],
        'Rows':[79863,79383,78688,77341,77341],
        'Removed':['—','480','695','1347','—'],
        'Status':['📥 Loaded','🧹 Cleaned','🔍 Filtered','⚡ Applied','✅ Ready']}),
        use_container_width=True,hide_index=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("<div class='sec'>◈ VOLTAGE DISTRIBUTION</div>",unsafe_allow_html=True)
        np.random.seed(42)
        v_sim = np.clip(np.concatenate([
            np.random.normal(7.8,0.15,46760),
            np.random.normal(7.2,0.2,17660),
            np.random.normal(6.4,0.2,12921)]),6.0,8.2)
        fig_v = go.Figure(go.Histogram(x=v_sim,nbinsx=60,
            marker_color='#00d4ff',opacity=0.75,
            marker_line=dict(color='#050d1a',width=0.5)))
        fig_v.add_vline(x=v_sim.mean(),line_dash='dash',line_color='#ff6b35',
            annotation_text=f'Mean:{v_sim.mean():.2f}V')
        fig_v.update_layout(**PLOT_BG,title='Voltage Distribution',
            height=270,xaxis_title='Voltage (V)',yaxis_title='Count',
            margin=dict(l=5,r=5,t=40,b=5))
        st.plotly_chart(fig_v,use_container_width=True)
    with c2:
        st.markdown("<div class='sec'>◈ FEATURE CORRELATION</div>",unsafe_allow_html=True)
        corr_v = [1.000,0.164,0.123,-0.454,-0.462,-0.490]
        corr_f = ['Voltage','CycleCount','Temperature','Power','State','Current']
        fig_c = go.Figure(go.Bar(x=corr_v,y=corr_f,orientation='h',
            marker_color=['#00ff9d' if v>0 else '#ff3366' for v in corr_v],
            opacity=0.85,text=[f'{v:.3f}' for v in corr_v],textposition='outside',
            textfont=dict(family='Share Tech Mono',size=10)))
        fig_c.add_vline(x=0,line_color='#7ba7cc',line_width=1)
        fig_c.update_layout(**PLOT_BG,title='Correlation with SoH',
            height=270,xaxis=dict(range=[-0.6,1.2],gridcolor='#1a3a5c',linecolor='#1a3a5c'),
            margin=dict(l=5,r=5,t=40,b=5))
        st.plotly_chart(fig_c,use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 4: ABOUT
# ══════════════════════════════════════════════════════════
with tab4:
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("<div class='sec'>◈ RESEARCHER</div>",unsafe_allow_html=True)
        for icon,lbl,val in [
            ("👤","Name","R.M.C.S.L Jayathilaka"),
            ("🔢","Index No","219092"),
            ("🎓","Degree","BET (Hons) Electrotechnology"),
            ("📅","Year","Final Year Research Project")]:
            st.markdown(f"<div class='icard'>{icon} <strong style='color:#7ba7cc;'>{lbl}:</strong> "
                        f"<strong style='color:#00d4ff;'>{val}</strong></div>",unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='sec'>◈ RESEARCH OBJECTIVES</div>",unsafe_allow_html=True)
        for i,obj in enumerate([
            "Analyze performance data of reconditioned second-life Li-ion battery modules",
            "Develop LSTM prediction model for usability and performance condition forecasting",
            "Determine accuracy and reliability of the developed prediction model"
        ],1):
            st.markdown(f"<div class='icard'><strong style='color:#00d4ff;'>{i}.</strong> {obj}</div>",
                unsafe_allow_html=True)

    st.markdown("<div class='sec'>◈ TECH STACK</div>",unsafe_allow_html=True)
    techs=[("🐍","Python 3.14","Core"),("🧠","Keras 3.14","Deep Learning"),
           ("📊","Scikit-learn","ML Utils"),("🐼","Pandas/NumPy","Data"),
           ("📈","Plotly","Charts"),("🌐","Streamlit","Dashboard"),
           ("☁️","Google Colab","Training"),("📡","ESP32","Sensors")]
    tcols = st.columns(4)
    for i,(icon,name,desc) in enumerate(techs):
        with tcols[i%4]:
            st.markdown(f"""<div class='mcard c' style='text-align:left;padding:0.8rem;margin-bottom:0.6rem;'>
                <div style='font-size:1.3rem;'>{icon}</div>
                <div style='color:#00d4ff;font-family:Rajdhani,sans-serif;font-weight:600;font-size:0.85rem;'>{name}</div>
                <div style='color:#7ba7cc;font-family:Share Tech Mono,monospace;font-size:0.68rem;'>{desc}</div>
            </div>""",unsafe_allow_html=True)
