import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import os

st.set_page_config(
    page_title="Battery Usability Predictor",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
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
    [data-testid="stSidebar"]{background:#060e1c;border-right:1px solid var(--border);}
    [data-testid="stSidebarNav"]{display:none;}
    #MainMenu,footer,header{visibility:hidden;}
    .block-container{padding:1.5rem 2rem;}
    .hero{background:linear-gradient(135deg,#0a1628,#0d1f3c);border:1px solid var(--border);
          border-top:2px solid var(--cyan);border-radius:12px;padding:2rem 2.5rem;margin-bottom:1.5rem;}
    .hero-title{font-family:Rajdhani,sans-serif;font-size:2.2rem;font-weight:700;
                color:var(--cyan);letter-spacing:2px;text-shadow:0 0 30px rgba(0,212,255,0.3);margin:0;}
    .hero-sub{font-family:Share Tech Mono,monospace;color:var(--muted);font-size:0.82rem;
              margin-top:0.3rem;letter-spacing:1px;}
    .hero-badge{display:inline-block;background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.3);
                color:var(--cyan);padding:0.2rem 0.8rem;border-radius:20px;font-size:0.72rem;
                font-family:Share Tech Mono,monospace;margin-top:0.8rem;}
    .mcard{background:var(--card);border:1px solid var(--border);border-radius:10px;
           padding:1.2rem;text-align:center;position:relative;overflow:hidden;}
    .mcard::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
    .mcard.c::before{background:var(--cyan);}.mcard.g::before{background:var(--green);}
    .mcard.o::before{background:var(--orange);}.mcard.r::before{background:var(--red);}
    .mval{font-family:Rajdhani,sans-serif;font-size:2rem;font-weight:700;margin:0;}
    .mval.c{color:var(--cyan);}.mval.g{color:var(--green);}
    .mval.o{color:var(--orange);}.mval.r{color:var(--red);}
    .mlbl{font-family:Share Tech Mono,monospace;color:var(--muted);font-size:0.72rem;
          letter-spacing:1px;margin-top:0.3rem;}
    .sec{font-family:Rajdhani,sans-serif;font-size:1.2rem;font-weight:600;color:var(--cyan);
         letter-spacing:2px;border-bottom:1px solid var(--border);padding-bottom:0.4rem;
         margin:1.2rem 0 0.8rem 0;}
    .icard{background:var(--card2);border:1px solid var(--border);border-left:3px solid var(--cyan);
           border-radius:8px;padding:0.8rem 1rem;margin:0.4rem 0;font-size:0.88rem;color:var(--text);}
    .pred-box{border-radius:12px;padding:1.8rem;text-align:center;margin:0.8rem 0;border:1px solid;}
    .pred-g{background:rgba(0,255,157,0.05);border-color:var(--green);}
    .pred-f{background:rgba(255,107,53,0.05);border-color:var(--orange);}
    .pred-p{background:rgba(255,51,102,0.05);border-color:var(--red);}
    .ptitle{font-family:Rajdhani,sans-serif;font-size:0.9rem;color:var(--muted);letter-spacing:2px;}
    .pval{font-family:Rajdhani,sans-serif;font-size:2.8rem;font-weight:700;margin:0.4rem 0;}
    .psub{font-family:Share Tech Mono,monospace;font-size:1rem;color:var(--muted);}
    .stButton>button{background:linear-gradient(135deg,#00d4ff20,#00ff9d20)!important;
        border:1px solid var(--cyan)!important;color:var(--cyan)!important;
        font-family:Rajdhani,sans-serif!important;font-size:1rem!important;
        font-weight:600!important;letter-spacing:2px!important;
        padding:0.6rem 2rem!important;border-radius:6px!important;width:100%!important;}
    .stTabs [data-baseweb="tab"]{font-family:Rajdhani,sans-serif!important;
        font-size:1rem!important;color:var(--muted)!important;}
    .stTabs [aria-selected="true"]{color:var(--cyan)!important;
        border-bottom-color:var(--cyan)!important;}
    .nav-btn{background:var(--card);border:1px solid var(--border);border-radius:8px;
             padding:0.6rem 1rem;margin:0.3rem 0;width:100%;text-align:left;
             color:var(--text);font-family:Rajdhani,sans-serif;font-size:1rem;
             cursor:pointer;letter-spacing:1px;}
    p,li{color:var(--text)!important;}
    h1,h2,h3{color:var(--cyan)!important;font-family:Rajdhani,sans-serif!important;}
    label{color:var(--muted)!important;font-family:Share Tech Mono,monospace!important;
          font-size:0.78rem!important;letter-spacing:1px!important;}
    [data-testid="stSidebar"] *{color:var(--text)!important;}
    section[data-testid="stSidebar"] > div {min-width: 250px !important;}
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

lstm_reg, lstm_cls, scaler, loaded = load_models()

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0;'>
        <div style='font-size:2.5rem;'>🔋</div>
        <div style='font-family:Rajdhani,sans-serif;font-size:1.1rem;color:#00d4ff;
                    font-weight:700;letter-spacing:2px;'>BATTERY AI</div>
        <div style='font-family:Share Tech Mono,monospace;font-size:0.68rem;
                    color:#7ba7cc;margin-top:0.2rem;'>219092 | v1.0</div>
    </div>
    <hr style='border-color:#1a3a5c;'>
    <div style='font-family:Rajdhani,sans-serif;font-size:0.8rem;color:#7ba7cc;
                letter-spacing:1px;padding:0.3rem 0;'>NAVIGATION</div>
    """, unsafe_allow_html=True)

    # Navigation buttons
    pages = ["🏠 Dashboard","🔮 Live Prediction","📊 Model Performance",
             "🔍 Data Analysis","ℹ️ About"]

    if 'page' not in st.session_state:
        st.session_state.page = "🏠 Dashboard"

    for p in pages:
        if st.button(p, key=f"nav_{p}", use_container_width=True):
            st.session_state.page = p

    page = st.session_state.page

    st.markdown(f"""
    <hr style='border-color:#1a3a5c;'>
    <div style='font-family:Share Tech Mono,monospace;font-size:0.7rem;
                color:#7ba7cc;padding:0.5rem;'>
        <div style='color:#00d4ff;margin-bottom:0.4rem;'>MODEL STATUS</div>
        <div>⚡ LSTM Regression</div>
        <div>⚡ LSTM Classification</div>
        <div style='margin-top:0.4rem;
                    color:{"#00ff9d" if loaded else "#ff3366"};'>
            {"● LOADED" if loaded else "● NOT LOADED"}
        </div>
    </div>""", unsafe_allow_html=True)

# ── DASHBOARD ──────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.markdown("""<div class='hero'>
        <div class='hero-title'>⚡ BATTERY USABILITY PREDICTOR</div>
        <div class='hero-sub'>LSTM-BASED DUAL OUTPUT PREDICTION SYSTEM</div>
        <div class='hero-badge'>219092 | R.M.C.S.L JAYATHILAKA | FINAL YEAR RESEARCH</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec'>◈ MODEL PERFORMANCE METRICS</div>",unsafe_allow_html=True)
    cols = st.columns(6)
    for col,(val,lbl,clr) in zip(cols,[
        ("94.73%","R² SCORE","c"),("1.31%","MAE","g"),("4.02%","RMSE","o"),
        ("97.78%","ACCURACY","c"),("97.79%","PRECISION","g"),("97.78%","F1-SCORE","o")]):
        with col:
            st.markdown(f"<div class='mcard {clr}'><div class='mval {clr}'>{val}</div>"
                        f"<div class='mlbl'>{lbl}</div></div>",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("<div class='sec'>◈ RESEARCH OVERVIEW</div>",unsafe_allow_html=True)
        for icon,txt in [
            ("📡","Real battery data via ESP32 sensors"),
            ("📊","77,341 readings — 11 charge/discharge cycles"),
            ("🧠","Dual LSTM: SoH Regression + Usability Classification"),
            ("✅","No overfitting — Test R² > Train R²"),
            ("🔁","5-Fold temporal cross-validation performed")]:
            st.markdown(f"<div class='icard'>{icon} {txt}</div>",unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='sec'>◈ CLASS DISTRIBUTION</div>",unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=['Good (60.5%)','Fair (22.8%)','Poor (16.7%)'],
            values=[60.5,22.8,16.7], hole=0.6,
            marker=dict(colors=['#00ff9d','#ff6b35','#ff3366'],
                        line=dict(color='#050d1a',width=2)),
            textfont=dict(family='Share Tech Mono',size=10)))
        fig.update_layout(**PLOT_BG,height=270,showlegend=True,
            margin=dict(l=5,r=5,t=5,b=5),
            annotations=[dict(text='DATA',x=0.5,y=0.5,
                font=dict(size=11,color='#7ba7cc',family='Share Tech Mono'),
                showarrow=False)])
        st.plotly_chart(fig,use_container_width=True)

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

# ── LIVE PREDICTION ────────────────────────────────────────
elif page == "🔮 Live Prediction":
    st.markdown("""<div class='hero'>
        <div class='hero-title'>🔮 LIVE PREDICTION ENGINE</div>
        <div class='hero-sub'>REAL-TIME BATTERY SoH & USABILITY PREDICTION</div>
    </div>""",unsafe_allow_html=True)

    tab1,tab2 = st.tabs(["⚙️ MANUAL INPUT","📁 CSV UPLOAD"])

    with tab1:
        st.markdown("<div class='sec'>◈ BATTERY PARAMETERS</div>",unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            voltage     = st.slider("⚡ VOLTAGE (V)",6.0,8.2,7.4,0.01)
            current     = st.slider("🔌 CURRENT (A)",-5.0,5.0,1.2,0.01)
            power       = st.slider("💡 POWER (W)",-30.0,30.0,float(round(7.4*1.2,2)),0.01)
        with c2:
            temperature = st.slider("🌡️ TEMPERATURE (°C)",20.0,60.0,35.0,0.1)
            cycle_count = st.slider("🔄 CYCLE COUNT",1,20,5,1)
            state       = st.selectbox("🔋 STATE",['CHARGING','DISCHARGING'])
            state_enc   = 1 if state=="CHARGING" else 0

        soh_est = max(0,min(100,((voltage-V_MIN)/(V_MAX-V_MIN))*100))
        st.markdown(f"""<div class='icard'>
            💡 Voltage SoH estimate: <strong style='color:#00d4ff;'>{soh_est:.1f}%</strong>
            &nbsp;|&nbsp; State: <strong style='color:#00ff9d;'>{state}</strong>
            &nbsp;|&nbsp; Cycle: <strong style='color:#ff6b35;'>{cycle_count}</strong>
        </div>""",unsafe_allow_html=True)

        if st.button("🔮 PREDICT NOW"):
            if not loaded:
                st.error("⚠️ Models not loaded!")
            else:
                with st.spinner("Running LSTM prediction..."):
                    soh,usability,probs = predict_one(
                        voltage,current,power,temperature,
                        cycle_count,state_enc,scaler,lstm_reg,lstm_cls)
                css   = {'Good':'pred-g','Fair':'pred-f','Poor':'pred-p'}[usability]
                color = CLASS_COLORS[usability]
                emoji = {'Good':'✅','Fair':'⚠️','Poor':'❌'}[usability]
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown(f"""<div class='pred-box {css}'>
                        <div class='ptitle'>STATE OF HEALTH</div>
                        <div class='pval' style='color:{color};'>{soh:.1f}%</div>
                        <div class='psub'>SoH PREDICTION</div>
                    </div>""",unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div class='pred-box {css}'>
                        <div class='ptitle'>USABILITY STATUS</div>
                        <div class='pval' style='color:{color};'>{emoji}</div>
                        <div class='psub' style='color:{color};font-size:1.2rem;'>{usability.upper()}</div>
                    </div>""",unsafe_allow_html=True)

                st.markdown("<div class='sec'>◈ CLASS PROBABILITIES</div>",unsafe_allow_html=True)
                fig_p = go.Figure()
                for i,(cls,prob) in enumerate(zip(CLASS_NAMES,probs)):
                    fig_p.add_trace(go.Bar(
                        x=[prob*100],y=[cls],orientation='h',
                        marker_color=['#ff3366','#ff6b35','#00ff9d'][i],opacity=0.85,
                        text=f"{prob*100:.1f}%",textposition='inside',
                        textfont=dict(size=13,family='Rajdhani',color='white')))
                fig_p.update_layout(**PLOT_BG,height=180,showlegend=False,
                    xaxis=dict(range=[0,100],gridcolor='#1a3a5c',
                               linecolor='#1a3a5c',ticksuffix='%'),
                    margin=dict(l=5,r=5,t=5,b=5))
                st.plotly_chart(fig_p,use_container_width=True)

    with tab2:
        st.markdown("<div class='sec'>◈ CSV FILE UPLOAD</div>",unsafe_allow_html=True)
        st.markdown("""<div class='icard'>
            📋 Required: <strong>Voltage, Current, Power,
            Temperature, CycleCount, State_encoded</strong><br>
            State_encoded: 1=CHARGING, 0=DISCHARGING
        </div>""",unsafe_allow_html=True)
        uploaded = st.file_uploader("DROP CSV FILE",type=['csv'])
        if uploaded:
            df_up   = pd.read_csv(uploaded)
            missing = [f for f in FEATURES if f not in df_up.columns]
            if missing:
                st.error(f"❌ Missing: {missing}")
            else:
                st.success(f"✅ {len(df_up)} rows loaded!")
                if st.button("🔮 BATCH PREDICT") and loaded:
                    results = []
                    bar = st.progress(0)
                    X   = scaler.transform(df_up[FEATURES].values)
                    for i in range(len(X)):
                        seq = X[max(0,i-SEQUENCE_LENGTH+1):i+1]
                        if len(seq)<SEQUENCE_LENGTH:
                            seq = np.vstack([np.tile(seq[0],(SEQUENCE_LENGTH-len(seq),1)),seq])
                        seq  = seq.reshape(1,SEQUENCE_LENGTH,len(FEATURES))
                        soh  = float(np.clip(lstm_reg.predict(seq,verbose=0)[0][0],0,100))
                        prob = lstm_cls.predict(seq,verbose=0)[0]
                        results.append({'SoH':soh,'Usability':CLASS_NAMES[int(np.argmax(prob))]})
                        if i%max(1,len(X)//20)==0: bar.progress(i/len(X))
                    bar.progress(1.0)
                    res_df = pd.DataFrame(results)
                    df_out = pd.concat([df_up.reset_index(drop=True),res_df],axis=1)
                    c1,c2,c3 = st.columns(3)
                    for col,cls,clr in zip([c1,c2,c3],['Good','Fair','Poor'],['g','o','r']):
                        cnt = (res_df['Usability']==cls).sum()
                        with col:
                            st.markdown(f"<div class='mcard {clr}'>"
                                f"<div class='mval {clr}'>{cnt}</div>"
                                f"<div class='mlbl'>{cls.upper()}</div></div>",
                                unsafe_allow_html=True)
                    fig_t = go.Figure(go.Scatter(y=res_df['SoH'],mode='lines',
                        line=dict(color='#00d4ff',width=2),fill='tozeroy',
                        fillcolor='rgba(0,212,255,0.05)'))
                    fig_t.update_layout(**PLOT_BG,title='Predicted SoH Timeline',
                        height=280,margin=dict(l=5,r=5,t=40,b=5),
                        yaxis=dict(range=[0,105],gridcolor='#1a3a5c',linecolor='#1a3a5c'))
                    st.plotly_chart(fig_t,use_container_width=True)
                    st.download_button("📥 DOWNLOAD RESULTS",
                        df_out.to_csv(index=False),"predictions.csv",
                        "text/csv",use_container_width=True)

# ── MODEL PERFORMANCE ──────────────────────────────────────
elif page == "📊 Model Performance":
    st.markdown("""<div class='hero'>
        <div class='hero-title'>📊 MODEL PERFORMANCE</div>
        <div class='hero-sub'>COMPREHENSIVE EVALUATION & VALIDATION RESULTS</div>
    </div>""",unsafe_allow_html=True)

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

    c1,c2 = st.columns(2)
    with c1:
        fig_o = go.Figure(go.Bar(x=['Train R²','Test R²'],y=[90.56,94.73],
            marker_color=['#3498db','#00ff9d'],
            text=['90.56%','94.73%'],textposition='outside',
            textfont=dict(size=12,family='Rajdhani')))
        fig_o.add_hline(y=90,line_dash='dash',line_color='#ff3366')
        fig_o.update_layout(**PLOT_BG,title='R² No Overfitting ✅',
            height=270,yaxis=dict(range=[0,110],gridcolor='#1a3a5c',linecolor='#1a3a5c'),
            margin=dict(l=5,r=5,t=40,b=5))
        st.plotly_chart(fig_o,use_container_width=True)
    with c2:
        fig_o2 = go.Figure(go.Bar(x=['Train Acc','Test Acc'],y=[98.08,97.78],
            marker_color=['#3498db','#00ff9d'],
            text=['98.08%','97.78%'],textposition='outside',
            textfont=dict(size=12,family='Rajdhani')))
        fig_o2.add_hline(y=90,line_dash='dash',line_color='#ff3366')
        fig_o2.update_layout(**PLOT_BG,title='Accuracy Gap=0.30% ✅',
            height=270,yaxis=dict(range=[0,110],gridcolor='#1a3a5c',linecolor='#1a3a5c'),
            margin=dict(l=5,r=5,t=40,b=5))
        st.plotly_chart(fig_o2,use_container_width=True)

# ── DATA ANALYSIS ──────────────────────────────────────────
elif page == "🔍 Data Analysis":
    st.markdown("""<div class='hero'>
        <div class='hero-title'>🔍 DATA ANALYSIS</div>
        <div class='hero-sub'>EXPLORATORY DATA ANALYSIS & DATASET INSIGHTS</div>
    </div>""",unsafe_allow_html=True)

    cols = st.columns(4)
    for col,(val,lbl,clr) in zip(cols,[
        ("77,341","SAMPLES","c"),("11","CYCLES","g"),
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

# ── ABOUT ──────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown("""<div class='hero'>
        <div class='hero-title'>ℹ️ ABOUT THIS RESEARCH</div>
        <div class='hero-sub'>LSTM-BASED BATTERY USABILITY PREDICTION SYSTEM</div>
    </div>""",unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("<div class='sec'>◈ RESEARCHER</div>",unsafe_allow_html=True)
        for icon,lbl,val in [
            ("👤","Name","R.M.C.S.L Jayathilaka"),("🔢","Index","219092"),
            ("🎓","Degree","BSc (Hons) Information Technology"),
            ("📅","Year","Final Year Research Project")]:
            st.markdown(f"<div class='icard'>{icon} <strong style='color:#7ba7cc;'>{lbl}:</strong> "
                        f"<strong style='color:#00d4ff;'>{val}</strong></div>",unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='sec'>◈ RESEARCH SUMMARY</div>",unsafe_allow_html=True)
        for icon,lbl,txt in [
            ("🎯","Objective","Predict battery SoH & usability using LSTM"),
            ("📡","Data","Real-time ESP32 sensor — 77,341 readings"),
            ("🧠","Model","Dual LSTM (128→64→32), Huber + Softmax"),
            ("✅","Validation","5-fold temporal CV, no overfitting")]:
            st.markdown(f"<div class='icard'>{icon} <strong style='color:#7ba7cc;'>{lbl}:</strong> {txt}</div>",
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
