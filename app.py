import streamlit as st
import numpy as np
import joblib
import time

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    color: #e2e8f0;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }

/* Main container */
.main-header {
    text-align: center;
    padding: 2rem 0 1rem;
}

.main-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}

.main-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 300;
    letter-spacing: 0.02em;
}

/* Card style */
.glass-card {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1.2rem;
}

.section-title {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Sliders */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #38bdf8, #818cf8) !important;
}

/* Input labels */
label {
    color: #cbd5e1 !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
}

/* Number inputs */
.stNumberInput input {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
}

/* Result box */
.result-safe {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.1));
    border: 1px solid rgba(16, 185, 129, 0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}

.result-risk {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(185, 28, 28, 0.1));
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}

.result-title {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.result-subtitle {
    font-size: 0.95rem;
    opacity: 0.8;
}

.prob-bar-container {
    background: rgba(15, 23, 42, 0.5);
    border-radius: 50px;
    height: 14px;
    margin: 1.2rem 0 0.4rem;
    overflow: hidden;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 50px;
    transition: width 0.6s ease;
}

/* Predict Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #38bdf8, #818cf8) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    margin-top: 0.5rem;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(56, 189, 248, 0.35) !important;
}

/* Info chips */
.info-chip {
    display: inline-block;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 50px;
    padding: 0.25rem 0.75rem;
    font-size: 0.78rem;
    color: #38bdf8;
    margin: 0.2rem;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid rgba(148, 163, 184, 0.1);
    margin: 1.5rem 0;
}

/* Select box */
.stSelectbox > div > div {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
}

/* Metric card */
.metric-mini {
    background: rgba(15, 23, 42, 0.4);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-mini-val {
    font-size: 1.4rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-mini-label {
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)

# ── Load Models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    model  = joblib.load("diabetes_model.pkl")
    return scaler, model

scaler, model = load_models()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="main-title">🩺 Diabetes Risk Predictor</div>
    <div class="main-subtitle">Enter your health parameters below to assess your diabetes risk using AI</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Layout ───────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    # ── Section 1: Basic Info ──
    st.markdown("""<div class="glass-card">
    <div class="section-title">📋 Basic Information</div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age (years)", 10, 100, 35, help="Your current age")
        pregnancies = st.slider("Pregnancies", 0, 20, 1, help="Number of times pregnant (0 if male)")
    with c2:
        bmi = st.slider("BMI", 10.0, 70.0, 25.0, step=0.1, help="Body Mass Index")

        # BMI category display
        if bmi < 18.5:
            bmi_label = "🔵 Underweight"
        elif bmi < 25:
            bmi_label = "🟢 Healthy"
        elif bmi < 30:
            bmi_label = "🟡 Overweight"
        else:
            bmi_label = "🔴 Obese"
        st.markdown(f"<div style='color:#94a3b8; font-size:0.82rem; margin-top:-0.5rem;'>BMI Category: <b>{bmi_label}</b></div>", unsafe_allow_html=True)

        dpf = st.number_input("Diabetes Pedigree Function", 0.00, 3.00, 0.30, step=0.01,
                               help="A function that scores likelihood of diabetes based on family history")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 2: Blood & Vitals ──
    st.markdown("""<div class="glass-card">
    <div class="section-title">🩸 Blood & Vitals</div>""", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        glucose = st.slider("Glucose Level (mg/dL)", 50, 250, 110,
                             help="Plasma glucose concentration (2-hour oral glucose tolerance test)")
        blood_pressure = st.slider("Blood Pressure (mmHg)", 30, 130, 72,
                                    help="Diastolic blood pressure")
    with c4:
        insulin = st.slider("Insulin (μU/mL)", 0, 900, 80,
                             help="2-Hour serum insulin level")
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20,
                                    help="Triceps skinfold thickness")

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    # ── Summary Panel ──
    st.markdown("""<div class="glass-card">
    <div class="section-title">📊 Your Inputs Summary</div>""", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""<div class="metric-mini">
            <div class="metric-mini-val">{age}</div>
            <div class="metric-mini-label">Age</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-mini">
            <div class="metric-mini-val">{bmi:.1f}</div>
            <div class="metric-mini-label">BMI</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-mini">
            <div class="metric-mini-val">{glucose}</div>
            <div class="metric-mini-label">Glucose</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    m4, m5, m6 = st.columns(3)
    with m4:
        st.markdown(f"""<div class="metric-mini">
            <div class="metric-mini-val">{blood_pressure}</div>
            <div class="metric-mini-label">BP</div></div>""", unsafe_allow_html=True)
    with m5:
        st.markdown(f"""<div class="metric-mini">
            <div class="metric-mini-val">{insulin}</div>
            <div class="metric-mini-label">Insulin</div></div>""", unsafe_allow_html=True)
    with m6:
        st.markdown(f"""<div class="metric-mini">
            <div class="metric-mini-val">{pregnancies}</div>
            <div class="metric-mini-label">Preg.</div></div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Predict Button ──
    predict_clicked = st.button("🔍 Predict My Diabetes Risk", use_container_width=True)

    # ── Result Panel ──
    if predict_clicked:
        with st.spinner("Analyzing your health data..."):
            time.sleep(0.8)

        # BMI encoding
        bmi_healthy    = 1 if 18.5 <= bmi < 25 else 0
        bmi_overweight = 1 if 25   <= bmi < 30 else 0
        bmi_obese      = 1 if bmi >= 30          else 0

        input_data = np.array([[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age,
            bmi_healthy, bmi_overweight, bmi_obese
        ]])

        scaled   = scaler.transform(input_data)
        pred     = model.predict(scaled)[0]
        proba    = model.predict_proba(scaled)[0]
        risk_pct = round(proba[1] * 100, 1)
        safe_pct = round(proba[0] * 100, 1)

        if pred == 0:
            bar_color = "linear-gradient(90deg, #10b981, #059669)"
            result_class = "result-safe"
            icon  = "✅"
            title = "Low Risk"
            desc  = f"Your profile suggests a <b>{safe_pct}%</b> probability of not having diabetes. Keep maintaining a healthy lifestyle!"
        else:
            bar_color = "linear-gradient(90deg, #f97316, #ef4444)"
            result_class = "result-risk"
            icon  = "⚠️"
            title = "Elevated Risk"
            desc  = f"Your profile suggests a <b>{risk_pct}%</b> probability of diabetes. Please consult a healthcare professional."

        st.markdown(f"""
        <div class="{result_class}">
            <div class="result-title">{icon} {title}</div>
            <div class="result-subtitle">{desc}</div>
            <div class="prob-bar-container">
                <div class="prob-bar-fill" style="width:{risk_pct}%; background:{bar_color};"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#94a3b8;">
                <span>No Diabetes: {safe_pct}%</span>
                <span>Diabetes Risk: {risk_pct}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk factors
        st.markdown("""<div class="glass-card">
        <div class="section-title">💡 Key Risk Indicators</div>""", unsafe_allow_html=True)

        alerts = []
        if glucose > 140:
            alerts.append("🔴 High Glucose")
        if bmi >= 30:
            alerts.append("🟠 Obese BMI")
        elif bmi >= 25:
            alerts.append("🟡 Overweight")
        if blood_pressure > 90:
            alerts.append("🔴 High BP")
        if insulin > 200:
            alerts.append("🟠 High Insulin")
        if dpf > 0.5:
            alerts.append("🔴 Family History")
        if age > 45:
            alerts.append("🟡 Age Factor")

        if not alerts:
            alerts.append("🟢 All metrics in healthy range")

        chips_html = "".join([f'<span class="info-chip">{a}</span>' for a in alerts])
        st.markdown(chips_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Placeholder before prediction
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding: 3rem 1.5rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🔬</div>
            <div style="color: #64748b; font-size: 0.92rem;">
                Fill in your health parameters on the left<br>and click <b style="color:#38bdf8;">Predict</b> to see your risk analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#475569; font-size:0.78rem; padding-bottom:1rem;">
    ⚕️ This tool is for informational purposes only and does not replace professional medical advice.<br>
    Model: Logistic Regression &nbsp;|&nbsp; Dataset: Pima Indians Diabetes
</div>
""", unsafe_allow_html=True)
