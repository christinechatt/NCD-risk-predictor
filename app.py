import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NCD Risk Predictor — Nigerian Women",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a5276, #2e86c1);
        color: white; padding: 2rem; border-radius: 12px;
        margin-bottom: 1.5rem; text-align: center;
    }
    .main-header h1 { color: white; font-size: 1.8rem; margin-bottom: 0.3rem; }
    .main-header p  { color: #d6eaf8; font-size: 0.95rem; margin: 0; }

    .risk-low    { background:#1a5c30; border-left:6px solid #2ecc71;
                   padding:1.2rem; border-radius:8px; margin:1rem 0; color:#fff; }
    .risk-medium { background:#7a4a00; border-left:6px solid #f39c12;
                   padding:1.2rem; border-radius:8px; margin:1rem 0; color:#fff; }
    .risk-high   { background:#7b1a1a; border-left:6px solid #e74c3c;
                   padding:1.2rem; border-radius:8px; margin:1rem 0; color:#fff; }

    .metric-box {
        background:#0D2137; border-radius:10px;
        padding:1rem; text-align:center; border:1px solid #028090;
    }
    .metric-box h3 { font-size:1.6rem; margin:0; color:#02C39A; }
    .metric-box p  { font-size:0.8rem; color:#d6eaf8; margin:0; }

    .section-header {
        border-bottom:2px solid #2e86c1; padding-bottom:0.4rem;
        margin:1.5rem 0 1rem 0; color:#1a5276; font-weight:bold;
    }
    .disclaimer {
        background:#eaf2ff; border:1px solid #2e86c1;
        border-radius:8px; padding:0.8rem 1rem;
        font-size:0.82rem; color:#1a5276; margin-top:1rem;
    }
    .footer {
        text-align:center; color:#aaa; font-size:0.78rem;
        margin-top:2rem; border-top:1px solid #eee; padding-top:1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD ARTEFACTS
# ─────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    model  = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    with open('feature_columns.json') as f:
        config = json.load(f)
    with open('display_feature_names.json') as f:
        name_cfg = json.load(f)
    shap_explainer = pickle.load(open('shap_explainer.pkl', 'rb'))
    training_data  = np.load('lime_training_data.npy')
    from lime import lime_tabular
    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data=training_data,
        feature_names=name_cfg['display_names'],
        class_names=['Low Risk', 'Medium Risk', 'High Risk'],
        mode='classification',
        random_state=42
    )
    return (model, scaler, config['features'],
            name_cfg['display_names'], name_cfg['name_map'],
            shap_explainer, lime_explainer)

artefacts_loaded = False
try:
    (model, scaler, feature_columns, display_names,
     name_map, shap_explainer, lime_explainer) = load_artefacts()
    artefacts_loaded = True
except Exception as e:
    load_error = str(e)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🩺 NCD Risk Prediction System for Nigerian Women</h1>
    <p>An Explainable Machine Learning System | Miva Open University</p>
    <p>Christiana Chatt Richards &middot; 2023/A/DSC/0001 &middot; Supervisor: Mr. Obaje Williams Usman</p>
</div>
""", unsafe_allow_html=True)

if not artefacts_loaded:
    st.error(f"Model files could not be loaded. Ensure all .pkl and .json files "
             f"are in the same folder as app.py.\n\nError: {load_error}")
    st.stop()

# ─────────────────────────────────────────────
# SIDEBAR — INPUT FORM
# ─────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/000000/caduceus.png", width=60)
st.sidebar.title("Lifestyle Assessment")
st.sidebar.markdown("Complete all sections, then click **Predict Risk**.")

freq_map      = {'Never':0,'Rarely':1,'1-2 times/week':2,'3-4 times/week':3,'Daily':4}
oil_map       = {'Never':0,'Rarely':1,'Sometimes':2,'Often':3,'Always':4}
exercise_map  = {'None (0 days)':0,'1-2 days':1,'3-4 days':2,'5+ days':3}
mins_map      = {'Less than 30 min':0,'30-60 min':1,'More than 60 min':2}
sitting_map   = {'Less than 3 hrs':0,'3-5 hrs':1,'6-8 hrs':2,'More than 8 hrs':3}
sleep_dur_map = {'Less than 5 hrs':0,'5-7 hrs':1,'7-9 hrs':2,'More than 9 hrs':3}
sleep_q_map   = {'Poor':0,'Fair':1,'Good':2,'Excellent':3}
stress_map    = {'Low':0,'Moderate':1,'High':2,'Very High':3}
overwhelm_map = {'Never':0,'Rarely':1,'Sometimes':2,'Often':3,'Always':4}
alcohol_map   = {'Never':0,'Occasionally':1,'Regularly':2,'Frequently':3}
yn_map        = {'No':0,'Yes':1}

with st.sidebar.expander("Body Measurements", expanded=True):
    weight = st.number_input("Weight (kg)", 30.0, 200.0, 65.0, 0.5)
    height = st.number_input("Height (cm)", 100.0, 220.0, 162.0, 0.5)
    bmi    = round(weight / ((height / 100) ** 2), 1)
    bmi_cat = ("Underweight" if bmi < 18.5 else
               "Normal"      if bmi < 25   else
               "Overweight"  if bmi < 30   else "Obese")
    st.metric("Calculated BMI", bmi, delta=bmi_cat)

with st.sidebar.expander("Diet and Nutrition"):
    fried     = st.selectbox("Fried food frequency",     list(freq_map))
    processed = st.selectbox("Processed food frequency", list(freq_map))
    sugary    = st.selectbox("Sugary drinks frequency",  list(freq_map))
    oil_reuse = st.selectbox("Cooking oil reuse",        list(oil_map))
    fruit     = st.slider("Fruit servings/day",     0, 5, 2)
    veg       = st.slider("Vegetable servings/day", 0, 5, 2)

with st.sidebar.expander("Physical Activity"):
    ex_days = st.selectbox("Exercise days/week", list(exercise_map))
    ex_mins = st.selectbox("Exercise duration",  list(mins_map))
    sitting = st.selectbox("Daily sitting time", list(sitting_map))
    sed_job = st.radio("Sedentary job?", ['No','Yes'], horizontal=True)

with st.sidebar.expander("Sleep"):
    sleep_dur = st.selectbox("Sleep duration",       list(sleep_dur_map))
    sleep_q   = st.selectbox("Sleep quality",        list(sleep_q_map))
    insomnia  = st.radio("Struggle with insomnia?",  ['No','Yes'], horizontal=True)

with st.sidebar.expander("Stress and Mental Health"):
    stress    = st.selectbox("Daily stress level",     list(stress_map))
    overwhelm = st.selectbox("How often overwhelmed?", list(overwhelm_map))

with st.sidebar.expander("Lifestyle Behaviours"):
    smoking    = st.radio("Do you smoke?",                 ['No','Yes'], horizontal=True)
    alcohol    = st.selectbox("Alcohol consumption",        list(alcohol_map))
    secondhand = st.radio("Exposed to secondhand smoke?",  ['No','Yes'], horizontal=True)

with st.sidebar.expander("Medical History"):
    fam_hist   = st.radio("Family history of NCD?",             ['No','Yes'], horizontal=True)
    diagnosed  = st.radio("Ever diagnosed with NCD?",           ['No','Yes'], horizontal=True)
    headaches  = st.radio("Frequent headaches?",                ['No','Yes'], horizontal=True)
    breath     = st.radio("Shortness of breath (mild activity)?",['No','Yes'], horizontal=True)
    medication = st.radio("On long-term medication?",           ['No','Yes'], horizontal=True)
    health     = st.slider("Self-rated health (1=Poor, 5=Excellent)", 1, 5, 3)

predict_btn = st.sidebar.button("Predict My Risk", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# BUILD INPUT VECTOR (only when button pressed)
# ─────────────────────────────────────────────
if predict_btn:
    input_data = {
        'bmi':                  bmi,
        'weight_kg':            weight,
        'fried_food_enc':       freq_map[fried],
        'processed_food_enc':   freq_map[processed],
        'sugary_drinks_enc':    freq_map[sugary],
        'oil_reuse_enc':        oil_map[oil_reuse],
        'fruit_servings':       fruit,
        'veg_servings':         veg,
        'exercise_days_enc':    exercise_map[ex_days],
        'exercise_minutes_enc': mins_map[ex_mins],
        'sitting_hours_enc':    sitting_map[sitting],
        'sedentary_job':        yn_map[sed_job],
        'sleep_duration_enc':   sleep_dur_map[sleep_dur],
        'sleep_quality_enc':    sleep_q_map[sleep_q],
        'insomnia':             yn_map[insomnia],
        'stress_level_enc':     stress_map[stress],
        'overwhelmed_enc':      overwhelm_map[overwhelm],
        'smoking':              yn_map[smoking],
        'alcohol_enc':          alcohol_map[alcohol],
        'secondhand_smoke':     yn_map[secondhand],
        'family_history_flag':  yn_map[fam_hist],
        'diagnosed_flag':       yn_map[diagnosed],
        'frequent_headaches':   yn_map[headaches],
        'shortness_of_breath':  yn_map[breath],
        'long_term_medication': yn_map[medication],
        'self_rated_health':    health,
    }

    input_df     = pd.DataFrame([input_data])[feature_columns]
    input_arr    = input_df.values
    input_scaled = scaler.transform(input_arr)

    prediction  = model.predict(input_scaled)[0]
    proba       = model.predict_proba(input_scaled)[0]
    label_map   = {0:'Low Risk', 1:'Medium Risk', 2:'High Risk'}
    risk_label  = label_map[prediction]
    confidence  = proba[prediction]

    st.session_state['input_arr']    = input_arr
    st.session_state['input_scaled'] = input_scaled
    st.session_state['input_df']     = input_df
    st.session_state['prediction']   = prediction
    st.session_state['proba']        = proba
    st.session_state['risk_label']   = risk_label
    st.session_state['bmi']          = bmi

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["About", "Risk Prediction", "Explanations", "Recommendations"])

# ── TAB 1: ABOUT ─────────────────────────────
with tab1:
    st.markdown('<p class="section-header">About This System</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Research Overview**

This tool was built as part of a final year Data Science project at Miva Open University.
It predicts the likelihood of non-communicable disease (NCD) risk, specifically hypertension,
type 2 diabetes, and obesity, among Nigerian women aged 18 to 55, using lifestyle information.

**Why This Matters**

Nigeria lacks localised, gender-specific predictive tools for NCD risk. Most existing models
are trained on Western populations and miss Nigeria-specific risk factors such as cooking oil
reuse and traditional dietary patterns.

**How It Works**

A short lifestyle questionnaire collects information across six domains: diet, physical
activity, sleep, stress, behavioural factors, and medical history. A trained Random Forest
model predicts whether the user falls into Low, Medium, or High NCD risk. Two Explainable AI
methods, SHAP and LIME, then show which specific factors drove the result.
        """)
    with col2:
        st.markdown("""
**Model Performance**

| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Random Forest (selected) | 82.93% | 0.9287 |
| SVM | 82.93% | 0.9164 |
| Gradient Boosting | 80.49% | 0.9007 |
| Logistic Regression | 78.05% | 0.8998 |

**Dataset**

203 Nigerian women completed the lifestyle survey covering diet, physical activity, sleep,
stress, smoking, alcohol, family health history, and general health status.

**Explanation Methods**

SHAP (SHapley Additive exPlanations) shows which features matter most globally and for each
individual prediction. LIME (Local Interpretable Model-agnostic Explanations) generates
simple condition-based rules explaining a single prediction.
        """)
    st.markdown('<div class="disclaimer">This system provides risk assessment only and does not constitute a clinical diagnosis. Results should be discussed with a qualified healthcare provider.</div>',
                unsafe_allow_html=True)

# ── TAB 2: PREDICTION ────────────────────────
with tab2:
    if 'prediction' not in st.session_state:
        st.info("Complete the lifestyle assessment in the sidebar and click Predict My Risk.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="risk-low"><h4>Low Risk</h4><p>Lifestyle patterns indicate a lower probability of developing NCDs. Preventive habits should be maintained.</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="risk-medium"><h4>Medium Risk</h4><p>Some lifestyle factors are elevating NCD risk. Targeted modifications in diet, activity, and stress are recommended.</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="risk-high"><h4>High Risk</h4><p>Multiple risk factors are present. Clinical consultation and immediate lifestyle intervention are strongly advised.</p></div>', unsafe_allow_html=True)
    else:
        risk_label = st.session_state['risk_label']
        proba      = st.session_state['proba']
        confidence = proba[st.session_state['prediction']]
        bmi_v      = st.session_state['bmi']

        risk_class = {'Low Risk':'risk-low','Medium Risk':'risk-medium','High Risk':'risk-high'}[risk_label]
        risk_emoji = {'Low Risk':'🟢','Medium Risk':'🟡','High Risk':'🔴'}[risk_label]

        st.markdown(
            f'<div class="{risk_class}"><h2>{risk_emoji} {risk_label}</h2>'
            f'<p>Model confidence: <strong>{confidence:.0%}</strong></p></div>',
            unsafe_allow_html=True)

        st.markdown('<p class="section-header">Probability Breakdown</p>', unsafe_allow_html=True)
        cols = st.columns(3)
        for col, lbl, prob in zip(cols, ['Low Risk','Medium Risk','High Risk'], proba):
            with col:
                st.markdown(f'<div class="metric-box"><h3>{prob:.0%}</h3><p>{lbl}</p></div>',
                            unsafe_allow_html=True)

        st.markdown('<p class="section-header">BMI Summary</p>', unsafe_allow_html=True)
        bmi_clr = ("#2ecc71" if bmi_v < 25 else "#f39c12" if bmi_v < 30 else "#e74c3c")
        bmi_cat = ("Normal" if bmi_v < 25 else "Overweight" if bmi_v < 30 else "Obese")
        st.markdown(f'**BMI: {bmi_v}** — <span style="color:{bmi_clr};font-weight:bold">{bmi_cat}</span>',
                    unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(8, 0.6))
        ax.barh(['BMI'], [bmi_v], color=bmi_clr, height=0.4)
        ax.axvline(18.5, color='gray',   linestyle='--', linewidth=0.8)
        ax.axvline(25,   color='orange', linestyle='--', linewidth=0.8)
        ax.axvline(30,   color='red',    linestyle='--', linewidth=0.8)
        ax.set_xlim(10, 50)
        ax.set_xlabel('BMI')
        ax.set_yticks([])
        ax.spines[['top','right','left']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.success("Prediction complete. Navigate to the Explanations tab to understand the key factors driving this result.")

# ── TAB 3: EXPLANATIONS ──────────────────────
with tab3:
    if 'prediction' not in st.session_state:
        st.info("Run a prediction first using the sidebar.")
    else:
        pred   = st.session_state['prediction']
        inp    = st.session_state['input_arr']
        rlabel = st.session_state['risk_label']

        st.markdown(f'<p class="section-header">Explaining the {rlabel} Prediction</p>',
                    unsafe_allow_html=True)

        xai_tab1, xai_tab2 = st.tabs(["SHAP Explanation", "LIME Explanation"])

        # ── SHAP ──
        with xai_tab1:
            st.markdown("**SHAP** quantifies each lifestyle factor's contribution using game-theory principles. Red bars increase risk; green bars decrease it.")
            try:
                input_scaled = st.session_state['input_scaled']
                shap_values  = shap_explainer.shap_values(input_scaled)

                if isinstance(shap_values, list):
                    sv = np.array(shap_values[pred][0]).flatten()
                elif len(np.array(shap_values).shape) == 3:
                    sv = np.array(shap_values)[0, pred, :]
                else:
                    sv = np.array(shap_values[0]).flatten()

                colors     = ['#e74c3c' if float(v) > 0 else '#2ecc71' for v in sv]
                sorted_idx = np.argsort(np.abs(sv))[::-1][:10]

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(
                    [display_names[i] for i in sorted_idx],
                    [sv[i] for i in sorted_idx],
                    color=[colors[i] for i in sorted_idx]
                )
                ax.axvline(0, color='black', linewidth=0.8)
                ax.set_xlabel("SHAP Value (impact on prediction)")
                ax.set_title("Top 10 Features Driving This Prediction")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            except Exception as e:
                st.warning("SHAP explanation unavailable. Showing feature importance instead.")
                importances = model.feature_importances_
                sorted_idx  = np.argsort(importances)[::-1][:10]
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(
                    [display_names[i] for i in sorted_idx],
                    [importances[i] for i in sorted_idx],
                    color='#028090'
                )
                ax.set_xlabel("Feature Importance Score")
                ax.set_title("Top 10 Most Important Features (Feature Importance Fallback)")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # ── LIME ──
        with xai_tab2:
            st.markdown("**LIME** approximates the model locally around this prediction and expresses each factor's contribution as a simple rule.")
            try:
                exp      = lime_explainer.explain_instance(
                               inp[0], model.predict_proba,
                               num_features=10, labels=[pred])
                exp_list = exp.as_list(label=pred)
                feats    = [e[0] for e in exp_list]
                vals     = [e[1] for e in exp_list]
                colors   = ['#e74c3c' if v > 0 else '#3498db' for v in vals]

                fig, ax = plt.subplots(figsize=(9, 5))
                ax.barh(feats[::-1], vals[::-1], color=colors[::-1], edgecolor='white')
                ax.axvline(0, color='black', linewidth=0.8)
                ax.set_title(f'LIME Feature Contributions — {rlabel}', fontweight='bold')
                ax.set_xlabel('LIME Coefficient (positive = increases risk)')
                red_p  = mpatches.Patch(color='#e74c3c', label='Increases risk')
                blue_p = mpatches.Patch(color='#3498db', label='Decreases risk')
                ax.legend(handles=[red_p, blue_p])
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            except Exception as e:
                st.warning(f"LIME explanation could not be generated: {e}")

# ── TAB 4: RECOMMENDATIONS ───────────────────
with tab4:
    if 'risk_label' not in st.session_state:
        st.info("Run a prediction first using the sidebar.")
    else:
        rlabel = st.session_state['risk_label']
        bmi_v  = st.session_state['bmi']

        st.markdown(f'<p class="section-header">Personalised Recommendations — {rlabel}</p>',
                    unsafe_allow_html=True)

        if rlabel == 'Low Risk':
            st.markdown('<div class="risk-low">', unsafe_allow_html=True)
            st.markdown("""
### Current lifestyle patterns are largely protective. Keep it up.

**Maintain these habits:**
- Continue regular physical activity (at least 150 minutes per week)
- Maintain a balanced diet rich in fruits, vegetables, and whole grains
- Prioritise 7 to 9 hours of quality sleep consistently
- Keep stress manageable through rest and social support

**Prevention reminders:**
- Schedule an NCD screening at least once every two years
- Monitor blood pressure and blood sugar annually if aged 35 and above
- Limit cooking oil reuse to reduce chronic inflammation risk
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        elif rlabel == 'Medium Risk':
            st.markdown('<div class="risk-medium">', unsafe_allow_html=True)
            st.markdown("""
### Several modifiable risk factors have been identified. Action is recommended.

**Diet:**
- Reduce fried and processed food to no more than twice per week
- Increase fruit and vegetable intake to at least 5 servings per day
- Stop reusing cooking oil — use fresh oil each time or switch to smaller quantities

**Physical Activity:**
- Aim for at least 30 minutes of moderate activity on 5 days per week
- Reduce continuous sitting — take a short walk every hour

**Sleep:**
- Target 7 to 9 hours of uninterrupted sleep per night
- Establish a consistent bedtime routine

**Stress:**
- Identify and address primary stressors
- Consider prayer, deep breathing, or light exercise for stress relief

**Clinical actions:**
- Schedule an NCD screening within the next 6 months
- Discuss family history findings with a healthcare provider
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown('<div class="risk-high">', unsafe_allow_html=True)
            st.markdown("""
### Multiple high-impact risk factors are present. Immediate action is advised.

**Urgent clinical actions:**
- Consult a healthcare provider as soon as possible for blood pressure and blood sugar testing
- If already diagnosed, ensure medication and treatment plan are current
- Request a comprehensive NCD risk assessment from your doctor

**Diet (high priority):**
- Eliminate daily fried and processed food immediately
- Switch to boiling, steaming, or grilling as cooking methods
- Increase water intake and eliminate sugary drinks

**Physical Activity:**
- Begin with 15 to 20 minutes of walking daily and increase gradually
- Set hourly movement reminders to avoid prolonged sitting

**Weight Management:**
- Work with a nutritionist on a safe weight reduction plan if BMI is 30 or above
- A 5 to 10 percent weight loss significantly reduces NCD risk

**Sleep and Stress:**
- Prioritise sleep as a health intervention — poor sleep worsens all NCD risk factors
- Seek support for chronic stress management

**Support resources:**
- National Health Insurance Authority (NHIA) facilities across Nigeria
- Federal Ministry of Health NCD prevention clinics
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        if bmi_v >= 25:
            st.warning(f"BMI of {bmi_v} is above the healthy range. Weight management is an important component of NCD risk reduction.")

        st.markdown('<div class="disclaimer">These recommendations are generated based on lifestyle data provided and are for informational purposes only. They do not replace clinical diagnosis or medical advice from a qualified healthcare professional.</div>',
                    unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    An Explainable Machine Learning System for Predicting Lifestyle-Based Health Risks Among Nigerian Women<br>
    Christiana Chatt Richards &middot; Department of Data Science &middot; Miva Open University &middot; 2026<br>
    <em>For academic demonstration purposes only</em>
</div>
""", unsafe_allow_html=True)
