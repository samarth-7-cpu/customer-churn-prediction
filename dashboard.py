# pyright: ignore [reportMissingImports]
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import ast
from pathlib import Path

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
REPORTS = BASE / "reports"
FIGURES = REPORTS / "figures"
MODELS = BASE / "models"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-card: #111827;
    --bg-card-hover: #1a2236;
    --accent-blue: #3b82f6;
    --accent-purple: #8b5cf6;
    --accent-emerald: #10b981;
    --accent-rose: #f43f5e;
    --accent-amber: #f59e0b;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border: #1e293b;
    --gradient-1: linear-gradient(135deg, #3b82f6, #8b5cf6);
    --gradient-2: linear-gradient(135deg, #10b981, #3b82f6);
    --gradient-3: linear-gradient(135deg, #f43f5e, #f59e0b);
}

.stApp {
    font-family: 'Inter', sans-serif !important;
}

.hero-banner {
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 50%, #042f2e 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(99, 102, 241, 0.15);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    font-weight: 400;
    line-height: 1.6;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}
.kpi-card {
    background: linear-gradient(145deg, #111827, #1a1f35);
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.kpi-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.1);
}
.kpi-card .kpi-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
.kpi-card .kpi-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}
.kpi-card .kpi-label {
    font-size: 0.85rem;
    color: #64748b;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.3rem;
}

.section-header {
    font-size: 1.35rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 2rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.insight-card {
    background: linear-gradient(145deg, #111827, #161f32);
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    transition: all 0.3s ease;
}
.insight-card:hover { border-color: rgba(99, 102, 241, 0.3); }
.insight-card .insight-num {
    display: inline-block;
    background: var(--gradient-1);
    color: white;
    font-weight: 700;
    font-size: 0.75rem;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    margin-right: 0.5rem;
}
.insight-card .insight-text {
    color: #cbd5e1;
    font-size: 0.92rem;
    line-height: 1.6;
}

.prediction-result {
    border-radius: 20px;
    padding: 2rem 2.5rem;
    text-align: center;
    margin-top: 1.5rem;
    border: 1px solid;
}
.prediction-result.high-risk {
    background: linear-gradient(145deg, #1c0a0a, #2d1219);
    border-color: rgba(244, 63, 94, 0.3);
}
.prediction-result.low-risk {
    background: linear-gradient(145deg, #051e12, #0a2e1a);
    border-color: rgba(16, 185, 129, 0.3);
}
.prediction-result .pred-emoji { font-size: 3rem; margin-bottom: 0.5rem; }
.prediction-result .pred-label { font-size: 1.5rem; font-weight: 800; margin-bottom: 0.5rem; }
.prediction-result.high-risk .pred-label { color: #fb7185; }
.prediction-result.low-risk .pred-label { color: #34d399; }
.prediction-result .pred-prob { font-size: 3rem; font-weight: 800; letter-spacing: -0.03em; }
.prediction-result.high-risk .pred-prob {
    background: linear-gradient(135deg, #f43f5e, #fb923c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.prediction-result.low-risk .pred-prob {
    background: linear-gradient(135deg, #10b981, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.risk-factor {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(244, 63, 94, 0.1);
    border: 1px solid rgba(244, 63, 94, 0.2);
    border-radius: 999px; padding: 0.35rem 0.9rem;
    font-size: 0.82rem; color: #fda4af; margin: 0.25rem;
}
.safe-factor {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 999px; padding: 0.35rem 0.9rem;
    font-size: 0.82rem; color: #6ee7b7; margin: 0.25rem;
}

.winner-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(251, 146, 60, 0.1));
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 999px; padding: 0.4rem 1rem;
    font-size: 0.9rem; color: #fcd34d; font-weight: 600;
}

.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
.stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 0.5rem 1.5rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_train_data():
    X = pd.read_csv(DATA / "X_train.csv")
    y = pd.read_csv(DATA / "y_train.csv")
    return pd.concat([X, y], axis=1)


@st.cache_data
def load_test_data():
    X = pd.read_csv(DATA / "X_test.csv")
    y = pd.read_csv(DATA / "y_test.csv")
    return X, y


@st.cache_data
def load_model_comparison():
    path = REPORTS / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def load_model_and_pipeline():
    try:
        import joblib
        pipe = joblib.load(MODELS / "preprocessing_pipeline.joblib")
        model = joblib.load(MODELS / "final_model.joblib")
        return pipe, model
    except Exception:
        return None, None


CLR = {
    "primary": "#818cf8", "secondary": "#a78bfa",
    "emerald": "#10b981", "rose": "#f43f5e", "amber": "#f59e0b",
    "blue": "#3b82f6", "cyan": "#22d3ee",
    "bg": "#0f1729", "card": "#111827",
    "text": "#e2e8f0", "text_dim": "#64748b", "grid": "#1e293b",
}
PALETTE = ["#818cf8", "#f472b6", "#34d399", "#fbbf24", "#38bdf8", "#fb923c", "#a78bfa", "#f87171"]


def apply_layout(fig, title="", height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=CLR["text"], family="Inter"), x=0.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=CLR["text_dim"], size=12),
        height=height,
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", font=dict(color=CLR["text_dim"])),
        xaxis=dict(gridcolor=CLR["grid"], zerolinecolor=CLR["grid"]),
        yaxis=dict(gridcolor=CLR["grid"], zerolinecolor=CLR["grid"]),
    )
    return fig


# hero banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🔮 Customer Churn Predictor</div>
    <div class="hero-subtitle">
        AI-powered analytics dashboard for predicting and understanding bank customer churn.
        Built with machine learning models tuned on 10,000 customer records.
    </div>
</div>
""", unsafe_allow_html=True)

# sidebar
with st.sidebar:
    st.markdown("### ⚡ Navigation")
    st.markdown("---")
    st.markdown("**Team**")
    st.markdown("""
    - 🧑‍💻 **Samarth** — Pipeline Lead
    - 📊 **Kartik** — EDA Lead
    - 🤖 **Hitesh** — Modeling Lead
    """)
    st.markdown("---")
    st.markdown("**Tech Stack**")
    st.markdown("""
    `Python` · `scikit-learn` · `XGBoost`  
    `Streamlit` · `Plotly` · `Pandas`
    """)
    st.markdown("---")
    st.markdown(
        "<p style='color:#64748b; font-size:0.78rem; text-align:center;'>"
        "Customer Churn Prediction v1.0<br/>Bank Dataset • 10K Customers</p>",
        unsafe_allow_html=True,
    )

# tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview & EDA",
    "🎯 Individual Predictor",
    "📦 Batch Analysis",
    "🏆 Model Comparison",
])

# ── TAB 1: Overview & EDA ──
with tab1:
    df_train = load_train_data()
    total = len(df_train)
    churned = df_train["Exited"].sum()
    stayed = total - churned
    churn_pct = churned / total * 100

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-icon">👥</div>
            <div class="kpi-value">{total:,}</div>
            <div class="kpi-label">Total Customers</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">🚪</div>
            <div class="kpi-value">{churned:,}</div>
            <div class="kpi-label">Churned</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">✅</div>
            <div class="kpi-value">{stayed:,}</div>
            <div class="kpi-label">Retained</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">📈</div>
            <div class="kpi-value">{churn_pct:.1f}%</div>
            <div class="kpi-label">Churn Rate</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 Churn Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=["Stayed", "Churned"], y=[stayed, churned],
            marker=dict(color=[CLR["emerald"], CLR["rose"]], line=dict(width=0)),
            text=[stayed, churned], textposition="outside",
            textfont=dict(color=CLR["text"], size=14, family="Inter"),
        ))
        apply_layout(fig_bar, "Customer Count by Status", height=380)
        fig_bar.update_layout(xaxis_title="", yaxis_title="Count", bargap=0.4)
        st.plotly_chart(fig_bar, width='stretch')

    with col2:
        fig_pie = go.Figure()
        fig_pie.add_trace(go.Pie(
            labels=["Stayed", "Churned"], values=[stayed, churned],
            marker=dict(colors=[CLR["emerald"], CLR["rose"]]),
            hole=0.55, textinfo="label+percent",
            textfont=dict(size=13, family="Inter"),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
        ))
        apply_layout(fig_pie, "Churn Rate Breakdown", height=380)
        st.plotly_chart(fig_pie, width='stretch')

    st.markdown('<div class="section-header">🔍 Churn by Category</div>', unsafe_allow_html=True)
    cat_feat = st.selectbox(
        "Select feature to analyze:",
        ["Geography", "Gender", "NumOfProducts", "HasCrCard", "IsActiveMember", "Card Type"],
        key="eda_cat_select",
    )

    grouped = df_train.groupby(cat_feat)["Exited"].agg(["sum", "count"]).reset_index()
    grouped.columns = [cat_feat, "Churned", "Total"]
    grouped["Churn Rate (%)"] = (grouped["Churned"] / grouped["Total"] * 100).round(1)
    grouped["Stayed"] = grouped["Total"] - grouped["Churned"]

    col_a, col_b = st.columns(2)
    with col_a:
        fig_cat = go.Figure()
        fig_cat.add_trace(go.Bar(
            x=grouped[cat_feat].astype(str), y=grouped["Stayed"],
            name="Stayed", marker_color=CLR["emerald"],
            text=grouped["Stayed"], textposition="inside",
        ))
        fig_cat.add_trace(go.Bar(
            x=grouped[cat_feat].astype(str), y=grouped["Churned"],
            name="Churned", marker_color=CLR["rose"],
            text=grouped["Churned"], textposition="inside",
        ))
        apply_layout(fig_cat, f"Churn Count by {cat_feat}", height=400)
        fig_cat.update_layout(barmode="stack", xaxis_title=cat_feat, yaxis_title="Count")
        st.plotly_chart(fig_cat, width='stretch')

    with col_b:
        fig_rate = go.Figure()
        fig_rate.add_trace(go.Bar(
            x=grouped[cat_feat].astype(str), y=grouped["Churn Rate (%)"],
            marker=dict(
                color=grouped["Churn Rate (%)"],
                colorscale=[[0, CLR["emerald"]], [0.5, CLR["amber"]], [1, CLR["rose"]]],
                line=dict(width=0),
            ),
            text=grouped["Churn Rate (%)"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            textfont=dict(color=CLR["text"], size=13),
        ))
        apply_layout(fig_rate, f"Churn Rate by {cat_feat}", height=400)
        fig_rate.update_layout(xaxis_title=cat_feat, yaxis_title="Churn Rate (%)")
        fig_rate.add_hline(
            y=churn_pct, line_dash="dash", line_color=CLR["text_dim"],
            annotation_text=f"Avg: {churn_pct:.1f}%",
            annotation_font=dict(color=CLR["text_dim"], size=11),
        )
        st.plotly_chart(fig_rate, width='stretch')

    st.markdown('<div class="section-header">📈 Age vs Churn</div>', unsafe_allow_html=True)
    col_age1, col_age2 = st.columns(2)

    with col_age1:
        fig_age = go.Figure()
        for label, color, val in [("Stayed", CLR["emerald"], 0), ("Churned", CLR["rose"], 1)]:
            subset = df_train[df_train["Exited"] == val]["Age"]
            fig_age.add_trace(go.Histogram(x=subset, name=label, marker_color=color, opacity=0.7, nbinsx=30))
        apply_layout(fig_age, "Age Distribution by Churn", height=400)
        fig_age.update_layout(barmode="overlay", xaxis_title="Age", yaxis_title="Count")
        st.plotly_chart(fig_age, width='stretch')

    with col_age2:
        df_train["_age_bucket"] = pd.cut(
            df_train["Age"], bins=[0, 30, 40, 50, 60, 100], labels=["18-30", "31-40", "41-50", "51-60", "61+"],
        )
        age_churn = df_train.groupby("_age_bucket", observed=True)["Exited"].mean().reset_index()
        age_churn.columns = ["Age Group", "Churn Rate"]
        age_churn["Churn Rate"] = (age_churn["Churn Rate"] * 100).round(1)

        fig_age_bar = go.Figure()
        fig_age_bar.add_trace(go.Bar(
            x=age_churn["Age Group"], y=age_churn["Churn Rate"],
            marker=dict(
                color=age_churn["Churn Rate"],
                colorscale=[[0, CLR["emerald"]], [0.5, CLR["amber"]], [1, CLR["rose"]]],
            ),
            text=age_churn["Churn Rate"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            textfont=dict(color=CLR["text"], size=13),
        ))
        apply_layout(fig_age_bar, "Churn Rate by Age Group", height=400)
        fig_age_bar.update_layout(xaxis_title="Age Group", yaxis_title="Churn Rate (%)")
        fig_age_bar.add_hline(
            y=churn_pct, line_dash="dash", line_color=CLR["text_dim"],
            annotation_text=f"Avg: {churn_pct:.1f}%",
            annotation_font=dict(color=CLR["text_dim"], size=11),
        )
        st.plotly_chart(fig_age_bar, width='stretch')
        df_train.drop(columns=["_age_bucket"], inplace=True, errors="ignore")

    st.markdown('<div class="section-header">🔗 Feature Correlations with Churn</div>', unsafe_allow_html=True)
    num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    corr_churn = df_train[num_cols].corr()["Exited"].drop("Exited").sort_values()

    fig_corr = go.Figure()
    fig_corr.add_trace(go.Bar(
        y=corr_churn.index, x=corr_churn.values, orientation="h",
        marker=dict(
            color=corr_churn.values,
            colorscale=[[0, CLR["emerald"]], [0.5, "#334155"], [1, CLR["rose"]]],
            cmid=0,
        ),
        text=corr_churn.values.round(3), textposition="outside",
        textfont=dict(color=CLR["text_dim"], size=11),
    ))
    apply_layout(fig_corr, "Feature Correlation with Churn (Exited)", height=450)
    fig_corr.update_layout(xaxis_title="Pearson Correlation", yaxis_title="")
    fig_corr.add_vline(x=0, line_color=CLR["text_dim"], line_dash="dash")
    st.plotly_chart(fig_corr, width='stretch')

    st.markdown('<div class="section-header">💡 Key EDA Insights</div>', unsafe_allow_html=True)
    insights = [
        ("Class Imbalance", "Only 20.4% of customers churned — models must use F1 score and class-weight balancing."),
        ("Age = #1 Predictor", "Customers aged 40-60 churn at dramatically higher rates. Age has the highest positive correlation with churn."),
        ("Germany Effect", "German customers churn at ~33%, roughly double France (16%) and Spain (17%). Targeted retention needed."),
        ("Multi-Product Risk", "3-4 product customers churn at 83-100%, but they are rare. 2-product customers are safest (7.6% churn)."),
        ("Inactive Members", "Inactive members churn at 27% vs 15% for active. Engagement programs could reduce churn significantly."),
        ("Balance Paradox", "Non-zero balance customers churn more (24%) than zero-balance (14%). Female customers churn more across all geographies."),
    ]
    for i, (title, text) in enumerate(insights, 1):
        st.markdown(f"""
        <div class="insight-card">
            <span class="insight-num">{i}</span>
            <strong style="color:#e2e8f0;">{title}:</strong>
            <span class="insight-text">{text}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🖼️ EDA Visualizations Gallery</div>', unsafe_allow_html=True)
    figures = sorted(FIGURES.glob("*.png")) if FIGURES.exists() else []
    if figures:
        fig_names = {f.stem: f for f in figures}
        selected = st.selectbox(
            "Select visualization:", list(fig_names.keys()),
            format_func=lambda x: x.replace("_", " ").title(), key="eda_fig_select",
        )
        st.image(str(fig_names[selected]), width='stretch')
    else:
        st.info("No EDA figures found. Run `python src/eda_churn.py` to generate them.")


# ── TAB 2: Individual Predictor ──
with tab2:
    st.markdown('<div class="section-header">🎯 Individual Customer Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94a3b8; margin-bottom:1.5rem;'>"
        "Enter customer details below to predict their churn probability. "
        "The model uses a Random Forest classifier (F1 = 0.635) with engineered features.</p>",
        unsafe_allow_html=True,
    )

    pipe, model = load_model_and_pipeline()
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("##### 📋 Customer Profile")
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            credit_score = st.slider("Credit Score", 350, 850, 650, key="pred_cs")
            age = st.slider("Age", 18, 92, 38, key="pred_age")
            tenure = st.slider("Tenure (years)", 0, 10, 5, key="pred_tenure")
            satisfaction = st.slider("Satisfaction Score", 1, 5, 3, key="pred_sat")
        with r1c2:
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"], key="pred_geo")
            gender = st.selectbox("Gender", ["Male", "Female"], key="pred_gender")
            card_type = st.selectbox("Card Type", ["SILVER", "GOLD", "PLATINUM", "DIAMOND"], key="pred_card")
            num_products = st.selectbox("Num of Products", [1, 2, 3, 4], key="pred_np")
        with r1c3:
            balance = st.number_input("Balance ($)", 0.0, 300000.0, 50000.0, step=1000.0, key="pred_bal")
            salary = st.number_input("Est. Salary ($)", 0.0, 250000.0, 100000.0, step=5000.0, key="pred_sal")
            has_card = st.radio("Has Credit Card?", ["Yes", "No"], key="pred_cc", horizontal=True)
            is_active = st.radio("Active Member?", ["Yes", "No"], key="pred_active", horizontal=True)
            points = st.number_input("Points Earned", 0, 1200, 500, step=50, key="pred_pts")

    input_df = pd.DataFrame([{
        "CreditScore": credit_score, "Geography": geography, "Gender": gender,
        "Age": age, "Tenure": tenure, "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": salary, "Satisfaction Score": satisfaction,
        "Card Type": card_type, "Point Earned": points,
    }])

    with col_right:
        if pipe is not None and model is not None:
            X_proc = pipe.transform(input_df)
            proba = model.predict_proba(X_proc)[0]
            churn_prob = proba[1] * 100
        else:
            # heuristic fallback when models aren't available
            churn_prob = 20.0
            if 40 <= age <= 60: churn_prob += 18
            elif age > 60: churn_prob += 12
            if geography == "Germany": churn_prob += 14
            if gender == "Female": churn_prob += 5
            if num_products >= 3: churn_prob += 30
            if is_active == "No": churn_prob += 8
            if balance > 100000: churn_prob += 5
            elif balance == 0: churn_prob -= 5
            churn_prob = max(2, min(98, churn_prob))

        high_risk = churn_prob >= 50
        risk_cls = "high-risk" if high_risk else "low-risk"
        emoji = "⚠️" if high_risk else "✅"
        label = "HIGH RISK" if high_risk else "LOW RISK"

        st.markdown(f"""
        <div class="prediction-result {risk_cls}">
            <div class="pred-emoji">{emoji}</div>
            <div class="pred-label">{label}</div>
            <div class="pred-prob">{churn_prob:.1f}%</div>
            <p style="color:#94a3b8; font-size:0.9rem; margin-top:0.5rem;">Churn Probability</p>
        </div>
        """, unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=churn_prob,
            number=dict(suffix="%", font=dict(size=28, color=CLR["text"])),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=CLR["text_dim"]),
                bar=dict(color=CLR["rose"] if high_risk else CLR["emerald"]),
                bgcolor=CLR["card"], bordercolor=CLR["grid"],
                steps=[
                    dict(range=[0, 30], color="rgba(16,185,129,0.15)"),
                    dict(range=[30, 60], color="rgba(245,158,11,0.15)"),
                    dict(range=[60, 100], color="rgba(244,63,94,0.15)"),
                ],
                threshold=dict(line=dict(color=CLR["text"], width=2), value=churn_prob),
            ),
        ))
        apply_layout(fig_gauge, "", height=250)
        fig_gauge.update_layout(margin=dict(l=30, r=30, t=20, b=20))
        st.plotly_chart(fig_gauge, width='stretch')

        risks = []
        safes = []
        if age >= 40: risks.append("🔴 Age ≥ 40")
        else: safes.append("🟢 Young age")
        if geography == "Germany": risks.append("🔴 Germany")
        else: safes.append("🟢 Non-Germany")
        if num_products >= 3: risks.append("🔴 3+ Products")
        if is_active == "No": risks.append("🔴 Inactive")
        else: safes.append("🟢 Active member")
        if balance > 100000: risks.append("🔴 High balance")
        if gender == "Female": risks.append("🟡 Female")

        if risks:
            st.markdown("**Risk Factors:**")
            st.markdown(" ".join(f'<span class="risk-factor">{f}</span>' for f in risks), unsafe_allow_html=True)
        if safes:
            st.markdown("**Protective Factors:**")
            st.markdown(" ".join(f'<span class="safe-factor">{f}</span>' for f in safes), unsafe_allow_html=True)

        if pipe is None:
            st.warning(
                "⚠️ Models not found. Using heuristic estimates. "
                "Run `python src/preprocessing_pipeline.py --verify` and "
                "`python src/model_training.py --verify` to train models."
            )


# ── TAB 3: Batch Analysis ──
with tab3:
    st.markdown('<div class="section-header">📦 Batch Prediction & Segmentation</div>', unsafe_allow_html=True)

    source = st.radio("Data source:", ["Use test dataset", "Upload CSV"], horizontal=True, key="batch_source")

    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload customer CSV", type=["csv"], key="batch_upload")
        batch_df = pd.read_csv(uploaded) if uploaded else None
    else:
        X_test, y_test = load_test_data()
        batch_df = X_test.copy()
        batch_df["Exited_Actual"] = y_test["Exited"].values

    if batch_df is not None:
        st.markdown(f"**Dataset:** {len(batch_df):,} rows × {len(batch_df.columns)} columns")

        pipe, model = load_model_and_pipeline()
        feat_cols = [
            "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
            "Satisfaction Score", "Card Type", "Point Earned",
        ]
        has_feats = all(c in batch_df.columns for c in feat_cols)

        if has_feats and pipe is not None and model is not None:
            X_batch = batch_df[feat_cols]
            X_proc = pipe.transform(X_batch)
            probas = model.predict_proba(X_proc)[:, 1]
            batch_df["Churn_Probability"] = (probas * 100).round(2)
            batch_df["Risk_Level"] = pd.cut(
                batch_df["Churn_Probability"], bins=[0, 30, 60, 100], labels=["Low", "Medium", "High"],
            )
        elif has_feats:
            def heuristic(row):
                p = 20.0
                if 40 <= row["Age"] <= 60: p += 18
                elif row["Age"] > 60: p += 12
                if row["Geography"] == "Germany": p += 14
                if row["Gender"] == "Female": p += 5
                if row["NumOfProducts"] >= 3: p += 30
                if row["IsActiveMember"] == 0: p += 8
                if row["Balance"] > 100000: p += 5
                elif row["Balance"] == 0: p -= 5
                return max(2, min(98, p))

            batch_df["Churn_Probability"] = batch_df.apply(heuristic, axis=1)
            batch_df["Risk_Level"] = pd.cut(
                batch_df["Churn_Probability"], bins=[0, 30, 60, 100], labels=["Low", "Medium", "High"],
            )
            st.warning("⚠️ Using heuristic predictions. Train models for accurate results.")

        if "Churn_Probability" in batch_df.columns:
            risk_counts = batch_df["Risk_Level"].value_counts()
            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1: st.metric("🔴 High Risk", risk_counts.get("High", 0))
            with rc2: st.metric("🟡 Medium Risk", risk_counts.get("Medium", 0))
            with rc3: st.metric("🟢 Low Risk", risk_counts.get("Low", 0))
            with rc4: st.metric("📊 Avg Probability", f"{batch_df['Churn_Probability'].mean():.1f}%")

            col_b1, col_b2 = st.columns(2)
            with col_b1:
                fig_risk = go.Figure()
                for level, color in [("Low", CLR["emerald"]), ("Medium", CLR["amber"]), ("High", CLR["rose"])]:
                    count = risk_counts.get(level, 0)
                    fig_risk.add_trace(go.Bar(
                        x=[level], y=[count], name=level, marker_color=color,
                        text=[count], textposition="outside",
                        textfont=dict(color=CLR["text"], size=14),
                    ))
                apply_layout(fig_risk, "Risk Level Distribution", height=380)
                fig_risk.update_layout(showlegend=False, xaxis_title="Risk Level", yaxis_title="Count")
                st.plotly_chart(fig_risk, width='stretch')

            with col_b2:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=batch_df["Churn_Probability"], nbinsx=30, marker_color=CLR["primary"], opacity=0.8,
                ))
                apply_layout(fig_hist, "Churn Probability Distribution", height=380)
                fig_hist.update_layout(xaxis_title="Churn Probability (%)", yaxis_title="Count")
                st.plotly_chart(fig_hist, width='stretch')

            st.markdown('<div class="section-header">🔬 Segment Analysis</div>', unsafe_allow_html=True)
            seg_col = st.selectbox(
                "Segment by:", ["Geography", "Gender", "NumOfProducts", "IsActiveMember"], key="batch_segment",
            )
            seg = batch_df.groupby(seg_col)["Churn_Probability"].agg(["mean", "count"]).reset_index()
            seg.columns = [seg_col, "Avg Churn Prob (%)", "Count"]
            seg["Avg Churn Prob (%)"] = seg["Avg Churn Prob (%)"].round(1)

            fig_seg = go.Figure()
            fig_seg.add_trace(go.Bar(
                x=seg[seg_col].astype(str), y=seg["Avg Churn Prob (%)"],
                marker=dict(
                    color=seg["Avg Churn Prob (%)"],
                    colorscale=[[0, CLR["emerald"]], [0.5, CLR["amber"]], [1, CLR["rose"]]],
                ),
                text=seg["Avg Churn Prob (%)"].apply(lambda x: f"{x:.1f}%"),
                textposition="outside",
                textfont=dict(color=CLR["text"], size=13),
            ))
            apply_layout(fig_seg, f"Avg Churn Probability by {seg_col}", height=380)
            fig_seg.update_layout(xaxis_title=seg_col, yaxis_title="Avg Churn Probability (%)")
            st.plotly_chart(fig_seg, width='stretch')

            st.markdown('<div class="section-header">📋 Detailed Results</div>', unsafe_allow_html=True)
            sort_by = st.selectbox("Sort by:", ["Churn_Probability", "Age", "Balance", "CreditScore"], key="batch_sort")
            risk_filter = st.multiselect("Filter risk level:", ["High", "Medium", "Low"], default=["High", "Medium", "Low"], key="batch_filter")
            display_df = batch_df[batch_df["Risk_Level"].isin(risk_filter)].sort_values(sort_by, ascending=False)
            st.dataframe(display_df.head(100), width='stretch', height=400)

            csv = display_df.to_csv(index=False)
            st.download_button("⬇️ Download Results as CSV", csv, "churn_predictions.csv", "text/csv", key="batch_download")
        else:
            st.info("Upload a CSV with the required features to see predictions.")
    else:
        st.info("Upload a CSV file to start batch prediction.")


# ── TAB 4: Model Comparison ──
with tab4:
    st.markdown('<div class="section-header">🏆 Model Performance Comparison</div>', unsafe_allow_html=True)
    comp_df = load_model_comparison()

    if comp_df is not None:
        st.markdown('<div class="winner-badge">🥇 Best Model: Random Forest (Test F1 = 0.635)</div>', unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)

        disp = comp_df.copy()
        col_map = {}
        for c in disp.columns:
            cl = c.lower()
            if "model" in cl: col_map[c] = "Model"
            elif "cv" in cl and "f1" in cl: col_map[c] = "CV F1 (Train)"
            elif "test" in cl and "f1" in cl: col_map[c] = "Test F1"
            elif "param" in cl: col_map[c] = "Best Params"
        disp = disp.rename(columns=col_map)

        if "Model" in disp.columns:
            models = disp["Model"].tolist()
            cv_f1 = disp["CV F1 (Train)"].tolist() if "CV F1 (Train)" in disp.columns else [0] * len(models)
            test_f1 = disp["Test F1"].tolist() if "Test F1" in disp.columns else [0] * len(models)

            col_m1, col_m2 = st.columns([3, 2])
            with col_m1:
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    x=models, y=cv_f1, name="CV F1 (Train)", marker_color=CLR["primary"],
                    text=[f"{v:.3f}" for v in cv_f1], textposition="outside",
                    textfont=dict(color=CLR["text"], size=12),
                ))
                fig_comp.add_trace(go.Bar(
                    x=models, y=test_f1, name="Test F1", marker_color=CLR["cyan"],
                    text=[f"{v:.3f}" for v in test_f1], textposition="outside",
                    textfont=dict(color=CLR["text"], size=12),
                ))
                apply_layout(fig_comp, "F1 Score Comparison", height=420)
                fig_comp.update_layout(
                    barmode="group", bargap=0.3,
                    xaxis_title="Model", yaxis_title="F1 Score",
                    yaxis_range=[0, max(max(cv_f1), max(test_f1)) * 1.15],
                )
                st.plotly_chart(fig_comp, width='stretch')

            with col_m2:
                categories = ["F1 Score", "Generalization", "Robustness"]
                fig_radar = go.Figure()
                model_clrs = [CLR["primary"], CLR["rose"], CLR["emerald"]]
                for i, m in enumerate(models):
                    gen = 1 - abs(cv_f1[i] - test_f1[i]) / max(cv_f1[i], 0.001)
                    vals = [test_f1[i], gen, test_f1[i]]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]], theta=categories + [categories[0]],
                        name=m, line=dict(color=model_clrs[i % len(model_clrs)]),
                        fill="toself", opacity=0.3,
                    ))
                apply_layout(fig_radar, "Model Capabilities", height=420)
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(visible=True, range=[0, 1], gridcolor=CLR["grid"], tickfont=dict(color=CLR["text_dim"])),
                        angularaxis=dict(gridcolor=CLR["grid"], tickfont=dict(color=CLR["text_dim"])),
                    )
                )
                st.plotly_chart(fig_radar, width='stretch')

            st.markdown('<div class="section-header">📋 Detailed Results</div>', unsafe_allow_html=True)
            st.dataframe(disp, width='stretch')

            st.markdown('<div class="section-header">🔍 Model Details</div>', unsafe_allow_html=True)
            medals = ["🥇", "🥈", "🥉"]
            descriptions = {
                "RandomForest": "Ensemble of decision trees with bagging. Handles non-linear relationships well.",
                "XGBoost": "Gradient boosted trees with regularization. Excellent for tabular data.",
                "LogisticRegression": "Linear model with L2 regularization. Provides interpretable coefficients.",
            }

            sorted_disp = disp.sort_values("Test F1", ascending=False) if "Test F1" in disp.columns else disp
            model_cols = st.columns(len(sorted_disp))
            for idx, (_, row) in enumerate(sorted_disp.iterrows()):
                with model_cols[idx]:
                    medal = medals[idx] if idx < 3 else ""
                    m_name = row.get("Model", "Unknown")
                    m_f1 = row.get("Test F1", 0)
                    m_cv = row.get("CV F1 (Train)", 0)
                    desc = descriptions.get(m_name, "")

                    st.markdown(f"""
                    <div class="insight-card" style="text-align:center; padding:1.5rem;">
                        <div style="font-size:2rem;">{medal}</div>
                        <div style="font-size:1.2rem; font-weight:700; color:#e2e8f0; margin:0.5rem 0;">{m_name}</div>
                        <div style="font-size:0.85rem; color:#94a3b8; margin-bottom:1rem;">{desc}</div>
                        <div style="font-size:2rem; font-weight:800; background:linear-gradient(135deg,#818cf8,#a78bfa); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                            {m_f1:.4f}
                        </div>
                        <div style="font-size:0.8rem; color:#64748b; margin-top:0.3rem;">Test F1 Score</div>
                        <hr style="border-color:#1e293b; margin:1rem 0;">
                        <div style="font-size:0.8rem; color:#94a3b8;">CV F1: {m_cv:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # feature importances
        pipe, model = load_model_and_pipeline()
        if model is not None and hasattr(model, "feature_importances_"):
            st.markdown('<div class="section-header">📊 Feature Importances (Best Model)</div>', unsafe_allow_html=True)
            importances = model.feature_importances_

            try:
                feat_names = pipe.named_steps["col_trans"].get_feature_names_out()
            except Exception:
                feat_names = [f"Feature {i}" for i in range(len(importances))]

            fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
            fi_df = fi_df.sort_values("Importance", ascending=True).tail(15)

            fig_fi = go.Figure()
            fig_fi.add_trace(go.Bar(
                y=fi_df["Feature"], x=fi_df["Importance"], orientation="h",
                marker=dict(color=fi_df["Importance"], colorscale=[[0, CLR["blue"]], [1, CLR["primary"]]]),
                text=fi_df["Importance"].round(4), textposition="outside",
                textfont=dict(color=CLR["text_dim"], size=11),
            ))
            apply_layout(fig_fi, "Top 15 Feature Importances", height=500)
            fig_fi.update_layout(xaxis_title="Importance", yaxis_title="")
            st.plotly_chart(fig_fi, width='stretch')
    else:
        st.warning("No model comparison data found. Run `python src/model_training.py --verify` to generate results.")
        st.markdown('<div class="section-header">🧪 Training Methodology</div>', unsafe_allow_html=True)
        st.markdown("""
        | Aspect | Details |
        |--------|---------|
        | **CV Strategy** | 5-fold Stratified Cross-Validation |
        | **Scoring Metric** | F1 Score (handles class imbalance) |
        | **Tuning** | GridSearchCV with exhaustive parameter grids |
        | **Imbalance Handling** | `class_weight='balanced'` (LR, RF), `scale_pos_weight` (XGB) |
        | **Models Tested** | Logistic Regression, Random Forest, XGBoost |
        """)


# footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#475569; font-size:0.8rem;'>"
    "Built with ❤️ by Samarth, Kartik & Hitesh · "
    "Powered by Streamlit & scikit-learn · "
    "Bank Customer Churn Dataset (Kaggle)"
    "</p>",
    unsafe_allow_html=True,
)
