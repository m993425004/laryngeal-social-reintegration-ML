# -*- coding: utf-8 -*-
"""
Final LR Risk Calculator - single-file local Streamlit APP\n\nRelease version: publication supplementary calculator

Features:
1. Single-file version with embedded logistic regression coefficients.
2. No external model file is required.
3. UI style adjusted to match the previous calculator layout.
4. Times New Roman is applied throughout the interface.
5. No raw patient-level dataset is required.

Run:
streamlit run "Final_LR_Risk_Calculator_MATCHED_STYLE.py"
"""

import math
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. Page settings
# ============================================================

st.set_page_config(
    page_title="LR Risk Calculator",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"], .stApp {
        font-family: "Times New Roman", Times, serif !important;
    }
    h1, h2, h3, h4, h5, h6, p, label {
        font-family: "Times New Roman", Times, serif !important;
    }

    .stButton button, .stSelectbox, .stNumberInput, .stMetric {
        font-family: "Times New Roman", Times, serif !important;
    }

    /* Prevent Streamlit expander title overlap */
    [data-testid="stExpander"] summary {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 1rem !important;
        line-height: 1.4 !important;
    }

    [data-testid="stExpander"] summary p {
        font-size: 1rem !important;
        line-height: 1.4 !important;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .section-card {
        border: 1px solid rgba(49, 51, 63, 0.15);
        border-radius: 0.75rem;
        padding: 0.9rem 1rem 0.2rem 1rem;
        margin-bottom: 1rem;
        background: rgba(255,255,255,0.02);
    }
    .app-subtitle {
        color: #666666;
        margin-top: -0.25rem;
        margin-bottom: 1rem;
    }
    .risk-chip-low {
        font-size: 2rem;
        font-weight: 700;
        color: #1f7a1f;
    }
    .risk-chip-mid {
        font-size: 2rem;
        font-weight: 700;
        color: #b8860b;
    }
    .risk-chip-high {
        font-size: 2rem;
        font-weight: 700;
        color: #b22222;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. Embedded LR model parameters
# ============================================================

INTERCEPT = -2.9226278

NUM_MEAN = {
    "Age": 62.4332893,
    "OSSS3_Total": 10.04491413,
    "VHI10_Total": 11.3989432,
    "EAT10_Total": 5.79128137,
    "HADS_A": 6.51783355,
    "HADS_D": 7.59048877,
}

NUM_SCALE = {
    "Age": 9.65411604,
    "OSSS3_Total": 2.18027452,
    "VHI10_Total": 6.68655853,
    "EAT10_Total": 4.96340937,
    "HADS_A": 3.21345611,
    "HADS_D": 4.19993262,
}

NUM_COEF = {
    "Age": 0.3818646093,
    "OSSS3_Total": -1.0178355619,
    "VHI10_Total": 1.5957352777,
    "EAT10_Total": 1.9686954067,
    "HADS_A": 1.0378193267,
    "HADS_D": 1.1614761458,
}

CAT_COEF = {
    "Education=1": -0.4286017671,
    "Income_Level=1": -0.1127256632,
    "Income_Level=2": -0.3838235229,
    "Symptom_Duration=1": 0.1361432195,
    "Symptom_Duration=2": 0.5257157662,
    "Postoperative_adjuvant_therapy=1": 0.7986889491,
    "Postoperative_adjuvant_therapy=2": 1.4915650053,
    "Preoperative_Pathology=1": 0.5153624036,
    "Preoperative_Pathology=2": 0.6635838499,
    "Tumor_Location=1": 1.4997551422,
    "Tumor_Location=2": 1.2087331185,
    "Tumor_Location=3": 0.3646943898,
    "Vocal_Cord_Mobility=1": 0.2887550473,
    "Vocal_Cord_Mobility=2": 0.5978657725,
    "Clinical_T_Stage=1": 0.9524358564,
    "Resection_Extent=1": 0.7863757317,
    "Neck_Dissection=1": 0.5098423621,
    "Minimally_Invasive_Surgery=1": -1.7014972261,
}

CONTINUOUS_VARS = [
    "OSSS3_Total",
    "VHI10_Total",
    "EAT10_Total",
    "HADS_A",
    "HADS_D",
    "Age",
]

SELECTED_FEATURES = [
    "OSSS3_Total",
    "VHI10_Total",
    "EAT10_Total",
    "HADS_A",
    "HADS_D",
    "Education",
    "Income_Level",
    "Symptom_Duration",
    "Preoperative_Pathology",
    "Tumor_Location",
    "Vocal_Cord_Mobility",
    "Clinical_T_Stage",
    "Postoperative_adjuvant_therapy",
    "Resection_Extent",
    "Neck_Dissection",
    "Minimally_Invasive_Surgery",
    "Age",
]

PRETTY_LABELS = {
    "Age": "Age",
    "OSSS3_Total": "OSSS-3 total",
    "VHI10_Total": "VHI-10 total",
    "EAT10_Total": "EAT-10 total",
    "HADS_A": "HADS-A",
    "HADS_D": "HADS-D",
    "Education": "Education",
    "Income_Level": "Income level",
    "Symptom_Duration": "Symptom duration",
    "Preoperative_Pathology": "Preoperative pathology",
    "Tumor_Location": "Tumor location",
    "Vocal_Cord_Mobility": "Vocal cord mobility",
    "Clinical_T_Stage": "Clinical T stage",
    "Postoperative_adjuvant_therapy": "Postoperative adjuvant therapy",
    "Resection_Extent": "Resection extent",
    "Neck_Dissection": "Neck dissection",
    "Minimally_Invasive_Surgery": "Minimally invasive surgery",
}

VARIABLE_GROUPS = {
    "Patient-reported and psychosocial variables": [
        "OSSS3_Total",
        "VHI10_Total",
        "EAT10_Total",
        "HADS_A",
        "HADS_D",
    ],
    "Socioeconomic variables": [
        "Education",
        "Income_Level",
    ],
    "Tumor-related variables": [
        "Symptom_Duration",
        "Preoperative_Pathology",
        "Tumor_Location",
        "Vocal_Cord_Mobility",
        "Clinical_T_Stage",
    ],
    "Surgical variables": [
        "Postoperative_adjuvant_therapy",
        "Resection_Extent",
        "Neck_Dissection",
        "Minimally_Invasive_Surgery",
    ],
    "Demographic variable": [
        "Age",
    ],
}

NUMERIC_RANGES = {
    "Age": (18.0, 100.0, 1.0),
    "OSSS3_Total": (3.0, 14.0, 1.0),
    "VHI10_Total": (0.0, 40.0, 1.0),
    "EAT10_Total": (0.0, 40.0, 1.0),
    "HADS_A": (0.0, 21.0, 1.0),
    "HADS_D": (0.0, 21.0, 1.0),
}

DEFAULT_NUMERIC_VALUES = {
    "OSSS3_Total": 10.0,
    "VHI10_Total": 13.0,
    "EAT10_Total": 5.0,
    "HADS_A": 7.0,
    "HADS_D": 8.0,
    "Age": 63.0,
}

CATEGORY_VALUE_LABELS = {
    "Education": {
        "0": "High school or below",
        "1": "College or above",
    },
    "Income_Level": {
        "0": "<=2000 RMB",
        "1": "2001-5000 RMB",
        "2": ">5000 RMB",
    },
    "Symptom_Duration": {
        "0": "<=0.5 year",
        "1": "0.5-1 year",
        "2": ">1 year",
    },
    "Preoperative_Pathology": {
        "0": "Highly differentiated",
        "1": "Moderately differentiated",
        "2": "Poorly differentiated",
    },
    "Tumor_Location": {
        "0": "Unilateral vocal cord",
        "1": "Bilateral vocal cord",
        "2": "Supraglottic",
        "3": "Subglottic",
    },
    "Vocal_Cord_Mobility": {
        "0": "Normal",
        "1": "Limited",
        "2": "Fixed",
    },
    "Clinical_T_Stage": {
        "0": "T1/T2",
        "1": "T3/4",
    },
    "Postoperative_adjuvant_therapy": {
        "0": "No",
        "1": "Yes",
    },
    "Resection_Extent": {
        "0": "Partial laryngectomy",
        "1": "Total laryngectomy",
    },
    "Neck_Dissection": {
        "0": "No",
        "1": "Yes",
    },
    "Minimally_Invasive_Surgery": {
        "0": "No",
        "1": "Yes",
    },
}

DEFAULT_CATEGORICAL_VALUES = {
    "Education": "1",
    "Income_Level": "2",
    "Symptom_Duration": "0",
    "Preoperative_Pathology": "0",
    "Tumor_Location": "0",
    "Vocal_Cord_Mobility": "0",
    "Clinical_T_Stage": "0",
    "Postoperative_adjuvant_therapy": "0",
    "Resection_Extent": "0",
    "Neck_Dissection": "0",
    "Minimally_Invasive_Surgery": "0",
}


# ============================================================
# 3. Helper functions
# ============================================================

def pretty_label(var_name):
    return PRETTY_LABELS.get(var_name, var_name.replace("_", " "))


def display_category_option(var_name, value):
    return CATEGORY_VALUE_LABELS.get(var_name, {}).get(str(value), str(value))


def display_input_value(var_name, value):
    if var_name in CATEGORY_VALUE_LABELS:
        return display_category_option(var_name, value)
    return value


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def calculate_probability(data):
    score = INTERCEPT

    for key in NUM_COEF:
        z = (float(data[key]) - NUM_MEAN[key]) / NUM_SCALE[key]
        score += NUM_COEF[key] * z

    for key, value in data.items():
        if key in CATEGORY_VALUE_LABELS and str(value) != "0":
            coef_key = f"{key}={value}"
            if coef_key in CAT_COEF:
                score += CAT_COEF[coef_key]

    return float(sigmoid(score))


def risk_category(prob):
    if prob < 0.30:
        return "Low risk", "risk-chip-low", "🟢"
    if prob < 0.60:
        return "Intermediate risk", "risk-chip-mid", "🟡"
    return "High risk", "risk-chip-high", "🔴"


def create_group_inputs(group_vars):
    values = {}
    cols = st.columns(2)

    for i, var in enumerate(group_vars):
        with cols[i % 2]:
            if var in CONTINUOUS_VARS:
                mn, mx, step = NUMERIC_RANGES[var]
                values[var] = st.number_input(
                    pretty_label(var),
                    min_value=float(mn),
                    max_value=float(mx),
                    value=float(DEFAULT_NUMERIC_VALUES.get(var, mn)),
                    step=float(step),
                    key=f"num_{var}",
                )
            else:
                options = list(CATEGORY_VALUE_LABELS[var].keys())
                default_value = DEFAULT_CATEGORICAL_VALUES.get(var, options[0])
                default_index = options.index(default_value) if default_value in options else 0
                values[var] = st.selectbox(
                    pretty_label(var),
                    options=options,
                    index=default_index,
                    format_func=lambda x, var=var: display_category_option(var, x),
                    key=f"cat_{var}",
                )
    return values


def build_input_summary(input_values):
    rows = []
    for var in SELECTED_FEATURES:
        rows.append(
            {
                "Predictor": pretty_label(var),
                "Value": display_input_value(var, input_values[var]),
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# 4. Main APP
# ============================================================

def main():
    st.title("Logistic Regression Risk Calculator")
    st.markdown(
        '<div class="app-subtitle">Prediction of poor social reintegration after surgery for laryngeal cancer</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Model information")
        st.write("Final model: **Logistic Regression**")
        st.write(f"Number of predictors: **{len(SELECTED_FEATURES)}**")

        with st.expander("Selected predictors", expanded=False):
            for i, v in enumerate(SELECTED_FEATURES, start=1):
                st.write(f"{i}. {pretty_label(v)}")

        with st.expander("Categorical options", expanded=False):
            for v in [x for x in SELECTED_FEATURES if x in CATEGORY_VALUE_LABELS]:
                opts = [display_category_option(v, x) for x in CATEGORY_VALUE_LABELS[v].keys()]
                st.write(f"{pretty_label(v)}: {', '.join(opts)}")

        st.markdown("---")
        st.caption("For research use only. External validation is required before clinical implementation.")

    st.markdown(
        "Enter patient information below. The calculator returns the predicted probability of "
        "poor social reintegration based on the final LR model."
    )

    with st.form("lr_risk_form"):
        form_values = {}

        for group_name, vars_in_group in VARIABLE_GROUPS.items():
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown(f"### {group_name}")
            form_values.update(create_group_inputs(vars_in_group))
            st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("Calculate predicted risk")

    if submitted:
        prob = calculate_probability(form_values)
        category, css_class, icon = risk_category(prob)

        st.markdown("---")
        left, right = st.columns([1, 1])

        with left:
            st.subheader("Predicted risk")
            st.write("Probability of poor social reintegration")
            st.markdown(f"## {prob * 100:.1f}%")
            st.progress(min(max(prob, 0.0), 1.0))

        with right:
            st.subheader("Risk category")
            st.markdown(
                f'<div class="{css_class}">{icon} {category}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("### Input summary")
        summary_df = build_input_summary(form_values)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    else:
        st.markdown("---")
        st.info("Fill in the predictors and click **Calculate predicted risk**.")


if __name__ == "__main__":
    main()
