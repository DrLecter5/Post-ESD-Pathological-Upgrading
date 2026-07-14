from pathlib import Path
import io
import math

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Pathological Upgrade Risk Calculator",
    page_icon="🩺",
    layout="wide",
)


st.markdown(
    """
    <style>
    .main-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .sub-title {
        color: #555;
        margin-bottom: 18px;
    }
    .hero-card {
        border-radius: 20px;
        padding: 22px 24px;
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        border: 1px solid #e5e7eb;
        box-shadow: 0 10px 30px rgba(0,0,0,0.06);
    }
    .mini-card {
        border-radius: 16px;
        padding: 16px 18px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 18px rgba(0,0,0,0.04);
    }
    .risk-bar-wrap {
        width: 100%;
        height: 20px;
        background: #e5e7eb;
        border-radius: 999px;
        overflow: hidden;
        margin-top: 8px;
        margin-bottom: 8px;
    }
    .risk-bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #22c55e 0%, #f59e0b 55%, #ef4444 100%);
    }
    .tag-pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 6px;
    }
    .tag-high {
        background: #fee2e2;
        color: #991b1b;
    }
    .tag-low {
        background: #dcfce7;
        color: #166534;
    }
    .muted {
        color: #6b7280;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Public deployment files
# ---------------------------------------------------------------------
TRAIN_PATH = Path("Training_Cohort_Patients.csv")
VALIDATION_PATH = Path("Internal_Validation_Patients.csv")
EXTERNAL_PATH = Path("External_Validation_Patients.csv")


# ---------------------------------------------------------------------
# Input mappings
# ---------------------------------------------------------------------
MORPH_OPTIONS = ["a", "c", "b"]
MORPH_MAP = {"a": 0, "c": 1, "b": 2}

IPCL_OPTIONS = ["A", "B1", "B2", "B3"]
IPCL_MAP = {"A": 1, "B1": 2, "B2": 3, "B3": 4}


# ---------------------------------------------------------------------
# Locked coefficients and preprocessing values
# The public calculator does not refit any model online.
# ---------------------------------------------------------------------
CLIN_MEANS = {
    "ipcl": 1.8681318681318682,
    "intralesional_ulcer": 0.12454212454212454,
    "surface_hyperemia_erosion": 0.2454212454212454,
    "nodule": 0.42124542124542125,
    "lesion_size": 3.389364303178483,
    "morphology_code": 0.8534798534798534,
    "siri": 0.8761221643312102,
}
CLIN_STDS = {
    "ipcl": 0.9295214098351234,
    "intralesional_ulcer": 0.3301885321736297,
    "surface_hyperemia_erosion": 0.43033861235242114,
    "nodule": 0.49375615490669355,
    "lesion_size": 1.5307717780603904,
    "morphology_code": 0.8966603961424088,
    "siri": 0.8868546218967986,
}
CLIN_COEF = {
    "const": 0.6651062805711763,
    "ipcl": 1.6811844946252706,
    "intralesional_ulcer": 0.3952093844683807,
    "surface_hyperemia_erosion": 0.5309375665678092,
    "nodule": 0.10002743037434834,
    "lesion_size": 0.8754363954074059,
    "morphology_code": -0.43043818254493084,
    "siri": 0.14665729221137015,
}

FUSION_MEANS = {
    "ipcl": 1.8681318681318682,
    "intralesional_ulcer": 0.12454212454212454,
    "surface_hyperemia_erosion": 0.2454212454212454,
    "nodule": 0.42124542124542125,
    "lesion_size": 3.389364303178483,
    "morphology_code": 0.8534798534798534,
    "siri": 0.8761221643312102,
    "cnn_score": 0.7366758687626536,
}
FUSION_STDS = {
    "ipcl": 0.9295214098351234,
    "intralesional_ulcer": 0.3301885321736297,
    "surface_hyperemia_erosion": 0.43033861235242114,
    "nodule": 0.49375615490669355,
    "lesion_size": 1.5307717780603904,
    "morphology_code": 0.8966603961424088,
    "siri": 0.8868546218967986,
    "cnn_score": 0.14950197590783872,
}
FUSION_COEF = {
    "const": 0.51926929103262,
    "ipcl": 1.3569965009859741,
    "intralesional_ulcer": 0.13790272535939233,
    "surface_hyperemia_erosion": 0.36513800459113693,
    "nodule": -0.09792609128090361,
    "lesion_size": 0.8961129826861076,
    "morphology_code": -0.6300639499778794,
    "siri": 0.2729278505820768,
    "cnn_score": 2.576734824602877,
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def standardize(value: float, mean: float, std: float) -> float:
    if std == 0:
        return 0.0
    return (value - mean) / std


def to_float(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        if isinstance(x, str):
            x = x.replace("cm", "").replace("mm", "").strip()
        return float(x)
    except Exception:
        return default


def score_logit(raw_values: dict, means: dict, stds: dict, coefs: dict) -> float:
    total = coefs["const"]
    for key in means:
        total += coefs[key] * standardize(raw_values[key], means[key], stds[key])
    return total


def clinical_probability(raw_values: dict) -> float:
    return sigmoid(score_logit(raw_values, CLIN_MEANS, CLIN_STDS, CLIN_COEF))


def fusion_probability(raw_values: dict) -> float:
    return sigmoid(score_logit(raw_values, FUSION_MEANS, FUSION_STDS, FUSION_COEF))


def risk_badge(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return '<span class="tag-pill tag-high">High-risk prediction</span>'
    return '<span class="tag-pill tag-low">Low-risk prediction</span>'


def risk_bar(probability: float) -> str:
    pct = max(0, min(100, probability * 100))
    return f'<div class="risk-bar-wrap"><div class="risk-bar-fill" style="width:{pct:.1f}%;"></div></div>'


def explain_contributions(raw_values: dict, means: dict, stds: dict, coefs: dict):
    display_names = {
        "ipcl": "IPCL",
        "intralesional_ulcer": "Intralesional ulcer",
        "surface_hyperemia_erosion": "Surface hyperemia or erosion",
        "nodule": "Nodule",
        "lesion_size": "Lesion size",
        "morphology_code": "Morphology",
        "siri": "SIRI",
        "cnn_score": "Image-derived CNN score",
    }
    rows = []
    for key in means:
        z_value = standardize(raw_values[key], means[key], stds[key])
        contribution = z_value * coefs[key]
        rows.append({
            "Feature": display_names.get(key, key),
            "Contribution": contribution,
            "Direction": "Higher risk" if contribution >= 0 else "Lower risk",
        })
    return pd.DataFrame(rows).sort_values("Contribution", ascending=False).reset_index(drop=True)


def load_cohort_summary():
    rows = []
    for name, path in [
        ("Training", TRAIN_PATH),
        ("Validation", VALIDATION_PATH),
        ("External validation", EXTERNAL_PATH),
    ]:
        if path.exists():
            try:
                rows.append({"Cohort": name, "N": len(pd.read_csv(path))})
            except Exception:
                rows.append({"Cohort": name, "N": "Read error"})
        else:
            rows.append({"Cohort": name, "N": "Not available"})
    return pd.DataFrame(rows)


def build_raw_values(
    ipcl_label,
    ulcer,
    hyperemia,
    nodule,
    lesion_size,
    morphology,
    siri,
    cnn_score,
):
    return {
        "ipcl": float(IPCL_MAP[ipcl_label]),
        "intralesional_ulcer": float(ulcer),
        "surface_hyperemia_erosion": float(hyperemia),
        "nodule": float(nodule),
        "lesion_size": float(lesion_size),
        "morphology_code": float(MORPH_MAP.get(morphology, 0)),
        "siri": float(siri),
        "cnn_score": float(cnn_score),
    }


def batch_predict(df_input: pd.DataFrame):
    result_rows = []
    for _, row in df_input.iterrows():
        ipcl_raw = row.get("IPCL", "A")
        if isinstance(ipcl_raw, str):
            ipcl_value = float(IPCL_MAP.get(ipcl_raw, 1))
        else:
            ipcl_value = to_float(ipcl_raw, 1.0)

        morphology = str(row.get("Morphology", row.get("morphology", "a")))
        raw = {
            "ipcl": ipcl_value,
            "intralesional_ulcer": to_float(row.get("Intralesional_ulcer", row.get("intralesional_ulcer", 0)), 0.0),
            "surface_hyperemia_erosion": to_float(row.get("Surface_hyperemia_or_erosion", row.get("surface_hyperemia_or_erosion", 0)), 0.0),
            "nodule": to_float(row.get("Nodule", row.get("nodule", 0)), 0.0),
            "lesion_size": to_float(row.get("Lesion_size", row.get("lesion_size", 3.0)), 3.0),
            "morphology_code": float(MORPH_MAP.get(morphology, 0)),
            "siri": to_float(row.get("SIRI", row.get("siri", 1.0)), 1.0),
            "cnn_score": to_float(row.get("CNN_score", row.get("CNNpred_Mean", row.get("cnn_score", 0.5))), 0.5),
        }

        clinical_prob = clinical_probability(raw)
        fusion_prob = fusion_probability(raw)
        result_rows.append({
            **row.to_dict(),
            "Clinical_probability": clinical_prob,
            "Fusion_probability": fusion_prob,
            "Risk_label_at_0_50": "High risk" if fusion_prob >= 0.5 else "Low risk",
        })
    return pd.DataFrame(result_rows)


# ---------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------
st.markdown(
    '<div class="main-title">Multimodal Risk Calculator for Pathological Upgrade after ESD</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Interactive risk estimation using structured clinical variables and a patient-level image-derived CNN score.</div>',
    unsafe_allow_html=True,
)

tab_single, tab_batch, tab_info = st.tabs([
    "Single-patient prediction",
    "Batch prediction",
    "Model information",
])

with st.sidebar:
    threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.01)
    st.success("Public calculator loaded")
    st.caption("This public version does not perform online image inference.")
    st.caption("Enter the patient-level CNN score generated by the locked image model.")

with tab_single:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown("#### Clinical and image-derived inputs")
        with st.form("single_patient_form"):
            patient_id = st.text_input("Patient identifier", value="Demo_Patient")
            c1, c2 = st.columns(2)
            with c1:
                ipcl_label = st.selectbox("IPCL classification", IPCL_OPTIONS, index=0)
                ulcer = st.selectbox("Intralesional ulcer", [0, 1], index=0)
                hyperemia = st.selectbox("Surface hyperemia or erosion", [0, 1], index=0)
                nodule = st.selectbox("Nodule", [0, 1], index=0)
            with c2:
                morphology = st.selectbox("Morphology", MORPH_OPTIONS, index=0)
                lesion_size = st.number_input("Lesion size", min_value=0.0, value=3.0, step=0.1, format="%.2f")
                siri = st.number_input("SIRI", min_value=0.0, value=1.00, step=0.01, format="%.2f")
                cnn_score = st.slider(
                    "Image-derived CNN score",
                    0.00,
                    1.00,
                    0.50,
                    0.01,
                    help="Enter the patient-level CNN score generated by the locked image model.",
                )
            submitted = st.form_submit_button("Calculate risk", use_container_width=True)

        if submitted:
            raw = build_raw_values(
                ipcl_label,
                ulcer,
                hyperemia,
                nodule,
                lesion_size,
                morphology,
                siri,
                cnn_score,
            )
            clinical_prob = clinical_probability(raw)
            fusion_prob = fusion_probability(raw)

            st.markdown('<div class="hero-card">', unsafe_allow_html=True)
            st.markdown("### Final result: fusion model")
            st.markdown(
                f"#### Predicted probability: **{fusion_prob:.3f}** &nbsp;&nbsp; {risk_badge(fusion_prob, threshold)}",
                unsafe_allow_html=True,
            )
            st.markdown(risk_bar(fusion_prob), unsafe_allow_html=True)
            st.markdown(
                f'<div class="muted">Clinical-only probability: {clinical_prob:.3f}; image-derived CNN score: {cnn_score:.3f}; threshold: {threshold:.2f}.</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("#### Feature contribution summary")
            st.dataframe(
                explain_contributions(raw, FUSION_MEANS, FUSION_STDS, FUSION_COEF).round(4),
                use_container_width=True,
                hide_index=True,
            )

    with right:
        st.markdown('<div class="mini-card">', unsafe_allow_html=True)
        st.markdown("#### Cohort summary")
        st.dataframe(load_cohort_summary(), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="mini-card">', unsafe_allow_html=True)
        st.markdown("#### Input coding")
        st.markdown(
            """
            - Binary variables: 0 = absent, 1 = present  
            - IPCL: A, B1, B2, or B3  
            - Morphology: a, b, or c  
            - Image-derived CNN score: patient-level score from the locked image model  
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)


with tab_batch:
    st.subheader("Batch prediction")
    st.caption(
        "Upload a CSV or Excel file containing: IPCL, Intralesional_ulcer, "
        "Surface_hyperemia_or_erosion, Nodule, Lesion_size, Morphology, SIRI, CNN_score."
    )

    sample_df = pd.DataFrame([{
        "Patient_ID": "Example",
        "IPCL": "B1",
        "Intralesional_ulcer": 0,
        "Surface_hyperemia_or_erosion": 1,
        "Nodule": 0,
        "Lesion_size": 3.5,
        "Morphology": "a",
        "SIRI": 1.23,
        "CNN_score": 0.67,
    }])
    buffer = io.BytesIO()
    sample_df.to_csv(buffer, index=False)
    st.download_button(
        "Download CSV template",
        data=buffer.getvalue(),
        file_name="sample_input_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload a prediction file", type=["csv", "xlsx"])
    if uploaded is not None:
        if uploaded.name.lower().endswith(".csv"):
            df_in = pd.read_csv(uploaded)
        else:
            df_in = pd.read_excel(uploaded)

        required_cols = [
            "IPCL",
            "Intralesional_ulcer",
            "Surface_hyperemia_or_erosion",
            "Nodule",
            "Lesion_size",
            "Morphology",
            "SIRI",
            "CNN_score",
        ]
        missing_cols = [col for col in required_cols if col not in df_in.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            out_df = batch_predict(df_in)
            st.success("Batch prediction completed")
            st.dataframe(out_df, use_container_width=True)

            out_buffer = io.BytesIO()
            out_df.to_excel(out_buffer, index=False)
            st.download_button(
                "Download prediction results",
                data=out_buffer.getvalue(),
                file_name="batch_prediction_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


with tab_info:
    st.subheader("Model information")
    st.markdown(
        """
        This web application provides an interactive calculator for pathological
        upgrade after endoscopic submucosal dissection. The fusion model combines
        structured clinical variables and a patient-level image-derived CNN score.

        The public calculator uses locked preprocessing values and coefficients.
        It does not refit the model during deployment and does not perform online
        image inference. The CNN score should be generated offline using the
        locked image model and then entered into this calculator.
        """
    )
