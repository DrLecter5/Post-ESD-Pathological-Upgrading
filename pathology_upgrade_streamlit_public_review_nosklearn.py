from pathlib import Path
import io
import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="病理升级在线预测平台", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.main-title {font-size: 34px; font-weight: 700; margin-bottom: 6px;}
.sub-title {color: #555; margin-bottom: 18px;}
.hero-card {border-radius: 20px; padding: 22px 24px; background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%); border: 1px solid #e5e7eb; box-shadow: 0 10px 30px rgba(0,0,0,0.06);}
.mini-card {border-radius: 16px; padding: 16px 18px; background: #ffffff; border: 1px solid #e5e7eb; box-shadow: 0 6px 18px rgba(0,0,0,0.04);}
.risk-bar-wrap {width: 100%; height: 20px; background: #e5e7eb; border-radius: 999px; overflow: hidden; margin-top: 8px; margin-bottom: 8px;}
.risk-bar-fill {height: 100%; border-radius: 999px; background: linear-gradient(90deg, #22c55e 0%, #f59e0b 55%, #ef4444 100%);}
.tag-pill {display: inline-block; padding: 6px 12px; border-radius: 999px; font-size: 13px; font-weight: 600; margin-right: 8px; margin-top: 6px;}
.tag-high {background: #fee2e2; color: #991b1b;}
.tag-low {background: #dcfce7; color: #166534;}
.muted {color: #6b7280; font-size: 14px;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Public-review no-sklearn version
# This version uses locked logistic-style coefficients hardcoded from
# the re-built training cohort, avoiding online sklearn fitting.
# -------------------------------------------------------------------

COHORT_FILES = {
    "Training": Path("Training_Cohort_Patients.csv"),
    "Internal validation": Path("Internal_Validation_Patients.csv"),
    "External validation": Path("External_Validation_Patients.csv"),
}

MORPH_OPTIONS = ["a", "c", "b"]
MORPH_MAP = {"a": 0, "c": 1, "b": 2}
IPCL_OPTIONS = [1, 2, 3, 4]

# Locked preprocessing values derived offline from the rebuilt training cohort
CLINICAL_FEATURES = ["IPCL", "瘤内溃疡", "表面充血糜烂", "结节", "大小", "病变形态_code", "SIRI"]
FUSION_FEATURES = CLINICAL_FEATURES + ["CNNpred_Mean"]

CLIN_MEANS = {
    "IPCL": 1.8681318681318682,
    "瘤内溃疡": 0.12454212454212454,
    "表面充血糜烂": 0.2454212454212454,
    "结节": 0.42124542124542125,
    "大小": 3.389364303178483,
    "病变形态_code": 0.8534798534798534,
    "SIRI": 0.8761221643312102,
}
CLIN_STDS = {
    "IPCL": 0.9295214098351234,
    "瘤内溃疡": 0.3301885321736297,
    "表面充血糜烂": 0.43033861235242114,
    "结节": 0.49375615490669355,
    "大小": 1.5307717780603904,
    "病变形态_code": 0.8966603961424088,
    "SIRI": 0.8868546218967986,
}
CLIN_COEF = {
    "const": 0.6651062805711763,
    "IPCL": 1.6811844946252706,
    "瘤内溃疡": 0.3952093844683807,
    "表面充血糜烂": 0.5309375665678092,
    "结节": 0.10002743037434834,
    "大小": 0.8754363954074059,
    "病变形态_code": -0.43043818254493084,
    "SIRI": 0.14665729221137015,
}

FUSION_MEANS = {
    "IPCL": 1.8681318681318682,
    "瘤内溃疡": 0.12454212454212454,
    "表面充血糜烂": 0.2454212454212454,
    "结节": 0.42124542124542125,
    "大小": 3.389364303178483,
    "病变形态_code": 0.8534798534798534,
    "SIRI": 0.8761221643312102,
    "CNNpred_Mean": 0.7366758687626536,
}
FUSION_STDS = {
    "IPCL": 0.9295214098351234,
    "瘤内溃疡": 0.3301885321736297,
    "表面充血糜烂": 0.43033861235242114,
    "结节": 0.49375615490669355,
    "大小": 1.5307717780603904,
    "病变形态_code": 0.8966603961424088,
    "SIRI": 0.8868546218967986,
    "CNNpred_Mean": 0.14950197590783872,
}
FUSION_COEF = {
    "const": 0.51926929103262,
    "IPCL": 1.3569965009859741,
    "瘤内溃疡": 0.13790272535939233,
    "表面充血糜烂": 0.36513800459113693,
    "结节": -0.09792609128090361,
    "大小": 0.8961129826861076,
    "病变形态_code": -0.6300639499778794,
    "SIRI": 0.2729278505820768,
    "CNNpred_Mean": 2.576734824602877,
}


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


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
    s = coefs["const"]
    for key in means.keys():
        v = standardize(raw_values[key], means[key], stds[key])
        s += coefs[key] * v
    return s


def clinical_probability(raw_values: dict) -> float:
    return sigmoid(score_logit(raw_values, CLIN_MEANS, CLIN_STDS, CLIN_COEF))


def fusion_probability(raw_values: dict) -> float:
    return sigmoid(score_logit(raw_values, FUSION_MEANS, FUSION_STDS, FUSION_COEF))


def risk_tag(prob, threshold):
    if prob >= threshold:
        return '<span class="tag-pill tag-high">预测：病理升级高风险</span>'
    return '<span class="tag-pill tag-low">预测：病理升级低风险</span>'


def risk_bar(prob):
    pct = max(0, min(100, prob * 100))
    return f'<div class="risk-bar-wrap"><div class="risk-bar-fill" style="width:{pct:.1f}%;"></div></div>'


def explain_contributions(raw_values: dict, means: dict, stds: dict, coefs: dict):
    rows = []
    for key in means.keys():
        z = standardize(raw_values[key], means[key], stds[key])
        c = z * coefs[key]
        rows.append({
            "Feature": key.replace("_code", ""),
            "Contribution": c,
            "Direction": "↑风险" if c >= 0 else "↓风险",
        })
    out = pd.DataFrame(rows).sort_values("Contribution", ascending=False)
    return out.reset_index(drop=True)


def load_cohort_summary():
    rows = []
    for label, path in COHORT_FILES.items():
        if path.exists():
            try:
                df = pd.read_csv(path)
                rows.append({"Cohort": label, "N": len(df)})
            except Exception:
                rows.append({"Cohort": label, "N": "read_error"})
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame([{"Cohort": "Not provided", "N": "-"}])


def batch_predict(df_input: pd.DataFrame):
    result_rows = []
    for _, row in df_input.iterrows():
        morph = str(row.get("病变形态", "a"))
        morph_code = MORPH_MAP.get(morph, 0)
        raw = {
            "IPCL": to_float(row.get("IPCL", 1), 1.0),
            "瘤内溃疡": to_float(row.get("瘤内溃疡", 0), 0.0),
            "表面充血糜烂": to_float(row.get("表面充血糜烂", 0), 0.0),
            "结节": to_float(row.get("结节", 0), 0.0),
            "大小": to_float(row.get("大小", 3.0), 3.0),
            "病变形态_code": float(morph_code),
            "SIRI": to_float(row.get("SIRI", 1.0), 1.0),
            "CNNpred_Mean": to_float(row.get("CNNpred_Mean", 0.5), 0.5),
        }
        cp = clinical_probability(raw)
        fp = fusion_probability(raw)
        result_rows.append({
            **row.to_dict(),
            "Clinical_LR_like": cp,
            "Fusion_LR_like": fp,
            "Risk_Label@0.50": "病理升级高风险" if fp >= 0.5 else "病理升级低风险",
        })
    return pd.DataFrame(result_rows)

# -----------------------------
# UI
# -----------------------------
st.markdown('<div class="main-title">病理升级在线预测平台（审稿人公开版）</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">该版本不依赖 sklearn 或 PyTorch，直接使用锁定的 logistic-style 系数进行在线风险计算，更适合稳定公开部署。</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["单例预测", "批量预测", "说明"])

with st.sidebar:
    threshold = st.slider("判定阈值", 0.10, 0.90, 0.50, 0.01)
    st.success("No-sklearn 公开版已加载")
    st.caption("在线端不重新拟合模型，仅调用锁定系数进行预测。")

with tab1:
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        with st.form("reviewer_predict_form"):
            name = st.text_input("患者标识（可选）", value="Reviewer_Demo")
            c1, c2 = st.columns(2)
            with c1:
                ipcl = st.selectbox("IPCL", IPCL_OPTIONS, index=0)
                ulcer = st.selectbox("瘤内溃疡", [0, 1], index=0)
                hyperemia = st.selectbox("表面充血糜烂", [0, 1], index=0)
                nodule = st.selectbox("结节", [0, 1], index=0)
            with c2:
                morph = st.selectbox("病变形态", MORPH_OPTIONS, index=0)
                size = st.number_input("大小", min_value=0.0, value=3.0, step=0.1, format="%.2f")
                siri = st.number_input("SIRI", min_value=0.0, value=1.00, step=0.01, format="%.2f")
                cnnprob = st.slider("CNNpred_Mean", 0.00, 1.00, 0.50, 0.01)
            submitted = st.form_submit_button("开始预测", use_container_width=True)

        if submitted:
            raw = {
                "IPCL": float(ipcl),
                "瘤内溃疡": float(ulcer),
                "表面充血糜烂": float(hyperemia),
                "结节": float(nodule),
                "大小": float(size),
                "病变形态_code": float(MORPH_MAP.get(morph, 0)),
                "SIRI": float(siri),
                "CNNpred_Mean": float(cnnprob),
            }
            clin_prob = clinical_probability(raw)
            fusion_prob = fusion_probability(raw)

            st.markdown('<div class="hero-card">', unsafe_allow_html=True)
            st.markdown("### 最终推荐结果：Fusion LR-like")
            st.markdown(
                f"#### 风险概率：**{fusion_prob:.3f}** &nbsp;&nbsp; {risk_tag(fusion_prob, threshold)}",
                unsafe_allow_html=True,
            )
            st.markdown(risk_bar(fusion_prob), unsafe_allow_html=True)
            st.markdown(
                f'<div class="muted">当前输入 CNNpred_Mean：{cnnprob:.3f}；当前阈值：{threshold:.2f}。</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            m1.metric("Clinical LR-like", f"{clin_prob:.3f}")
            m2.metric("Fusion LR-like", f"{fusion_prob:.3f}")

            st.markdown("#### Fusion 主要驱动因素")
            exp_df = explain_contributions(raw, FUSION_MEANS, FUSION_STDS, FUSION_COEF)
            st.dataframe(exp_df.round(4), use_container_width=True, hide_index=True)

    with right:
        st.markdown('<div class="mini-card">', unsafe_allow_html=True)
        st.markdown("#### 当前网站使用的数据口径")
        st.dataframe(load_cohort_summary(), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="mini-card">', unsafe_allow_html=True)
        st.markdown("#### 公开版说明")
        st.markdown(
            """
            - 该版本适合 Streamlit Cloud 公开部署  
            - 不依赖 sklearn / PyTorch  
            - 不在云端重新训练模型  
            - 直接使用锁定的系数进行临床 + CNN 风险计算  
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("批量预测")
    st.caption("上传包含以下字段的 CSV/Excel：IPCL、瘤内溃疡、表面充血糜烂、结节、大小、病变形态、SIRI、CNNpred_Mean")

    sample_df = pd.DataFrame([{
        "姓名": "示例患者",
        "IPCL": 2,
        "瘤内溃疡": 0,
        "表面充血糜烂": 1,
        "结节": 0,
        "大小": 3.5,
        "病变形态": "a",
        "SIRI": 1.23,
        "CNNpred_Mean": 0.67,
    }])
    buf = io.BytesIO()
    sample_df.to_csv(buf, index=False)
    st.download_button("下载模板 CSV", data=buf.getvalue(), file_name="sample_input_template.csv", mime="text/csv")

    uploaded = st.file_uploader("上传待预测文件", type=["csv", "xlsx"])
    if uploaded is not None:
        if uploaded.name.lower().endswith(".csv"):
            df_in = pd.read_csv(uploaded)
        else:
            df_in = pd.read_excel(uploaded)

        need_cols = ["IPCL", "瘤内溃疡", "表面充血糜烂", "结节", "大小", "病变形态", "SIRI", "CNNpred_Mean"]
        missing_cols = [c for c in need_cols if c not in df_in.columns]
        if missing_cols:
            st.error(f"缺少字段：{missing_cols}")
        else:
            out_df = batch_predict(df_in)
            st.success("批量预测完成")
            st.dataframe(out_df, use_container_width=True)

            out_buf = io.BytesIO()
            out_df.to_excel(out_buf, index=False)
            st.download_button(
                "下载预测结果 Excel",
                data=out_buf.getvalue(),
                file_name="batch_prediction_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

with tab3:
    st.subheader("说明")
    st.markdown(
        """
        **为什么要做 no-sklearn 公开版：**
        1. Streamlit Community Cloud 当前环境对部分 `scikit-learn` 版本构建不稳定。  
        2. 公开版的核心目标是让审稿人稳定打开并演示预测流程。  
        3. 因此在线端采用锁定的 logistic-style 系数，而不是云端实时重新拟合。  

        **这版适合：**
        - 审稿人公开访问  
        - 论文补充材料链接  
        - 课题汇报演示  

        **内部完整版仍可保留：**
        - 本地/Cloud Studio 图像推理版  
        - 含 5 个 CNN 权重的自动图像推理流程  
        """
    )

st.markdown("---")
st.caption("Reviewer-facing public demo without sklearn. Internal image-inference version should be kept separately.")
