from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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

TARGET = "group"
CLINICAL_FEATURES = ["IPCL", "瘤内溃疡", "表面充血糜烂", "结节", "大小", "病变形态", "SIRI"]
FUSION_FEATURES = CLINICAL_FEATURES + ["CNNpred_Mean"]

TRAIN_TABLES = [Path("训练集.xlsx"), Path("验证集.xlsx"), Path("训练集.csv"), Path("验证集.csv")]
CNN_OOF_PATH = Path("Patient_CNN_Predictions_OOF.csv")
TRAIN_COHORT_PATH = Path("Training_Cohort_Patients.csv")
INTERNAL_VAL_PATH = Path("Internal_Validation_Patients.csv")
EXTERNAL_VAL_PATH = Path("External_Validation_Patients.csv")

def coerce_binary_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    mapping = {"是":1, "否":0, "有":1, "无":0, "yes":1, "no":0, "true":1, "false":0, "阳性":1, "阴性":0}
    return s.astype(str).str.strip().str.lower().map(mapping).fillna(pd.to_numeric(s, errors="coerce"))

def coerce_size_col(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("cm", "", regex=False).str.replace("mm", "", regex=False), errors="coerce")

def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)

def normalize_name(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("（","(", regex=False).str.replace("）",")", regex=False).str.replace(r"\s+","", regex=True).str.replace("_","", regex=False).str.strip()

def normalize_cnn_table(df_cnn: pd.DataFrame) -> pd.DataFrame:
    df = df_cnn.copy()
    if "PatientID" in df.columns and "姓名" not in df.columns:
        df = df.rename(columns={"PatientID": "姓名"})
    pred_candidates = ["CNNpred_Mean", "Pred_CNN_topk2", "pred", "prob", "probability", "score"]
    pred_col = None
    for c in pred_candidates:
        if c in df.columns:
            pred_col = c
            break
    if pred_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            pred_col = numeric_cols[-1]
    if pred_col is None or "姓名" not in df.columns:
        raise ValueError("CNN OOF 文件缺少必要列")
    out = df[["姓名", pred_col]].copy().rename(columns={pred_col: "CNNpred_Mean"})
    out["姓名_norm"] = normalize_name(out["姓名"])
    return out

def build_master_pool():
    dfs = []
    for path in TRAIN_TABLES:
        if path.exists():
            dt = read_table(path)
            dt["__source_file__"] = path.name
            dfs.append(dt)
    if not dfs:
        raise FileNotFoundError("未找到训练/验证原始数据表")
    pool = pd.concat(dfs, axis=0, ignore_index=True)
    if "姓名" not in pool.columns:
        raise ValueError("原始数据表缺少“姓名”列")
    pool["姓名_norm"] = normalize_name(pool["姓名"])

    cnn = normalize_cnn_table(pd.read_csv(CNN_OOF_PATH))
    pool = pool.merge(cnn[["姓名_norm", "CNNpred_Mean"]], on="姓名_norm", how="left")
    if "CNNpred_Mean" not in pool.columns:
        pool["CNNpred_Mean"] = 0.5
    return pool

def subset_by_cohort(pool: pd.DataFrame, cohort_path: Path):
    if not cohort_path.exists():
        return None
    cohort = pd.read_csv(cohort_path)
    if "姓名" not in cohort.columns:
        raise ValueError(f"{cohort_path.name} 缺少“姓名”列")
    cohort["姓名_norm"] = normalize_name(cohort["姓名"])
    out = pool.merge(cohort[["姓名_norm"]].drop_duplicates(), on="姓名_norm", how="inner")
    return out

def prep_common(df: pd.DataFrame):
    dt = df.copy()
    for col in ["瘤内溃疡", "表面充血糜烂", "结节"]:
        if col in dt.columns:
            dt[col] = coerce_binary_series(dt[col])
    if "大小" in dt.columns:
        dt["大小"] = coerce_size_col(dt["大小"])
    if TARGET in dt.columns:
        dt[TARGET] = pd.to_numeric(dt[TARGET], errors="coerce")
    return dt

@st.cache_resource(show_spinner=False)
def load_state():
    if not CNN_OOF_PATH.exists():
        return {"ok": False, "message": f"未找到 CNN OOF 文件：{CNN_OOF_PATH}"}
    try:
        pool = prep_common(build_master_pool())
    except Exception as e:
        return {"ok": False, "message": str(e)}

    if TRAIN_COHORT_PATH.exists():
        df_train = subset_by_cohort(pool, TRAIN_COHORT_PATH)
        split_mode = "按新的 Training_Cohort_Patients.csv 训练"
    else:
        if Path("训练集.xlsx").exists():
            df_train = prep_common(read_table(Path("训练集.xlsx")))
            df_train["姓名_norm"] = normalize_name(df_train["姓名"])
            cnn = normalize_cnn_table(pd.read_csv(CNN_OOF_PATH))
            df_train = df_train.merge(cnn[["姓名_norm", "CNNpred_Mean"]], on="姓名_norm", how="left")
        else:
            df_train = pool.copy()
        split_mode = "按原训练表训练"

    if df_train is None or len(df_train) == 0:
        return {"ok": False, "message": "训练队列为空，请检查 Training_Cohort_Patients.csv 与原始数据表的姓名是否匹配。"}
    if "CNNpred_Mean" not in df_train.columns:
        df_train["CNNpred_Mean"] = 0.5

    df_internal = subset_by_cohort(pool, INTERNAL_VAL_PATH) if INTERNAL_VAL_PATH.exists() else None
    df_external = subset_by_cohort(pool, EXTERNAL_VAL_PATH) if EXTERNAL_VAL_PATH.exists() else None

    missing = [c for c in FUSION_FEATURES + [TARGET] if c not in df_train.columns]
    if missing:
        return {"ok": False, "message": f"训练集缺少字段：{missing}"}

    morph_vals = df_train["病变形态"].dropna().astype(str).tolist() if "病变形态" in df_train.columns else []
    morph_options = list(dict.fromkeys(morph_vals)) or ["a", "b", "c"]
    morph_map = {v: i for i, v in enumerate(morph_options)}

    ipcl_vals = pd.to_numeric(df_train["IPCL"], errors="coerce").dropna().astype(int).unique().tolist() if "IPCL" in df_train.columns else []
    ipcl_options = sorted(ipcl_vals) if ipcl_vals else [1, 2, 3]

    def prep(df, features, fit=False, imp=None, scl=None):
        dt = df.copy()
        dt["病变形态_code"] = dt["病变形态"].astype(str).map(morph_map).fillna(-1)
        actual = [f if f != "病变形态" else "病变形态_code" for f in features]
        for c in actual:
            dt[c] = pd.to_numeric(dt[c], errors="coerce")
        X = dt[actual].copy()
        y = pd.to_numeric(dt[TARGET], errors="coerce")
        if fit:
            imp = SimpleImputer(strategy="median")
            X_imp = imp.fit_transform(X)
            scl = StandardScaler()
            X_scl = scl.fit_transform(X_imp)
            return X_scl, y, actual, imp, scl
        X_imp = imp.transform(X)
        X_scl = scl.transform(X_imp)
        return X_scl, y, actual

    Xc, yc, actual_c, imp_c, scl_c = prep(df_train, CLINICAL_FEATURES, fit=True)
    Xf, yf, actual_f, imp_f, scl_f = prep(df_train, FUSION_FEATURES, fit=True)

    model_clin = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000).fit(Xc, yc)
    model_fuse = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000).fit(Xf, yf)
    model_rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=4, random_state=42, class_weight="balanced").fit(Xf, yf)
    model_dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=8, random_state=42, class_weight="balanced").fit(Xf, yf)

    summary_rows = [{"Cohort": "Training", "N": len(df_train)}]
    if df_internal is not None and len(df_internal) > 0:
        summary_rows.append({"Cohort": "Internal validation", "N": len(df_internal)})
    if df_external is not None and len(df_external) > 0:
        summary_rows.append({"Cohort": "External validation", "N": len(df_external)})
    summary_df = pd.DataFrame(summary_rows)

    return {
        "ok": True,
        "split_mode": split_mode,
        "summary_df": summary_df,
        "morph_options": morph_options,
        "ipcl_options": ipcl_options,
        "morph_map": morph_map,
        "imp_c": imp_c, "scl_c": scl_c,
        "imp_f": imp_f, "scl_f": scl_f,
        "model_clin": model_clin,
        "model_fuse": model_fuse,
        "model_rf": model_rf,
        "model_dt": model_dt,
        "actual_c": actual_c,
        "actual_f": actual_f,
    }

def prepare_row(name, ipcl, ulcer, hyperemia, nodule, size, morph, siri, cnnprob):
    return pd.DataFrame([{
        "姓名": name,
        "IPCL": ipcl,
        "瘤内溃疡": ulcer,
        "表面充血糜烂": hyperemia,
        "结节": nodule,
        "大小": size,
        "病变形态": morph,
        "SIRI": siri,
        "CNNpred_Mean": cnnprob,
    }])

def predict_one(row_df, state):
    dt = row_df.copy()
    dt["大小"] = coerce_size_col(dt["大小"])
    dt["病变形态_code"] = dt["病变形态"].astype(str).map(state["morph_map"]).fillna(-1)
    actual_c = [f if f != "病变形态" else "病变形态_code" for f in CLINICAL_FEATURES]
    actual_f = [f if f != "病变形态" else "病变形态_code" for f in FUSION_FEATURES]
    Xc = dt[actual_c].copy()
    Xf = dt[actual_f].copy()
    for c in actual_c:
        Xc[c] = pd.to_numeric(Xc[c], errors="coerce")
    for c in actual_f:
        Xf[c] = pd.to_numeric(Xf[c], errors="coerce")
    Xc = state["scl_c"].transform(state["imp_c"].transform(Xc))
    Xf = state["scl_f"].transform(state["imp_f"].transform(Xf))
    return {
        "CNN Only": float(pd.to_numeric(row_df["CNNpred_Mean"], errors="coerce").fillna(0.5).iloc[0]),
        "Clinical LR": float(state["model_clin"].predict_proba(Xc)[:, 1][0]),
        "Fusion LR": float(state["model_fuse"].predict_proba(Xf)[:, 1][0]),
        "Fusion RF": float(state["model_rf"].predict_proba(Xf)[:, 1][0]),
        "Fusion DT": float(state["model_dt"].predict_proba(Xf)[:, 1][0]),
    }

def explain_fusion_lr(row_df, state, top_k=6):
    dt = row_df.copy()
    dt["大小"] = coerce_size_col(dt["大小"])
    dt["病变形态_code"] = dt["病变形态"].astype(str).map(state["morph_map"]).fillna(-1)
    actual_f = [f if f != "病变形态" else "病变形态_code" for f in FUSION_FEATURES]
    Xf = dt[actual_f].copy()
    for c in actual_f:
        Xf[c] = pd.to_numeric(Xf[c], errors="coerce")
    Xf = state["scl_f"].transform(state["imp_f"].transform(Xf))
    coefs = state["model_fuse"].coef_[0]
    contrib = Xf[0] * coefs
    out = pd.DataFrame({
        "Feature": [x.replace("_code", "") for x in actual_f],
        "Contribution": contrib,
        "Direction": ["↑风险" if c >= 0 else "↓风险" for c in contrib]
    }).sort_values("Contribution", ascending=False)
    return out.head(top_k).reset_index(drop=True)

def risk_tag(prob, threshold):
    if prob >= threshold:
        return '<span class="tag-pill tag-high">预测：病理升级高风险</span>'
    return '<span class="tag-pill tag-low">预测：病理升级低风险</span>'

def risk_bar(prob):
    pct = max(0, min(100, prob * 100))
    return f'<div class="risk-bar-wrap"><div class="risk-bar-fill" style="width:{pct:.1f}%;"></div></div>'

st.markdown('<div class="main-title">病理升级在线预测平台（审稿人公开版）</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">该公开版优先按新的训练/内部/外部名单重建训练队列；网页输入临床变量与患者级 CNN 风险分数（CNNpred_Mean）后输出风险。</div>', unsafe_allow_html=True)

state = load_state()

with st.sidebar:
    threshold = st.slider("判定阈值", 0.10, 0.90, 0.50, 0.01)
    if state["ok"]:
        st.success("公开版模型已加载")
        st.caption(state["split_mode"])
    else:
        st.error(state["message"])

if not state["ok"]:
    st.stop()

left, right = st.columns([1.05, 1], gap="large")

with left:
    with st.form("reviewer_predict_form"):
        name = st.text_input("患者标识（可选）", value="Reviewer_Demo")
        c1, c2 = st.columns(2)
        with c1:
            ipcl = st.selectbox("IPCL", state["ipcl_options"], index=0)
            ulcer = st.selectbox("瘤内溃疡", [0, 1], index=0)
            hyperemia = st.selectbox("表面充血糜烂", [0, 1], index=0)
            nodule = st.selectbox("结节", [0, 1], index=0)
        with c2:
            morph = st.selectbox("病变形态", state["morph_options"], index=0)
            size = st.number_input("大小", min_value=0.0, value=3.0, step=0.1, format="%.2f")
            siri = st.number_input("SIRI", min_value=0.0, value=1.00, step=0.01, format="%.2f")
            cnnprob = st.slider("CNNpred_Mean", 0.00, 1.00, 0.50, 0.01)
        submitted = st.form_submit_button("开始预测", use_container_width=True)

    if submitted:
        row_df = prepare_row(name, ipcl, ulcer, hyperemia, nodule, size, morph, siri, cnnprob)
        result = predict_one(row_df, state)
        fusion_prob = result["Fusion LR"]

        st.markdown('<div class="hero-card">', unsafe_allow_html=True)
        st.markdown("### 最终推荐结果：Fusion LR")
        st.markdown(f"#### 风险概率：**{fusion_prob:.3f}** &nbsp;&nbsp; {risk_tag(fusion_prob, threshold)}", unsafe_allow_html=True)
        st.markdown(risk_bar(fusion_prob), unsafe_allow_html=True)
        st.markdown(f'<div class="muted">当前输入 CNNpred_Mean：{cnnprob:.3f}；当前阈值：{threshold:.2f}。</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("CNN Only", f"{result['CNN Only']:.3f}")
        m2.metric("Clinical LR", f"{result['Clinical LR']:.3f}")
        m3.metric("Fusion LR", f"{result['Fusion LR']:.3f}")
        m4.metric("Fusion RF", f"{result['Fusion RF']:.3f}")
        m5.metric("Fusion DT", f"{result['Fusion DT']:.3f}")

        st.markdown("#### Fusion LR 主要驱动因素")
        st.dataframe(explain_fusion_lr(row_df, state).round(4), use_container_width=True, hide_index=True)

with right:
    st.markdown('<div class="mini-card">', unsafe_allow_html=True)
    st.markdown("#### 当前网站使用的数据口径")
    st.dataframe(state["summary_df"], use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
