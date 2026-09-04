import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import mode


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CyberLens",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.20);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

.logo-box {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 5px;
}

.logo-icon {
    font-size: 35px;
}

.logo-title {
    font-size: 25px;
    font-weight: 750;
}

.logo-subtitle {
    font-size: 12px;
    opacity: 0.60;
    margin-bottom: 25px;
}

.page-title {
    font-size: 38px;
    font-weight: 750;
    letter-spacing: -1px;
    margin-bottom: 3px;
}

.page-subtitle {
    font-size: 15px;
    opacity: 0.60;
    margin-bottom: 28px;
}

.metric-card {
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 14px;
    padding: 20px;
    min-height: 125px;
    background: rgba(128,128,128,0.035);
}

.metric-label {
    font-size: 13px;
    opacity: 0.60;
}

.metric-value {
    font-size: 30px;
    font-weight: 750;
    margin-top: 8px;
}

.metric-description {
    font-size: 12px;
    opacity: 0.55;
    margin-top: 5px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 15px;
}

.status-box {
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 10px;
    padding: 12px;
    margin-top: 10px;
    font-size: 13px;
}

.hero {
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 25px;
    background: rgba(128,128,128,0.035);
}

.hero-title {
    font-size: 25px;
    font-weight: 700;
}

.hero-text {
    opacity: 0.65;
    margin-top: 8px;
    line-height: 1.6;
}

.model-card {
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    min-height: 115px;
    background: rgba(128,128,128,0.035);
}

.model-name {
    font-weight: 700;
    font-size: 16px;
}

.model-status {
    font-size: 13px;
    opacity: 0.60;
    margin-top: 8px;
}

.manual-info {
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 15px;
    background: rgba(128,128,128,0.035);
}

.cyber-footer {
    text-align: center;
    opacity: 0.45;
    font-size: 12px;
    padding-top: 45px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = "cyberlens_models"


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    dt_model = joblib.load(
        f"{MODEL_DIR}/decision_tree.pkl"
    )

    rf_model = joblib.load(
        f"{MODEL_DIR}/random_forest.pkl"
    )

    xgb_model = joblib.load(
        f"{MODEL_DIR}/xgboost.pkl"
    )

    scaler = joblib.load(
        f"{MODEL_DIR}/scaler.pkl"
    )

    with open(
        f"{MODEL_DIR}/metadata.json",
        "r"
    ) as f:
        metadata = json.load(f)

    return (
        dt_model,
        rf_model,
        xgb_model,
        scaler,
        metadata
    )


try:

    (
        dt_model,
        rf_model,
        xgb_model,
        scaler,
        metadata
    ) = load_models()

    feature_names = metadata["feature_names"]
    class_names = metadata["class_names"]

except Exception as e:

    st.error("Unable to load CyberLens models.")
    st.exception(e)
    st.stop()


# ============================================================
# SHAP EXPLAINER
# ============================================================

@st.cache_resource
def create_explainer():

    return shap.TreeExplainer(xgb_model)


explainer = create_explainer()


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "data" not in st.session_state:
    st.session_state.data = None

if "predictions" not in st.session_state:
    st.session_state.predictions = None

if "manual_mode" not in st.session_state:
    st.session_state.manual_mode = False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="logo-box">
            <div class="logo-icon">🛡️</div>
            <div class="logo-title">CyberLens</div>
        </div>

        <div class="logo-subtitle">
            Explainable Network Intrusion Detection
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Navigation")

    if st.button(
        "🏠  Dashboard",
        use_container_width=True
    ):
        st.session_state.page = "Dashboard"

    if st.button(
        "🔍  Detection",
        use_container_width=True
    ):
        st.session_state.page = "Detection"

    if st.button(
        "📊  Model Analysis",
        use_container_width=True
    ):
        st.session_state.page = "Model Analysis"

    if st.button(
        "🧠  Explainability",
        use_container_width=True
    ):
        st.session_state.page = "Explainability"

    if st.button(
        "📄  Reports",
        use_container_width=True
    ):
        st.session_state.page = "Reports"

    st.markdown("---")

    st.markdown("### System Status")

    st.success("● Models Loaded")

    st.markdown(
        f"""
        <div class="status-box">
        <b>Feature Space</b><br>
        {len(feature_names)} network features
        </div>

        <div class="status-box">
        <b>Classes</b><br>
        {len(class_names)} traffic categories
        </div>

        <div class="status-box">
        <b>Ensemble</b><br>
        Majority Voting
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="page-title">CyberLens</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="page-subtitle">'
    'Explainable Ensemble Learning Framework for '
    'Intelligent Network Intrusion Detection'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.page == "Dashboard":

    if st.session_state.predictions is None:

        st.markdown(
            """
            <div class="hero">

            <div class="hero-title">
            Intelligent Network Security Analysis
            </div>

            <div class="hero-text">
            CyberLens combines Decision Tree, Random Forest,
            and XGBoost classifiers through majority voting
            to identify malicious network traffic while
            providing explainable predictions using SHAP.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">System Capabilities</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.markdown(
                """
                ### 🔍 Traffic Detection

                Analyze network-flow records and
                classify them into normal traffic
                or attack categories.
                """
            )

        with c2:

            st.markdown(
                """
                ### 🗳️ Ensemble Intelligence

                Combine predictions from three
                independent machine learning
                models using majority voting.
                """
            )

        with c3:

            st.markdown(
                """
                ### 🧠 Explainable AI

                Use SHAP to understand which
                network features influenced
                individual predictions.
                """
            )

        st.info(
            "Start by opening **Detection** and either "
            "uploading a CSV or entering a network flow manually."
        )

    else:

        predictions = st.session_state.predictions

        ensemble_pred = predictions["ensemble"]

        total = len(ensemble_pred)

        normal_index = class_names.index("Normal")

        normal_count = np.sum(
            ensemble_pred == normal_index
        )

        attack_count = total - normal_count

        attack_rate = (
            attack_count / total
        ) * 100

        dt_pred = predictions["decision_tree"]
        rf_pred = predictions["random_forest"]
        xgb_pred = predictions["xgboost"]

        agreement = np.mean(
            (
                (dt_pred == rf_pred)
                &
                (rf_pred == xgb_pred)
            )
        ) * 100

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Security Overview'
            '</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)

        metrics = [
            ("Total Traffic", f"{total:,}", "Analyzed records"),
            ("Normal Traffic", f"{normal_count:,}", "Benign records"),
            ("Potential Attacks", f"{attack_count:,}", "Detected threats"),
            ("Attack Rate", f"{attack_rate:.2f}%", "Traffic requiring attention")
        ]

        for col, metric in zip(
            [c1, c2, c3, c4],
            metrics
        ):

            with col:

                st.markdown(
                    f"""
                    <div class="metric-card">

                    <div class="metric-label">
                    {metric[0]}
                    </div>

                    <div class="metric-value">
                    {metric[1]}
                    </div>

                    <div class="metric-description">
                    {metric[2]}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ----------------------------------------------------
        # CHARTS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Traffic Analysis'
            '</div>',
            unsafe_allow_html=True
        )

        labels = [
            class_names[int(x)]
            for x in ensemble_pred
        ]

        counts = (
            pd.Series(labels)
            .value_counts()
            .reset_index()
        )

        counts.columns = [
            "Class",
            "Records"
        ]

        c1, c2 = st.columns(2)

        with c1:

            fig = px.bar(
                counts,
                x="Class",
                y="Records",
                title="Traffic Classification"
            )

            fig.update_layout(
                height=420,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with c2:

            fig = px.pie(
                counts,
                names="Class",
                values="Records",
                title="Traffic Distribution",
                hole=0.45
            )

            fig.update_layout(
                height=420,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ----------------------------------------------------
        # MODEL AGREEMENT
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Ensemble Model Status'
            '</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)

        model_cards = [
            ("🌳", "Decision Tree"),
            ("🌲", "Random Forest"),
            ("⚡", "XGBoost"),
            ("🗳️", "Majority Voting")
        ]

        for col, model in zip(
            [c1, c2, c3, c4],
            model_cards
        ):

            with col:

                st.markdown(
                    f"""
                    <div class="model-card">

                    <div style="font-size:25px">
                    {model[0]}
                    </div>

                    <div class="model-name">
                    {model[1]}
                    </div>

                    <div class="model-status">
                    ✓ Active
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.metric(
            "Model Agreement",
            f"{agreement:.2f}%"
        )


# ============================================================
# DETECTION
# ============================================================

elif st.session_state.page == "Detection":

    st.header("🔍 Network Traffic Detection")

    st.write(
        "Analyze network traffic using either an uploaded CSV "
        "dataset or manually entered network-flow features."
    )

    # ========================================================
    # INPUT MODE
    # ========================================================

    input_mode = st.radio(
        "Choose Detection Method",
        [
            "📁 Upload CSV Dataset",
            "⌨️ Manual Network Flow Input"
        ],
        horizontal=True
    )

    # ========================================================
    # CSV MODE
    # ========================================================

    if input_mode == "📁 Upload CSV Dataset":

        uploaded_file = st.file_uploader(
            "Upload Network Traffic CSV",
            type=["csv"]
        )

        if uploaded_file:

            try:

                data = pd.read_csv(
                    uploaded_file
                )

            except Exception as e:

                st.error("Unable to read the CSV file.")
                st.exception(e)
                st.stop()

            st.success(
                f"Dataset loaded successfully — "
                f"{len(data):,} records × "
                f"{len(data.columns)} columns"
            )

            missing = [
                feature
                for feature in feature_names
                if feature not in data.columns
            ]

            if missing:

                st.error(
                    f"{len(missing)} required features are missing."
                )

                st.write(missing)

                st.stop()

            st.subheader("Dataset Preview")

            st.dataframe(
                data.head(10),
                use_container_width=True
            )

            st.subheader("Dataset Information")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Records",
                    f"{len(data):,}"
                )

            with c2:
                st.metric(
                    "Features",
                    len(feature_names)
                )

            with c3:
                st.metric(
                    "Models",
                    3
                )

            if st.button(
                "🚀 Analyze Network Traffic",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "Running ensemble detection..."
                ):

                    X_input = data[
                        feature_names
                    ].copy()

                    X_input = X_input.apply(
                        pd.to_numeric,
                        errors="coerce"
                    ).fillna(0)

                    X_scaled = scaler.transform(
                        X_input
                    )

                    dt_pred = dt_model.predict(
                        X_scaled
                    )

                    rf_pred = rf_model.predict(
                        X_scaled
                    )

                    xgb_pred = xgb_model.predict(
                        X_scaled
                    )

                    all_predictions = np.column_stack(
                        [
                            dt_pred,
                            rf_pred,
                            xgb_pred
                        ]
                    )

                    ensemble_pred = mode(
                        all_predictions,
                        axis=1,
                        keepdims=False
                    ).mode

                    probabilities = (
                        xgb_model.predict_proba(
                            X_scaled
                        )
                    )

                st.session_state.data = data

                st.session_state.predictions = {

                    "decision_tree": dt_pred,
                    "random_forest": rf_pred,
                    "xgboost": xgb_pred,
                    "ensemble": ensemble_pred,
                    "probabilities": probabilities,
                    "X_scaled": X_scaled

                }

                st.session_state.manual_mode = False

                st.success(
                    "Network traffic analysis completed!"
                )

                st.session_state.page = "Dashboard"

                st.rerun()

    # ========================================================
    # MANUAL INPUT MODE
    # ========================================================

    else:

        st.markdown(
            """
            <div class="manual-info">

            <b>⌨️ Manual Network Flow Analysis</b><br>

            Enter the values of a single network-flow record.
            CyberLens will process all 67 features using the
            same preprocessing pipeline used during model training.

            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            "All 67 network-flow features are required. "
            "Fields are organized into logical groups to keep "
            "the interface easy to use."
        )

        manual_values = {}

        # ====================================================
        # BASIC FLOW INFORMATION
        # ====================================================

        with st.expander(
            "📡 Basic Flow Information",
            expanded=True
        ):

            c1, c2, c3 = st.columns(3)

            with c1:

                manual_values["Destination Port"] = st.number_input(
                    "Destination Port",
                    min_value=0,
                    value=80,
                    step=1
                )

                manual_values["Flow Duration"] = st.number_input(
                    "Flow Duration",
                    min_value=0,
                    value=100000,
                    step=1
                )

                manual_values["Total Fwd Packets"] = st.number_input(
                    "Total Fwd Packets",
                    min_value=0,
                    value=10,
                    step=1
                )

                manual_values["Total Backward Packets"] = st.number_input(
                    "Total Backward Packets",
                    min_value=0,
                    value=8,
                    step=1
                )

            with c2:

                manual_values["Total Length of Fwd Packets"] = st.number_input(
                    "Total Length of Fwd Packets",
                    min_value=0.0,
                    value=500.0
                )

                manual_values["Total Length of Bwd Packets"] = st.number_input(
                    "Total Length of Bwd Packets",
                    min_value=0.0,
                    value=400.0
                )

                manual_values["Fwd Packet Length Max"] = st.number_input(
                    "Fwd Packet Length Max",
                    min_value=0.0,
                    value=100.0
                )

                manual_values["Fwd Packet Length Min"] = st.number_input(
                    "Fwd Packet Length Min",
                    min_value=0.0,
                    value=20.0
                )

            with c3:

                manual_values["Fwd Packet Length Mean"] = st.number_input(
                    "Fwd Packet Length Mean",
                    min_value=0.0,
                    value=50.0
                )

                manual_values["Fwd Packet Length Std"] = st.number_input(
                    "Fwd Packet Length Std",
                    min_value=0.0,
                    value=10.0
                )

                manual_values["Bwd Packet Length Max"] = st.number_input(
                    "Bwd Packet Length Max",
                    min_value=0.0,
                    value=100.0
                )

                manual_values["Bwd Packet Length Min"] = st.number_input(
                    "Bwd Packet Length Min",
                    min_value=0.0,
                    value=20.0
                )

        # ====================================================
        # PACKET STATISTICS
        # ====================================================

        with st.expander(
            "📦 Packet Statistics"
        ):

            c1, c2, c3 = st.columns(3)

            with c1:

                manual_values["Bwd Packet Length Mean"] = st.number_input(
                    "Bwd Packet Length Mean",
                    min_value=0.0,
                    value=50.0
                )

                manual_values["Bwd Packet Length Std"] = st.number_input(
                    "Bwd Packet Length Std",
                    min_value=0.0,
                    value=10.0
                )

                manual_values["Flow Bytes/s"] = st.number_input(
                    "Flow Bytes/s",
                    min_value=0.0,
                    value=1000.0
                )

                manual_values["Flow Packets/s"] = st.number_input(
                    "Flow Packets/s",
                    min_value=0.0,
                    value=100.0
                )

            with c2:

                manual_values["Flow IAT Mean"] = st.number_input(
                    "Flow IAT Mean",
                    min_value=0.0,
                    value=1000.0
                )

                manual_values["Flow IAT Std"] = st.number_input(
                    "Flow IAT Std",
                    min_value=0.0,
                    value=500.0
                )

                manual_values["Flow IAT Max"] = st.number_input(
                    "Flow IAT Max",
                    min_value=0.0,
                    value=5000.0
                )

                manual_values["Flow IAT Min"] = st.number_input(
                    "Flow IAT Min",
                    min_value=0.0,
                    value=100.0
                )

            with c3:

                manual_values["Fwd IAT Total"] = st.number_input(
                    "Fwd IAT Total",
                    min_value=0.0,
                    value=5000.0
                )

                manual_values["Fwd IAT Mean"] = st.number_input(
                    "Fwd IAT Mean",
                    min_value=0.0,
                    value=500.0
                )

                manual_values["Fwd IAT Std"] = st.number_input(
                    "Fwd IAT Std",
                    min_value=0.0,
                    value=200.0
                )

                manual_values["Fwd IAT Max"] = st.number_input(
                    "Fwd IAT Max",
                    min_value=0.0,
                    value=2000.0
                )

        # ====================================================
        # IAT / TIMING
        # ====================================================

        with st.expander(
            "⏱️ Backward Timing Features"
        ):

            c1, c2, c3 = st.columns(3)

            with c1:

                manual_values["Fwd IAT Min"] = st.number_input(
                    "Fwd IAT Min",
                    min_value=0.0,
                    value=100.0
                )

                manual_values["Bwd IAT Total"] = st.number_input(
                    "Bwd IAT Total",
                    min_value=0.0,
                    value=4000.0
                )

                manual_values["Bwd IAT Mean"] = st.number_input(
                    "Bwd IAT Mean",
                    min_value=0.0,
                    value=500.0
                )

            with c2:

                manual_values["Bwd IAT Std"] = st.number_input(
                    "Bwd IAT Std",
                    min_value=0.0,
                    value=200.0
                )

                manual_values["Bwd IAT Max"] = st.number_input(
                    "Bwd IAT Max",
                    min_value=0.0,
                    value=2000.0
                )

                manual_values["Bwd IAT Min"] = st.number_input(
                    "Bwd IAT Min",
                    min_value=0.0,
                    value=100.0
                )

            with c3:

                manual_values["Fwd PSH Flags"] = st.number_input(
                    "Fwd PSH Flags",
                    min_value=0,
                    value=0,
                    step=1
                )

                manual_values["Fwd Header Length"] = st.number_input(
                    "Fwd Header Length",
                    min_value=0,
                    value=320,
                    step=1
                )

                manual_values["Bwd Header Length"] = st.number_input(
                    "Bwd Header Length",
                    min_value=0,
                    value=256,
                    step=1
                )

        # ====================================================
        # PACKET RATE / LENGTH
        # ====================================================

        with st.expander(
            "📊 Packet Rate and Length Features"
        ):

            c1, c2, c3 = st.columns(3)

            with c1:

                manual_values["Fwd Packets/s"] = st.number_input(
                    "Fwd Packets/s",
                    min_value=0.0,
                    value=50.0
                )

                manual_values["Bwd Packets/s"] = st.number_input(
                    "Bwd Packets/s",
                    min_value=0.0,
                    value=40.0
                )

                manual_values["Min Packet Length"] = st.number_input(
                    "Min Packet Length",
                    min_value=0.0,
                    value=20.0
                )

                manual_values["Max Packet Length"] = st.number_input(
                    "Max Packet Length",
                    min_value=0.0,
                    value=100.0
                )

            with c2:

                manual_values["Packet Length Mean"] = st.number_input(
                    "Packet Length Mean",
                    min_value=0.0,
                    value=50.0
                )

                manual_values["Packet Length Std"] = st.number_input(
                    "Packet Length Std",
                    min_value=0.0,
                    value=10.0
                )

                manual_values["Packet Length Variance"] = st.number_input(
                    "Packet Length Variance",
                    min_value=0.0,
                    value=100.0
                )

                manual_values["FIN Flag Count"] = st.number_input(
                    "FIN Flag Count",
                    min_value=0,
                    value=0,
                    step=1
                )

            with c3:

                manual_values["SYN Flag Count"] = st.number_input(
                    "SYN Flag Count",
                    min_value=0,
                    value=0,
                    step=1
                )

                manual_values["RST Flag Count"] = st.number_input(
                    "RST Flag Count",
                    min_value=0,
                    value=0,
                    step=1
                )

                manual_values["PSH Flag Count"] = st.number_input(
                    "PSH Flag Count",
                    min_value=0,
                    value=0,
                    step=1
                )

                manual_values["ACK Flag Count"] = st.number_input(
                    "ACK Flag Count",
                    min_value=0,
                    value=0,
                    step=1
                )

        # ====================================================
        # FLAGS / RATIOS
        # ====================================================

        with st.expander(
            "🔐 TCP Flags and Flow Ratios"
        ):

            c1, c2, c3 = st.columns(3)

            with c1:

                manual_values["URG Flag Count"] = st.number_input(
                    "URG Flag Count",
                    min_value=0,
                    value=0,
                    step=1
                )

                manual_values["ECE Flag Count"] = st.number_input(
                    "ECE Flag Count",
                    min_value=0,
                    value=0,
                    step=1
                )

                manual_values["Down/Up Ratio"] = st.number_input(
                    "Down/Up Ratio",
                    min_value=0.0,
                    value=1.0
                )

            with c2:

                manual_values["Average Packet Size"] = st.number_input(
                    "Average Packet Size",
                    min_value=0.0,
                    value=50.0
                )

                manual_values["Avg Fwd Segment Size"] = st.number_input(
                    "Avg Fwd Segment Size",
                    min_value=0.0,
                    value=50.0
                )

                manual_values["Avg Bwd Segment Size"] = st.number_input(
                    "Avg Bwd Segment Size",
                    min_value=0.0,
                    value=50.0
                )

            with c3:

                manual_values["Subflow Fwd Packets"] = st.number_input(
                    "Subflow Fwd Packets",
                    min_value=0,
                    value=10,
                    step=1
                )

                manual_values["Subflow Fwd Bytes"] = st.number_input(
                    "Subflow Fwd Bytes",
                    min_value=0,
                    value=500,
                    step=1
                )

                manual_values["Subflow Bwd Packets"] = st.number_input(
                    "Subflow Bwd Packets",
                    min_value=0,
                    value=8,
                    step=1
                )

        # ====================================================
        # SUBFLOW / WINDOW
        # ====================================================

        with st.expander(
            "🪟 Window and Segment Information"
        ):

            c1, c2, c3 = st.columns(3)

            with c1:

                manual_values["Subflow Bwd Bytes"] = st.number_input(
                    "Subflow Bwd Bytes",
                    min_value=0,
                    value=400,
                    step=1
                )

                manual_values["Init_Win_bytes_forward"] = st.number_input(
                    "Init_Win_bytes_forward",
                    min_value=0,
                    value=8192,
                    step=1
                )

            with c2:

                manual_values["Init_Win_bytes_backward"] = st.number_input(
                    "Init_Win_bytes_backward",
                    min_value=0,
                    value=8192,
                    step=1
                )

                manual_values["act_data_pkt_fwd"] = st.number_input(
                    "act_data_pkt_fwd",
                    min_value=0,
                    value=5,
                    step=1
                )

            with c3:

                manual_values["min_seg_size_forward"] = st.number_input(
                    "min_seg_size_forward",
                    min_value=0,
                    value=20,
                    step=1
                )

        # ====================================================
        # ACTIVE FEATURES
        # ====================================================

        with st.expander(
            "⚡ Active Flow Features"
        ):

            c1, c2, c3 = st.columns(3)

            with c1:

                manual_values["Active Mean"] = st.number_input(
                    "Active Mean",
                    min_value=0.0,
                    value=1000.0
                )

                manual_values["Active Std"] = st.number_input(
                    "Active Std",
                    min_value=0.0,
                    value=100.0
                )

            with c2:

                manual_values["Active Max"] = st.number_input(
                    "Active Max",
                    min_value=0.0,
                    value=2000.0
                )

                manual_values["Active Min"] = st.number_input(
                    "Active Min",
                    min_value=0.0,
                    value=100.0
                )

            with c3:

                manual_values["Idle Mean"] = st.number_input(
                    "Idle Mean",
                    min_value=0.0,
                    value=5000.0
                )

        # ====================================================
        # IDLE FEATURES
        # ====================================================

        with st.expander(
            "💤 Idle Flow Features"
        ):

            c1, c2, c3 = st.columns(3)

            with c1:

                manual_values["Idle Std"] = st.number_input(
                    "Idle Std",
                    min_value=0.0,
                    value=500.0
                )

                manual_values["Idle Max"] = st.number_input(
                    "Idle Max",
                    min_value=0.0,
                    value=10000.0
                )

            with c2:

                manual_values["Idle Min"] = st.number_input(
                    "Idle Min",
                    min_value=0.0,
                    value=1000.0
                )

        # ====================================================
        # VERIFY FEATURES
        # ====================================================

        missing_manual = [
            feature
            for feature in feature_names
            if feature not in manual_values
        ]

        if missing_manual:

            st.error(
                "Some required features are missing from the manual input form:"
            )

            st.write(missing_manual)

        else:

            st.markdown("---")

            st.subheader("Manual Input Summary")

            manual_df = pd.DataFrame(
                [manual_values],
                columns=feature_names
            )

            st.dataframe(
                manual_df,
                use_container_width=True
            )

            # =================================================
            # ANALYZE
            # =================================================

            if st.button(
                "🚀 Analyze Manual Network Flow",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "Analyzing network flow with ensemble models..."
                ):

                    X_input = manual_df.copy()

                    X_input = X_input.apply(
                        pd.to_numeric,
                        errors="coerce"
                    ).fillna(0)

                    X_scaled = scaler.transform(
                        X_input
                    )

                    dt_pred = dt_model.predict(
                        X_scaled
                    )

                    rf_pred = rf_model.predict(
                        X_scaled
                    )

                    xgb_pred = xgb_model.predict(
                        X_scaled
                    )

                    all_predictions = np.column_stack(
                        [
                            dt_pred,
                            rf_pred,
                            xgb_pred
                        ]
                    )

                    ensemble_pred = mode(
                        all_predictions,
                        axis=1,
                        keepdims=False
                    ).mode

                    probabilities = (
                        xgb_model.predict_proba(
                            X_scaled
                        )
                    )

                st.session_state.data = manual_df

                st.session_state.predictions = {

                    "decision_tree": dt_pred,
                    "random_forest": rf_pred,
                    "xgboost": xgb_pred,
                    "ensemble": ensemble_pred,
                    "probabilities": probabilities,
                    "X_scaled": X_scaled

                }

                st.session_state.manual_mode = True

                predicted_class = class_names[
                    int(ensemble_pred[0])
                ]

                st.success(
                    f"Analysis completed — CyberLens prediction: "
                    f"**{predicted_class}**"
                )

                st.session_state.page = "Dashboard"

                st.rerun()


# ============================================================
# MODEL ANALYSIS
# ============================================================

elif st.session_state.page == "Model Analysis":

    st.header("📊 Model Analysis")

    st.write(
        "Detailed evaluation of the individual classifiers "
        "and the final Voting Ensemble across all traffic classes."
    )

    st.subheader("Overall Model Performance")

    model_data = pd.DataFrame({

        "Model": [
            "Decision Tree",
            "Random Forest",
            "XGBoost",
            "Voting Ensemble"
        ],

        "Accuracy": [
            0.9993,
            0.9992,
            0.9993,
            0.9994
        ],

        "Macro Precision": [
            0.9422,
            0.9364,
            0.9395,
            0.9448
        ],

        "Macro Recall": [
            0.9922,
            0.9920,
            0.9972,
            0.9955
        ],

        "Macro F1": [
            0.9632,
            0.9584,
            0.9626,
            0.9655
        ],

        "Weighted F1": [
            0.9994,
            0.9993,
            0.9994,
            0.9994
        ]
    })

    st.dataframe(
        model_data.style.format({
            "Accuracy": "{:.4f}",
            "Macro Precision": "{:.4f}",
            "Macro Recall": "{:.4f}",
            "Macro F1": "{:.4f}",
            "Weighted F1": "{:.4f}"
        }),
        use_container_width=True
    )

    st.success(
        "🏆 Final Model Selection: Voting Ensemble"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Macro F1",
            "0.9655"
        )

    with c2:
        st.metric(
            "Macro Precision",
            "0.9448"
        )

    with c3:
        st.metric(
            "Macro Recall",
            "0.9955"
        )

    st.subheader("Macro F1 Comparison")

    fig = px.bar(
        model_data,
        x="Model",
        y="Macro F1",
        text="Macro F1",
        title="Macro F1 Score by Model"
    )

    fig.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_range=[0.90, 1.00],
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Class-wise F1-Score")

    class_data = pd.DataFrame({

        "Class": [
            "Botnet",
            "Brute Force",
            "DDoS",
            "Normal",
            "Port Scan",
            "Web Attack"
        ],

        "Decision Tree": [
            0.8004,
            0.9984,
            0.9997,
            0.9996,
            0.9996,
            0.9815
        ],

        "Random Forest": [
            0.7672,
            0.9992,
            0.9998,
            0.9996,
            0.9998,
            0.9849
        ],

        "XGBoost": [
            0.7846,
            0.9997,
            0.9999,
            0.9996,
            0.9998,
            0.9919
        ],

        "Voting Ensemble": [
            0.8025,
            0.9995,
            0.9999,
            0.9997,
            0.9998,
            0.9919
        ]
    })

    st.dataframe(
        class_data.style.format({
            "Decision Tree": "{:.4f}",
            "Random Forest": "{:.4f}",
            "XGBoost": "{:.4f}",
            "Voting Ensemble": "{:.4f}"
        }),
        use_container_width=True
    )

    chart_data = class_data.melt(
        id_vars="Class",
        var_name="Model",
        value_name="F1 Score"
    )

    fig = px.bar(
        chart_data,
        x="Class",
        y="F1 Score",
        color="Model",
        barmode="group",
        title="Class-wise F1-Score Comparison"
    )

    fig.update_layout(
        yaxis_range=[0.70, 1.01],
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Minority Attack Analysis")

    st.info(
        "Minority-class performance is particularly important "
        "because overall accuracy can be dominated by the large "
        "Normal traffic class."
    )

    minority_data = pd.DataFrame({

        "Attack Class": [
            "Botnet",
            "Web Attack"
        ],

        "Decision Tree": [
            0.8004,
            0.9815
        ],

        "Random Forest": [
            0.7672,
            0.9849
        ],

        "XGBoost": [
            0.7846,
            0.9919
        ],

        "Voting Ensemble": [
            0.8025,
            0.9919
        ]
    })

    st.dataframe(
        minority_data.style.format({
            "Decision Tree": "{:.4f}",
            "Random Forest": "{:.4f}",
            "XGBoost": "{:.4f}",
            "Voting Ensemble": "{:.4f}"
        }),
        use_container_width=True
    )

    st.subheader("Ensemble Improvement")

    xgb_f1 = 0.999389
    ensemble_f1 = 0.999444

    xgb_accuracy = 0.999325
    ensemble_accuracy = 0.999393

    f1_improvement = ensemble_f1 - xgb_f1
    accuracy_improvement = ensemble_accuracy - xgb_accuracy

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Weighted F1 Improvement",
            f"+{f1_improvement:.6f}"
        )

    with c2:
        st.metric(
            "Accuracy Improvement",
            f"+{accuracy_improvement:.6f}"
        )

    st.caption(
        "Comparison of the final Voting Ensemble against XGBoost."
    )


# ============================================================
# EXPLAINABILITY
# ============================================================

elif st.session_state.page == "Explainability":

    st.header("🧠 Explainable AI")

    st.write(
        "SHAP explanations identify the network features "
        "that contributed most strongly to an individual prediction."
    )

    if st.session_state.predictions is None:

        st.warning(
            "Please run a detection first."
        )

    else:

        predictions = st.session_state.predictions

        ensemble_pred = predictions[
            "ensemble"
        ]

        probabilities = predictions[
            "probabilities"
        ]

        X_scaled = predictions[
            "X_scaled"
        ]

        record = st.number_input(
            "Select Network Record",
            min_value=0,
            max_value=len(ensemble_pred) - 1,
            value=0,
            step=1
        )

        record = int(record)

        predicted_class = class_names[
            int(
                ensemble_pred[record]
            )
        ]

        confidence = probabilities[
            record
        ][
            int(
                ensemble_pred[record]
            )
        ]

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Ensemble Prediction",
                predicted_class
            )

        with c2:

            st.metric(
                "Model Confidence",
                f"{confidence:.2%}"
            )

        st.subheader(
            "Individual Model Predictions"
        )

        c1, c2, c3 = st.columns(3)

        individual_models = [
            (
                c1,
                "🌳",
                "Decision Tree",
                predictions["decision_tree"]
            ),
            (
                c2,
                "🌲",
                "Random Forest",
                predictions["random_forest"]
            ),
            (
                c3,
                "⚡",
                "XGBoost",
                predictions["xgboost"]
            )
        ]

        for col, icon, name, pred in individual_models:

            with col:

                prediction_name = class_names[
                    int(pred[record])
                ]

                st.markdown(
                    f"""
                    <div class="model-card">

                    <div style="font-size:28px">
                    {icon}
                    </div>

                    <div class="model-name">
                    {name}
                    </div>

                    <div class="model-status">
                    Prediction: <b>{prediction_name}</b>
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.subheader(
            "SHAP Feature Importance"
        )

        with st.spinner(
            "Generating SHAP explanation..."
        ):

            shap_values = explainer.shap_values(
                X_scaled[
                    record:record + 1
                ]
            )

        if isinstance(
            shap_values,
            list
        ):

            values = shap_values[
                int(
                    ensemble_pred[record]
                )
            ][0]

        elif np.ndim(shap_values) == 3:

            values = shap_values[
                0,
                :,
                int(
                    ensemble_pred[record]
                )
            ]

        else:

            values = shap_values[0]

        explanation = pd.DataFrame({

            "Feature": feature_names,

            "SHAP Value": values,

            "Impact": np.abs(values)

        })

        explanation = (
            explanation
            .sort_values(
                "Impact",
                ascending=False
            )
            .head(10)
        )

        st.dataframe(
            explanation.style.format({
                "SHAP Value": "{:.6f}",
                "Impact": "{:.6f}"
            }),
            use_container_width=True
        )

        plot_data = (
            explanation
            .sort_values(
                "SHAP Value"
            )
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=plot_data["SHAP Value"],
                y=plot_data["Feature"],
                orientation="h"
            )
        )

        fig.add_vline(
            x=0
        )

        fig.update_layout(
            title=f"Features Influencing: {predicted_class}",
            xaxis_title="SHAP Value",
            yaxis_title="Feature",
            height=550
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# REPORTS
# ============================================================

elif st.session_state.page == "Reports":

    st.header("📄 CyberLens Reports")

    st.write(
        "Generate and export a complete network intrusion "
        "detection report."
    )

    if st.session_state.predictions is None:

        st.warning(
            "Please run a detection first from the Detection page."
        )

    else:

        data = st.session_state.data
        predictions = st.session_state.predictions

        ensemble_pred = predictions["ensemble"]

        total_records = len(ensemble_pred)

        normal_index = class_names.index("Normal")

        normal_count = np.sum(
            ensemble_pred == normal_index
        )

        attack_count = (
            total_records - normal_count
        )

        attack_rate = (
            attack_count / total_records
        ) * 100

        dt_pred = predictions["decision_tree"]
        rf_pred = predictions["random_forest"]
        xgb_pred = predictions["xgboost"]

        agreement = np.mean(
            (
                (dt_pred == rf_pred)
                &
                (rf_pred == xgb_pred)
            )
        ) * 100

        st.subheader("Detection Summary")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Total Records",
                f"{total_records:,}"
            )

        with c2:
            st.metric(
                "Normal Traffic",
                f"{normal_count:,}"
            )

        with c3:
            st.metric(
                "Potential Attacks",
                f"{attack_count:,}"
            )

        with c4:
            st.metric(
                "Attack Rate",
                f"{attack_rate:.2f}%"
            )

        st.subheader("Prediction Distribution")

        labels = [
            class_names[int(x)]
            for x in ensemble_pred
        ]

        distribution = (
            pd.Series(labels)
            .value_counts()
            .reset_index()
        )

        distribution.columns = [
            "Class",
            "Records"
        ]

        st.dataframe(
            distribution,
            use_container_width=True
        )

        fig = px.pie(
            distribution,
            names="Class",
            values="Records",
            hole=0.45,
            title="Ensemble Prediction Distribution"
        )

        fig.update_layout(
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Model Performance")

        performance = pd.DataFrame({

            "Model": [
                "Decision Tree",
                "Random Forest",
                "XGBoost",
                "Voting Ensemble"
            ],

            "Accuracy": [
                0.9993,
                0.9992,
                0.9993,
                0.9994
            ],

            "Macro F1": [
                0.9632,
                0.9584,
                0.9626,
                0.9655
            ],

            "Weighted F1": [
                0.9994,
                0.9993,
                0.9994,
                0.9994
            ]
        })

        st.dataframe(
            performance.style.format({
                "Accuracy": "{:.4f}",
                "Macro F1": "{:.4f}",
                "Weighted F1": "{:.4f}"
            }),
            use_container_width=True
        )

        st.subheader("Final Model Selection")

        st.success(
            "Voting Ensemble selected as the final CyberLens model."
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Macro F1",
                "0.9655"
            )

        with c2:
            st.metric(
                "Macro Precision",
                "0.9448"
            )

        with c3:
            st.metric(
                "Macro Recall",
                "0.9955"
            )

        st.metric(
            "Model Agreement",
            f"{agreement:.2f}%"
        )

        st.subheader("Detection Results")

        result = data.copy()

        result["Decision Tree"] = [
            class_names[int(x)]
            for x in dt_pred
        ]

        result["Random Forest"] = [
            class_names[int(x)]
            for x in rf_pred
        ]

        result["XGBoost"] = [
            class_names[int(x)]
            for x in xgb_pred
        ]

        result["Ensemble Prediction"] = [
            class_names[int(x)]
            for x in ensemble_pred
        ]

        probabilities = predictions["probabilities"]

        result["Model Confidence"] = [
            float(
                np.max(probabilities[i])
            )
            for i in range(
                len(probabilities)
            )
        ]

        st.dataframe(
            result.head(100),
            use_container_width=True
        )

        st.subheader("Export Results")

        csv_data = result.to_csv(
            index=False
        )

        st.download_button(
            label="⬇️ Download Detection Results (CSV)",
            data=csv_data,
            file_name="CyberLens_Detection_Results.csv",
            mime="text/csv",
            use_container_width=True
        )

        # ====================================================
        # HTML REPORT
        # ====================================================

        report_html = f"""
        <!DOCTYPE html>

        <html>

        <head>

        <meta charset="UTF-8">

        <title>CyberLens Detection Report</title>

        <style>

        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            line-height: 1.6;
        }}

        h1 {{
            margin-bottom: 5px;
        }}

        h2 {{
            margin-top: 35px;
        }}

        .summary {{
            display: flex;
            gap: 20px;
            margin: 25px 0;
        }}

        .card {{
            border: 1px solid #ccc;
            padding: 18px;
            border-radius: 8px;
            min-width: 150px;
        }}

        .value {{
            font-size: 25px;
            font-weight: bold;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            margin-top: 15px;
        }}

        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}

        th {{
            font-weight: bold;
        }}

        </style>

        </head>

        <body>

        <h1>CyberLens</h1>

        <p>
        Explainable Ensemble Learning Framework for
        Intelligent Network Intrusion Detection
        </p>

        <hr>

        <h2>Detection Summary</h2>

        <div class="summary">

        <div class="card">
        Total Records
        <div class="value">{total_records:,}</div>
        </div>

        <div class="card">
        Normal Traffic
        <div class="value">{normal_count:,}</div>
        </div>

        <div class="card">
        Potential Attacks
        <div class="value">{attack_count:,}</div>
        </div>

        <div class="card">
        Attack Rate
        <div class="value">{attack_rate:.2f}%</div>
        </div>

        </div>

        <h2>Model Performance</h2>

        {performance.to_html(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )}

        <h2>Final Model</h2>

        <p>
        <b>Voting Ensemble</b>
        </p>

        <p>
        Macro F1: <b>0.9655</b><br>
        Macro Precision: <b>0.9448</b><br>
        Macro Recall: <b>0.9955</b><br>
        Model Agreement: <b>{agreement:.2f}%</b>
        </p>

        <h2>Prediction Distribution</h2>

        {distribution.to_html(index=False)}

        <h2>Methodology</h2>

        <p>
        CyberLens uses three supervised machine learning
        classifiers: Decision Tree, Random Forest and XGBoost.
        Their predictions are combined using majority voting
        to produce the final ensemble prediction.
        </p>

        <p>
        SHAP-based explainability is used to identify the
        network-flow features contributing to individual
        predictions.
        </p>

        <hr>

        <p>
        Generated by CyberLens.
        </p>

        </body>

        </html>
        """

        st.download_button(
            label="📄 Download Evaluation Report (HTML)",
            data=report_html,
            file_name="CyberLens_Evaluation_Report.html",
            mime="text/html",
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="cyber-footer">
    CyberLens — Explainable Ensemble Learning Framework
    for Intelligent Network Intrusion Detection
    </div>
    """,
    unsafe_allow_html=True
)