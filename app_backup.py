import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
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
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Main title */

.cyber-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 0;
}

.cyber-subtitle {
    font-size: 17px;
    opacity: 0.65;
    margin-bottom: 30px;
}


/* Cards */

.metric-card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.25);
    background-color: rgba(128,128,128,0.05);
}

.metric-title {
    font-size: 14px;
    opacity: 0.65;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    margin-top: 5px;
}


/* Section */

.section-header {
    font-size: 25px;
    font-weight: 650;
    margin-top: 25px;
    margin-bottom: 15px;
}


/* Sidebar */

[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.2);
}


/* Buttons */

.stButton > button {
    border-radius: 8px;
    font-weight: 600;
}


/* Footer */

.footer {
    text-align: center;
    opacity: 0.5;
    margin-top: 40px;
    padding-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL DIRECTORY
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
    ) as file:

        metadata = json.load(file)

    return (
        dt_model,
        rf_model,
        xgb_model,
        scaler,
        metadata
    )


# ============================================================
# LOAD MODELS
# ============================================================

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

    st.error(
        "Unable to load CyberLens models."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SHAP
# ============================================================

@st.cache_resource
def create_explainer():

    return shap.TreeExplainer(
        xgb_model
    )


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


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# 🛡️ CyberLens"
    )

    st.caption(
        "Intelligent Network Intrusion Detection"
    )

    st.markdown("---")

    st.subheader("Navigation")

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

    st.subheader("System")

    st.success("● Models Loaded")

    st.caption(
        f"Features: {len(feature_names)}"
    )

    st.caption(
        f"Classes: {len(class_names)}"
    )

    st.caption(
        "Ensemble: Majority Voting"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="cyber-title">CyberLens</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="cyber-subtitle">'
    'Explainable Ensemble Learning Framework for '
    'Intelligent Network Intrusion Detection'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.page == "Dashboard":

    st.markdown(
        '<div class="section-header">'
        'Security Overview'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.predictions is None:

        st.info(
            "No network traffic has been analyzed yet. "
            "Go to Detection and upload a CSV dataset."
        )

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(
                """
                ### 🔍 Detection

                Upload network traffic data and
                identify potential attacks using
                three machine learning models.
                """
            )

        with col2:

            st.markdown(
                """
                ### 🗳️ Ensemble

                Combine Decision Tree, Random Forest
                and XGBoost predictions using
                majority voting.
                """
            )

        with col3:

            st.markdown(
                """
                ### 🧠 Explainability

                Understand why the model made a
                particular prediction using SHAP.
                """
            )

    else:

        data = st.session_state.data

        predictions = st.session_state.predictions

        ensemble_predictions = predictions[
            "ensemble"
        ]

        total_records = len(
            ensemble_predictions
        )

        normal_index = class_names.index(
            "Normal"
        )

        normal_count = np.sum(
            ensemble_predictions == normal_index
        )

        attack_count = (
            total_records - normal_count
        )

        attack_percentage = (
            attack_count / total_records
        ) * 100


        # ----------------------------------------------------
        # METRIC CARDS
        # ----------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Total Traffic",
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
                f"{attack_percentage:.2f}%"
            )


        st.markdown("---")


        # ----------------------------------------------------
        # DISTRIBUTION
        # ----------------------------------------------------

        col1, col2 = st.columns(
            [1, 1]
        )

        with col1:

            st.subheader(
                "Traffic Classification"
            )

            labels = [
                class_names[int(x)]
                for x in ensemble_predictions
            ]

            counts = (
                pd.Series(labels)
                .value_counts()
            )

            st.dataframe(
                counts.rename(
                    "Records"
                ),
                use_container_width=True
            )


        with col2:

            st.subheader(
                "Attack Distribution"
            )

            fig, ax = plt.subplots(
                figsize=(7, 5)
            )

            counts.plot(
                kind="bar",
                ax=ax
            )

            ax.set_xlabel(
                "Traffic Class"
            )

            ax.set_ylabel(
                "Number of Records"
            )

            ax.set_title(
                "Network Traffic Classification"
            )

            plt.xticks(
                rotation=30,
                ha="right"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


        # ----------------------------------------------------
        # MODEL AGREEMENT
        # ----------------------------------------------------

        st.subheader(
            "Model Agreement"
        )

        dt_pred = predictions[
            "decision_tree"
        ]

        rf_pred = predictions[
            "random_forest"
        ]

        xgb_pred = predictions[
            "xgboost"
        ]

        agreement = np.mean(
            (
                (dt_pred == rf_pred)
                &
                (rf_pred == xgb_pred)
            )
        ) * 100


        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Decision Tree",
                "Active"
            )

        with c2:

            st.metric(
                "Random Forest",
                "Active"
            )

        with c3:

            st.metric(
                "XGBoost",
                "Active"
            )

        with c4:

            st.metric(
                "Agreement",
                f"{agreement:.2f}%"
            )


# ============================================================
# DETECTION PAGE
# ============================================================

elif st.session_state.page == "Detection":

    st.header(
        "🔍 Network Traffic Detection"
    )

    st.write(
        "Upload network-flow data to classify "
        "traffic using the CyberLens ensemble."
    )


    uploaded_file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"]
    )


    if uploaded_file:

        try:

            data = pd.read_csv(
                uploaded_file
            )

            st.success(
                f"Dataset loaded: "
                f"{len(data):,} records × "
                f"{len(data.columns)} columns"
            )

        except Exception as e:

            st.error(
                "Unable to read CSV."
            )

            st.exception(e)

            st.stop()


        missing_features = [
            f for f in feature_names
            if f not in data.columns
        ]


        if missing_features:

            st.error(
                f"{len(missing_features)} required "
                f"features are missing."
            )

            st.stop()


        X_input = data[
            feature_names
        ].copy()


        X_input = X_input.apply(
            pd.to_numeric,
            errors="coerce"
        ).fillna(0)


        st.dataframe(
            data.head(10),
            use_container_width=True
        )


        if st.button(
            "🚀 Analyze Traffic",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing network traffic..."
            ):

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


            st.success(
                "Analysis completed successfully!"
            )

            st.info(
                "Go to Dashboard to view the "
                "security overview."
            )


# ============================================================
# MODEL ANALYSIS PAGE
# ============================================================

elif st.session_state.page == "Model Analysis":

    st.header(
        "📊 Model Analysis"
    )

    st.write(
        "Performance comparison of the models "
        "used in the CyberLens framework."
    )


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


    st.subheader(
        "Macro F1 Comparison"
    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    ax.bar(
        model_data["Model"],
        model_data["Macro F1"]
    )


    ax.set_ylabel(
        "Macro F1"
    )

    ax.set_ylim(
        0.90,
        1.00
    )

    ax.set_title(
        "Model Performance Comparison"
    )


    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    st.success(
        "Final Model: Voting Ensemble"
    )


# ============================================================
# EXPLAINABILITY PAGE
# ============================================================

elif st.session_state.page == "Explainability":

    st.header(
        "🧠 Explainable AI"
    )

    st.write(
        "SHAP-based explanations showing which "
        "network features influenced a prediction."
    )


    if st.session_state.predictions is None:

        st.warning(
            "Run a detection first."
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


        record_number = st.number_input(
            "Select Record",
            min_value=0,
            max_value=len(ensemble_pred) - 1,
            value=0
        )


        record_number = int(
            record_number
        )


        predicted_class = class_names[
            int(
                ensemble_pred[
                    record_number
                ]
            )
        ]


        confidence = probabilities[
            record_number
        ][
            int(
                ensemble_pred[
                    record_number
                ]
            )
        ]


        c1, c2 = st.columns(2)


        with c1:

            st.metric(
                "Prediction",
                predicted_class
            )


        with c2:

            st.metric(
                "Confidence",
                f"{confidence:.2%}"
            )


        st.subheader(
            "Model Predictions"
        )


        c1, c2, c3 = st.columns(3)


        with c1:

            st.info(
                "🌳 Decision Tree\n\n"
                +
                class_names[
                    int(
                        predictions[
                            "decision_tree"
                        ][record_number]
                    )
                ]
            )


        with c2:

            st.info(
                "🌲 Random Forest\n\n"
                +
                class_names[
                    int(
                        predictions[
                            "random_forest"
                        ][record_number]
                    )
                ]
            )


        with c3:

            st.info(
                "⚡ XGBoost\n\n"
                +
                class_names[
                    int(
                        predictions[
                            "xgboost"
                        ][record_number]
                    )
                ]
            )


        st.subheader(
            "SHAP Feature Importance"
        )


        with st.spinner(
            "Generating explanation..."
        ):

            shap_values = explainer.shap_values(
                X_scaled[
                    record_number:
                    record_number + 1
                ]
            )


        if isinstance(
            shap_values,
            list
        ):

            values = shap_values[
                int(
                    ensemble_pred[
                        record_number
                    ]
                )
            ][0]

        elif np.ndim(shap_values) == 3:

            values = shap_values[
                0,
                :,
                int(
                    ensemble_pred[
                        record_number
                    ]
                )
            ]

        else:

            values = shap_values[
                0
            ]


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


        fig, ax = plt.subplots(
            figsize=(11, 6)
        )


        ax.barh(
            plot_data["Feature"],
            plot_data["SHAP Value"]
        )


        ax.axvline(
            0,
            linewidth=1
        )


        ax.set_xlabel(
            "SHAP Value"
        )


        ax.set_title(
            f"Why CyberLens Predicted "
            f"{predicted_class}"
        )


        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


# ============================================================
# REPORTS PAGE
# ============================================================

elif st.session_state.page == "Reports":

    st.header(
        "📄 Reports"
    )

    st.write(
        "Export CyberLens detection results."
    )


    if st.session_state.predictions is None:

        st.warning(
            "Run a detection before generating "
            "a report."
        )

    else:

        data = st.session_state.data

        predictions = st.session_state.predictions

        result = data.copy()


        result[
            "Decision Tree"
        ] = [
            class_names[int(x)]
            for x in predictions[
                "decision_tree"
            ]
        ]


        result[
            "Random Forest"
        ] = [
            class_names[int(x)]
            for x in predictions[
                "random_forest"
            ]
        ]


        result[
            "XGBoost"
        ] = [
            class_names[int(x)]
            for x in predictions[
                "xgboost"
            ]
        ]


        result[
            "Ensemble Prediction"
        ] = [
            class_names[int(x)]
            for x in predictions[
                "ensemble"
            ]
        ]


        st.dataframe(
            result[
                [
                    "Decision Tree",
                    "Random Forest",
                    "XGBoost",
                    "Ensemble Prediction"
                ]
            ].head(100),
            use_container_width=True
        )


        csv_data = result.to_csv(
            index=False
        )


        st.download_button(
            "⬇️ Download Detection Report",
            data=csv_data,
            file_name="CyberLens_Detection_Report.csv",
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
    CyberLens — Explainable Ensemble Learning Framework
    for Intelligent Network Intrusion Detection
    </div>
    """,
    unsafe_allow_html=True
)