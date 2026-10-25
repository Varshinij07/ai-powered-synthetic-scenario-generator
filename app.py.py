import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from data_generator import generate_network_data


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="AI powered synthetic scenario generator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# PROFESSIONAL DARK THEME
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background-color: #111827 !important;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #e5e7eb;
    }

    h1, h2, h3 {
        color: #93c5fd !important;
    }

    [data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #475569;
        padding: 18px;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700;
    }

    .stMarkdown p,
    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: #cbd5e1 !important;
    }

    .stSelectbox label,
    .stMultiSelect label,
    .stSlider label,
    .stNumberInput label {
        color: #f1f5f9 !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #475569;
        border-radius: 10px;
    }

    .hero {
        background: linear-gradient(120deg, #1e3a8a, #172033);
        border: 1px solid #475569;
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: #ffffff !important;
    }

    .hero p {
        color: #e2e8f0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# PATHS AND CONSTANTS
# --------------------------------------------------
DATA_PATH = "data/network_traffic.csv"
METRICS_PATH = "data/model_metrics.json"
MATRIX_PATH = "data/confusion_matrix.csv"
REPORT_PATH = "data/classification_report.json"
MODEL_PATH = "models/best_model.joblib"

# Keep this order consistent with train_model.py.
FEATURES = [
    "duration",
    "packets",
    "bytes",
    "failed_connections",
    "connection_rate",
    "bytes_per_packet",
    "failed_connection_ratio",
    "traffic_intensity",
]

CLASSES = [
    "Normal",
    "DoS",
    "Brute Force",
    "Port Scan",
    "Data Exfiltration",
]


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------
required_files = [
    DATA_PATH,
    METRICS_PATH,
    MATRIX_PATH,
    MODEL_PATH,
]

missing_files = [path for path in required_files if not os.path.exists(path)]

st.sidebar.title("🛡️ Navigation")
page = st.sidebar.radio(
    "Choose a section",
    [
        "Security Overview",
        "Model Performance",
        "Scenario Generator",
        "Dataset Explorer",
        "About Project",
    ],
)

st.sidebar.divider()
st.sidebar.caption("AI powered synthetic scenario generator")
st.sidebar.caption("Educational cybersecurity simulation")

if missing_files:
    st.error("Some required project files are missing.")
    st.write("Missing files:", missing_files)
    st.code("python data_generator.py\npython train_model.py")
    st.stop()


# --------------------------------------------------
# LOAD PROJECT FILES
# --------------------------------------------------
try:
    df = pd.read_csv(DATA_PATH)

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        model_metrics = json.load(file)

    confusion_df = pd.read_csv(MATRIX_PATH, index_col=0)
    model = joblib.load(MODEL_PATH)

    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r", encoding="utf-8") as file:
            class_report = json.load(file)
    else:
        class_report = {}

    missing_features = [feature for feature in FEATURES if feature not in df.columns]
    if missing_features:
        st.error(
            "The dataset does not contain all eight expected features. "
            "Regenerate the dataset and retrain the model."
        )
        st.write("Missing features:", missing_features)
        st.code("python data_generator.py\npython train_model.py")
        st.stop()

except Exception as exc:
    st.error(f"Unable to load project data: {exc}")
    st.stop()


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🛡️ AI powered synthetic scenario generator</h1>
        <p>
        Intelligent network traffic simulation, machine-learning
        classification, model evaluation and cybersecurity analytics.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# SECURITY OVERVIEW
# --------------------------------------------------
if page == "Security Overview":
    st.header("Security Operations Overview")

    total_records = len(df)
    attack_records = int((df["label"] != "Normal").sum())
    attack_share = attack_records / total_records * 100 if total_records else 0

    metric_table = pd.DataFrame(model_metrics).T

    if "cv_f1_macro_mean" in metric_table.columns:
        best_name = metric_table["cv_f1_macro_mean"].astype(float).idxmax()
        best_f1 = float(metric_table.loc[best_name, "cv_f1_macro_mean"])
        selection_label = "Best CV Macro F1"
    elif "f1_macro" in metric_table.columns:
        best_name = metric_table["f1_macro"].astype(float).idxmax()
        best_f1 = float(metric_table.loc[best_name, "f1_macro"])
        selection_label = "Best Test Macro F1"
    else:
        best_name = "Not available"
        best_f1 = 0.0
        selection_label = "Best Macro F1"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Traffic Records", f"{total_records:,}")
    c2.metric("Attack Records", f"{attack_records:,}")
    c3.metric("Attack Share", f"{attack_share:.1f}%")
    c4.metric(selection_label, f"{best_f1:.4f}")

    st.caption(
        f"Selected model by cross-validation: {best_name}. "
        "Scores describe the current synthetic-data experiment."
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Traffic Class Distribution")
        class_counts = df["label"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        class_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Traffic class")
        ax.set_ylabel("Number of records")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Connection Rate by Class")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df, x="label", y="connection_rate", ax=ax)
        ax.set_xlabel("Traffic class")
        ax.set_ylabel("Connection rate")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Recent Synthetic Traffic Records")
    st.dataframe(df.tail(15), use_container_width=True, hide_index=True)


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------
elif page == "Model Performance":
    st.header("Machine Learning Model Evaluation")

    st.warning(
        "These results use a held-out test split from a synthetic dataset. "
        "They do not establish performance on real network traffic."
    )

    metric_df = pd.DataFrame(model_metrics).T

    display_columns = [
        col for col in [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "f1_weighted",
            "roc_auc_ovr_macro",
            "false_positive_rate_macro",
            "cv_f1_macro_mean",
            "cv_f1_macro_std",
            "training_seconds",
        ]
        if col in metric_df.columns
    ]

    st.subheader("Model Comparison")
    st.dataframe(
        metric_df[display_columns].style.format("{:.4f}"),
        use_container_width=True,
    )

    st.caption(
        "CV macro F1 is the mean score across five stratified folds on the "
        "training portion. The final test set is kept separate from model selection."
    )

    available_metrics = [
        col for col in [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "cv_f1_macro_mean",
        ]
        if col in metric_df.columns
    ]

    if available_metrics:
        selected_metric = st.selectbox(
            "Compare models using",
            available_metrics,
            format_func=lambda value: value.replace("_", " ").title(),
        )

        fig, ax = plt.subplots(figsize=(9, 4))
        metric_df[selected_metric].sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel(selected_metric.replace("_", " ").title())
        ax.set_xlim(0, 1)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Per-Class Detection Performance")

    class_rows = {
        name: values
        for name, values in class_report.items()
        if isinstance(values, dict)
        and all(
            key in values
            for key in ["precision", "recall", "f1-score", "support"]
        )
    }

    if class_rows:
        class_df = pd.DataFrame(class_rows).T
        class_df = class_df.rename(
            columns={
                "precision": "Precision",
                "recall": "Recall",
                "f1-score": "F1-score",
                "support": "Test Samples",
            }
        )

        st.dataframe(
            class_df[
                ["Precision", "Recall", "F1-score", "Test Samples"]
            ].style.format({
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-score": "{:.4f}",
                "Test Samples": "{:.0f}",
            }),
            use_container_width=True,
        )
    else:
        st.info(
            "The per-class report is unavailable. "
            "Run python train_model.py to regenerate it."
        )

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(confusion_df, annot=True, fmt="g", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "Download Model Metrics",
        data=json.dumps(model_metrics, indent=4),
        file_name="model_metrics.json",
        mime="application/json",
    )


# --------------------------------------------------
# SCENARIO GENERATOR
# --------------------------------------------------
elif page == "Scenario Generator":
    st.header("Synthetic Attack Scenario Generator")
    st.write(
        "Generate simulated network traffic for controlled "
        "cybersecurity experiments."
    )

    with st.form("scenario_form"):
        selected_class = st.selectbox("Select scenario category", CLASSES)
        number = st.slider(
            "Number of synthetic records",
            min_value=1,
            max_value=500,
            value=25,
        )
        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=999999,
            value=42,
            step=1,
        )
        generate = st.form_submit_button("Generate Scenario", type="primary")

    if generate:
        batch = generate_network_data(
            samples_per_class=int(number),
            random_state=int(seed),
        )

        generated = batch[batch["label"] == selected_class].copy()

        if generated.empty:
            st.error("No synthetic records were generated.")
        else:
            # Pass exactly the same eight features and order used for training.
            scenario_features = generated[FEATURES]
            generated["predicted_class"] = model.predict(scenario_features)

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(scenario_features)
                generated["prediction_confidence"] = probabilities.max(axis=1)

            generated["prediction_matches_label"] = (
                generated["label"] == generated["predicted_class"]
            )

            st.session_state["generated_scenarios"] = generated

    if "generated_scenarios" in st.session_state:
        generated = st.session_state["generated_scenarios"]

        st.subheader("Generated Scenario Results")
        match_rate = generated["prediction_matches_label"].mean() * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Generated Records", len(generated))
        c2.metric("Selected Scenario", generated["label"].iloc[0])
        c3.metric("Prediction Agreement", f"{match_rate:.1f}%")

        st.dataframe(generated, use_container_width=True, hide_index=True)

        st.download_button(
            "Download Scenario CSV",
            data=generated.to_csv(index=False),
            file_name="synthetic_attack_scenarios.csv",
            mime="text/csv",
        )


# --------------------------------------------------
# DATASET EXPLORER
# --------------------------------------------------
elif page == "Dataset Explorer":
    st.header("Synthetic Dataset Explorer")

    selected_classes = st.multiselect(
        "Filter by traffic class",
        options=sorted(df["label"].unique()),
        default=sorted(df["label"].unique()),
    )

    filtered = df[df["label"].isin(selected_classes)]
    st.write(f"Records displayed: {len(filtered):,}")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Filtered Dataset",
        data=filtered.to_csv(index=False),
        file_name="filtered_network_traffic.csv",
        mime="text/csv",
    )

    st.subheader("Feature Summary")
    numeric_summary = filtered[FEATURES].describe().T
    st.dataframe(numeric_summary, use_container_width=True)


# --------------------------------------------------
# ABOUT
# --------------------------------------------------
elif page == "About Project":
    st.header("About This Project")

    st.markdown(
        """
        **AI powered synthetic scenario generator** is an educational
        cybersecurity analytics application.

        ### Main capabilities

        - Generate labeled synthetic network traffic.
        - Engineer additional traffic features from generated values.
        - Train and compare machine-learning classifiers.
        - Evaluate models with cross-validation and held-out test metrics.
        - Explore traffic classes and numerical features.
        - Generate downloadable synthetic attack scenarios.

        ### Current limitations

        - The data is simulated and is not a real network capture.
        - Synthetic-data scores may overestimate real-world performance.
        - The application is a research and demonstration tool, not a
          production intrusion detection system.
        """
    )

    st.info(
        "For a stronger real-world evaluation, test the pipeline on a "
        "recognized intrusion-detection dataset with appropriate "
        "preprocessing and leakage prevention."
    )


st.sidebar.divider()
st.sidebar.caption("Built with Python, Streamlit and scikit-learn")
