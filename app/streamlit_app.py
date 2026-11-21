from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st

from transformer_pm.config import (
    DEFAULT_COST_CONFIG_PATH,
    DEFAULT_MODEL_BUNDLE_PATH,
    FEATURE_COLUMNS,
)
from transformer_pm.inference import load_model_bundle, score_input_frame
from transformer_pm.maintenance import build_maintenance_plan, load_cost_config


st.set_page_config(page_title="Transformer Predictive Maintenance", layout="wide")
st.title("Transformer Predictive Maintenance")
st.caption("DGA fault classification and maintenance prioritization")


@st.cache_resource
def load_bundle():
    return load_model_bundle(DEFAULT_MODEL_BUNDLE_PATH)


@st.cache_resource
def load_config():
    return load_cost_config(DEFAULT_COST_CONFIG_PATH)


try:
    bundle = load_bundle()
    cost_config = load_config()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

tab_manual, tab_file = st.tabs(["Manual analysis", "Batch spreadsheet"])

with tab_manual:
    st.subheader("Dissolved gas concentrations")
    col1, col2, col3, col4, col5 = st.columns(5)
    values = [
        col1.number_input("H2", min_value=0.0, value=100.0),
        col2.number_input("CH4", min_value=0.0, value=50.0),
        col3.number_input("C2H6", min_value=0.0, value=25.0),
        col4.number_input("C2H4", min_value=0.0, value=30.0),
        col5.number_input("C2H2", min_value=0.0, value=10.0),
    ]

    if st.button("Analyze transformer", type="primary"):
        raw_df = pd.DataFrame([dict(zip(FEATURE_COLUMNS, values))])
        scored_df = score_input_frame(bundle, raw_df)
        plan_df, summary = build_maintenance_plan(
            scored_df,
            labels=bundle.labels,
            cost_config=cost_config,
        )
        row = plan_df.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted fault", row["Predicted Fault Type"])
        c2.metric("Priority", row["Priority"])
        c3.metric("Confidence", f"{row['Prediction Confidence']:.1%}")
        st.info(row["Recommended Action"])

with tab_file:
    uploaded = st.file_uploader("Upload an Excel or CSV file", type=["xlsx", "xls", "csv"])
    if uploaded is not None:
        if uploaded.name.lower().endswith(".csv"):
            raw_df = pd.read_csv(uploaded)
        else:
            raw_df = pd.read_excel(BytesIO(uploaded.getvalue()))
        try:
            scored_df = score_input_frame(bundle, raw_df)
            plan_df, summary = build_maintenance_plan(
                scored_df,
                labels=bundle.labels,
                cost_config=cost_config,
            )
        except (KeyError, ValueError) as exc:
            st.error(f"Could not process the spreadsheet: {exc}")
        else:
            st.success(f"Analyzed {len(plan_df)} transformer samples.")
            st.dataframe(plan_df, use_container_width=True)
            st.json(summary)
            st.download_button(
                "Download maintenance plan",
                plan_df.to_csv(index=False).encode("utf-8"),
                file_name="maintenance_plan.csv",
                mime="text/csv",
            )
