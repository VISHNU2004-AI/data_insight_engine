import streamlit as st
import pandas as pd
import numpy as np


def load_data(uploaded_file):
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported format. Please upload a CSV or Excel file.")
            return None

        if df.empty:
            st.error("The uploaded file is empty.")
            return None

        if df.shape[1] == 0:
            st.error("The file has no columns.")
            return None

        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def _cat_cols(df):
    return df.select_dtypes(include=["object", "string", "category"]).columns.tolist()


def _num_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def show_dataset_info(df):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Duplicate Rows", int(df.duplicated().sum()))
    with col4:
        total_missing = int(df.isnull().sum().sum())
        st.metric("Total Missing", total_missing)

    st.subheader("Column Information")
    type_df = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str).values,
        "Non-Null": df.notnull().sum().values,
        "Null": df.isnull().sum().values,
        "Null %": (df.isnull().sum() / len(df) * 100).round(1).astype(str) + "%",
        "Unique Values": [df[c].nunique() for c in df.columns],
        "Sample": [str(df[c].dropna().iloc[0]) if df[c].notnull().any() else "—" for c in df.columns],
    })
    st.dataframe(type_df, width="stretch", hide_index=True)

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        st.subheader("Missing Values Detail")
        missing_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing %": (missing / len(df) * 100).round(1).values
        }).sort_values("Missing Count", ascending=False)
        st.dataframe(missing_df, width="stretch", hide_index=True)
    else:
        st.success("No missing values — dataset is complete!")
