import streamlit as st
import pandas as pd
import numpy as np


def _cat_cols(df):
    return df.select_dtypes(include=["object", "string", "category"]).columns.tolist()


def _num_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    return df, removed


def handle_missing_numeric(df, strategy="median"):
    for col in _num_cols(df):
        null_count = df[col].isnull().sum()
        if null_count == 0:
            continue
        if strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "zero":
            df[col] = df[col].fillna(0)
        elif strategy == "drop":
            df = df.dropna(subset=[col])
    return df


def handle_missing_categorical(df, strategy="mode"):
    for col in _cat_cols(df):
        null_count = df[col].isnull().sum()
        if null_count == 0:
            continue
        if strategy == "mode":
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
        elif strategy == "unknown":
            df[col] = df[col].fillna("Unknown")
        elif strategy == "drop":
            df = df.dropna(subset=[col])
    return df


def detect_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    return outliers, lower, upper


def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df


def clean_data_auto(df):
    log = []

    # For large datasets, show progress
    if len(df) > 100000:
        st.info(f"🔄 Cleaning {len(df):,} rows... This may take a moment.")

    # Remove duplicates efficiently
    before = len(df)
    df = df.drop_duplicates(subset=None, keep='first')
    dup_removed = before - len(df)
    if dup_removed > 0:
        log.append(f"Removed {dup_removed} duplicate rows.")

    # Clean numeric columns
    numeric_cols = _num_cols(df)
    for col in numeric_cols:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            # Use median for large datasets (faster)
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            log.append(f"Filled {null_count} missing values in '{col}' with median.")

    # Clean categorical columns
    cat_cols = _cat_cols(df)
    for col in cat_cols:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
                log.append(f"Filled {null_count} missing values in '{col}' with mode.")
        
        # Convert to string for Arrow compatibility
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)

    if not log:
        log.append("Dataset is already clean — no issues found.")

    return df, log


def show_cleaning_ui(df):
    st.subheader("Cleaning Options")
    mode = st.radio("Cleaning Mode", ["Automatic", "Manual"], horizontal=True)

    if mode == "Automatic":
        st.markdown(
            "Auto-clean performs: **remove duplicates → fill numeric NaNs with median "
            "→ fill categorical NaNs with mode**."
        )
        if st.button("Run Auto-Clean", type="primary"):
            with st.spinner("Cleaning data..."):
                cleaned_df, log = clean_data_auto(df.copy())
                st.session_state["cleaned_df"] = cleaned_df
                st.success("Auto-cleaning complete!")
                for entry in log:
                    st.write(f"• {entry}")
            return cleaned_df

    else:
        cleaned = df.copy()

        st.markdown("#### Step 1 — Remove Duplicates")
        remove_dups = st.checkbox("Remove duplicate rows", value=True)
        if remove_dups:
            cleaned, removed = remove_duplicates(cleaned)
            if removed > 0:
                st.info(f"Will remove **{removed}** duplicate row(s).")

        st.markdown("#### Step 2 — Handle Missing Numeric Values")
        numeric_strategy = st.selectbox(
            "Strategy for numeric columns",
            ["median", "mean", "zero", "drop"],
        )

        st.markdown("#### Step 3 — Handle Missing Categorical Values")
        cat_strategy = st.selectbox(
            "Strategy for categorical columns",
            ["mode", "unknown", "drop"],
        )

        st.markdown("#### Step 4 — Outlier Removal (IQR Method)")
        numeric_cols = _num_cols(cleaned)
        outlier_cols = st.multiselect(
            "Select columns to remove outliers from",
            numeric_cols,
            help="Rows where values fall outside 1.5 × IQR from Q1/Q3 will be dropped."
        )

        if outlier_cols:
            preview_rows = 0
            tmp = cleaned.copy()
            for col in outlier_cols:
                _, lo, hi = detect_outliers_iqr(tmp, col)
                n_out = ((tmp[col] < lo) | (tmp[col] > hi)).sum()
                preview_rows += n_out
            st.info(f"~{preview_rows} outlier row(s) will be removed across selected columns.")

        if st.button("Apply Cleaning", type="primary"):
            with st.spinner("Applying cleaning steps..."):
                cleaned = handle_missing_numeric(cleaned, numeric_strategy)
                cleaned = handle_missing_categorical(cleaned, cat_strategy)
                if outlier_cols:
                    before = len(cleaned)
                    cleaned = remove_outliers(cleaned, outlier_cols)
                    removed_out = before - len(cleaned)
                    if removed_out > 0:
                        st.info(f"Removed {removed_out} outlier row(s).")
                st.session_state["cleaned_df"] = cleaned
                st.success("Cleaning applied successfully!")
            return cleaned

    return st.session_state.get("cleaned_df", df)
