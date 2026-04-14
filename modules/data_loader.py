import streamlit as st
import pandas as pd
import numpy as np
import io

# File size limits (in MB)
MAX_FILE_SIZE_MB = 50  # Reasonable limit for web app
MAX_ROWS_PREVIEW = 100000  # Max rows to load initially
CHUNK_SIZE = 10000  # For chunked reading


def load_data(uploaded_file):
    try:
        # Check file size first
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)

        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"File too large! Maximum allowed size is {MAX_FILE_SIZE_MB}MB. Your file is {file_size_mb:.1f}MB.")
            st.info("💡 Try splitting your data into smaller files or sampling a subset of rows.")
            return None

        if file_size_mb > 10:
            st.warning(f"⚠️ Large file detected ({file_size_mb:.1f}MB). Loading may take time...")

        name = uploaded_file.name.lower()

        with st.spinner("Loading data..."):
            if name.endswith(".csv"):
                df = _load_csv_smart(uploaded_file, file_size_mb)
            elif name.endswith((".xlsx", ".xls")):
                df = _load_excel_smart(uploaded_file, file_size_mb)
            else:
                st.error("Unsupported format. Please upload a CSV or Excel file.")
                return None

        if df is None or df.empty:
            st.error("The uploaded file is empty or could not be loaded.")
            return None

        if df.shape[1] == 0:
            st.error("The file has no columns.")
            return None

        # Show loading summary
        st.success(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns ({file_size_mb:.1f}MB)")

        return df

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.info("💡 Try checking your file format or reducing file size.")
        return None


def _load_csv_smart(uploaded_file, file_size_mb):
    """Smart CSV loading with chunking for large files"""
    try:
        # For very large files, try chunked reading first to estimate size
        if file_size_mb > 20:
            # Read first chunk to check structure
            sample_df = pd.read_csv(uploaded_file, nrows=1000)
            total_rows = _estimate_csv_rows(uploaded_file)

            if total_rows > MAX_ROWS_PREVIEW:
                st.warning(f"📊 Large dataset detected ({total_rows:,} rows). Loading first {MAX_ROWS_PREVIEW:,} rows for preview.")
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Load in chunks
                chunks = []
                chunk_count = 0
                for chunk in pd.read_csv(uploaded_file, chunksize=CHUNK_SIZE):
                    chunks.append(chunk)
                    chunk_count += 1
                    progress = min(chunk_count * CHUNK_SIZE / MAX_ROWS_PREVIEW, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Loading chunk {chunk_count}...")

                    if len(pd.concat(chunks)) >= MAX_ROWS_PREVIEW:
                        break

                progress_bar.empty()
                status_text.empty()
                df = pd.concat(chunks).head(MAX_ROWS_PREVIEW)
            else:
                df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        return df

    except Exception as e:
        st.error(f"CSV loading failed: {str(e)}")
        return None


def _load_excel_smart(uploaded_file, file_size_mb):
    """Smart Excel loading with memory optimization"""
    try:
        # Excel files are often smaller but can still be memory intensive
        if file_size_mb > 30:
            st.warning("📊 Large Excel file detected. Loading may take time...")

        df = pd.read_excel(uploaded_file)

        # If too many rows, take a sample
        if len(df) > MAX_ROWS_PREVIEW:
            st.warning(f"📊 Large dataset ({len(df):,} rows). Showing first {MAX_ROWS_PREVIEW:,} rows.")
            df = df.head(MAX_ROWS_PREVIEW)

        return df

    except Exception as e:
        st.error(f"Excel loading failed: {str(e)}")
        return None


def _estimate_csv_rows(uploaded_file):
    """Estimate total rows in CSV without loading fully"""
    try:
        content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
        lines = content.split('\n')
        # Subtract header row
        return max(0, len(lines) - 1)
    except:
        return 0


def _cat_cols(df):
    return df.select_dtypes(include=["object", "string", "category"]).columns.tolist()


def _num_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def show_dataset_info(df):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")
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
