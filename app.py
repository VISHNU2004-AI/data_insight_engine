import io
import streamlit as st
import pandas as pd

from modules.data_loader   import load_data, show_dataset_info
from modules.data_cleaner  import show_cleaning_ui
from modules.data_analysis import (
    show_summary_statistics, show_correlation_analysis,
    show_distributions, show_trend_analysis
)
from modules.visualizations import auto_visualize, custom_visualization
from modules.insights       import display_insights
from modules.ml_module      import show_ml_module

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Scientist",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── SESSION STATE INIT ────────────────────────────────────────────────────────
for key in ("df", "cleaned_df", "last_file_name"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 AI Data Scientist")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx", "xls"],
        help="Supports CSV and Excel files (any size)"
    )

    if uploaded_file:
        if uploaded_file.name != st.session_state["last_file_name"]:
            df_loaded = load_data(uploaded_file)
            if df_loaded is not None:
                st.session_state["df"]             = df_loaded
                st.session_state["cleaned_df"]     = df_loaded.copy()
                st.session_state["last_file_name"] = uploaded_file.name
                st.session_state.pop("ml_trained", None)
                st.session_state.pop("ml_results", None)

    if st.session_state["df"] is not None:
        df_info = st.session_state["df"]
        st.success(f"✅ {st.session_state['last_file_name']}")
        st.caption(f"{df_info.shape[0]:,} rows × {df_info.shape[1]} columns")
        miss = int(df_info.isnull().sum().sum())
        if miss:
            st.warning(f"⚠️ {miss:,} missing values detected")
        else:
            st.caption("✅ No missing values")
        if st.session_state.get("ml_trained"):
            st.caption("🤖 Model trained")

    st.markdown("---")

    if st.session_state["cleaned_df"] is not None:
        cleaned = st.session_state["cleaned_df"]
        st.markdown("### 📥 Downloads")

        csv_buf = io.StringIO()
        cleaned.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Cleaned CSV",
            data=csv_buf.getvalue(),
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

        xl_buf = io.BytesIO()
        cleaned.to_excel(xl_buf, index=False, engine="openpyxl")
        st.download_button(
            "⬇️ Cleaned Excel",
            data=xl_buf.getvalue(),
            file_name="cleaned_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    st.markdown("---")
    st.caption("Built with Streamlit · Plotly · scikit-learn · python-pptx")

# ── LANDING PAGE ──────────────────────────────────────────────────────────────
if st.session_state["df"] is None:
    st.title("🤖 AI Data Scientist Web App")
    st.markdown("#### Upload any CSV or Excel file and get instant analysis, visualizations, ML predictions, and a downloadable report.")
    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.info("**🔍 Data Preview**\n\nColumn types, missing values, and raw data at a glance.")
    with c2:
        st.info("**🧹 Smart Cleaning**\n\nAuto or manual: dedup, fill NaNs, remove outliers.")
    with c3:
        st.info("**📊 Deep Analysis**\n\nCorrelation heatmaps, distributions, trends, statistics.")
    with c4:
        st.info("**🤖 ML Prediction**\n\nAuto-detect regression vs classification, train & predict.")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.info("**📈 Visualization**\n\nAuto-charts + 12-type custom builder. Download any chart.")
    with c6:
        st.info("**💡 Insights**\n\nAutomated insight engine — correlations, outliers, skew & more.")
    with c7:
        st.info("**📑 PPT Report**\n\nOne-click professional PowerPoint with charts & ML results.")
    with c8:
        st.info("**💾 Export**\n\nDownload cleaned data (CSV/Excel), charts, model, and report.")

    st.markdown("---")
    st.markdown("**Accepted formats:** CSV (`.csv`)  ·  Excel (`.xlsx`, `.xls`)")
    st.info("👈 Use the sidebar to upload your dataset and get started.")
    st.stop()

# ── DATA REFERENCES ───────────────────────────────────────────────────────────
df         = st.session_state["df"]
working_df = st.session_state.get("cleaned_df", df)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔍 Data Preview",
    "🧹 Data Cleaning",
    "📊 Analysis",
    "📈 Visualization",
    "🤖 Prediction",
    "💡 Insights",
    "📑 Report"
])

# ── TAB 1: DATA PREVIEW ───────────────────────────────────────────────────────
with tab1:
    st.header("Data Preview")
    show_dataset_info(df)
    st.markdown("---")
    st.subheader("Dataset Sample")
    n = st.slider("Rows to display", 5, min(500, len(df)), min(20, len(df)), key="prev_rows")
    st.dataframe(df.head(n), width="stretch", hide_index=True)

# ── TAB 2: DATA CLEANING ──────────────────────────────────────────────────────
with tab2:
    st.header("Data Cleaning")
    result = show_cleaning_ui(df)
    if result is not None:
        st.session_state["cleaned_df"] = result
        working_df = result

    st.markdown("---")
    st.subheader("Cleaned Data Preview")
    cleaned = st.session_state["cleaned_df"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows",           cleaned.shape[0], delta=cleaned.shape[0] - df.shape[0])
    c2.metric("Columns",        cleaned.shape[1])
    c3.metric("Missing Values", int(cleaned.isnull().sum().sum()))
    st.dataframe(cleaned.head(20), width="stretch", hide_index=True)

# ── TAB 3: ANALYSIS ───────────────────────────────────────────────────────────
with tab3:
    st.header("Statistical Analysis")
    show_summary_statistics(working_df)
    st.markdown("---")
    show_correlation_analysis(working_df)
    st.markdown("---")
    show_distributions(working_df)
    st.markdown("---")
    show_trend_analysis(working_df)

# ── TAB 4: VISUALIZATION ──────────────────────────────────────────────────────
with tab4:
    st.header("Visualizations")
    mode = st.radio("Mode", ["Auto-Generated", "Custom Builder"], horizontal=True, key="viz_mode")
    if mode == "Auto-Generated":
        auto_visualize(working_df)
    else:
        custom_visualization(working_df)

# ── TAB 5: PREDICTION ─────────────────────────────────────────────────────────
with tab5:
    st.header("ML Prediction")
    show_ml_module(working_df)

# ── TAB 6: INSIGHTS ───────────────────────────────────────────────────────────
with tab6:
    st.header("Automated Insights")
    display_insights(working_df)

# ── TAB 7: REPORT ─────────────────────────────────────────────────────────────
with tab7:
    st.header("📑 Downloadable Report")
    st.markdown(
        "Generate a professional **PowerPoint presentation** with dataset overview, "
        "cleaning summary, visualizations, insights, and ML results."
    )

    ml_results = st.session_state.get("ml_results")
    if ml_results:
        st.success("🤖 ML results from the Prediction tab will be included automatically.")
    else:
        st.info("💡 Train a model in the **Prediction** tab to include ML results in the report.")

    file_label = st.session_state.get("last_file_name", "dataset").replace(".csv", "").replace(".xlsx", "")

    col_left, _ = st.columns([1, 2])
    with col_left:
        if st.button("🎨 Generate PowerPoint Report", type="primary", key="gen_ppt"):
            with st.spinner("Building your presentation… (this may take a few seconds)"):
                try:
                    from modules.insights       import generate_insights
                    from modules.report_generator import generate_ppt

                    insights_list = generate_insights(working_df)
                    ppt_buf = generate_ppt(
                        df        = df,
                        cleaned_df = working_df,
                        insights  = insights_list,
                        ml_results = ml_results,
                        file_name  = file_label
                    )
                    st.session_state["ppt_buf"]   = ppt_buf
                    st.session_state["ppt_label"] = file_label
                    st.success("✅ Presentation ready!")
                except Exception as e:
                    import traceback
                    st.error(f"Report generation failed: {e}")
                    st.code(traceback.format_exc())

    if st.session_state.get("ppt_buf"):
        st.download_button(
            label     = "⬇️ Download PowerPoint (.pptx)",
            data      = st.session_state["ppt_buf"],
            file_name = f"{st.session_state['ppt_label']}_report.pptx",
            mime      = "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            key       = "dl_ppt"
        )
        st.markdown("---")
        st.markdown("**Report contains:**")
        slides_list = [
            "1. Title slide",
            "2. Dataset overview (metrics + column table)",
            "3. Data cleaning summary",
            "4. Feature distributions (chart)",
            "5. Correlation heatmap",
            "6. Key automated insights",
        ]
        if ml_results:
            slides_list.append("7. ML results (metrics + feature importance)")
        slides_list.append(f"{'8' if ml_results else '7'}. Conclusion & next steps")
        for s in slides_list:
            st.markdown(f"  • {s}")
