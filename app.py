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
from modules.stock_analyzer import show_stock_analyzer

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Insight Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_styles(theme: str = "dark") -> None:
    if theme == "light":
        root_bg = "#f2faf2"
        body_bg = "#ffffff"
        app_bg = "#ffffff"
        sidebar_bg = "#f7fff7"
        card_bg = "#ffffff"
        text_color = "#0f172a"
        subtitle_color = "#22663f"
        border_color = "rgba(16, 185, 129, 0.18)"
        badge_bg = "rgba(16, 185, 129, 0.12)"
        badge_border = "rgba(16, 185, 129, 0.30)"
        button_bg = "linear-gradient(135deg, #16a34a 0%, #22c55e 100%)"
        button_color = "#ffffff"
    else:
        root_bg = "#020617"
        body_bg = "#020617"
        app_bg = "#071123"
        sidebar_bg = "rgba(12, 16, 31, 0.96)"
        card_bg = "rgba(15, 23, 42, 0.92)"
        text_color = "#e2e8f0"
        subtitle_color = "#94a3b8"
        border_color = "rgba(56, 189, 248, 0.16)"
        badge_bg = "rgba(56, 189, 248, 0.10)"
        badge_border = "rgba(56, 189, 248, 0.26)"
        button_bg = "linear-gradient(135deg, #38bdf8 0%, #6366f1 100%)"
        button_color = "#ffffff"

    st.markdown(
        f"""
        <style>
        :root {{
            color-scheme: {theme};
            color: {text_color};
            background-color: {root_bg};
            font-family: 'Inter', 'Segoe UI', sans-serif;
            --shadow-sm: 0 2px 8px rgba(15, 23, 42, 0.06);
            --shadow-md: 0 8px 20px rgba(15, 23, 42, 0.10);
            --shadow-lg: 0 20px 48px rgba(15, 23, 42, 0.12);
            --shadow-xl: 0 28px 60px rgba(15, 23, 42, 0.16);
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        * {{
            transition: var(--transition);
        }}

        body {{
            background: {body_bg} !important;
            font-weight: 400;
            letter-spacing: 0.3px;
        }}

        [data-testid="stAppViewContainer"] {{
            background: {app_bg} !important;
            color: {text_color} !important;
        }}

        [data-testid="stMainContainer"] {{
            background: transparent !important;
        }}

        .main .block-container {{
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            background: transparent !important;
        }}

        .css-18e3th9, .css-1d391kg, .css-1lcbmhc {{
            background: transparent !important;
        }}

        header,
        [data-testid="stToolbar"],
        [data-testid="stTopBar"],
        [data-testid="MainMenu"],
        .css-1lsmgbg,
        .css-17lntkn {{
            background: transparent !important;
            box-shadow: none !important;
            border: none !important;
            color: inherit !important;
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: {text_color} !important;
            font-weight: 800 !important;
            letter-spacing: -0.5px !important;
            line-height: 1.2 !important;
        }}

        h1 {{
            font-size: 3.1rem !important;
            margin-bottom: 0.8rem !important;
        }}

        h2 {{
            font-size: 2.4rem !important;
            margin-bottom: 0.6rem !important;
        }}

        h3 {{
            font-size: 1.9rem !important;
            margin-bottom: 0.5rem !important;
        }}

        h4, h5, h6 {{
            font-weight: 700 !important;
            letter-spacing: -0.25px !important;
        }}

        p, span, a, label, li,
        .stMarkdown, .stText {{
            color: {text_color} !important;
            letter-spacing: 0.3px !important;
            line-height: 1.65 !important;
        }}

        .stButton>button {{
            border-radius: 999px !important;
            padding: 0.95rem 1.55rem !important;
            background: {button_bg} !important;
            color: {button_color} !important;
            border: none !important;
            box-shadow: var(--shadow-md) !important;
            font-weight: 700 !important;
            letter-spacing: 0.3px !important;
            transition: var(--transition) !important;
        }}

        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg) !important;
        }}

        .stButton>button:active {{
            transform: translateY(0px);
            box-shadow: var(--shadow-sm) !important;
        }}

        [data-testid="stSidebar"] {{
            background: {sidebar_bg} !important;
            color: {text_color} !important;
        }}

        [data-testid="stSidebar"] .block-container,
        [data-testid="stSidebar"] .sidebar-content,
        [data-testid="stSidebar"] .css-1d391kg,
        [data-testid="stSidebar"] .css-18e3th9,
        [data-testid="stSidebar"] .css-1lcbmhc,
        [data-testid="stSidebar"] section {{
            background: {sidebar_bg} !important;
            color: {text_color} !important;
            border: 1px solid {border_color} !important;
        }}

        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] select,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] button {{
            color: {text_color} !important;
            background: transparent !important;
        }}

        [data-testid="stSidebar"] .stRadio,
        [data-testid="stSidebar"] .stFileUploader,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stText {{
            color: {text_color} !important;
        }}

        [data-testid="stSidebar"] .css-8h2u6s {{
            padding-top: 0 !important;
        }}

        [data-testid="stSidebar"] .block-container {{
            background: {sidebar_bg} !important;
            border: 1px solid {border_color};
            border-radius: 24px;
            padding: 1.35rem 1.35rem 1rem 1.35rem;
            box-shadow: 0 28px 60px rgba(15, 23, 42, 0.12);
        }}

        .sidebar-title {{
            color: {text_color};
            font-size: 1.75rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }}

        .sidebar-subtitle {{
            color: {subtitle_color};
            margin-bottom: 1.3rem;
            font-size: 0.95rem;
        }}

        .sidebar-badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.5rem 0.95rem;
            border-radius: 999px;
            border: 1px solid {badge_border};
            background: {badge_bg};
            color: {text_color};
            font-size: 0.88rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }}

        .section-card, .hero-card, .info-card {{
            background: {card_bg};
            border: 1px solid {border_color};
            border-radius: 24px;
            padding: 1.6rem;
            box-shadow: 0 22px 40px rgba(15, 23, 42, 0.08);
            transition: var(--transition);
        }}

        .section-card:hover, .hero-card:hover, .info-card:hover {{
            box-shadow: var(--shadow-lg);
            transform: translateY(-4px);
            border-color: {button_bg};
        }}

        .hero-title {{
            color: {text_color};
            font-size: 2.9rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
        }}

        .hero-subtitle, .section-subtitle {{
            color: {subtitle_color};
            font-size: 1rem;
            line-height: 1.75;
        }}

        button[role="tab"] {{
            border-radius: 999px !important;
            padding: 0.8rem 1.25rem !important;
            margin-right: 0.35rem !important;
            border: 1px solid {border_color} !important;
            background: transparent !important;
            color: {text_color} !important;
            font-weight: 600 !important;
            box-shadow: none !important;
            transition: var(--transition) !important;
        }}

        button[role="tab"]:hover {{
            background: rgba(56, 189, 248, 0.05) !important;
            border-color: rgba(56, 189, 248, 0.3) !important;
        }}

        button[role="tab"][aria-selected="true"] {{
            background: {button_bg} !important;
            color: #ffffff !important;
            border-color: transparent !important;
            box-shadow: var(--shadow-md) !important;
        }}

        .stDataFrame div[data-testid="stDataFrame"] {{
            background: transparent !important;
        }}

        .stDataFrame tbody tr td, .stDataFrame thead tr th {{
            color: {text_color} !important;
        }}

        .metric-card {{
            background: {card_bg} !important;
            border: 1px solid {border_color};
            border-radius: 18px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            box-shadow: var(--shadow-sm);
            transition: var(--transition);
        }}

        .metric-card:hover {{
            box-shadow: var(--shadow-md);
            transform: translateY(-1px);
        }}

        .metric-card strong {{
            color: {text_color};
            font-weight: 700;
        }}

        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > select,
        .stTextArea > div > div > textarea {{
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid {border_color} !important;
            color: {text_color} !important;
            border-radius: 12px !important;
            padding: 0.75rem 1rem !important;
            font-weight: 500 !important;
            transition: var(--transition) !important;
        }}

        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus,
        .stTextArea > div > div > textarea:focus {{
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15) !important;
            border-color: rgba(56, 189, 248, 0.5) !important;
        }}

        .stCheckbox, .stRadio {{
            margin: 0.5rem 0 !important;
        }}

        .stSelectbox, .stSlider {{
            margin-bottom: 1rem !important;
        }}

        .hero-badges {{
            display: flex;
            gap: 0.75rem;
            margin-top: 1.4rem;
            flex-wrap: wrap;
        }}

        .hero-badge {{
            display: inline-flex;
            align-items: center;
            padding: 0.55rem 1.35rem;
            border-radius: 999px;
            border: 1px solid rgba(56, 189, 248, 0.3);
            background: rgba(56, 189, 248, 0.08);
            color: {text_color};
            font-size: 0.88rem;
            font-weight: 700;
            letter-spacing: 0.4px;
        }}

        .section-separator {{
            border: none !important;
            border-top: 2px solid {border_color} !important;
            margin: 2rem 0 !important;
        }}

        .stAlert {{
            border-radius: 16px !important;
            border: 1px solid currentColor !important;
            padding: 1.2rem !important;
            background: rgba(56, 189, 248, 0.06) !important;
        }}

        .stAlert > div {{
            font-weight: 500 !important;
            letter-spacing: 0.2px !important;
        }}

        .stSuccess {{
            border-color: rgba(34, 197, 94, 0.4) !important;
            background: rgba(34, 197, 94, 0.08) !important;
        }}

        .stWarning {{
            border-color: rgba(251, 146, 60, 0.4) !important;
            background: rgba(251, 146, 60, 0.08) !important;
        }}

        .stError {{
            border-color: rgba(239, 68, 68, 0.4) !important;
            background: rgba(239, 68, 68, 0.08) !important;
        }}

        .stInfo {{
            border-color: rgba(56, 189, 248, 0.4) !important;
            background: rgba(56, 189, 248, 0.08) !important;
        }}

        .stCaption {{
            color: {subtitle_color} !important;
            font-weight: 500 !important;
        }}

        .stDataFrame {{
            border-radius: 16px !important;
        }}

        dt {{
            font-weight: 700 !important;
            color: {text_color} !important;
        }}

        dd {{
            color: {subtitle_color} !important;
        }}

        .stSlider > div > div > div > div {{
            background: {button_bg} !important;
            border-radius: 999px !important;
        }}

        .stSlider > div > div > div > input {{
            border-radius: 999px !important;
        }}

        [data-testid="stHorizontalBlock"] {{
            gap: 1rem !important;
        }}

        [data-testid="stVerticalBlock"] {{
            gap: 1rem !important;
        }}

        .stMetric {{
            background: transparent !important;
        }}

        .metric {{
            background: {card_bg};
            border: 1px solid {border_color};
            border-radius: 18px;
            padding: 1.2rem;
            box-shadow: var(--shadow-sm);
        }}

        .stPlotlyChart {{
            border-radius: 20px;
            overflow: hidden;
        }}

        .plotly-graph-div {{
            border-radius: 20px !important;
        }}

        .expander {{
            border-radius: 18px !important;
            border: 1px solid {border_color} !important;
        }}

        .stExpander {{
            border-radius: 18px !important;
        }}

        [data-testid="stExpander"] {{
            border-radius: 18px !important;
            border: 1px solid {border_color} !important;
            margin-bottom: 0.8rem !important;
        }}

        [data-testid="stExpander"] > details > summary {{
            padding: 1rem 1.2rem !important;
            font-weight: 700 !important;
            color: {text_color} !important;
        }}

        [data-testid="stExpander"] > details > summary:hover {{
            background: rgba(56, 189, 248, 0.04) !important;
        }}

        [data-testid="stExpander"] > details {{
            border-radius: 18px !important;
        }}

        .css-1kyxreq {{
            padding: 0 !important;
        }}

        [data-testid="stNumberInput"],
        [data-testid="stTextInput"],
        [data-testid="stSelectbox"] {{
            border-radius: 12px !important;
        }}

        .stDownloadButton > button {{
            border-radius: 999px !important;
            padding: 0.8rem 1.4rem !important;
            background: linear-gradient(135deg, #16a34a 0%, #22c55e 100%) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            box-shadow: var(--shadow-md) !important;
        }}

        .stDownloadButton > button:hover {{
            box-shadow: var(--shadow-lg) !important;
            transform: translateY(-2px);
        }}
        
        
        
        </style>
        """,
        unsafe_allow_html=True,
    )

# ── SESSION STATE INIT ────────────────────────────────────────────────────────
for key in ("df", "cleaned_df", "last_file_name", "ui_theme"):
    if key not in st.session_state:
        st.session_state[key] = "Dark" if key == "ui_theme" else None

current_theme = st.session_state["ui_theme"]
inject_custom_styles(current_theme.lower())

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class='sidebar-title'>Data Insight Studio</div>
        <div class='sidebar-subtitle'>Executive analytics and premium reporting for every dataset.</div>
        <div class='sidebar-badge'>Enterprise-ready</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    theme_choice = st.radio(
        "Theme",
        ["Dark", "Light"],
        index=0 if st.session_state.get("ui_theme", "Dark") == "Dark" else 1,
        horizontal=False,
        key="ui_theme",
        help="Choose between a premium dark or light theme for the app interface."
    )
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
    st.markdown(
        """
        <div class='hero-card'>
            <div class='hero-title'>Data Insight Studio</div>
            <div class='hero-subtitle'>Executive-grade analytics, automated intelligence, and premium reporting in one polished workspace.</div>
            <div class='hero-badges'>
                <div class='hero-badge'>Enterprise-ready</div>
                <div class='hero-badge'>Smart Automation</div>
                <div class='hero-badge'>Professional Reports</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='section-subtitle'>Upload a dataset and immediately unlock high-value insights, interactive charts, ML modeling, and executive-ready export options.</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            "<div class='info-card'><h4>Data Preview</h4><p>Deep dataset profiling with intelligent column summaries and instant filtering.</p></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div class='info-card'><h4>Smart Cleaning</h4><p>Automated cleanup, missing value handling and outlier detection for clean outputs.</p></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            "<div class='info-card'><h4>Advanced Analysis</h4><p>Business-grade statistics, trends, correlations and distribution insights.</p></div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            "<div class='info-card'><h4>ML Predictions</h4><p>Seamless regression or classification modeling with instant evaluation metrics.</p></div>",
            unsafe_allow_html=True,
        )

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown(
            "<div class='info-card'><h4>Visualizations</h4><p>Premium charting with auto-generated and custom builders for executive dashboards.</p></div>",
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            "<div class='info-card'><h4>Automated Insights</h4><p>Data-driven findings surfaced automatically with professional commentary.</p></div>",
            unsafe_allow_html=True,
        )
    with c7:
        st.markdown(
            "<div class='info-card'><h4>Report Builder</h4><p>Create polished PPT summaries ready for stakeholder presentations.</p></div>",
            unsafe_allow_html=True,
        )
    with c8:
        st.markdown(
            "<div class='info-card'><h4>Export Hub</h4><p>Download clean data, model outputs, and reports with confidence.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='section-separator'>", unsafe_allow_html=True)
    st.markdown("**Accepted formats:** CSV (`.csv`) · Excel (`.xlsx`, `.xls`)")
    st.markdown("<div class='sidebar-subtitle'>👈 Use the sidebar to upload your dataset and start your professional analytics workflow.</div>", unsafe_allow_html=True)
    st.stop()

# ── DATA REFERENCES ───────────────────────────────────────────────────────────
df         = st.session_state["df"]
working_df = st.session_state.get("cleaned_df", df)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🔍 Data Preview",
    "🧹 Data Cleaning",
    "📊 Analysis",
    "📈 Visualization",
    "🤖 Prediction",
    "📈 Stock Analyzer",
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

# ── TAB 6: STOCK ANALYZER ─────────────────────────────────────────────────────
with tab6:
    show_stock_analyzer()

# ── TAB 7: INSIGHTS ───────────────────────────────────────────────────────────
with tab7:
    st.header("Automated Insights")
    display_insights(working_df)

# ── TAB 8: REPORT ─────────────────────────────────────────────────────────
with tab8:
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
