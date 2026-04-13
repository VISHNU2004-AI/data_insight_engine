import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def _num_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _cat_cols(df):
    return df.select_dtypes(include=["object", "string", "category"]).columns.tolist()


def _download_chart(fig, name="chart"):
    try:
        html_bytes = fig.to_html(include_plotlyjs="cdn").encode()
        st.download_button(
            "⬇️ Download Chart (HTML)",
            data=html_bytes,
            file_name=f"{name}.html",
            mime="text/html",
            key=f"dl_{name}_{id(fig)}"
        )
    except Exception:
        pass


@st.cache_data(show_spinner=False)
def _get_corr(df_json):
    df = pd.read_json(io.StringIO(df_json))
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[num].corr() if len(num) >= 2 else None


def auto_visualize(df):
    st.subheader("Auto-Generated Visualizations")
    numeric_cols = _num_cols(df)
    cat_cols     = _cat_cols(df)

    if not numeric_cols and not cat_cols:
        st.warning("No columns available to visualize.")
        return

    # ── Histograms ────────────────────────────────────────────────────────────
    if numeric_cols:
        st.markdown("#### Histograms")
        display = numeric_cols[:6]
        ncols = min(3, len(display))
        for i in range(0, len(display), ncols):
            row = st.columns(ncols)
            for j, col_name in enumerate(display[i:i + ncols]):
                with row[j]:
                    fig = px.histogram(
                        df, x=col_name, marginal="box",
                        title=col_name,
                        template="plotly_dark",
                        color_discrete_sequence=["#4F8BF9"]
                    )
                    fig.update_layout(height=300, showlegend=False,
                                      margin=dict(t=45, b=20, l=20, r=20))
                    st.plotly_chart(fig, width="stretch")
                    _download_chart(fig, f"hist_{col_name}")

    # ── Bar Charts ────────────────────────────────────────────────────────────
    if cat_cols:
        st.markdown("#### Value Counts (Categorical)")
        display = cat_cols[:6]
        ncols = min(3, len(display))
        for i in range(0, len(display), ncols):
            row = st.columns(ncols)
            for j, col_name in enumerate(display[i:i + ncols]):
                with row[j]:
                    vc = df[col_name].value_counts().head(10)
                    fig = px.bar(
                        x=vc.index.astype(str), y=vc.values,
                        title=col_name,
                        labels={"x": col_name, "y": "Count"},
                        template="plotly_dark",
                        color_discrete_sequence=["#FF6B6B"]
                    )
                    fig.update_layout(height=300, showlegend=False,
                                      margin=dict(t=45, b=20, l=20, r=20))
                    st.plotly_chart(fig, width="stretch")
                    _download_chart(fig, f"bar_{col_name}")

    # ── Correlation Heatmap ───────────────────────────────────────────────────
    if len(numeric_cols) >= 2:
        st.markdown("#### Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, aspect="auto"
        )
        fig.update_layout(
            title="Correlation Heatmap",
            height=max(380, len(corr) * 44 + 90),
            template="plotly_dark"
        )
        st.plotly_chart(fig, width="stretch")
        _download_chart(fig, "correlation_heatmap")

    # ── Scatter Matrix ────────────────────────────────────────────────────────
    if len(numeric_cols) >= 2:
        st.markdown("#### Scatter Matrix")
        scatter_cols = numeric_cols[:4]
        sample = df.sample(min(600, len(df)), random_state=42) if len(df) > 600 else df
        cat_color = next((c for c in cat_cols if df[c].nunique() <= 10), None)
        fig = px.scatter_matrix(
            sample, dimensions=scatter_cols, color=cat_color,
            title=f"Scatter Matrix — {len(scatter_cols)} columns",
            template="plotly_dark"
        )
        fig.update_layout(height=580)
        fig.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.55))
        st.plotly_chart(fig, width="stretch")
        _download_chart(fig, "scatter_matrix")

    # ── Box Plots ─────────────────────────────────────────────────────────────
    if numeric_cols:
        st.markdown("#### Box Plots")
        fig = go.Figure()
        for col_name in numeric_cols[:8]:
            fig.add_trace(go.Box(
                y=df[col_name].dropna(), name=col_name,
                boxpoints="outliers", marker_size=3
            ))
        fig.update_layout(
            title="Box Plots — Numeric Columns",
            template="plotly_dark",
            height=430
        )
        st.plotly_chart(fig, width="stretch")
        _download_chart(fig, "boxplots")


def custom_visualization(df):
    st.subheader("Custom Visualization Builder")

    all_cols  = df.columns.tolist()
    num_cols  = _num_cols(df)
    cat_cols  = _cat_cols(df)

    if not all_cols:
        st.warning("No columns available.")
        return

    chart_type = st.selectbox(
        "Chart Type",
        ["Scatter", "Line", "Bar", "Histogram", "Box Plot", "Violin",
         "Pie", "Area", "Bubble", "Funnel", "Sunburst", "Strip"],
        key="cv_type"
    )

    fig = None
    chart_key = chart_type.lower().replace(" ", "_")

    if chart_type == "Histogram":
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Column", num_cols if num_cols else all_cols, key=f"{chart_key}_x")
        with col2:
            bins = st.slider("Bins", 5, 100, 30, key=f"{chart_key}_bins")
        color_col = st.selectbox("Color by", [None] + cat_cols, key=f"{chart_key}_col")
        filt_col, filt_vals = _filter_widget(df, all_cols, chart_key)
        pdf = _apply_filter(df, filt_col, filt_vals)
        fig = px.histogram(pdf, x=x_col, nbins=bins,
                           color=color_col or None,
                           marginal="box",
                           title=f"Histogram — {x_col}",
                           template="plotly_dark")

    elif chart_type == "Box Plot":
        col1, col2 = st.columns(2)
        with col1:
            y_col = st.selectbox("Y (numeric)", num_cols if num_cols else all_cols, key=f"{chart_key}_y")
        with col2:
            x_col = st.selectbox("Group by", [None] + cat_cols, key=f"{chart_key}_x")
        filt_col, filt_vals = _filter_widget(df, all_cols, chart_key)
        pdf = _apply_filter(df, filt_col, filt_vals)
        fig = px.box(pdf, x=x_col or None, y=y_col,
                     color=x_col or None,
                     title=f"Box Plot — {y_col}" + (f" by {x_col}" if x_col else ""),
                     template="plotly_dark", points="outliers")

    elif chart_type == "Violin":
        col1, col2 = st.columns(2)
        with col1:
            y_col = st.selectbox("Y (numeric)", num_cols if num_cols else all_cols, key=f"{chart_key}_y")
        with col2:
            x_col = st.selectbox("Group by", [None] + cat_cols, key=f"{chart_key}_x")
        filt_col, filt_vals = _filter_widget(df, all_cols, chart_key)
        pdf = _apply_filter(df, filt_col, filt_vals)
        fig = px.violin(pdf, x=x_col or None, y=y_col,
                        color=x_col or None, box=True,
                        title=f"Violin — {y_col}",
                        template="plotly_dark")

    elif chart_type == "Pie":
        if not cat_cols:
            st.warning("Pie chart requires a categorical column.")
            return
        col1, col2 = st.columns(2)
        with col1:
            names_col = st.selectbox("Labels", cat_cols, key=f"{chart_key}_n")
        with col2:
            val_col = st.selectbox("Values", num_cols if num_cols else all_cols, key=f"{chart_key}_v")
        filt_col, filt_vals = _filter_widget(df, all_cols, chart_key)
        pdf = _apply_filter(df, filt_col, filt_vals)
        agg = pdf.groupby(names_col)[val_col].sum().reset_index().nlargest(15, val_col)
        fig = px.pie(agg, values=val_col, names=names_col,
                     title=f"Pie — {val_col} by {names_col}",
                     template="plotly_dark")

    elif chart_type == "Bubble":
        if len(num_cols) < 3:
            st.warning("Bubble chart needs ≥ 3 numeric columns.")
            return
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("X", num_cols, key=f"{chart_key}_x")
        with col2:
            y_col = st.selectbox("Y", [c for c in num_cols if c != x_col], key=f"{chart_key}_y")
        with col3:
            sz_col = st.selectbox("Size", [c for c in num_cols if c not in [x_col, y_col]], key=f"{chart_key}_sz")
        color_col = st.selectbox("Color by", [None] + cat_cols, key=f"{chart_key}_col")
        filt_col, filt_vals = _filter_widget(df, all_cols, chart_key)
        pdf = _apply_filter(df, filt_col, filt_vals)
        fig = px.scatter(pdf, x=x_col, y=y_col,
                         size=pdf[sz_col].clip(lower=0),
                         color=color_col or None,
                         title=f"Bubble — {x_col} vs {y_col} (size={sz_col})",
                         template="plotly_dark", size_max=60)

    elif chart_type == "Funnel":
        if not cat_cols or not num_cols:
            st.warning("Funnel needs ≥1 categorical and ≥1 numeric column.")
            return
        col1, col2 = st.columns(2)
        with col1:
            stage = st.selectbox("Stage", cat_cols, key=f"{chart_key}_stage")
        with col2:
            val   = st.selectbox("Value", num_cols, key=f"{chart_key}_val")
        filt_col, filt_vals = _filter_widget(df, all_cols, chart_key)
        pdf = _apply_filter(df, filt_col, filt_vals)
        agg = pdf.groupby(stage)[val].sum().reset_index().sort_values(val, ascending=False)
        fig = px.funnel(agg, x=val, y=stage,
                        title=f"Funnel — {val} by {stage}",
                        template="plotly_dark")

    elif chart_type == "Sunburst":
        if not cat_cols:
            st.warning("Sunburst needs ≥1 categorical column.")
            return
        path_cols = st.multiselect("Hierarchy (outer → inner)", cat_cols,
                                   default=cat_cols[:2], key=f"{chart_key}_path")
        val_col   = st.selectbox("Value", [None] + num_cols, key=f"{chart_key}_val")
        if not path_cols:
            st.info("Select at least one hierarchy column.")
            return
        filt_col, filt_vals = _filter_widget(df, all_cols, chart_key)
        pdf = _apply_filter(df, filt_col, filt_vals)
        fig = px.sunburst(pdf, path=path_cols, values=val_col or None,
                          title="Sunburst", template="plotly_dark")

    elif chart_type == "Strip":
        col1, col2 = st.columns(2)
        with col1:
            y_col = st.selectbox("Y (numeric)", num_cols if num_cols else all_cols, key=f"{chart_key}_y")
        with col2:
            x_col = st.selectbox("Group by", [None] + cat_cols, key=f"{chart_key}_x")
        filt_col, filt_vals = _filter_widget(df, all_cols, chart_key)
        pdf = _apply_filter(df, filt_col, filt_vals)
        fig = px.strip(pdf, x=x_col or None, y=y_col,
                       color=x_col or None,
                       title=f"Strip — {y_col}",
                       template="plotly_dark")

    else:
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X", all_cols, key=f"{chart_key}_x")
        with col2:
            y_remaining = [c for c in all_cols if c != x_col]
            y_col = st.selectbox("Y", y_remaining if y_remaining else all_cols, key=f"{chart_key}_y")
        color_col = st.selectbox("Color by", [None] + cat_cols, key=f"{chart_key}_col")
        filt_col, filt_vals = _filter_widget(df, all_cols, chart_key)
        pdf = _apply_filter(df, filt_col, filt_vals)
        fn_map = {"Scatter": px.scatter, "Line": px.line,
                  "Bar": px.bar, "Area": px.area}
        fn = fn_map.get(chart_type, px.scatter)
        fig = fn(pdf, x=x_col, y=y_col, color=color_col or None,
                 title=f"{chart_type} — {x_col} vs {y_col}",
                 template="plotly_dark")

    if fig is not None:
        fig.update_layout(height=520)
        st.plotly_chart(fig, width="stretch")
        _download_chart(fig, f"custom_{chart_key}")


def _filter_widget(df, all_cols, key_prefix):
    with st.expander("🔍 Filter Data (optional)"):
        f_col = st.selectbox("Filter column", [None] + all_cols, key=f"{key_prefix}_fcol")
        f_vals = None
        if f_col:
            unique = df[f_col].dropna().unique()
            display = unique.tolist() if len(unique) <= 50 else \
                      df[f_col].value_counts().head(50).index.tolist()
            if len(unique) > 50:
                st.caption(f"Showing top 50 of {len(unique)} unique values.")
            f_vals = st.multiselect(f"Keep where {f_col} is:", display, key=f"{key_prefix}_fval")
    return f_col, f_vals


def _apply_filter(df, f_col, f_vals):
    if f_col and f_vals:
        return df[df[f_col].isin(f_vals)]
    return df
