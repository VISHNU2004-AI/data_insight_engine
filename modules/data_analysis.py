import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def _num_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _cat_cols(df):
    return df.select_dtypes(include=["object", "string", "category"]).columns.tolist()


def show_summary_statistics(df):
    st.subheader("Summary Statistics")
    numeric_cols = _num_cols(df)
    if not numeric_cols:
        st.warning("No numeric columns found.")
        return
    numeric_df = df[numeric_cols]
    stats = numeric_df.describe().round(4)
    stats.loc["skewness"] = numeric_df.skew().round(4)
    stats.loc["kurtosis"] = numeric_df.kurtosis().round(4)
    st.dataframe(stats, width="stretch")


def show_correlation_analysis(df):
    st.subheader("Correlation Heatmap")
    numeric_cols = _num_cols(df)
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
        return

    # Limit columns for performance on large datasets
    MAX_CORR_COLS = 20
    if len(numeric_cols) > MAX_CORR_COLS:
        st.warning(f"⚠️ Too many numeric columns ({len(numeric_cols)}). Showing correlation for first {MAX_CORR_COLS} columns only.")
        numeric_cols = numeric_cols[:MAX_CORR_COLS]

    # For very large datasets, sample rows for correlation
    if len(df) > 50000:
        st.info("📊 Large dataset detected. Computing correlation on sample for performance.")
        sample_df = df.sample(min(50000, len(df)), random_state=42)
        corr = sample_df[numeric_cols].corr().round(3)
    else:
        corr = df[numeric_cols].corr().round(3)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto"
    )
    fig.update_layout(
        title="Correlation Heatmap",
        height=max(400, len(corr) * 45 + 100),
        template="plotly_dark"
    )
    st.plotly_chart(fig, width="stretch")

    strong, moderate = [], []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            val = corr.iloc[i, j]
            if pd.isna(val):
                continue
            direction = "positive" if val > 0 else "negative"
            entry = {
                "Column 1": corr.columns[i],
                "Column 2": corr.columns[j],
                "r": round(val, 3),
                "Direction": direction
            }
            if abs(val) >= 0.8:
                strong.append(entry)
            elif abs(val) >= 0.5:
                moderate.append(entry)

    if strong:
        st.markdown("**Strong correlations (|r| ≥ 0.8)**")
        st.dataframe(pd.DataFrame(strong), width="stretch", hide_index=True)
    if moderate:
        st.markdown("**Moderate correlations (0.5 ≤ |r| < 0.8)**")
        st.dataframe(pd.DataFrame(moderate), width="stretch", hide_index=True)
    if not strong and not moderate:
        st.info("No notable correlations found (|r| < 0.5 for all pairs).")


def show_distributions(df):
    st.subheader("Feature Distributions")
    numeric_cols = _num_cols(df)
    cat_cols = _cat_cols(df)

    if numeric_cols:
        selected_num = st.selectbox("Select numeric column", numeric_cols, key="dist_num")
        col_data = df[selected_num].dropna()

        fig = px.histogram(
            df, x=selected_num, marginal="box",
            title=f"Distribution of {selected_num}",
            template="plotly_dark",
            color_discrete_sequence=["#4F8BF9"]
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mean", f"{col_data.mean():.3f}")
        c2.metric("Median", f"{col_data.median():.3f}")
        c3.metric("Std Dev", f"{col_data.std():.3f}")
        c4.metric("Skewness", f"{col_data.skew():.3f}")
        c5.metric("Kurtosis", f"{col_data.kurtosis():.3f}")

    if cat_cols:
        st.markdown("---")
        selected_cat = st.selectbox("Select categorical column", cat_cols, key="dist_cat")
        value_counts = df[selected_cat].value_counts().head(20)
        fig = px.bar(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            title=f"Value Counts: {selected_cat} (Top 20)",
            labels={"x": selected_cat, "y": "Count"},
            template="plotly_dark",
            color_discrete_sequence=["#4F8BF9"]
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

        total = len(df[selected_cat].dropna())
        pct_df = pd.DataFrame({
            "Value": value_counts.index.astype(str),
            "Count": value_counts.values,
            "Percentage": (value_counts.values / total * 100).round(1)
        })
        st.dataframe(pct_df, width="stretch", hide_index=True)


def show_trend_analysis(df):
    st.subheader("Trend & Scatter Analysis")
    numeric_cols = _num_cols(df)

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for trend analysis.")
        return

    cat_cols = _cat_cols(df)
    col1, col2, col3 = st.columns(3)
    with col1:
        x_col = st.selectbox("X-axis", numeric_cols, key="trend_x")
    with col2:
        y_options = [c for c in numeric_cols if c != x_col]
        y_col = st.selectbox("Y-axis", y_options, key="trend_y")
    with col3:
        color_col = st.selectbox("Color by (optional)", [None] + cat_cols, key="trend_color")

    valid_color = color_col if (color_col and df[color_col].nunique() <= 20) else None
    plot_data = df[[x_col, y_col] + ([valid_color] if valid_color else [])].dropna()

    if len(plot_data) < 2:
        st.warning("Not enough non-null data points to plot.")
        return

    fig = px.scatter(
        plot_data, x=x_col, y=y_col,
        color=valid_color,
        title=f"{x_col} vs {y_col}",
        template="plotly_dark",
        opacity=0.75
    )

    try:
        x_vals = plot_data[x_col].values.astype(float)
        y_vals = plot_data[y_col].values.astype(float)
        coeffs = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
        y_line = np.polyval(coeffs, x_line)
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            name=f"Trend (slope={coeffs[0]:.4f})",
            line=dict(color="#FF6B6B", dash="dash", width=2)
        ))
        corr_val = float(np.corrcoef(x_vals, y_vals)[0, 1])
        direction = "positive" if corr_val > 0 else "negative"
        strength = "strong" if abs(corr_val) > 0.7 else "moderate" if abs(corr_val) > 0.4 else "weak"
        st.caption(
            f"Pearson r = **{corr_val:.3f}** ({strength} {direction} relationship) | "
            f"Slope = **{coeffs[0]:.4f}** | Intercept = **{coeffs[1]:.4f}**"
        )
    except Exception:
        pass

    fig.update_layout(height=480)
    st.plotly_chart(fig, width="stretch")
