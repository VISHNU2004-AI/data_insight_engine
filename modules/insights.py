import streamlit as st
import pandas as pd
import numpy as np


def _num_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _cat_cols(df):
    return df.select_dtypes(include=["object", "string", "category"]).columns.tolist()


def generate_insights(df):
    insights = []
    numeric_cols = _num_cols(df)
    cat_cols = _cat_cols(df)

    # Limit correlation analysis for performance
    MAX_CORR_COLS = 15
    if len(numeric_cols) > MAX_CORR_COLS:
        corr_cols = numeric_cols[:MAX_CORR_COLS]
    else:
        corr_cols = numeric_cols

    if len(corr_cols) >= 2:
        # Sample for large datasets
        if len(df) > 50000:
            sample_df = df.sample(min(50000, len(df)), random_state=42)
            corr = sample_df[corr_cols].corr()
        else:
            corr = df[corr_cols].corr()

        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if pd.isna(val):
                    continue
                direction = "positively" if val > 0 else "negatively"
                if abs(val) >= 0.8:
                    insights.append({
                        "type": "correlation",
                        "icon": "🔗",
                        "message": f"**{corr.columns[i]}** is strongly {direction} correlated with **{corr.columns[j]}** (r = {val:.3f})"
                    })
                elif abs(val) >= 0.5:
                    insights.append({
                        "type": "correlation",
                        "icon": "📊",
                        "message": f"**{corr.columns[i]}** is moderately {direction} correlated with **{corr.columns[j]}** (r = {val:.3f})"
                    })

    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) < 3:
            continue
        try:
            skew = float(col_data.skew())
            if abs(skew) > 1.5:
                direction = "right (positive)" if skew > 0 else "left (negative)"
                insights.append({
                    "type": "distribution",
                    "icon": "📈",
                    "message": f"**{col}** is highly skewed to the {direction} (skewness = {skew:.3f}) — consider log-transform"
                })
            elif abs(skew) > 0.8:
                direction = "right" if skew > 0 else "left"
                insights.append({
                    "type": "distribution",
                    "icon": "📈",
                    "message": f"**{col}** has moderate skew to the {direction} (skewness = {skew:.3f})"
                })
        except Exception:
            pass

    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) < 4:
            continue
        try:
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_count = int(((col_data < lower) | (col_data > upper)).sum())
            if outlier_count > 0:
                pct = outlier_count / len(col_data) * 100
                insights.append({
                    "type": "outlier",
                    "icon": "⚠️",
                    "message": f"**{col}** has {outlier_count} outlier(s) ({pct:.1f}% of data) beyond IQR bounds [{lower:.2f}, {upper:.2f}]"
                })
        except Exception:
            pass

    missing = df.isnull().sum()
    for col in missing[missing > 0].index:
        pct = missing[col] / len(df) * 100
        severity = "critical" if pct > 40 else "high" if pct > 20 else "moderate" if pct > 5 else "low"
        insights.append({
            "type": "missing",
            "icon": "❓",
            "message": f"**{col}** has {int(missing[col])} missing value(s) ({pct:.1f}%) — {severity} impact"
        })

    for col in cat_cols:
        nunique = df[col].nunique()
        total = len(df)
        if nunique == 0:
            continue
        elif nunique == 1:
            insights.append({
                "type": "constant",
                "icon": "📌",
                "message": f"**{col}** has only 1 unique value — adds no information, consider dropping"
            })
        elif nunique >= total * 0.95 and total > 20:
            insights.append({
                "type": "unique",
                "icon": "🆔",
                "message": f"**{col}** has nearly all unique values ({nunique}/{total}) — likely an ID column"
            })
        else:
            vals = df[col].value_counts()
            dominant_pct = vals.iloc[0] / len(df) * 100
            if dominant_pct > 80 and nunique >= 2:
                insights.append({
                    "type": "imbalance",
                    "icon": "⚖️",
                    "message": f"**{col}** is imbalanced — '{vals.index[0]}' makes up {dominant_pct:.1f}% of values"
                })

    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        try:
            mean_val = float(col_data.mean())
            std_val = float(col_data.std())
            if mean_val != 0 and not np.isnan(mean_val) and not np.isnan(std_val):
                cv = abs(std_val / mean_val) * 100
                if cv > 100:
                    insights.append({
                        "type": "variability",
                        "icon": "📉",
                        "message": f"**{col}** has very high variability (CV = {cv:.1f}%) — consider normalization"
                    })
        except Exception:
            pass

    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        insights.append({
            "type": "duplicates",
            "icon": "📋",
            "message": f"Dataset has **{dup_count} duplicate row(s)** ({dup_count / len(df) * 100:.1f}%) — cleaning recommended"
        })

    return insights


def display_insights(df):
    st.subheader("Automated Insights")

    if len(df) == 0:
        st.warning("Dataset is empty.")
        return

    insights = generate_insights(df)

    if not insights:
        st.success("No notable patterns or issues detected.")
        return

    type_order = ["correlation", "distribution", "outlier", "missing", "imbalance",
                  "constant", "unique", "variability", "duplicates"]
    type_labels = {
        "correlation": "🔗 Correlations",
        "distribution": "📈 Distribution Patterns",
        "outlier": "⚠️ Outliers",
        "missing": "❓ Missing Data",
        "imbalance": "⚖️ Data Imbalance",
        "constant": "📌 Constant Columns",
        "unique": "🆔 Identifier Columns",
        "variability": "📉 High Variability",
        "duplicates": "📋 Duplicates"
    }

    grouped = {}
    for insight in insights:
        grouped.setdefault(insight["type"], []).append(insight)

    st.write(f"Found **{len(insights)}** insight(s) across **{len(grouped)}** categories:")
    st.markdown("")

    for t in type_order:
        if t not in grouped:
            continue
        label = type_labels.get(t, t)
        with st.expander(f"{label} ({len(grouped[t])})", expanded=True):
            for ins in grouped[t]:
                st.markdown(f"- {ins['message']}")
