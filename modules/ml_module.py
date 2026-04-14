import io
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix,
    f1_score
)
import plotly.express as px
import plotly.graph_objects as go


def _num_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _cat_cols(df):
    return df.select_dtypes(include=["object", "string", "category"]).columns.tolist()


def _datetime_cols(df):
    return [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]


@st.cache_data(show_spinner=False)
def _auto_select_features(serialized_df, target_col, n=10):
    df = pd.read_json(io.StringIO(serialized_df))
    num  = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    cats = [c for c in df.select_dtypes(include=["object", "string", "category"]).columns if c != target_col]
    target_num = df[target_col].dtype != object and str(df[target_col].dtype) != "string"
    if target_num:
        scores = {}
        for col in num:
            try:
                corr = abs(df[[col, target_col]].dropna().corr().iloc[0, 1])
                scores[col] = 0 if np.isnan(corr) else corr
            except Exception:
                scores[col] = 0
        top_num = sorted(scores, key=lambda x: scores[x], reverse=True)[:n]
        return top_num + cats[:max(0, n - len(top_num))]
    return (num + cats)[:n]


def show_ml_module(df):
    st.subheader("🤖 ML Prediction")
    all_cols  = df.columns.tolist()
    num_cols  = _num_cols(df)

    if len(all_cols) < 2:
        st.warning("Need at least 2 columns.")
        return

    # ── CONFIGURATION ────────────────────────────────────────────────────────
    with st.expander("⚙️ Model Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Target variable", all_cols, key="ml_target")
        with col2:
            available = [c for c in all_cols if c != target_col]
            auto_feats = []
            try:
                auto_feats = _auto_select_features(
                    df.to_json(), target_col, n=10
                )
                auto_feats = [f for f in auto_feats if f in available]
            except Exception:
                auto_feats = [c for c in num_cols if c != target_col][:6]

            feature_cols = st.multiselect(
                "Feature columns  (auto-selected by correlation)",
                available,
                default=auto_feats,
                key="ml_features"
            )

        if not feature_cols:
            st.info("Select at least one feature column.")
            return

        target_series = df[target_col].dropna()
        n_unique = target_series.nunique()
        is_cls   = (
            str(target_series.dtype) in ("object", "string", "category")
            or n_unique <= 15
        )

        col3, col4 = st.columns(2)
        with col3:
            if is_cls:
                st.info(f"Task: **Classification** — {n_unique} class(es)")
                model_opts = ["Random Forest", "Gradient Boosting", "Decision Tree",
                              "Logistic Regression"]
            else:
                st.info(f"Task: **Regression** — continuous target")
                model_opts = ["Random Forest", "Decision Tree",
                              "Linear Regression", "Ridge Regression"]
            model_choice = st.selectbox("Algorithm", model_opts, key="ml_model")
        with col4:
            test_pct = st.slider("Test set %", 10, 40, 20, 5, key="ml_test")

        use_cv = st.checkbox("5-fold cross-validation", value=False, key="ml_cv")

    train_btn = st.button("🚀 Train Model", type="primary", key="ml_train")

    if train_btn:
        with st.spinner("Training…"):
            try:
                _run_training(df, target_col, feature_cols, is_cls,
                              model_choice, test_pct, use_cv)
            except Exception as e:
                import traceback
                st.error(f"Training failed: {e}")
                st.code(traceback.format_exc())

    # ── PREDICTION PANEL ─────────────────────────────────────────────────────
    if st.session_state.get("ml_trained"):
        st.markdown("---")
        _show_prediction_panel(df, feature_cols)


def _run_training(df, target_col, feature_cols, is_cls, model_choice, test_pct, use_cv):
    cols = feature_cols + [target_col]
    ml_df = df[cols].dropna()

    if len(ml_df) < 10:
        st.error(f"Only {len(ml_df)} complete rows — need ≥10. Try cleaning missing values first.")
        return

    X = ml_df[feature_cols].copy()
    y = ml_df[target_col].copy()

    # encode categorical features
    encoders = {}
    for col in _cat_cols(X):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # convert datetime features to numeric timestamps
    for col in _datetime_cols(X):
        X[col] = X[col].astype('int64') / 1e9

    X = X.astype(float)

    le_target = None
    if is_cls and (str(y.dtype) in ("object", "string", "category")):
        le_target = LabelEncoder()
        y_enc = le_target.fit_transform(y.astype(str))
    elif is_cls:
        y_enc = y.astype(int).values
    else:
        y_enc = y.astype(float).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models_cls = {
        "Random Forest":      RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting":  GradientBoostingClassifier(n_estimators=150, random_state=42),
        "Decision Tree":      DecisionTreeClassifier(max_depth=8, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    }
    models_reg = {
        "Random Forest":    RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Decision Tree":    DecisionTreeRegressor(max_depth=8, random_state=42),
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=1.0),
    }
    model = (models_cls if is_cls else models_reg)[model_choice]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y_enc, test_size=test_pct / 100, random_state=42
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    st.markdown("### 📊 Model Results")

    if is_cls:
        acc = accuracy_score(y_te, y_pred)
        f1  = f1_score(y_te, y_pred, average="weighted", zero_division=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",     f"{acc:.4f}")
        c2.metric("F1 (weighted)", f"{f1:.4f}")
        c3.metric("Train samples", len(X_tr))
        c4.metric("Test samples",  len(X_te))

        if use_cv:
            cv = cross_val_score(model, X_scaled, y_enc, cv=5, scoring="accuracy")
            st.metric("CV Accuracy (mean ± std)", f"{cv.mean():.3f} ± {cv.std():.3f}")

        report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
        st.markdown("**Classification Report**")
        st.dataframe(pd.DataFrame(report).transpose().round(3), width="stretch")

        cm = confusion_matrix(y_te, y_pred)
        labels = (
            [str(c) for c in le_target.classes_]
            if le_target is not None
            else [str(v) for v in sorted(set(y_te.tolist() + y_pred.tolist()))]
        )
        fig = px.imshow(
            cm, text_auto=True,
            x=labels[:cm.shape[1]], y=labels[:cm.shape[0]],
            labels=dict(x="Predicted", y="Actual"),
            title="Confusion Matrix",
            template="plotly_dark",
            color_continuous_scale="Blues"
        )
        fig.update_layout(height=max(300, len(labels) * 50 + 120))
        st.plotly_chart(fig, width="stretch")

        ml_metrics = {"Accuracy": acc, "F1 Weighted": f1, "Train": len(X_tr), "Test": len(X_te)}
    else:
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        mae  = float(mean_absolute_error(y_te, y_pred))
        r2   = float(r2_score(y_te, y_pred))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²",    f"{r2:.4f}")
        c2.metric("RMSE",  f"{rmse:.4f}")
        c3.metric("MAE",   f"{mae:.4f}")
        c4.metric("Test samples", len(X_te))

        if use_cv:
            cv = cross_val_score(model, X_scaled, y_enc, cv=5, scoring="r2")
            st.metric("CV R² (mean ± std)", f"{cv.mean():.3f} ± {cv.std():.3f}")

        y_te_l   = y_te.tolist() if hasattr(y_te, "tolist") else list(y_te)
        y_pred_l = y_pred.tolist()

        fig = px.scatter(
            x=y_te_l, y=y_pred_l,
            labels={"x": "Actual", "y": "Predicted"},
            title="Actual vs Predicted",
            template="plotly_dark",
            opacity=0.7,
            color_discrete_sequence=["#4F8BF9"]
        )
        mn, mx = min(y_te_l + y_pred_l), max(y_te_l + y_pred_l)
        fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                      line=dict(dash="dash", color="#FF6B6B", width=2))
        fig.update_layout(height=440)
        st.plotly_chart(fig, width="stretch")

        residuals = [a - p for a, p in zip(y_te_l, y_pred_l)]
        fig2 = px.scatter(
            x=y_pred_l, y=residuals,
            labels={"x": "Predicted", "y": "Residual"},
            title="Residual Plot",
            template="plotly_dark",
            color_discrete_sequence=["#FF6B6B"],
            opacity=0.7
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="white")
        fig2.update_layout(height=360)
        st.plotly_chart(fig2, width="stretch")

        ml_metrics = {"R²": r2, "RMSE": rmse, "MAE": mae, "Train": len(X_tr), "Test": len(X_te)}

    # Feature importance / coefficients
    imp_vals   = None
    imp_names  = feature_cols

    if hasattr(model, "feature_importances_"):
        imp_vals = model.feature_importances_
        imp_df   = pd.DataFrame({"Feature": feature_cols, "Importance": imp_vals})
        imp_df   = imp_df.sort_values("Importance", ascending=True)
        fig = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            title="Feature Importance",
            template="plotly_dark",
            color="Importance", color_continuous_scale="Blues"
        )
        fig.update_layout(height=max(320, len(feature_cols) * 30 + 90))
        st.plotly_chart(fig, width="stretch")

    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1 and coef.shape[0] == 1:
            coef = coef[0]
        if coef.ndim == 1 and len(coef) == len(feature_cols):
            coef_df = pd.DataFrame({"Feature": feature_cols, "Coefficient": coef})
            coef_df = coef_df.sort_values("Coefficient", ascending=True)
            fig = px.bar(
                coef_df, x="Coefficient", y="Feature", orientation="h",
                title="Feature Coefficients",
                template="plotly_dark",
                color="Coefficient", color_continuous_scale="RdBu"
            )
            fig.update_layout(height=max(320, len(feature_cols) * 30 + 90))
            st.plotly_chart(fig, width="stretch")

    # ── Save model to session state ──────────────────────────────────────────
    st.session_state["ml_trained"]       = True
    st.session_state["ml_model_obj"]     = model
    st.session_state["ml_scaler"]        = scaler
    st.session_state["ml_encoders"]      = encoders
    st.session_state["ml_le_target"]     = le_target
    st.session_state["ml_feature_cols"]  = feature_cols
    st.session_state["ml_target_col"]    = target_col
    st.session_state["ml_is_cls"]        = is_cls
    st.session_state["ml_results"]       = {
        "task":               "Classification" if is_cls else "Regression",
        "model":              model_choice,
        "target":             target_col,
        "features":           feature_cols,
        "metrics":            ml_metrics,
        "feature_importances": imp_vals.tolist() if imp_vals is not None else None,
        "feature_names":      imp_names,
    }

    # ── Save model file download ─────────────────────────────────────────────
    model_buf = io.BytesIO()
    joblib.dump({"model": model, "scaler": scaler,
                 "encoders": encoders, "le_target": le_target,
                 "feature_cols": feature_cols, "target_col": target_col,
                 "is_cls": is_cls}, model_buf)
    model_buf.seek(0)
    st.download_button(
        "💾 Download Trained Model (.pkl)",
        data=model_buf,
        file_name=f"model_{model_choice.lower().replace(' ', '_')}.pkl",
        mime="application/octet-stream",
        key="ml_download_model"
    )


def _show_prediction_panel(df, feature_cols):
    st.markdown("### 🔮 Make a Prediction")
    st.caption("Enter values for each feature and click **Predict** to get a result.")

    model    = st.session_state["ml_model_obj"]
    scaler   = st.session_state["ml_scaler"]
    encoders = st.session_state["ml_encoders"]
    le_tgt   = st.session_state["ml_le_target"]
    feats    = st.session_state["ml_feature_cols"]
    is_cls   = st.session_state["ml_is_cls"]

    input_vals = {}
    n_cols = min(3, len(feats))
    rows   = [feats[i:i + n_cols] for i in range(0, len(feats), n_cols)]

    for row in rows:
        cols_st = st.columns(len(row))
        for col_st, feat in zip(cols_st, row):
            with col_st:
                col_data = df[feat].dropna()
                if feat in encoders:
                    options = sorted(df[feat].dropna().unique().tolist())
                    input_vals[feat] = st.selectbox(feat, options, key=f"pred_{feat}")
                else:
                    min_v = float(col_data.min())
                    max_v = float(col_data.max())
                    mean_v = float(col_data.mean())
                    input_vals[feat] = st.number_input(
                        feat, min_value=min_v, max_value=max_v,
                        value=round(mean_v, 4),
                        step=round((max_v - min_v) / 100, 6) or 0.01,
                        key=f"pred_{feat}"
                    )

    if st.button("⚡ Predict", type="primary", key="ml_predict"):
        try:
            row_dict = {}
            for feat in feats:
                val = input_vals[feat]
                if feat in encoders:
                    le = encoders[feat]
                    val_str = str(val)
                    if val_str in le.classes_:
                        row_dict[feat] = float(le.transform([val_str])[0])
                    else:
                        row_dict[feat] = 0.0
                else:
                    row_dict[feat] = float(val)

            X_input = pd.DataFrame([row_dict])[feats].astype(float)
            X_scaled = scaler.transform(X_input)
            pred = model.predict(X_scaled)[0]

            if is_cls:
                if le_tgt is not None:
                    label = le_tgt.inverse_transform([int(pred)])[0]
                else:
                    label = str(pred)
                st.success(f"**Predicted Class: {label}**")

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_scaled)[0]
                    class_labels = (
                        [str(c) for c in le_tgt.classes_]
                        if le_tgt is not None
                        else [str(i) for i in range(len(probs))]
                    )
                    prob_df = pd.DataFrame({
                        "Class": class_labels, "Probability": probs
                    }).sort_values("Probability", ascending=False)
                    fig = px.bar(
                        prob_df, x="Class", y="Probability",
                        title="Class Probabilities",
                        template="plotly_dark",
                        color="Probability", color_continuous_scale="Blues",
                        range_y=[0, 1]
                    )
                    fig.update_layout(height=340)
                    st.plotly_chart(fig, width="stretch")
            else:
                st.success(f"**Predicted Value: {pred:.4f}**")
                st.metric("Prediction", f"{pred:,.4f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
