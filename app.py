"""Customer Churn Prediction & Retention Intelligence - Streamlit dashboard.
Run: streamlit run app.py
"""
import os
import subprocess
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

import ml
from retention import reasons_and_actions

st.set_page_config(page_title="Keepwise Churn Intelligence", page_icon="📉", layout="wide")
COLORS = {"High": "#e5484d", "Medium": "#e49b0f", "Low": "#1f9d74"}

st.sidebar.title("📉 Keepwise")
page = st.sidebar.radio("Navigate", ["1. Upload data", "2. Churn prediction", "3. Risk analysis", "4. Retention suggestions"])
high_t = st.sidebar.slider("High-risk threshold", 0.4, 0.9, 0.6, 0.05)
med_t = st.sidebar.slider("Medium-risk threshold", 0.1, 0.5, 0.3, 0.05)

S = st.session_state

def need(key, msg):
    if key not in S:
        st.info(msg)
        st.stop()

def scored():
    """All customers with churn probability and risk tier (cached per model/threshold)."""
    r = S["result"]
    out = r["df"].copy()
    out["ChurnProbability"] = r["pipe"].predict_proba(r["X"])[:, 1].round(3)
    out["RiskTier"] = ml.risk_tier(out["ChurnProbability"], high_t, med_t)
    return out

# ---------------------------------------------------------------- 1. Upload
if page.startswith("1"):
    st.title("Upload customer data")
    st.write("Upload a CSV with one row per customer and a column showing whether they churned.")
    up = st.file_uploader("CSV file", type="csv")
    if st.button("Use sample dataset"):
        if not os.path.exists("data/sample_churn.csv"):
            subprocess.run([sys.executable, "generate_sample_data.py"], check=True)
        S["raw"] = pd.read_csv("data/sample_churn.csv")
    if up is not None:
        S["raw"] = pd.read_csv(up)
    if "raw" in S:
        df = S["raw"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}"); c2.metric("Columns", df.shape[1]); c3.metric("Missing values", int(df.isna().sum().sum()))
        st.dataframe(df.head(50), use_container_width=True)
        cols = list(df.columns)
        guess = next((c for c in cols if c.lower() in ("churn", "churned", "exited", "attrition")), cols[-1])
        gid = next((c for c in cols if "id" in c.lower()), None)
        a, b, c = st.columns(3)
        target = a.selectbox("Churn column", cols, index=cols.index(guess))
        id_col = b.selectbox("Customer ID column", ["(none)"] + cols, index=(cols.index(gid) + 1) if gid else 0)
        model_name = c.selectbox("Model", list(ml.MODELS))
        if st.button("Train model", type="primary"):
            with st.spinner("Training..."):
                try:
                    S["result"] = ml.train(df, target, None if id_col == "(none)" else id_col, model_name)
                    st.success("Model trained. Open 'Churn prediction' in the sidebar.")
                except ValueError as e:
                    st.error(str(e))

# ---------------------------------------------------------------- 2. Prediction
elif page.startswith("2"):
    st.title("Churn prediction")
    need("result", "Upload a dataset and train a model first.")
    r = S["result"]
    st.caption(f"Model: {r['model_name']}, evaluated on a 20% held-out test set")
    cols = st.columns(5)
    for col, (k, v) in zip(cols, r["metrics"].items()):
        col.metric(k, f"{v:.2f}")
    l, rt = st.columns(2)
    cm = r["cm"]
    l.plotly_chart(px.imshow(cm, text_auto=True, x=["Stay", "Churn"], y=["Stay", "Churn"],
                             labels=dict(x="Predicted", y="Actual"), title="Confusion matrix",
                             color_continuous_scale="Blues"), use_container_width=True)
    fpr, tpr = r["roc"]
    roc = px.line(x=fpr, y=tpr, labels=dict(x="False positive rate", y="True positive rate"), title="ROC curve")
    roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
    rt.plotly_chart(roc, use_container_width=True)

    st.subheader("Predict one customer")
    X = r["X"]; num, cat = ml.split_columns(X)
    with st.form("single"):
        vals, grid = {}, st.columns(4)
        for i, c in enumerate(X.columns):
            with grid[i % 4]:
                if c in num:
                    vals[c] = st.number_input(c, value=float(X[c].median()))
                else:
                    vals[c] = st.selectbox(c, sorted(X[c].dropna().unique().tolist()))
        go = st.form_submit_button("Predict churn risk")
    if go:
        p = float(r["pipe"].predict_proba(pd.DataFrame([vals]))[0, 1])
        tier = ml.risk_tier(p, high_t, med_t).item()
        st.metric("Churn probability", f"{p:.0%}", help="Chance this customer leaves")
        st.markdown(f"**Risk tier:** :{ {'High':'red','Medium':'orange','Low':'green'}[tier] }[{tier}]")
        why, act = reasons_and_actions(pd.Series(vals), X.columns)
        if tier != "Low":
            st.write("**Why:** " + "; ".join(why)); st.write("**Do this:** " + "; ".join(act))

    st.subheader("Score all customers")
    out = scored()
    st.dataframe(out.sort_values("ChurnProbability", ascending=False).head(200), use_container_width=True)
    st.download_button("Download scored customers (CSV)", out.to_csv(index=False), "scored_customers.csv", "text/csv")

# ---------------------------------------------------------------- 3. Risk analysis
elif page.startswith("3"):
    st.title("Risk analysis")
    need("result", "Upload a dataset and train a model first.")
    out = scored()
    tiers = out["RiskTier"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
    c = st.columns(4)
    c[0].metric("Customers", f"{len(out):,}")
    c[1].metric("High risk", f"{int(tiers['High']):,}")
    c[2].metric("Average churn probability", f"{out['ChurnProbability'].mean():.0%}")
    if "MonthlyCharges" in out:
        at_risk = out.loc[out.RiskTier == "High", "MonthlyCharges"].sum() * 12
        c[3].metric("Annual revenue at risk", f"${at_risk:,.0f}")
    a, b = st.columns(2)
    a.plotly_chart(px.pie(values=tiers.values, names=tiers.index, hole=.5, color=tiers.index, color_discrete_map=COLORS,
                          title="Customers by risk tier"), use_container_width=True)
    imp = ml.importances(S["result"]["pipe"]).head(12)
    b.plotly_chart(px.bar(imp[::-1], x="importance", y="feature", orientation="h", title="What drives churn"), use_container_width=True)

    st.subheader("Risk by segment")
    cats = [x for x in out.columns if out[x].dtype == object and out[x].nunique() <= 12
            and x not in (S["result"]["target"], S["result"]["id_col"], "RiskTier")]
    if cats:
        seg = st.selectbox("Segment by", cats)
        g = out.groupby(seg)["ChurnProbability"].mean().reset_index()
        st.plotly_chart(px.bar(g, x=seg, y="ChurnProbability", title=f"Average churn probability by {seg}"), use_container_width=True)
    num = [x for x in out.select_dtypes("number").columns if x not in ("ChurnProbability",)]
    if num:
        n = st.selectbox("Distribution of", num)
        st.plotly_chart(px.histogram(out, x=n, color="RiskTier", color_discrete_map=COLORS, barmode="overlay", opacity=.7), use_container_width=True)

# ---------------------------------------------------------------- 4. Retention
else:
    st.title("Retention suggestions")
    need("result", "Upload a dataset and train a model first.")
    out = scored()
    tier = st.multiselect("Show risk tiers", ["High", "Medium", "Low"], default=["High"])
    top = out[out.RiskTier.isin(tier)].sort_values("ChurnProbability", ascending=False).head(100).copy()
    cols = S["result"]["X"].columns
    rec = top.apply(lambda r: reasons_and_actions(r, cols), axis=1)
    top["WhyAtRisk"] = rec.map(lambda x: "; ".join(x[0]))
    top["SuggestedActions"] = rec.map(lambda x: "; ".join(x[1]))
    idc = S["result"]["id_col"]
    show = ([idc] if idc else []) + ["ChurnProbability", "RiskTier", "WhyAtRisk", "SuggestedActions"]
    st.dataframe(top[show], use_container_width=True, hide_index=True)
    st.download_button("Download action list (CSV)", top[show].to_csv(index=False), "retention_actions.csv", "text/csv")
    st.subheader("Most common actions")
    cnt = top["SuggestedActions"].str.split("; ").explode().value_counts().head(8).reset_index()
    cnt.columns = ["Action", "Customers"]
    st.plotly_chart(px.bar(cnt[::-1], x="Customers", y="Action", orientation="h"), use_container_width=True)
