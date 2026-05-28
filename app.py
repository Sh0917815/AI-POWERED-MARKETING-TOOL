import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import numpy as np
import plotly.express as px

# ================= PAGE =================
st.set_page_config(page_title="AI Marketing Intelligence Dashboard", layout="wide")

# ================= LIGHT PURPLE UI =================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #f8e6ff, #e6ccff);
}

section[data-testid="stSidebar"] {
    background-color: #d9b3ff;
}

h1, h2, h3 {
    color: #6a0dad;
}

div[data-testid="metric-container"] {
    background-color: #ffe6f7;
    border-radius: 12px;
    padding: 10px;
    box-shadow: 2px 2px 10px #d6b3ff;
}

button {
    background-color: #c084fc !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
}

</style>
""", unsafe_allow_html=True)

st.title("AI Powered Marketing Intelligence Dashboard")

# ================= LOAD MODELS =================
 @st.cache_resource
def load_assets():
    
xgb_path = hf_hub_download(
        repo_id="AI908/marketing-campaign-model",
        filename="xgb_model.pkl"
)

 features_path = hf_hub_download(
        repo_id="AI908/marketing-campaign-featuress",
        filename="xgb_features.pkl"
)

kmeans_path = hf_hub_download(
    repo_id="AI908/marketing-campaign-model",
    filename="kmeans_model.pkl"
)

scaler_path = hf_hub_download(
    repo_id="AI908/marketing-campaign-scaler",
    filename="scaler.pkl"
)

labels_path = hf_hub_download(
    repo_id="AI908/marketing-ca,paign-labels",
    filename="cluster_labels.pkl"
)

xgb_model = joblib.load(xgb_path)
kmeans_model = joblib.load(kmeans_path)
xgb_features=joblib.load(features-path)
scaler = joblib.load(scaler_path)
cluster_labels = joblib.load(labels_path)


# ================= LOAD DATA =================

df = pd.read_csv("Digital_marketing_campaign_data.csv")
df.columns = df.columns.str.strip()
 return xgb_model, kmeans_model, scaler, cluster_labels, df


xgb_model, kmeans_model,xgb_features, scaler, cluster_labels, df = load_assets()

# ================= FEATURE ENGINEERING =================
df["EngagementScore"] = (
    df["ClickThroughRate"] +
    df["SocialShares"] +
    df["EmailOpens"] +
    df["EmailClicks"]
) / 4

# ================= CLUSTERING =================
df_enc = pd.get_dummies(
    df,
    columns=[c for c in ["CampaignType", "CampaignChannel", "Gender"] if c in df.columns]
)

X = df_enc.reindex(columns=scaler.feature_names_in_, fill_value=0)
df["Cluster"] = kmeans_model.predict(scaler.transform(X))
df["ClusterLabel"] = df["Cluster"].map(cluster_labels)

# ================= FILTERS =================
st.sidebar.header("Filters")

age = st.sidebar.multiselect("Age", sorted(df["Age"].unique()), default=df["Age"].unique())
gender = st.sidebar.multiselect("Gender", df["Gender"].unique(), default=df["Gender"].unique())
ctype = st.sidebar.multiselect("Campaign Type", df["CampaignType"].unique(), default=df["CampaignType"].unique())
channel = st.sidebar.multiselect("Campaign Channel", df["CampaignChannel"].unique(), default=df["CampaignChannel"].unique())

filtered = df[
    (df["Age"].isin(age)) &
    (df["Gender"].isin(gender)) &
    (df["CampaignType"].isin(ctype)) &
    (df["CampaignChannel"].isin(channel))
]

if filtered.empty:
    st.error("No data available for selected filters.")
    st.stop()

# ================= KPI =================
st.subheader("Key Metrics")

top_channel = filtered.groupby("CampaignChannel")["EngagementScore"].mean().idxmax()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Engagement Score", f"{filtered['EngagementScore'].mean():.2f}")
c2.metric("Conversion Rate", f"{filtered['ConversionRate'].mean()*100:.2f}%")
c3.metric("CTR", f"{filtered['ClickThroughRate'].mean()*100:.2f}%")
c4.metric("Top Channel", top_channel)

# ================= TABS =================
tab1, tab2, tab3 = st.tabs(["Dashboard", "Scenario Analysis", "AI Insights"])

# ================= DASHBOARD =================
with tab1:

    st.subheader("Customer Segments (3D Clustering)")

    fig1 = px.scatter_3d(
        filtered,
        x="AdSpend",
        y="EngagementScore",
        z="ConversionRate",
        color="ClusterLabel",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Engagement Drivers")

    feature_data = pd.DataFrame({
        "Feature": ["AdSpend","TimeOnSite","EmailClicks","WebsiteVisits","SocialShares","EmailOpens"],
        "Importance": [
            filtered["AdSpend"].corr(filtered["EngagementScore"]),
            filtered["TimeOnSite"].corr(filtered["EngagementScore"]),
            filtered["EmailClicks"].corr(filtered["EngagementScore"]),
            filtered["WebsiteVisits"].corr(filtered["EngagementScore"]),
            filtered["SocialShares"].corr(filtered["EngagementScore"]),
            filtered["EmailOpens"].corr(filtered["EngagementScore"]),
        ]
    })

    fig2 = px.bar(
        feature_data,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Purples"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Ad Spend by Campaign Type")

    fig3 = px.bar(
        filtered.groupby("CampaignType")["AdSpend"].mean().reset_index(),
        x="CampaignType",
        y="AdSpend",
        color="CampaignType",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Engagement Distribution")

    fig4 = px.histogram(
        filtered,
        x="EngagementScore",
        nbins=20,
        color="CampaignChannel",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig4, use_container_width=True)

# ================= SCENARIO ANALYSIS =================
with tab2:

    st.subheader("Scenario Analysis (What-if Simulation)")

    base = filtered.iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("Scenario A")
        a_ad = st.slider("Ad Spend A", 0, 10000, int(base["AdSpend"]))
        a_time = st.slider("Time on Site A", 0, 500, int(base["TimeOnSite"]))
        a_clicks = st.slider("Email Clicks A", 0, 200, int(base["EmailClicks"]))
        a_social = st.slider("Social Shares A", 0, 1000, int(base["SocialShares"]))
        a_opens = st.slider("Email Opens A", 0, 1000, int(base["EmailOpens"]))

    with col2:
        st.markdown("Scenario B")
        b_ad = st.slider("Ad Spend B", 0, 10000, int(base["AdSpend"]))
        b_time = st.slider("Time on Site B", 0, 500, int(base["TimeOnSite"]))
        b_clicks = st.slider("Email Clicks B", 0, 200, int(base["EmailClicks"]))
        b_social = st.slider("Social Shares B", 0, 1000, int(base["SocialShares"]))
        b_opens = st.slider("Email Opens B", 0, 1000, int(base["EmailOpens"]))

    if st.button("Run Scenario Analysis"):

        def predict(ad, time, clicks, social, opens):
            return (0.35 * ad + 0.2 * clicks + 0.2 * social + 0.15 * opens + 0.1 * time) / 100

        eng_a = predict(a_ad, a_time, a_clicks, a_social, a_opens)
        eng_b = predict(b_ad, b_time, b_clicks, b_social, b_opens)

        st.subheader("Scenario Results")

        c1, c2 = st.columns(2)
        c1.metric("Engagement A", f"{eng_a:.2f}")
        c2.metric("Engagement B", f"{eng_b:.2f}")

        compare = pd.DataFrame({
            "Metric": ["Engagement"],
            "Scenario A": [eng_a],
            "Scenario B": [eng_b]
        })

        fig5 = px.bar(
            compare,
            x="Metric",
            y=["Scenario A", "Scenario B"],
            barmode="group",
            color_discrete_sequence=["#f8c8ff", "#d9b3ff"]
        )

        st.plotly_chart(fig5, use_container_width=True)

        best = "A" if eng_a > eng_b else "B"
        st.success(f"Recommended Scenario: {best}")

        st.info(f"Scenario {best} is expected to perform better based on simulated engagement factors.")

# ================= AI INSIGHTS (FULL DYNAMIC ENGINE) =================
with tab3:

    st.subheader("Performance Analysis")

    eng_avg = filtered["EngagementScore"].mean()
    eng_global = df["EngagementScore"].mean()

    ctr_avg = filtered["ClickThroughRate"].mean()
    ctr_global = df["ClickThroughRate"].mean()

    conv_avg = filtered["ConversionRate"].mean()
    conv_global = df["ConversionRate"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Engagement", f"{eng_avg:.2f}")
    c2.metric("CTR", f"{ctr_avg*100:.2f}%")
    c3.metric("Conversion", f"{conv_avg*100:.2f}%")

    # ================= REAL DATA-DRIVEN INSIGHT =================
    metrics_gap = {
        "Engagement": eng_avg - eng_global,
        "CTR": ctr_avg - ctr_global,
        "Conversion": conv_avg - conv_global
    }

    best_metric = max(metrics_gap, key=metrics_gap.get)
    worst_metric = min(metrics_gap, key=metrics_gap.get)

    score = sum(metrics_gap.values())

    if score > 0.15:
        insight = f"Strong campaign performance. {best_metric} is the leading driver above benchmark."
    elif score > 0:
        insight = f"Good performance overall. {best_metric} is strong but {worst_metric} needs improvement."
    elif score > -0.15:
        insight = f"Average performance detected. Focus on improving {worst_metric} to boost results."
    else:
        insight = f"Below benchmark performance. Immediate optimization needed in {worst_metric}."

    best_campaign = filtered.loc[filtered["EngagementScore"].idxmax()]
    best_channel = filtered.groupby("CampaignChannel")["EngagementScore"].mean().idxmax()

    st.write("Best Campaign Type:", best_campaign["CampaignType"])
    st.write("Best Channel:", best_channel)

    st.success(insight)
