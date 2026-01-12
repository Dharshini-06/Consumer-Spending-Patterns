# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# --------------------------------------------------------
# PAGE STYLE + CUSTOM CSS
# --------------------------------------------------------

st.set_page_config(
    page_title="Consumer Spending Dashboard",
    page_icon="💳",
    layout="wide"
)

# Custom CSS
# -------------------------- LIGHT THEME CSS --------------------------
light_theme_css = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
    background-color: #ffffff !important;
}

/* Main App Background */
.stApp {
    background-color: #ffffff !important;
}

/* Sidebar */
.css-1d391kg, .css-12oz5g7, .css-hxt7ib {
    background-color: #f8f9fa !important; /* Slightly off-white */
    border-right: 1px solid #dddddd !important;
}

/* Sidebar Widget Labels (e.g., "Filter by Category") */
.sidebar .sidebar-content label {
    color: #333333 !important;
}

/* Multiselect background fix */
.css-1n76uvr, .css-1n76uvr * {
    background-color: #f0f0f0 !important;
    color: #333333 !important;
}

/* Dropdowns */
.stSelectbox div, .stMultiSelect div {
    color: #333333 !important;
}

/* Date Input */
.stDateInput input {
    background-color: #f0f0f0 !important;
    color: #333333 !important;
}

/* File Uploader Box */
.stFileUploader {
    background-color: #f8f9fa !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

/* File Uploader Label */
.stFileUploader label {
    color: #333333 !important;
}

/* File Uploader Inside Box */
.stFileUploader div[data-testid="stFileUploadDropzone"] {
    background-color: #ffffff !important;
    border: 2px dashed #007bff !important;
    color: #333333 !important;
    border-radius: 10px !important;
}

.stFileUploader div[data-testid="stFileUploadDropzone"] * {
    color: #333333 !important;
}

/* Browse Files Button */
.stFileUploader button {
    background-color: #007bff !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
}
.stFileUploader button:hover {
    background-color: #0056b3 !important;
    color: #ffffff !important;
}

/* Data Table */
.dataframe {
    background-color: #ffffff !important;
    color: #333333 !important;
}

/* Table Header */
thead tr th {
    background-color: #f8f9fa !important;
    color: #333333 !important;
}

/* KPI Cards */
.metric-card {
    background-color: #ffffff !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 16px !important;
    padding: 25px !important;
    text-align: center !important;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08) !important;
    transition: 0.2s ease-in-out !important;
}
.metric-card:hover {
    transform: translateY(-4px) !important;
}
.metric-card h4 {
    color: #007bff !important;
}
.metric-value {
    font-size: 32px !important;
    font-weight: 600 !important;
    color: #007bff !important;
}

/* Section Titles */
.section-header, h1, h2, h3, h4 {
    color: #0056b3 !important;
}

/* Buttons */
.stButton>button {
    background-color: #00eaff !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
.stButton>button:hover {
    background-color: #0056b3 !important;
    color: #ffffff !important;
}

/* Plot Background */
.js-plotly-plot, .plotly, .plot-container {
    background-color: #ffffff !important;
}
g {
    fill: #333333 !important;
    color: #333333 !important;
}

</style>
"""
st.markdown(light_theme_css, unsafe_allow_html=True)


# --------------------------------------------------------
# TITLE
# --------------------------------------------------------
st.title("💳 Consumer Spending Patterns — Dashboard")

# --------------------------------------------------------
# FILE UPLOADER
# --------------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV/Excel Dataset", type=["csv", "xlsx"])

if uploaded_file is not None:

    # Load dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ Dataset loaded successfully!")

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # --------------------------------------------------------
    # Cleaning
    # --------------------------------------------------------
    df = df.drop_duplicates()
    df = df.dropna()

    # --------------------------------------------------------
    # Sidebar Filters
    # --------------------------------------------------------
    st.sidebar.header("🔎 Filters")

    if "Category" in df.columns:
        selected_category = st.sidebar.multiselect(
            "Filter by Category:",
            options=df["Category"].unique(),
            default=df["Category"].unique()
        )
        df = df[df["Category"].isin(selected_category)]

    if "Gender" in df.columns:
        selected_gender = st.sidebar.multiselect(
            "Filter by Gender:",
            options=df["Gender"].unique(),
            default=df["Gender"].unique()
        )
        df = df[df["Gender"].isin(selected_gender)]

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        start_date = st.sidebar.date_input("Start Date:", df["Date"].min())
        end_date = st.sidebar.date_input("End Date:", df["Date"].max())
        df = df[(df["Date"] >= pd.to_datetime(start_date)) &
                (df["Date"] <= pd.to_datetime(end_date))]

    # Add a slider for K-Means
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 Model Settings")
    k_clusters = st.sidebar.slider("Select number of clusters (K)", 2, 10, 4)
    outlier_percentile = st.sidebar.slider("Outlier Sensitivity (Top %)", 1, 10, 5)


    # --------------------------------------------------------
    # KPI METRICS — Beautiful Cards
    # --------------------------------------------------------
    st.markdown("### 📊 Key Statistics")

    total_transactions = len(df)
    unique_customers = df["Customer ID"].nunique() if "Customer ID" in df.columns else 0
    total_revenue = df["Transaction Amount"].sum() if "Transaction Amount" in df.columns else 0
    unique_categories = df["Category"].nunique() if "Category" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f"""
        <div class="metric-card">
        <p style='font-size:22px; color:#333333;'>Transactions</p>
        <h2 style='color: #007bff;'>{total_transactions:,}</h2>
        </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
        <div class="metric-card">
        <p style='font-size:22px; color:#333333;'>Unique Customers</p>
        <h2 style='color: #007bff;'>{unique_customers:,}</h2>
        </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
        <div class="metric-card">
        <p style='font-size:22px; color:#333333;'>Total Revenue</p>
        <h2 style='color: #007bff;'>₹{total_revenue:,.2f}</h2>
        </div>
    """, unsafe_allow_html=True)

    col4.markdown(f"""
        <div class="metric-card">
        <p style='font-size:22px; color:#333333;'>Categories</p>
        <h2 style='color: #007bff;'>{unique_categories}</h2>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --------------------------------------------------------
    # PLOTS & CHARTS (Beautiful Dark Theme)
    # --------------------------------------------------------

    st.subheader("📈 Spending Trends & Insights")

    # Top Categories
    if "Transaction Amount" in df.columns and "Category" in df.columns:
        top_cat = df.groupby("Category")["Transaction Amount"].sum().sort_values(ascending=False).head(10)
        fig1 = px.bar(
            top_cat,
            x=top_cat.index,
            y=top_cat.values,
            title="Top 10 Spending Categories",
            color=top_cat.values,
            color_continuous_scale="teal",
            text_auto=True
        )
        fig1.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font_color='#333333')
        st.plotly_chart(fig1, use_container_width=True)

    # Monthly Revenue
    if "Date" in df.columns:
        df["month"] = df["Date"].dt.to_period("M")
        rev = df.groupby("month")["Transaction Amount"].sum().reset_index()
        rev["month"] = rev["month"].astype(str)

        fig2 = px.line(
            rev,
            x="month",
            y="Transaction Amount",
            title="Monthly Revenue Trend",
            markers=True,
        )
        fig2.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font_color='#333333')
        st.plotly_chart(fig2, use_container_width=True)

    # Gender comparison
    if "Gender" in df.columns and "Transaction Amount" in df.columns:
        fig3 = px.box(
            df,
            x="Gender",
            y="Transaction Amount",
            title="Spending Distribution by Gender",
            color="Gender",
        )
        fig3.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font_color='#333333')
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # --------------------------------------------------------
    # K-MEANS CLUSTERING SECTION
    # --------------------------------------------------------
    st.subheader("🤖 Customer Segmentation (K-Means Clustering)")

    # Select numeric columns for clustering
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 1:
        X = df[numeric_cols]

        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply K-Means
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        # --- Outlier Detection ---
        distances = kmeans.transform(X_scaled)
        min_distances = np.min(distances, axis=1)
        # Points with distance in the top X percentile are outliers
        threshold = np.percentile(min_distances, 100 - outlier_percentile)
        df['Outlier'] = np.where(min_distances > threshold, "Outlier", "Normal")
        # ---

        # Display cluster summary
        st.markdown("#### Cluster Summary (Average Values)")
        cluster_summary = df.groupby('Cluster')[numeric_cols].mean().reset_index()
        st.dataframe(cluster_summary, use_container_width=True)

        # Visualize clusters using PCA
        st.markdown("#### Cluster Visualization (PCA)")
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(X_scaled)
        
        viz_df = pd.DataFrame(pca_data, columns=['PCA1', 'PCA2'])
        viz_df['Cluster'] = df['Cluster'].astype(str) # Convert to string for discrete colors
        viz_df['Outlier'] = df['Outlier']

        fig_pca = px.scatter(
            viz_df,
            x='PCA1',
            y='PCA2',
            color='Cluster',
            symbol='Outlier', # Use symbol to mark outliers
            title='Customer Segments Visualized with PCA',
            color_discrete_sequence=px.colors.qualitative.Vivid,
            symbol_map={"Outlier": "x", "Normal": "circle"}
        )
        fig_pca.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font_color='#333333')
        st.plotly_chart(fig_pca, use_container_width=True)

        # --------------------------------------------------------
        # OUTLIER SECTION
        # --------------------------------------------------------
        st.subheader("🚨 Outlier Analysis")
        outlier_df = df[df['Outlier'] == 'Outlier']
        st.markdown(f"Found **{len(outlier_df)}** outliers (top {outlier_percentile}% most distant points from cluster centers).")

        if len(outlier_df) > 0:
            st.dataframe(outlier_df, use_container_width=True)

    else:
        st.warning("Not enough numeric data available for clustering.")

else:
    st.info("👆 Upload a CSV/Excel file to start.")
