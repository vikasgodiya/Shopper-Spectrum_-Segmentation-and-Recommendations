import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np


# --- Load Models and Data ---
kmeans = joblib.load("C:\\Users\\vikas\\OneDrive\\Desktop\\Labmentix\\LMS Customer Segmentation\\kmeans_rfm_model.pkl")
scaler = joblib.load("C:\\Users\\vikas\\OneDrive\\Desktop\\Labmentix\\LMS Customer Segmentation\\rfm_scaler.pkl")
product_names = pickle.load(open("C:\\Users\\vikas\\OneDrive\\Desktop\\Labmentix\\LMS Customer Segmentation\\product_name_map.pkl", "rb"))
product_similarity = pickle.load(open("C:\\Users\\vikas\\OneDrive\\Desktop\\Labmentix\\LMS Customer Segmentation\\product_similarity_matrix.pkl", "rb"))

# --- UI Styling ---
st.set_page_config(page_title="Shopper Spectrum App", layout="centered")
st.markdown("""
    <style>
        .stApp {
            background-color: #1c1c1e;
            color: #ffffff;
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            color: #ffffff;
            font-size: 40px;
            text-align: center;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px #000000;
        }
        .section-header {
            font-size: 22px;
            color: #00d084;
            margin-top: 40px;
            text-shadow: 1px 1px 2px #000000;
        }
        .stSelectbox label,
        .stTextInput label,
        .stNumberInput label,
        .stMarkdown p,
        .stMarkdown span,
        .st-expanderHeader {
            color: #ffffff !important;
            font-weight: 600;
        }
        .stButton > button {
            background-color: #222;
            color: white;
            border: 2px solid #00d084;
        }
        .stButton > button:hover {
            background-color: #00d084;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown('<div class="title">🛍️ Shopper Spectrum: RFM & Product Recommendations</div>', unsafe_allow_html=True)

# --- Project Overview ---
with st.expander("📌 Project Overview", expanded=False):
    st.write("""
        This capstone project aims to identify distinct customer segments using RFM (Recency, Frequency, Monetary) analysis
        and provide personalized product recommendations based on purchase behavior.

        **Modules Implemented:**
        - Customer segmentation using KMeans clustering
        - Item-based collaborative filtering using cosine similarity
        - Real-time prediction via Streamlit UI

        **Use Cases:**
        - Targeted marketing & loyalty campaigns
        - Customer retention insight
        - E-commerce personalization engine
    """)

# --- Tabs ---
tab1, tab2 = st.tabs(["📦 Product Recommender", "👤 Customer Segment Predictor"])

# === Tab 1: Product Recommendation ===
with tab1:
    st.markdown('<div class="section-header">Product Recommendation Module</div>', unsafe_allow_html=True)

    st.markdown("Select a product from the list to get similar recommendations:")
    valid_products = list(product_similarity.columns)
    selected_product = st.selectbox("Select Product Code:", valid_products)

    if st.button("🔍 Get Recommendations"):
        similar_scores = product_similarity[selected_product].sort_values(ascending=False)
        top_5 = similar_scores.iloc[1:6].index

        st.success("Top 5 Similar Products:")
        for idx, code in enumerate(top_5, 1):
            st.markdown(f"**{idx}.** {product_names.get(code, 'Unknown Product')} (`{code}`)")

# === Tab 2: Customer Segment Prediction ===
with tab2:
    st.markdown('<div class="section-header">Customer Segmentation Module</div>', unsafe_allow_html=True)

    r = st.number_input("Recency (days since last purchase):", min_value=0, value=30)
    f = st.number_input("Frequency (number of purchases):", min_value=0, value=5)
    m = st.number_input("Monetary (total spend in £):", min_value=0.0, value=500.0)

    if st.button("📊 Predict Segment"):
        rfm_input = scaler.transform(np.array([[r, f, m]]))
        cluster = kmeans.predict(rfm_input)[0]

        segment_map = {
            0: "Regular",
            1: "At-Risk",
            2: "High-Value",
            3: "Loyal"
        }
        predicted_segment = segment_map.get(cluster, "Unknown")

        st.success(f"Predicted Segment: **{predicted_segment}**")

# --- About Section ---
with st.expander("👨‍💼 About this Project"):
    st.markdown("""
    **Project Name:** Shopper Spectrum  
    **Team Members:** Vikas Hirender Godiya 
    **Capstone Cohort:** Machine Learning USVL  
    **Tools Used:** Python, Streamlit, Sklearn, Pandas, Seaborn, Plotly  
    **Date:** 03-08-2025
    """)