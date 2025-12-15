import streamlit as st
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Advanced Data Cleaning Platform",
    layout="wide"
)

st.title("🧼 Advanced Data Cleaning Platform")
st.write("Upload any dataset and clean it safely without crashes.")

# -----------------------------
# Upload Dataset
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Raw Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Fix numeric columns stored as strings
    # -----------------------------
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # -----------------------------
    # Column detection
    # -----------------------------
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    st.sidebar.header("⚙️ Cleaning Options")

    # -----------------------------
    # Missing Values
    # -----------------------------
    st.sidebar.subheader("🧩 Missing Values")

    num_strategy = st.sidebar.selectbox(
        "Numeric Imputation Strategy",
        ["mean", "median", "most_frequent"]
    )

    cat_strategy = st.sidebar.selectbox(
        "Categorical Imputation Strategy",
        ["most_frequent", "constant"]
    )

    # -----------------------------
    # Duplicates
    # -----------------------------
    remove_duplicates = st.sidebar.checkbox("🗑️ Remove Duplicates", value=True)

    # -----------------------------
    # Outliers
    # -----------------------------
    handle_outliers = st.sidebar.checkbox("📉 Remove Outliers (IQR)")

    # -----------------------------
    # Scaling
    # -----------------------------
    scale_data = st.sidebar.checkbox("📐 Scale Numeric Data")

    scaler_type = st.sidebar.selectbox(
        "Scaler Type",
        ["StandardScaler", "MinMaxScaler"],
        disabled=not scale_data
    )

    # -----------------------------
    # Encoding
    # -----------------------------
    encode_data = st.sidebar.checkbox("🔠 Encode Categorical Data")

    # -----------------------------
    # Apply Cleaning
    # -----------------------------
    if st.button("🚀 Apply Cleaning"):
        clean_df = df.copy()

        # ---------- Missing Values ----------
        if len(num_cols) > 0:
            num_imputer = SimpleImputer(strategy=num_strategy)
            clean_df[num_cols] = num_imputer.fit_transform(clean_df[num_cols])

        if len(cat_cols) > 0:
            if cat_strategy == "constant":
                cat_imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
            else:
                cat_imputer = SimpleImputer(strategy=cat_strategy)

            clean_df[cat_cols] = cat_imputer.fit_transform(clean_df[cat_cols])

        # ---------- Remove Duplicates ----------
        if remove_duplicates:
            clean_df.drop_duplicates(inplace=True)

        # ---------- Outlier Removal ----------
        if handle_outliers and len(num_cols) > 0:
            for col in num_cols:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                clean_df = clean_df[
                    (clean_df[col] >= lower) & (clean_df[col] <= upper)
                ]

        # ---------- Scaling ----------
        if scale_data and len(num_cols) > 0:
            if scaler_type == "StandardScaler":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()

            clean_df[num_cols] = scaler.fit_transform(clean_df[num_cols])

        # ---------- Encoding ----------
        if encode_data and len(cat_cols) > 0:
            encoder = LabelEncoder()
            for col in cat_cols:
                clean_df[col] = encoder.fit_transform(clean_df[col])

        # -----------------------------
        # Results
        # -----------------------------
        st.success("✅ Data cleaning completed successfully!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Cleaned Dataset Preview")
            st.dataframe(clean_df.head())

        with col2:
            st.subheader("📈 Dataset Info")
            st.write("Shape:", clean_df.shape)
            st.write("Missing values:", clean_df.isnull().sum().sum())

        # -----------------------------
        # Download
        # -----------------------------
        csv = clean_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Cleaned Dataset",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Upload a CSV file to get started.")
