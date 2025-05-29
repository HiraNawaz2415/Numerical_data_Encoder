import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer, Binarizer

st.set_page_config(page_title="Numerical Data Encoding", layout="wide")
st.title("🔢 Encoding Numerical Data (Binning & Binarization)")


def generate_csv_download(df, filename="encoded_data.csv"):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Encoded Data as CSV",
        data=csv,
        file_name=filename,
        mime='text/csv'
    )

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.dataframe(df)

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if numeric_cols:
        selected_col = st.selectbox("Select a numerical column", numeric_cols)
        method = st.radio("Choose encoding method", ["Equal-width Binning", "Equal-frequency Binning", "Custom Binning", "Binarization"])

        col_data = df[[selected_col]].copy()

        if method in ["Equal-width Binning", "Equal-frequency Binning"]:
            n_bins = st.slider("Number of bins", min_value=2, max_value=10, value=4)

            strategy = "uniform" if method == "Equal-width Binning" else "quantile"
            kbin = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
            df[f'{method} Result'] = kbin.fit_transform(col_data).astype(int)

            bin_labels = [f'Bin{i+1}' for i in range(n_bins)]
            df[f'{method} Label'] = pd.cut(col_data[selected_col], bins=np.unique(kbin.bin_edges_[0]), labels=bin_labels, include_lowest=True)

            st.dataframe(df[[selected_col, f'{method} Result', f'{method} Label']])
            generate_csv_download(df[[selected_col, f'{method} Result', f'{method} Label']], filename=f"{method.lower().replace(' ', '_')}.csv")

            st.subheader("📊 Histogram")
            fig, ax = plt.subplots()
            df[selected_col].hist(bins=n_bins, ax=ax, color="skyblue")
            plt.title(method)
            st.pyplot(fig)

        elif method == "Custom Binning":
            custom_edges = st.text_input("Enter custom bin edges separated by commas (e.g. 0,18,40,100)", "0,18,40,100")
            try:
                bins = [float(x.strip()) for x in custom_edges.split(',')]
                labels = [f"Bin{i+1}" for i in range(len(bins) - 1)]
                df['CustomBin'] = pd.cut(df[selected_col], bins=bins, labels=labels)
                st.dataframe(df[[selected_col, 'CustomBin']])
                generate_csv_download(df[[selected_col, 'CustomBin']], filename="custom_binning.csv")

                st.subheader("📊 Histogram of Custom Bins")
                fig, ax = plt.subplots()
                df[selected_col].hist(ax=ax, bins=10, color="orange")
                plt.title("Custom Binning")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Invalid input. Error: {e}")

        elif method == "Binarization":
            if col_data[selected_col].isnull().any():
                mean_val = col_data[selected_col].mean()
                col_data[selected_col].fillna(mean_val, inplace=True)
                st.info(f"Missing values in '{selected_col}' were filled with the mean: {mean_val:.2f}")

            threshold = st.slider(
                "Set binarization threshold",
                float(col_data[selected_col].min()),
                float(col_data[selected_col].max()),
                float(col_data[selected_col].mean()),
            )

            binarizer = Binarizer(threshold=threshold)
            df['Binarized'] = binarizer.fit_transform(col_data)

            st.dataframe(df[[selected_col, 'Binarized']])
            generate_csv_download(df[[selected_col, 'Binarized']], filename="binarized_data.csv")

            st.subheader("📊 Binarization Plot")
            fig, ax = plt.subplots()
            ax.hist(col_data[selected_col], bins=10, color='red')
            plt.axvline(threshold, color='black', linestyle='--', label=f"Threshold = {threshold}")
            plt.title("Binarization Threshold Split")
            plt.legend()
            st.pyplot(fig)

    else:
        st.warning("No numeric columns found in the dataset.")
else:
    st.info("Please upload a CSV file to begin.")
