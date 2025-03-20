import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
from sklearn.metrics import mean_squared_error
from fastdtw import fastdtw


# Title
st.title("Secure & Anonymous: Protecting Your Time Series Data with Markov Chains!")

# Introduction
st.markdown(
    """
    ### What does this website do?
    This platform helps anonymize time series data using Markov Chains to ensure privacy while maintaining data utility.
    
    ### Why use it?
    - Protects sensitive time series data.
    - Preserves essential patterns for analysis.
    - Provides privacy & utility analysis for better decision-making.
    
    ### How does it benefit you?
    - Enables secure sharing of time series data.
    - Reduces the risk of data breaches.
    - Allows easy evaluation of anonymization effectiveness.
    
    ### How to use:
    1. Upload your original time series dataset (CSV format).
    2. Select the numerical columns to anonymize.
    3. Set the number of states & noise level.
    4. Download the anonymized dataset.
    5. Analyze privacy & utility using provided metrics.
    """
)

# Sample Dataset Section
# Load Custom Sample Dataset
st.header("📂 Try a Test Dataset")

# Load dataset from file
sample_file_path = "https://raw.githubusercontent.com/therajkuwarTimeSeriesAnonymization/main/sample_dataset/patient_1_year_readings.csv"  # Update with your actual file name

try:
    sample_data = pd.read_csv(sample_file_path)
    csv = sample_data.to_csv(index=False).encode('utf-8')

    # Show dataset preview
    st.dataframe(sample_data.head())

    # Download button
    st.download_button(
        label="🔹 Download Sample Dataset",
        data=csv,
        file_name="sample_dataset.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error("⚠️ Error loading sample dataset. Make sure the file exists.")


# Navigation Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Module", ["Home", "Anonymization", "Privacy & Utility Check", "Temporal Pattern Check"])

# Home Page
if page == "Home":
    st.markdown("### Click below to get started!")
    st.button("Get Started")

# File uploader (Common for all modules)
st.sidebar.header("Upload Dataset")
orig_file = st.sidebar.file_uploader("Upload Original Dataset (CSV)", type="csv")
anon_file = st.sidebar.file_uploader("Upload Anonymized Dataset (CSV)", type="csv")

if page == "Anonymization":
    st.header("Time Series Data Anonymization")

    if orig_file:
        data = pd.read_csv(orig_file)
    else:
        st.info("No file uploaded. Using sample dataset.")
        data = load_sample_dataset()

    st.write("Dataset Preview:")
    st.write(data.head())

    # Select numerical columns
    num_columns = st.multiselect("Select Columns to Anonymize", data.select_dtypes(include=np.number).columns)
    num_states = st.slider("Number of States", 2, 10, 4)
    noise_scale = st.slider("Noise Scale", 0.01, 1.0, 0.1)

    # Functions for anonymization
    def discretize_column(values, num_states):
        qcut_result = pd.qcut(values, num_states, duplicates="drop")
        ranges = qcut_result.cat.categories
        return qcut_result.cat.codes, ranges

    def build_transition_matrix(states, num_states):
        matrix = np.zeros((num_states, num_states))
        for i in range(len(states) - 1):
            matrix[states[i], states[i + 1]] += 1
        return np.nan_to_num(matrix / matrix.sum(axis=1, keepdims=True))

    def add_exponential_noise(matrix, noise_scale):
        noisy_matrix = np.exp(matrix / noise_scale)
        return noisy_matrix / noisy_matrix.sum(axis=1, keepdims=True)

    def anonymize_column(values, states, ranges, noisy_matrix):
        anonymized_values = []
        for i in range(len(values)):
            current_state = states[i]
            sampled_state = np.random.choice(len(noisy_matrix[current_state]), p=noisy_matrix[current_state])
            range_min, range_max = ranges[sampled_state].left, ranges[sampled_state].right
            anonymized_values.append(range_min + (range_max - range_min) * np.random.uniform())
        return pd.Series(anonymized_values).rolling(window=3, min_periods=1).mean()

    # Anonymization process
    anonymized_data = data.copy()
    for col in num_columns:
        states, ranges = discretize_column(data[col], num_states)
        transition_matrix = build_transition_matrix(states, num_states)
        noisy_matrix = add_exponential_noise(transition_matrix, noise_scale)
        anonymized_data[col] = anonymize_column(data[col], states, ranges, noisy_matrix)

    st.write("Anonymized Dataset Preview:")
    st.write(anonymized_data.head())

    csv = anonymized_data.to_csv(index=False).encode()
    st.download_button("Download Anonymized Dataset", csv, "anonymized_dataset.csv", "text/csv")

elif page == "Privacy & Utility Check":
    st.header("Privacy & Utility Analysis")
    if orig_file and anon_file:
        orig_data = pd.read_csv(orig_file)
        anon_data = pd.read_csv(anon_file)
        common_columns = list(set(orig_data.columns) & set(anon_data.columns))
        num_columns = st.multiselect("Select Columns for Analysis", common_columns, default=common_columns)

        if num_columns:
            results = []
            for col in num_columns:
                orig_values, anon_values = orig_data[col].dropna().values, anon_data[col].dropna().values
                min_len = min(len(orig_values), len(anon_values))
                orig_values, anon_values = orig_values[:min_len], anon_values[:min_len]

                correlation = np.corrcoef(orig_values, anon_values)[0, 1]
                rmse = np.sqrt(mean_squared_error(orig_values, anon_values))
                dtw_dist, _ = fastdtw([(x, 0) for x in orig_values], [(x, 0) for x in anon_values], dist=euclidean)
                entropy_orig, entropy_anon = entropy(np.histogram(orig_values, bins=10)[0]), entropy(np.histogram(anon_values, bins=10)[0])

                results.append((col, correlation, rmse, dtw_dist, entropy_orig, entropy_anon, abs(entropy_orig - entropy_anon)))

            results_df = pd.DataFrame(results, columns=["Column", "Correlation", "RMSE", "DTW Distance", "Entropy (Original)", "Entropy (Anonymized)", "Entropy Diff"])
            st.dataframe(results_df)
            csv = results_df.to_csv(index=False).encode()
            st.download_button("Download Analysis Report", csv, "privacy_utility_analysis.csv", "text/csv")
