import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
from sklearn.metrics import mean_squared_error
from fastdtw import fastdtw

# Title
st.title("Time Series Data Anonymization & Analysis Using Markov Chains")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Module", ["Anonymization", "Privacy & Utility Check", "Temporal Pattern Check"])

# File uploader (Common for all modules)
st.sidebar.header("Upload Dataset")
orig_file = st.sidebar.file_uploader("Upload Original Dataset (CSV)", type="csv")
anon_file = st.sidebar.file_uploader("Upload Anonymized Dataset (CSV)", type="csv")

if page == "Anonymization":
    st.header("Time Series Data Anonymization")

    if orig_file:
        data = pd.read_csv(orig_file)
        st.write("Original Dataset Preview:")
        st.write(data.head())

        # Identify date column
        date_columns = [col for col in data.columns if data[col].dtype == 'object' and pd.to_datetime(data[col], errors='coerce').notnull().all()]
        date_column = st.selectbox("Select Date Column", options=date_columns) if date_columns else None
        if date_column:
            data[date_column] = pd.to_datetime(data[date_column])

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

elif page == "Temporal Pattern Check":
    st.header("Temporal Pattern Preservation Check")

    if orig_file and anon_file:
        orig_data = pd.read_csv(orig_file)
        anon_data = pd.read_csv(anon_file)
        date_column = st.sidebar.selectbox("Select Date Column", orig_data.columns)
        num_columns = st.sidebar.multiselect("Select Numerical Columns", orig_data.select_dtypes(include=np.number).columns)

        if date_column and num_columns:
            orig_data[date_column] = pd.to_datetime(orig_data[date_column])
            anon_data[date_column] = pd.to_datetime(anon_data[date_column])
            orig_data, anon_data = orig_data.sort_values(by=date_column), anon_data.sort_values(by=date_column)

            for col in num_columns:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(orig_data[date_column], orig_data[col], label="Original", linestyle="-")
                ax.plot(anon_data[date_column], anon_data[col], label="Anonymized", linestyle="--")
                ax.set_title(f"Temporal Pattern Comparison for {col}")
                ax.legend()
                st.pyplot(fig)
