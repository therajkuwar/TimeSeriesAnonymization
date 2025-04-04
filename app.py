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
sample_file_path = "https://raw.githubusercontent.com/therajkuwar/TimeSeriesAnonymization/main/sample_dataset/patient_1_year_readings.csv"  # Update with your actual file name

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
page = st.sidebar.radio("Select Module", ["Anonymization", "Privacy & Utility Check", "Temporal Pattern Check"])

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
        data = sample_data

    st.write("Dataset Preview:")
    st.write(data.head())

    # Select numerical columns
    num_columns = st.multiselect("Select Columns to Anonymize", data.select_dtypes(include=np.number).columns)
    num_states = st.slider("Number of States", 2, 10, 4)
    noise_scale = st.slider("Noise Scale", 0.1, 50.0, 0.1)

    # Functions for anonymization
    def discretize_column(values, num_states):
        qcut_result = pd.qcut(values, num_states, duplicates="drop")
        ranges = qcut_result.cat.categories
        return qcut_result.cat.codes, ranges

    def build_transition_matrix(states, num_states):
        matrix = np.zeros((num_states, num_states))

        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            if current_state < num_states and next_state < num_states:
                matrix[current_state, next_state] += 1

        matrix = np.nan_to_num(matrix / matrix.sum(axis=1, keepdims=True))  # Normalize
        return matrix

    def add_exponential_noise(matrix, noise_scale):
        noisy_matrix = np.exp(matrix / noise_scale)  # Apply exponential transformation
        noisy_matrix /= noisy_matrix.sum(axis=1, keepdims=True)  # Normalize rows
        return noisy_matrix

    def anonymize_column(values, states, ranges, noisy_matrix):
        anonymized_values = []
        
        for i, current_value in enumerate(values):
            current_state = states[i]
            range_min, range_max = ranges[current_state].left, ranges[current_state].right
            
            # Step 1: Use noisy transition probabilities to sample the next state
            state_probabilities = noisy_matrix[current_state]
            sampled_state = np.random.choice(np.arange(len(state_probabilities)), p=state_probabilities)

            # Step 2: Generate a value influenced by real trends
            noise_factor = np.random.normal(loc=0.5, scale=0.2)  # Gaussian noise
            noise_factor = np.clip(noise_factor, 0, 1)
            new_value = range_min + (range_max - range_min) * noise_factor

            anonymized_values.append(new_value)

        # Step 3: Apply smoothing to maintain temporal consistency
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
    # Function to compute privacy & utility scores
    def compute_privacy_utility(rmse, dtw, correlation, entropy_original, entropy_anonymized, 
                                rmse_min, rmse_max, dtw_min, dtw_max, entropy_diff_max):
        """Computes Privacy and Utility Scores based on given metrics."""
        rmse_score = (rmse - rmse_min) / (rmse_max - rmse_min) if rmse_max > rmse_min else 0
        dtw_score = (dtw - dtw_min) / (dtw_max - dtw_min) if dtw_max > dtw_min else 0
        correlation_score = 1 - abs(correlation)  # Invert correlation (higher correlation = lower privacy)
        
        entropy_diff = entropy_original - entropy_anonymized
        entropy_score = entropy_diff / entropy_diff_max if entropy_diff_max > 0 else 0

        # Assign weights (can be tuned)
        alpha, beta, gamma, delta = 0.25, 0.25, 0.25, 0.25  

        privacy_score = 100 * (alpha * rmse_score + beta * dtw_score + gamma * correlation_score + delta * entropy_score)
        privacy_score = max(0, min(100, privacy_score))  # Ensure between 0 and 100
        utility_score = 100 - privacy_score  # Utility is the inverse of privacy

        return round(privacy_score, 2), round(utility_score, 2)

    # Streamlit UI
    st.header("Privacy & Utility Analysis")
    if 'orig_file' in locals() and 'anon_file' in locals():
        orig_data = pd.read_csv(orig_file)
        anon_data = pd.read_csv(anon_file)

        common_columns = list(set(orig_data.columns) & set(anon_data.columns))
        num_columns = st.multiselect("Select Columns for Analysis", common_columns)

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

                # Compute Privacy & Utility Scores
                privacy, utility = compute_privacy_utility(
                    rmse, dtw_dist, correlation, entropy_orig, entropy_anon,
                    rmse_min=5, rmse_max=15, dtw_min=2000, dtw_max=3000, entropy_diff_max=0.5
                )

                results.append((col, correlation, rmse, dtw_dist, entropy_orig, entropy_anon, abs(entropy_orig - entropy_anon), privacy, utility))

            # Convert results to DataFrame
            results_df = pd.DataFrame(results, columns=["Column", "Correlation", "RMSE", "DTW Distance", 
                                                        "Entropy (Original)", "Entropy (Anonymized)", "Entropy Diff", 
                                                        "Privacy Score (%)", "Utility Score (%)"])
            
            # Display results
            st.dataframe(results_df)
            
            # Download option
            csv = results_df.to_csv(index=False).encode()
            st.download_button("Download Analysis Report", csv, "privacy_utility_analysis.csv", "text/csv")

elif page == "Temporal Pattern Check":
    st.header("Temporal Pattern Check")
    if orig_file and anon_file:
        orig_data = pd.read_csv(orig_file)
        anon_data = pd.read_csv(anon_file)        

        st.sidebar.header("Dataset Settings")

        # Select date column
        date_column = st.sidebar.selectbox("Select Date Column", orig_data.columns)

        # Select numerical columns
        num_columns = st.sidebar.multiselect(
            "Select Numerical Columns for Comparison",
            orig_data.select_dtypes(include=np.number).columns.tolist(),
        )

        if date_column and num_columns:
            # Convert date column to datetime
            orig_data[date_column] = pd.to_datetime(orig_data[date_column])
            anon_data[date_column] = pd.to_datetime(anon_data[date_column])

            # Sort datasets by date
            original_data = orig_data.sort_values(by=date_column)
            anonymized_data = anon_data.sort_values(by=date_column)

            # --- Temporal Pattern Comparison ---
            st.subheader("Temporal Pattern Comparison")
            for col in num_columns:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(original_data[date_column], original_data[col], label="Original Data", alpha=0.7)
                ax.plot(anonymized_data[date_column], anonymized_data[col], label="Anonymized Data", alpha=0.7)
                ax.set_title(f"Temporal Pattern Comparison for {col}")
                ax.set_xlabel("Date")
                ax.set_ylabel(col)
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

            # --- Aggregated Visualization ---
            st.subheader("Aggregated Temporal Pattern Comparison")

            def aggregate_data(data, date_column, columns, freq="W"):
                return data.set_index(date_column).resample(freq)[columns].mean().reset_index()

            freq = st.selectbox("Select Aggregation Frequency", ["Daily", "Weekly", "Monthly"], index=1)
            freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}

            aggregated_original = aggregate_data(original_data, date_column, num_columns, freq=freq_map[freq])
            aggregated_anonymized = aggregate_data(anonymized_data, date_column, num_columns, freq=freq_map[freq])

            for col in num_columns:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(aggregated_original[date_column], aggregated_original[col], label="Original (Aggregated)", alpha=0.7)
                ax.plot(aggregated_anonymized[date_column], aggregated_anonymized[col], label="Anonymized (Aggregated)", alpha=0.7)
                ax.set_title(f"Aggregated Temporal Pattern for {col} ({freq})")
                ax.set_xlabel("Date")
                ax.set_ylabel(col)
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

            # --- Correlation Analysis ---
            st.subheader("Correlation Analysis")
            for col in num_columns:
                correlation = np.corrcoef(original_data[col], anonymized_data[col])[0, 1]
                st.write(f"Correlation for {col}: {correlation:.2f}")
                if correlation > 0.8:
                    st.success(f"The temporal pattern for {col} is strongly preserved (correlation > 0.8).")
                elif correlation > 0.5:
                    st.warning(f"The temporal pattern for {col} is moderately preserved (correlation > 0.5).")
                else:
                    st.error(f"The temporal pattern for {col} is weakly preserved (correlation ≤ 0.5).")
        else:
            st.warning("Please select a date column and numerical columns for analysis.")
    else:
        st.info("Upload both original and anonymized datasets to proceed.")
