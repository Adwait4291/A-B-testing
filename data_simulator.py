# data_simulator.py
"""
Simulates A/B test data.
NOTE: This module is not directly used when analyzing marketing_AB.csv,
but kept for reference or potential future use.
The main execution block is commented out.
"""
import pandas as pd
import numpy as np
# import config # Keep commented unless running simulation specifically
import os

def simulate_data(n_samples, conv_rate_a, conv_rate_b):
    """
    Simulates A/B test data with two groups (A and B) and their conversions.

    Args:
        n_samples (int): Total number of samples (users) to simulate.
        conv_rate_a (float): Conversion rate for group A (control).
        conv_rate_b (float): Conversion rate for group B (treatment).

    Returns:
        pandas.DataFrame: DataFrame with columns 'user_id', 'group', 'converted'.
                         'converted' is 1 if the user converted, 0 otherwise.
    """
    # Assign users roughly 50/50 to Group A (control) or Group B (treatment)
    group = np.random.choice(['A', 'B'], size=n_samples, p=[0.5, 0.5])

    # Simulate conversions based on group-specific rates using binomial distribution
    converted = np.zeros(n_samples, dtype=int)
    mask_a = (group == 'A')
    converted[mask_a] = np.random.binomial(1, conv_rate_a, size=mask_a.sum())
    mask_b = (group == 'B')
    converted[mask_b] = np.random.binomial(1, conv_rate_b, size=mask_b.sum())

    # Create DataFrame
    df = pd.DataFrame({
        'user_id': range(n_samples),
        'group': group,
        'converted': converted
    })
    return df

# --- Main execution block commented out as we are using real data ---
# if __name__ == "__main__":
#     print("Starting data simulation...")
#
#     # Check if config is imported and has necessary variables
#     try:
#         import config
#         required_configs = ['SAMPLE_SIZE', 'CONVERSION_RATE_A', 'CONVERSION_RATE_B', 'SIMULATION_OUTPUT_FILE']
#         if not all(hasattr(config, attr) for attr in required_configs):
#             print("Error: Missing one or more required simulation variables in config.py")
#             exit(1) # Or raise an error
#
#         # Perform simulation
#         simulated_df = simulate_data(config.SAMPLE_SIZE,
#                                      config.CONVERSION_RATE_A,
#                                      config.CONVERSION_RATE_B)
#
#         print(f"Generated DataFrame with {len(simulated_df)} samples.")
#         print(simulated_df.head())
#         print("\nGroup distribution:")
#         print(simulated_df['group'].value_counts())
#         print("\nConversion summary:")
#         print(simulated_df.groupby('group')['converted'].mean())
#
#         # Save the data to CSV
#         try:
#             output_file = config.SIMULATION_OUTPUT_FILE
#             simulated_df.to_csv(output_file, index=False)
#             print(f"\nData simulation complete. Saved to {output_file}")
#         except IOError as e:
#             print(f"\nError: Could not write file to {output_file}")
#             print(f"System Error: {e}")
#         except Exception as e:
#             print(f"\nAn unexpected error occurred during file saving: {e}")
#
#     except ImportError:
#         print("Error: config.py not found or cannot be imported.")
#     except Exception as main_e:
#         print(f"An error occurred in the main simulation block: {main_e}")
# ------------------------------------------------------------------