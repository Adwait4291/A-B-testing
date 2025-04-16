# config.py
"""
Simplified configuration for A/B testing analysis.
"""
import os

# --- Input Data ---
# Get the absolute path of the directory where this config file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the input data file relative to this config file's directory
# Make sure 'finance_ab.csv' is in the same folder as this script
INPUT_DATA_FILE = os.path.join(BASE_DIR, 'finance_ab.csv') #

# --- Experiment Setup ---
# Column names MUST match your CSV file exactly
GROUP_COLUMN = 'Version'             # Column identifying the groups (e.g., 'A' or 'B') #
CONTROL_GROUP_VALUE = 'A'         # Value representing the Control group #
TREATMENT_GROUP_VALUE = 'B'       # Value representing the Treatment group #
CONVERSION_COLUMN = 'ApplicationCompleted' # Binary (True/False or 1/0) column for conversion #

# --- Metrics Configuration ---
# List other numeric columns you might want to compare
# REMOVED 'SessionDuration_seconds'
CONTINUOUS_METRICS = [] # Example: ['AnotherNumericColumn'] if you have others #

# --- Statistical Analysis Parameters ---
ALPHA = 0.05  # Significance level (commonly 5%)

# --- Dashboard Parameters ---
DASHBOARD_TITLE = 'Simple A/B Test Dashboard' #

print("Simplified config loaded. Continuous metrics updated.")