# config.py
"""
Configuration settings for the A/B testing analysis using marketing_AB.csv.
"""
import os

# --- Input Data ---
# Get the absolute path of the directory where this config file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the input data file relative to this config file's directory
INPUT_DATA_FILE = os.path.join(BASE_DIR, 'marketing_AB.csv')

# --- Experiment Setup ---
GROUP_COLUMN = 'test group'     # Column identifying the groups in marketing_AB.csv
CONTROL_GROUP_VALUE = 'psa'   # Value representing the Control group ('Public Service Announcement')
TREATMENT_GROUP_VALUE = 'ad'  # Value representing the Treatment group ('Advertisement')

# --- Metrics Configuration ---
# Primary metric (must be a binary 0/1 or boolean column for conversion analysis)
# In marketing_AB.csv, 'converted' is boolean. Analyzer handles bool/int conversion.
CONVERSION_COLUMN = 'converted'

# Other potential metrics to analyze (numeric columns from marketing_AB.csv)
# Note: Analyzer/Dashboard will check if these exist before using
CONTINUOUS_METRICS = ['total ads', 'most ads hour']

# --- Statistical Analysis Parameters ---
ALPHA = 0.05  # Significance level (commonly 5%)

# --- PDF Report Parameters ---
REPORT_FILENAME = os.path.join(BASE_DIR, "marketing_ab_test_report.pdf") # Default output filename
REPORT_TITLE = "Marketing A/B Test Analysis Report (Ad vs PSA)"

# --- Dashboard Parameters ---
DASHBOARD_TITLE = 'Marketing A/B Test Dashboard'

# --- (Optional) Segmentation Columns ---
# Define columns from marketing_AB.csv that might be used for filtering/segmentation
SEGMENTATION_COLUMNS = ['most ads day'] # Only 'most ads day' identified as potential segment


print(f"Config loaded. Input data: {INPUT_DATA_FILE}")
print(f"Group column: '{GROUP_COLUMN}', Control: '{CONTROL_GROUP_VALUE}', Treatment: '{TREATMENT_GROUP_VALUE}'")
print(f"Conversion metric: '{CONVERSION_COLUMN}'")
print(f"Potential continuous metrics: {CONTINUOUS_METRICS}")
print(f"Significance Level (Alpha): {ALPHA}")