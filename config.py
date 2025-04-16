# config.py
"""Configuration variables for the simulation and analysis."""

COUNTRIES = ['India', 'UK', 'Canada', 'USA', 'Germany']
CHANNELS = ['Organic Search', 'Paid Search', 'Email Marketing', 'Social Media', 'Referral']
DEVICES = ['Mobile', 'Desktop', 'Tablet']

# Base conversion rates (as probabilities) - Control Group
BASE_RATES = {
    'click_from_view': 0.15,
    'app_start_from_click': 0.25,
    'app_complete_from_start': 0.60
}

# Expected lift/effect from Treatment (can be positive or negative)
# Example: 0.02 means a 2 percentage point absolute increase for treatment
TREATMENT_EFFECT = {
    'click_from_view': 0.02,
    'app_start_from_click': 0.03,
    'app_complete_from_start': 0.05
}

# Base metrics for continuous variables (Control Group)
BASE_CONTINUOUS = {
    'time_on_page_mean': 60,  # seconds
    'time_on_page_std': 25,
    'bounce_rate': 0.40
}

# Treatment effect on continuous variables (relative change)
TREATMENT_EFFECT_CONTINUOUS = {
    'time_on_page_factor': 1.1, # Treatment users spend 10% more time
    'bounce_rate_factor': 0.9  # Treatment users bounce 10% less
}

# Simulation parameters
N_ROWS = 15000
CONTROL_GROUP_SIZE = 0.6 # 60% control, 40% treatment (imbalance)
NOISE_FACTOR = 0.1 # General noise level for rates