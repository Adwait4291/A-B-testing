# config.py
"""Configuration settings for the A/B test data simulator."""

import numpy as np
from datetime import datetime, timedelta

# --- Simulation Parameters ---
N_ROWS = 100_000  # Medium size dataset
CONTROL_GROUP_SIZE = 0.5 # Proportion of users in the control group (0.0 to 1.0)
NOISE_FACTOR = 0.2   # General variability factor (higher means more randomness)
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 3, 31)

# --- User Characteristics ---
COUNTRIES = ['US', 'UK', 'DE', 'CA', 'FR', 'AU', 'IN', 'BR']
CHANNELS = ['Paid Search', 'Organic Search', 'Social Media', 'Direct', 'Referral', 'Email']
DEVICES = ['Mobile', 'Desktop', 'Tablet']
DEVICE_DISTRIBUTION = [0.65, 0.30, 0.05] # Mobile slightly heavier
USER_TYPES = ['New', 'Returning']
USER_TYPE_DISTRIBUTION = [0.7, 0.3] # More new users
REFERRERS = ['google', 'facebook', 'instagram', 'bing', 'direct', 'partner_site', 'newsletter', 'other']
REFERRER_DISTRIBUTION = [0.35, 0.15, 0.10, 0.08, 0.12, 0.05, 0.1, 0.05]

# --- Base Funnel Conversion Rates (Control Group - Average) ---
# These represent the probability of moving from one step to the next
BASE_RATES = {
    'click_from_view': 0.15,         # 15% view-to-click rate
    'app_start_from_click': 0.30,    # 30% click-to-app_start rate
    'app_complete_from_start': 0.40 # 40% app_start-to-app_complete rate
}

# --- Treatment Effect (Absolute Lift/Decrease in Conversion Rate) ---
# Represents the *change* in probability for the treatment group
TREATMENT_EFFECT = {
    'click_from_view': 0.02,         # Treatment increases view-to-click by 2 percentage points
    'app_start_from_click': 0.05,    # Treatment increases click-to-start by 5 percentage points
    'app_complete_from_start': -0.03 # Treatment decreases start-to-complete by 3 percentage points (e.g., more friction)
}

# --- Base Continuous Metrics (Control Group - Average) ---
BASE_CONTINUOUS = {
    'time_on_page_mean': 90,  # seconds
    'time_on_page_std': 45,   # seconds
    'bounce_rate': 0.45,      # 45% bounce rate for users who *clicked* but didn't start app
    'scroll_depth_mean': 60,  # Percentage scroll depth
    'scroll_depth_std': 25,   # Percentage scroll depth std dev
}

# --- Treatment Effect (Multiplicative Factor for Continuous Metrics) ---
# Represents how the treatment *multiplies* the base value
TREATMENT_EFFECT_CONTINUOUS = {
    'time_on_page_factor': 1.15, # Treatment users spend 15% longer on page (if they click)
    'bounce_rate_factor': 0.90,  # Treatment users have 10% lower bounce rate (if they click)
    'scroll_depth_factor': 1.10, # Treatment users scroll 10% further down (if they click)
}

# --- Interaction Effects/Segment Adjustments (Multiplicative Factors) ---
# How different segments behave compared to the average base rates/effects
SEGMENT_ADJUSTMENTS = {
    'device': {
        'Mobile': {'rate_factor': 0.9, 'effect_factor': 1.0}, # Mobile users convert slightly less
        'Desktop': {'rate_factor': 1.1, 'effect_factor': 1.1}, # Desktop users convert slightly more, treatment more effective
        'Tablet': {'rate_factor': 1.0, 'effect_factor': 1.0}
    },
    'user_type': {
        'New': {'rate_factor': 0.95, 'effect_factor': 0.9}, # New users convert slightly less, treatment less effective
        'Returning': {'rate_factor': 1.1, 'effect_factor': 1.2} # Returning users convert more, treatment more effective
    }
    # Add more complex interactions if needed (e.g., country-specific)
}

# --- Cost Simulation ---
# Cost per click ranges (min, max) - potentially varying by channel/country
# Simplified: Varying by channel only for this example
CPC_RANGES = {
    'Paid Search': (0.8, 2.5),
    'Organic Search': (0.0, 0.0), # Organic has no direct click cost
    'Social Media': (0.4, 1.5),
    'Direct': (0.0, 0.0),
    'Referral': (0.1, 0.5), # May have partner costs
    'Email': (0.05, 0.2)    # Email platform costs distributed
}
# Default CPC if channel not found
DEFAULT_CPC_RANGE = (0.2, 0.6)