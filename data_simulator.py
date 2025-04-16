# data_simulator.py
"""Generates more complex simulated A/B test data for a digital acquisition funnel."""

import pandas as pd
import numpy as np
import random
from datetime import timedelta
from config import (
    COUNTRIES, CHANNELS, DEVICES, DEVICE_DISTRIBUTION, USER_TYPES, USER_TYPE_DISTRIBUTION,
    REFERRERS, REFERRER_DISTRIBUTION, BASE_RATES, TREATMENT_EFFECT,
    BASE_CONTINUOUS, TREATMENT_EFFECT_CONTINUOUS, N_ROWS, CONTROL_GROUP_SIZE,
    NOISE_FACTOR, START_DATE, END_DATE, SEGMENT_ADJUSTMENTS, CPC_RANGES, DEFAULT_CPC_RANGE
)

def get_segment_factors(device, user_type):
    """Calculates combined adjustment factors based on user segments."""
    device_adj = SEGMENT_ADJUSTMENTS['device'].get(device, {'rate_factor': 1.0, 'effect_factor': 1.0})
    user_type_adj = SEGMENT_ADJUSTMENTS['user_type'].get(user_type, {'rate_factor': 1.0, 'effect_factor': 1.0})

    # Combine factors (multiplicatively)
    combined_rate_factor = device_adj['rate_factor'] * user_type_adj['rate_factor']
    combined_effect_factor = device_adj['effect_factor'] * user_type_adj['effect_factor']

    return combined_rate_factor, combined_effect_factor

def simulate_cpc(channel):
    """Simulates cost per click based on channel."""
    min_cpc, max_cpc = CPC_RANGES.get(channel, DEFAULT_CPC_RANGE)
    if min_cpc == 0 and max_cpc == 0:
        return 0.0
    # Add some randomness within the range
    return max(0, np.random.uniform(min_cpc, max_cpc) + np.random.normal(0, (max_cpc - min_cpc) * NOISE_FACTOR / 2))


def simulate_ab_data(n_rows=N_ROWS):
    """
    Simulates more complex and realistic A/B test data for an international
    digital acquisition funnel, including user segments and interactions.

    Args:
        n_rows (int): Number of user sessions (rows) to simulate.

    Returns:
        pandas.DataFrame: DataFrame containing the simulated A/B test data.
    """
    data = []
    total_days = (END_DATE - START_DATE).days

    for user_id in range(n_rows):
        # --- Assign User Characteristics ---
        country = random.choice(COUNTRIES)
        channel = random.choice(CHANNELS)
        device = np.random.choice(DEVICES, p=DEVICE_DISTRIBUTION)
        user_type = np.random.choice(USER_TYPES, p=USER_TYPE_DISTRIBUTION)
        referrer = np.random.choice(REFERRERS, p=REFERRER_DISTRIBUTION)

        # Simulate timestamp
        random_days = random.randint(0, total_days)
        random_seconds = random.randint(0, 24*60*60 - 1)
        timestamp = START_DATE + timedelta(days=random_days, seconds=random_seconds)

        # --- Assign Treatment Group (with imbalance) ---
        group = 'Control' if np.random.rand() < CONTROL_GROUP_SIZE else 'Treatment'

        # --- Get Segment Adjustments ---
        rate_factor, effect_factor = get_segment_factors(device, user_type)

        # --- Simulate Funnel Conversion ---
        view = 1 # Every simulated session starts with a view

        # Click probability (apply base, segment, noise, treatment)
        noise = np.random.normal(0, NOISE_FACTOR / 5) # Smaller noise for rates
        base_click_prob_adj = BASE_RATES['click_from_view'] * rate_factor * (1 + noise)
        treatment_eff_adj = TREATMENT_EFFECT['click_from_view'] * effect_factor if group == 'Treatment' else 0
        click_prob = base_click_prob_adj + treatment_eff_adj
        click_prob = max(0, min(1, click_prob)) # Ensure probability is valid
        clicked = 1 if np.random.rand() < click_prob else 0

        # Application Started probability
        app_started = 0
        if clicked:
            noise = np.random.normal(0, NOISE_FACTOR / 4)
            base_app_start_prob_adj = BASE_RATES['app_start_from_click'] * rate_factor * (1 + noise)
            treatment_eff_adj = TREATMENT_EFFECT['app_start_from_click'] * effect_factor if group == 'Treatment' else 0
            app_start_prob = base_app_start_prob_adj + treatment_eff_adj
            app_start_prob = max(0, min(1, app_start_prob))
            app_started = 1 if np.random.rand() < app_start_prob else 0

        # Application Completed probability
        app_completed = 0
        if app_started:
            noise = np.random.normal(0, NOISE_FACTOR / 3)
            base_app_complete_prob_adj = BASE_RATES['app_complete_from_start'] * rate_factor * (1 + noise)
            treatment_eff_adj = TREATMENT_EFFECT['app_complete_from_start'] * effect_factor if group == 'Treatment' else 0
            app_complete_prob = base_app_complete_prob_adj + treatment_eff_adj
            app_complete_prob = max(0, min(1, app_complete_prob))
            app_completed = 1 if np.random.rand() < app_complete_prob else 0

        # --- Simulate Continuous Metrics & Cost ---
        time_on_page = 0
        scroll_depth = 0
        bounced = 1 # Default to bounced
        session_cost = 0

        if clicked:
            # --- Time on Page ---
            base_mean = BASE_CONTINUOUS['time_on_page_mean']
            std_dev = BASE_CONTINUOUS['time_on_page_std']
            treatment_factor = TREATMENT_EFFECT_CONTINUOUS['time_on_page_factor'] if group == 'Treatment' else 1.0
            # Apply segment effect factor to the *treatment* effect size (making factor relative to 1)
            treatment_factor_adj = 1 + (treatment_factor - 1) * effect_factor
            segment_mean = base_mean * treatment_factor_adj

            # Simulate time (e.g., using log-normal could be more realistic, using normal here)
            time_on_page = max(5, np.random.normal(segment_mean, std_dev) + np.random.normal(0, segment_mean * NOISE_FACTOR)) # Add noise relative to mean, minimum 5 sec

            # --- Scroll Depth ---
            base_scroll_mean = BASE_CONTINUOUS['scroll_depth_mean']
            scroll_std = BASE_CONTINUOUS['scroll_depth_std']
            scroll_treatment_factor = TREATMENT_EFFECT_CONTINUOUS['scroll_depth_factor'] if group == 'Treatment' else 1.0
            scroll_treatment_factor_adj = 1 + (scroll_treatment_factor - 1) * effect_factor
            segment_scroll_mean = base_scroll_mean * scroll_treatment_factor_adj

            scroll_depth = np.random.normal(segment_scroll_mean, scroll_std) + np.random.normal(0, segment_scroll_mean * NOISE_FACTOR / 2)
            scroll_depth = max(0, min(100, scroll_depth)) # Clamp between 0 and 100

            # --- Bounce ---
            base_bounce_prob = BASE_CONTINUOUS['bounce_rate']
            bounce_treatment_factor = TREATMENT_EFFECT_CONTINUOUS['bounce_rate_factor'] if group == 'Treatment' else 1.0
            bounce_treatment_factor_adj = 1 + (bounce_treatment_factor - 1) * effect_factor # Lower factor is better
            segment_bounce_prob = base_bounce_prob * bounce_treatment_factor_adj

            bounce_prob = max(0, min(1, segment_bounce_prob + np.random.normal(0, NOISE_FACTOR / 2)))
            bounced = 1 if np.random.rand() < bounce_prob else 0

            # Override bounce if they progressed in the funnel
            if app_started or app_completed:
                bounced = 0

            # Optional: If bounced, reduce time/scroll (could make bounce more meaningful)
            # if bounced:
            #     time_on_page = max(5, time_on_page * random.uniform(0.1, 0.4))
            #     scroll_depth = max(0, scroll_depth * random.uniform(0.1, 0.4))

            # --- Session Cost ---
            session_cost = simulate_cpc(channel) # Cost only incurred if clicked (CPC model)

        else:
            # If no click (immediate bounce)
            bounced = 1
            time_on_page = np.random.uniform(1, 10) # Very short time
            scroll_depth = np.random.uniform(0, 15) # Very little scroll
            session_cost = 0 # No click, no cost

        data.append({
            'user_id': user_id,
            'timestamp': timestamp,
            'country': country,
            'channel': channel,
            'referrer': referrer,
            'device': device,
            'user_type': user_type,
            'group': group,
            'views': view,
            'clicks': clicked,
            'applications_started': app_started,
            'applications_completed': app_completed,
            'time_on_page_seconds': time_on_page,
            'scroll_depth_percentage': scroll_depth,
            'bounced': bounced,
            'session_cost': round(session_cost, 4) # Round cost to 4 decimal places
        })

    df = pd.DataFrame(data)
    # Ensure correct dtypes
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['session_cost'] = df['session_cost'].astype(float)
    df['time_on_page_seconds'] = df['time_on_page_seconds'].astype(float)
    df['scroll_depth_percentage'] = df['scroll_depth_percentage'].astype(float)


    return df

# Example usage:
if __name__ == "__main__":
    print(f"Starting simulation for {N_ROWS} rows...")
    simulated_data = simulate_ab_data(N_ROWS)
    print(simulated_data.head())
    print(f"\nSimulated {len(simulated_data)} rows.")

    print("\n--- Basic Info ---")
    print(simulated_data.info())

    print("\n--- Group Distribution ---")
    print(simulated_data['group'].value_counts(normalize=True))

    print("\n--- Sample Metrics (Averages) ---")
    print(simulated_data.agg(
        avg_clicks=('clicks', 'mean'),
        avg_app_started=('applications_started', 'mean'),
        avg_app_completed=('applications_completed', 'mean'),
        avg_time_on_page=('time_on_page_seconds', 'mean'),
        avg_scroll_depth=('scroll_depth_percentage', 'mean'),
        avg_bounce_rate=('bounced', 'mean'),
        avg_session_cost=('session_cost', 'mean'),
        total_cost=('session_cost', 'sum')
    ))

    print("\n--- Sample Metrics by Group (Averages) ---")
    print(simulated_data.groupby('group').agg(
        sessions=('user_id', 'count'),
        avg_clicks=('clicks', 'mean'),
        avg_app_started=('applications_started', 'mean'),
        avg_app_completed=('applications_completed', 'mean'),
        avg_time_on_page=('time_on_page_seconds', 'mean'),
        avg_scroll_depth=('scroll_depth_percentage', 'mean'),
        avg_bounce_rate=('bounced', 'mean'),
        avg_session_cost=('session_cost', 'mean'),
        total_cost=('session_cost', 'sum')
    ).round(4))


    # Save to CSV for dashboard use
    output_filename = "simulated_ab_data_complex.csv"
    simulated_data.to_csv(output_filename, index=False)
    print(f"\nData saved to {output_filename}")