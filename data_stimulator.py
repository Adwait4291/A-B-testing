
# data_simulator.py
"""Generates simulated A/B test data for a digital acquisition funnel."""

import pandas as pd
import numpy as np
from config import (COUNTRIES, CHANNELS, DEVICES, BASE_RATES, TREATMENT_EFFECT,
                    BASE_CONTINUOUS, TREATMENT_EFFECT_CONTINUOUS, N_ROWS,
                    CONTROL_GROUP_SIZE, NOISE_FACTOR)
import random

def simulate_ab_data(n_rows=N_ROWS):
    """
    Simulates realistic A/B test data for an international digital acquisition funnel.

    Args:
        n_rows (int): Number of user sessions (rows) to simulate.

    Returns:
        pandas.DataFrame: DataFrame containing the simulated A/B test data.
    """
    data = []
    for user_id in range(n_rows):
        # --- Assign User Characteristics ---
        country = random.choice(COUNTRIES)
        channel = random.choice(CHANNELS)
        device = np.random.choice(DEVICES, p=[0.6, 0.35, 0.05]) # Mobile heavy

        # --- Assign Treatment Group (with imbalance) ---
        group = 'Control' if np.random.rand() < CONTROL_GROUP_SIZE else 'Treatment'

        # --- Simulate Funnel Conversion ---
        view = 1 # Every simulated session starts with a view

        # Click probability (add noise and treatment effect)
        noise = np.random.normal(0, NOISE_FACTOR / 5) # Smaller noise for rates
        base_click_prob = BASE_RATES['click_from_view'] * (1 + noise)
        click_prob = base_click_prob + (TREATMENT_EFFECT['click_from_view'] if group == 'Treatment' else 0)
        click_prob = max(0, min(1, click_prob)) # Ensure probability is valid
        clicked = 1 if np.random.rand() < click_prob else 0

        # Application Started probability
        app_started = 0
        if clicked:
            noise = np.random.normal(0, NOISE_FACTOR / 4)
            base_app_start_prob = BASE_RATES['app_start_from_click'] * (1 + noise)
            app_start_prob = base_app_start_prob + (TREATMENT_EFFECT['app_start_from_click'] if group == 'Treatment' else 0)
            app_start_prob = max(0, min(1, app_start_prob))
            app_started = 1 if np.random.rand() < app_start_prob else 0

        # Application Completed probability
        app_completed = 0
        if app_started:
            noise = np.random.normal(0, NOISE_FACTOR / 3)
            base_app_complete_prob = BASE_RATES['app_complete_from_start'] * (1 + noise)
            app_complete_prob = base_app_complete_prob + (TREATMENT_EFFECT['app_complete_from_start'] if group == 'Treatment' else 0)
            app_complete_prob = max(0, min(1, app_complete_prob))
            app_completed = 1 if np.random.rand() < app_complete_prob else 0

        # --- Simulate Continuous Metrics ---
        time_on_page = 0
        bounced = 1 # Default to bounced if no click
        if clicked: # Only relevant if they didn't bounce immediately (assume click means non-bounce for this sim)
            base_mean = BASE_CONTINUOUS['time_on_page_mean']
            std_dev = BASE_CONTINUOUS['time_on_page_std']
            if group == 'Treatment':
                base_mean *= TREATMENT_EFFECT_CONTINUOUS['time_on_page_factor']

            # Simulate time on page (e.g., using a log-normal or gamma, using normal here for simplicity)
            time_on_page = max(5, np.random.normal(base_mean, std_dev) + np.random.normal(0, base_mean * NOISE_FACTOR)) # Add noise relative to mean, minimum 5 sec

            # Simulate Bounce (as a separate event for clicked users)
            base_bounce_prob = BASE_CONTINUOUS['bounce_rate']
            if group == 'Treatment':
                base_bounce_prob *= TREATMENT_EFFECT_CONTINUOUS['bounce_rate_factor']
            bounce_prob = max(0, min(1, base_bounce_prob + np.random.normal(0, NOISE_FACTOR / 2)))
            bounced = 1 if np.random.rand() < bounce_prob else 0
            # Override bounce if they progressed in the funnel
            if app_started or app_completed:
                 bounced = 0
            # If bounced, maybe set time on page lower? (Optional complexity)
            # if bounced: time_on_page = max(5, time_on_page * 0.3)
        else:
            time_on_page = np.random.uniform(1, 10) # Short time if no click (immediate bounce)


        data.append({
            'user_id': user_id,
            'country': country,
            'channel': channel,
            'device': device,
            'group': group,
            'views': view,
            'clicks': clicked,
            'applications_started': app_started,
            'applications_completed': app_completed,
            'time_on_page_seconds': time_on_page if clicked else 0, # Only count time if clicked
            'bounced': bounced
        })

    df = pd.DataFrame(data)

    # --- Calculate Derived Metrics (Optional Here, Can be done in Analysis) ---
    # df['ctr'] = df['clicks'] / df['views'] # Careful with aggregation level
    # df['start_rate'] = df['applications_started'] / df['clicks'].replace(0, np.nan) # Avoid div by zero
    # df['completion_rate'] = df['applications_completed'] / df['applications_started'].replace(0, np.nan)

    return df

# Example usage:
if __name__ == "__main__":
    simulated_data = simulate_ab_data(N_ROWS)
    print(simulated_data.head())
    print(f"\nSimulated {len(simulated_data)} rows.")
    print("\nGroup distribution:")
    print(simulated_data['group'].value_counts(normalize=True))
    # Save to CSV for dashboard use
    simulated_data.to_csv("simulated_ab_data.csv", index=False)
    print("\nData saved to simulated_ab_data.csv")