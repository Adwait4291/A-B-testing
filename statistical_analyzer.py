# statistical_analyzer.py
"""Performs statistical analysis for A/B testing."""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_ztest
from statsmodels.stats.weightstats import ttest_ind

class ABAnalyzer:
    """
    Analyzes A/B test data using appropriate statistical methods.
    """
    def __init__(self, df, group_col='group', control_val='Control', treatment_val='Treatment'):
        """
        Initializes the analyzer.

        Args:
            df (pd.DataFrame): The A/B test data.
            group_col (str): Name of the column indicating the group (Control/Treatment).
            control_val (str): Value representing the Control group.
            treatment_val (str): Value representing the Treatment group.
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None.")
        if group_col not in df.columns:
             raise ValueError(f"Group column '{group_col}' not found in DataFrame.")

        self.df = df.copy()
        self.group_col = group_col
        self.control_val = control_val
        self.treatment_val = treatment_val
        self.control_df = self.df[self.df[self.group_col] == self.control_val]
        self.treatment_df = self.df[self.df[self.group_col] == self.treatment_val]

        if self.control_df.empty or self.treatment_df.empty:
            print(f"Warning: Control or Treatment group is empty.")
            # raise ValueError("Control or Treatment group is empty after filtering.")


    def _calculate_pooled_se(self, mean1, std1, n1, mean2, std2, n2):
        """Helper for calculating pooled standard error for t-test CIs."""
        return np.sqrt(std1**2 / n1 + std2**2 / n2)

    def analyze_conversion_rate(self, numerator_col, denominator_col, alpha=0.05):
        """
        Analyzes conversion rates using Z-test for proportions.

        Args:
            numerator_col (str): Column with the count of successes (e.g., 'clicks').
            denominator_col (str): Column defining the total population for the rate (e.g., 'views').
                                   If None, uses the total number of users in each group.
            alpha (float): Significance level (default 0.05).

        Returns:
            dict: Dictionary containing analysis results (p-value, lift, CIs, etc.).
                  Returns None if data is insufficient.
        """
        if self.control_df.empty or self.treatment_df.empty: return None
        if numerator_col not in self.df.columns:
            print(f"Warning: Numerator column '{numerator_col}' not found.")
            return None
        if denominator_col and denominator_col not in self.df.columns:
             print(f"Warning: Denominator column '{denominator_col}' not found.")
             return None

        if denominator_col:
             # Assumes 1 event per row for the denominator (e.g., 1 view per row)
             # More robust: Aggregate first if needed. For this sim, it's fine.
             control_successes = self.control_df[numerator_col].sum()
             control_total = self.control_df[denominator_col].sum()
             treatment_successes = self.treatment_df[numerator_col].sum()
             treatment_total = self.treatment_df[denominator_col].sum()
        else: # User-level conversion (e.g. % users who completed app)
             control_successes = self.control_df[numerator_col].sum() # Assumes 1/0 encoding
             control_total = len(self.control_df)
             treatment_successes = self.treatment_df[numerator_col].sum() # Assumes 1/0 encoding
             treatment_total = len(self.treatment_df)

        if control_total == 0 or treatment_total == 0:
            print(f"Warning: Zero denominator for metric '{numerator_col}' / '{denominator_col}'.")
            return None

        count = np.array([treatment_successes, control_successes])
        nobs = np.array([treatment_total, control_total])

        # Handle cases where success count exceeds total (shouldn't happen with good data)
        count = np.minimum(count, nobs)

        if np.any(nobs < 1): # Check if any group has no observations
             print(f"Warning: Insufficient observations for metric '{numerator_col}'.")
             return None

        # Check if success rates are identical and edge cases (0% or 100% for both)
        rate_c = control_successes / control_total if control_total > 0 else 0
        rate_t = treatment_successes / treatment_total if treatment_total > 0 else 0

        if rate_c == rate_t:
             p_value = 1.0
             z_stat = 0.0 # Or handle appropriately
        elif (control_successes == 0 and treatment_successes == 0) or \
             (control_successes == control_total and treatment_successes == treatment_total):
              p_value = 1.0 # No difference if both are 0% or 100%
              z_stat = 0.0
        else:
            try:
                z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
            except Exception as e:
                print(f"Error during Z-test for {numerator_col}: {e}")
                return None # Or handle more gracefully

        # Calculate Confidence Intervals for the difference in proportions
        # Note: confint_proportions_ztest gives CI for *each* proportion, not the difference.
        # We need to calculate the CI for the difference manually or use another method.
        # Using the standard formula: diff +/- z * SE_diff
        diff = rate_t - rate_c
        pooled_prop = (control_successes + treatment_successes) / (control_total + treatment_total)
        # Handle edge case where pooled_prop is 0 or 1
        if pooled_prop == 0 or pooled_prop == 1:
            se_diff = 0
        else:
            se_diff = np.sqrt(pooled_prop * (1 - pooled_prop) * (1 / control_total + 1 / treatment_total))

        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_lower = diff - z_crit * se_diff
        ci_upper = diff + z_crit * se_diff

        # Lift calculation
        lift = diff / rate_c if rate_c != 0 else np.inf

        # Significance check
        significant = (p_value < alpha) and (ci_lower * ci_upper > 0) # Check if CI excludes zero

        return {
            'metric': f"{numerator_col}{' / ' + denominator_col if denominator_col else ' Rate'}",
            'control_rate': rate_c,
            'treatment_rate': rate_t,
            'absolute_diff': diff,
            'lift (%)': lift * 100 if rate_c != 0 else 'N/A',
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': significant,
            'alpha': alpha,
            'control_n': control_total,
            'treatment_n': treatment_total,
            'control_conv': control_successes,
            'treatment_conv': treatment_successes,
            'test_type': 'Z-test (Proportions)'
        }

    def analyze_continuous_metric(self, metric_col, alpha=0.05, use_welch=True):
        """
        Analyzes continuous metrics using T-test.

        Args:
            metric_col (str): Column with the continuous metric (e.g., 'time_on_page_seconds').
            alpha (float): Significance level (default 0.05).
            use_welch (bool): Whether to use Welch's T-test (handles unequal variance, default True).

        Returns:
            dict: Dictionary containing analysis results (p-value, lift, CIs, etc.).
                  Returns None if data is insufficient.
        """
        if self.control_df.empty or self.treatment_df.empty: return None
        if metric_col not in self.df.columns:
             print(f"Warning: Metric column '{metric_col}' not found.")
             return None

        control_data = self.control_df[metric_col].dropna()
        treatment_data = self.treatment_df[metric_col].dropna()

        if len(control_data) < 2 or len(treatment_data) < 2:
            print(f"Warning: Insufficient data points for continuous metric '{metric_col}' after dropna.")
            return None

        mean_c = control_data.mean()
        mean_t = treatment_data.mean()
        std_c = control_data.std()
        std_t = treatment_data.std()
        n_c = len(control_data)
        n_t = len(treatment_data)

        # Perform T-test
        # statsmodels ttest_ind provides CI for difference directly
        try:
             t_stat, p_value, dof = ttest_ind(
                treatment_data,
                control_data,
                alternative='two-sided',
                usevar='pooled' if not use_welch else 'unequal' # 'unequal' corresponds to Welch's
             )
        except Exception as e:
            print(f"Error during T-test for {metric_col}: {e}")
            return None

        # Calculate Confidence Interval for the difference in means
        diff = mean_t - mean_c
        # Use statsmodels results if possible, otherwise calculate manually
        # Using Welch-Satterthwaite equation for degrees of freedom if use_welch=True
        # The `dof` returned by statsmodels ttest_ind is what we need
        se_diff = self._calculate_pooled_se(mean_t, std_t, n_t, mean_c, std_c, n_c) if use_welch else np.sqrt((((n_t-1)*std_t**2 + (n_c-1)*std_c**2)/(n_t+n_c-2)) * (1/n_t + 1/n_c)) # Pooled SE for Student's t

        t_crit = stats.t.ppf(1 - alpha / 2, df=dof)
        ci_lower = diff - t_crit * se_diff
        ci_upper = diff + t_crit * se_diff

        # Lift calculation
        lift = diff / mean_c if mean_c != 0 else np.inf

        # Significance check
        significant = (p_value < alpha) and (ci_lower * ci_upper > 0)

        return {
            'metric': metric_col,
            'control_mean': mean_c,
            'treatment_mean': mean_t,
            'absolute_diff': diff,
            'lift (%)': lift * 100 if mean_c != 0 else 'N/A',
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': significant,
            'alpha': alpha,
            'control_n': n_c,
            'treatment_n': n_t,
            'control_std': std_c,
            'treatment_std': std_t,
            'test_type': "Welch's T-test" if use_welch else "Student's T-test"
        }

    def run_analysis(self, conversion_metrics=None, continuous_metrics=None, alpha=0.05):
        """
        Runs analysis for specified conversion and continuous metrics.

        Args:
            conversion_metrics (list of tuples): List of (numerator, denominator) for conversion rates.
                                                  Use None for denominator for user-level rates.
                                                  e.g., [('clicks', 'views'), ('applications_completed', None)]
            continuous_metrics (list): List of column names for continuous metrics.
                                       e.g., ['time_on_page_seconds']
            alpha (float): Significance level.

        Returns:
            dict: A dictionary where keys are metric names and values are the result dictionaries.
        """
        results = {}
        if conversion_metrics:
            for num, den in conversion_metrics:
                res = self.analyze_conversion_rate(num, den, alpha)
                if res:
                    results[res['metric']] = res

        if continuous_metrics:
             # Special handling for bounce rate (it's binary 0/1 but often analyzed like conversion)
             if 'bounced' in continuous_metrics:
                 continuous_metrics.remove('bounced')
                 res = self.analyze_conversion_rate('bounced', None, alpha) # Treat as user-level conversion
                 if res:
                     # Rename metric for clarity
                     res['metric'] = 'Bounce Rate'
                     # Invert lift interpretation (lower bounce is better)
                     # lift = res.get('lift (%)', 0)
                     # if isinstance(lift, (int, float)):
                     #     res['lift (%)'] = -lift # Optional: Report lift direction consistently (improvement)
                     results['Bounce Rate'] = res


             for metric in continuous_metrics:
                res = self.analyze_continuous_metric(metric, alpha)
                if res:
                    results[res['metric']] = res

        return results

# Example usage:
if __name__ == "__main__":
    try:
        df = pd.read_csv("simulated_ab_data.csv")
        analyzer = ABAnalyzer(df, group_col='group', control_val='Control', treatment_val='Treatment')

        analysis_results = analyzer.run_analysis(
            conversion_metrics=[
                ('clicks', 'views'),                        # Click-Through Rate (CTR)
                ('applications_started', 'clicks'),       # App Start Rate (from Clicks)
                ('applications_completed', 'applications_started'), # App Completion Rate (from Starts)
                ('applications_completed', 'views')         # Overall Conversion Rate (from Views)
            ],
            continuous_metrics=['time_on_page_seconds', 'bounced'] # Analyze bounce rate as conversion
        )

        print("\nAnalysis Results:")
        import json
        print(json.dumps(analysis_results, indent=4, default=str)) # Use default=str for potential NaNs/Infs

    except FileNotFoundError:
        print("Error: simulated_ab_data.csv not found. Run data_simulator.py first.")
    except Exception as e:
        print(f"An error occurred: {e}")