# statistical_analyzer.py
"""
Performs simplified statistical analysis for A/B testing.
Focuses on Z-test for proportions and T-test for means.
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.weightstats import ttest_ind
import traceback


class SimpleABAnalyzer:
    """
    Analyzes A/B test data using Z-test for proportions and T-test for means.
    """
    def __init__(self, df, group_col, control_val, treatment_val):
        """
        Initializes the analyzer.
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None.")
        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found.")

        self.df = df.copy()
        self.group_col = group_col
        self.control_val = str(control_val)
        self.treatment_val = str(treatment_val)

        # Ensure group column is string type for filtering
        try:
            self.df[self.group_col] = self.df[self.group_col].astype(str)
        except Exception as e:
            raise ValueError(f"Could not convert group column '{group_col}' to string: {e}")

        self.control_df = self.df[self.df[self.group_col] == self.control_val]
        self.treatment_df = self.df[self.df[self.group_col] == self.treatment_val]

        if self.control_df.empty or self.treatment_df.empty:
            print(f"Warning: Data missing for one or both groups. Control: {len(self.control_df)}, Treatment: {len(self.treatment_df)}")


    def analyze_conversion_rate(self, conversion_col, alpha=0.05):
        """
        Analyzes conversion rates using Z-test. Assumes binary (0/1 or True/False).
        """
        metric_name = f"{conversion_col} Rate"
        if self.control_df.empty or self.treatment_df.empty or conversion_col not in self.df.columns:
            print(f"Skipping analysis for {metric_name} due to missing data or column.")
            return None

        # Convert boolean to int if needed
        if pd.api.types.is_bool_dtype(self.df[conversion_col]):
            self.df[conversion_col] = self.df[conversion_col].astype(int)
            # Re-filter after conversion
            self.control_df = self.df[self.df[self.group_col] == self.control_val]
            self.treatment_df = self.df[self.df[self.group_col] == self.treatment_val]


        if not pd.api.types.is_numeric_dtype(self.df[conversion_col]) or not self.df[conversion_col].dropna().isin([0, 1]).all():
             print(f"Warning: Column '{conversion_col}' is not binary (0/1). Cannot calculate rate.")
             return None

        control_data = self.control_df[conversion_col].dropna()
        treatment_data = self.treatment_df[conversion_col].dropna()

        control_successes = control_data.sum()
        control_total = len(control_data)
        treatment_successes = treatment_data.sum()
        treatment_total = len(treatment_data)

        if control_total == 0 or treatment_total == 0:
            print(f"Skipping {metric_name}: Zero samples in one group.")
            return None

        rate_c = control_successes / control_total
        rate_t = treatment_successes / treatment_total

        # Perform Z-test
        count = np.array([treatment_successes, control_successes])
        nobs = np.array([treatment_total, control_total])
        z_stat, p_value = np.nan, np.nan

        try:
            # Avoid division by zero or issues if rates are identical edge cases
            if nobs.sum() > 0 and count.sum() > 0 and count.sum() < nobs.sum() :
                 z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
            elif rate_c == rate_t:
                 z_stat, p_value = 0.0, 1.0
            else: # Handle cases where one rate is 0 or 1, and the other isn't - Z-test might still work
                 z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')

        except Exception as e:
            print(f"Error during Z-test for {metric_name}: {e}")
            # Fallback P-value to 1.0 if test fails but rates differ
            p_value = 1.0 if rate_c != rate_t else p_value


        lift_pct = ((rate_t - rate_c) / rate_c) * 100 if rate_c != 0 else np.inf if rate_t > 0 else 0.0
        significant = bool(pd.notna(p_value) and (p_value < alpha))

        return {
            'metric': metric_name,
            'control_rate': rate_c,
            'treatment_rate': rate_t,
            'lift (%)': lift_pct,
            'p_value': p_value,
            'significant': significant,
            'control_n': int(control_total),
            'treatment_n': int(treatment_total),
        }

    def analyze_continuous_metric(self, metric_col, alpha=0.05, use_welch=True):
        """
        Analyzes continuous metrics using T-test (Welch's default).
        """
        if self.control_df.empty or self.treatment_df.empty or metric_col not in self.df.columns:
            print(f"Skipping analysis for {metric_col} due to missing data or column.")
            return None
        if not pd.api.types.is_numeric_dtype(self.df[metric_col]):
            print(f"Skipping non-numeric metric: {metric_col}")
            return None

        control_data = self.control_df[metric_col].dropna()
        treatment_data = self.treatment_df[metric_col].dropna()
        n_c, n_t = len(control_data), len(treatment_data)

        if n_c < 2 or n_t < 2: # Need at least 2 points for variance/ttest
            print(f"Skipping {metric_col}: Insufficient data (N < 2) after dropping NAs. Control: {n_c}, Treatment: {n_t}")
            return None

        mean_c, mean_t = control_data.mean(), treatment_data.mean()

        # Perform T-test
        t_stat, p_value = np.nan, np.nan
        variance_type = 'unequal' if use_welch else 'pooled'
        try:
             # Check if variances are zero before calling ttest_ind
             if control_data.var() == 0 and treatment_data.var() == 0 and mean_c == mean_t:
                  t_stat, p_value = 0.0, 1.0
             elif control_data.var() == 0 and treatment_data.var() == 0 and mean_c != mean_t:
                   t_stat, p_value = np.inf, 0.0 # Or handle as needed
             else:
                  t_stat, p_value, _ = ttest_ind(treatment_data, control_data,
                                                alternative='two-sided', usevar=variance_type)
        except Exception as e:
             print(f"Error during T-test for {metric_col}: {e}")
             p_value = 1.0 # Fallback

        lift_pct = ((mean_t - mean_c) / mean_c) * 100 if mean_c != 0 else np.inf if mean_t > 0 else 0.0
        significant = bool(pd.notna(p_value) and (p_value < alpha))

        return {
            'metric': metric_col,
            'control_mean': mean_c,
            'treatment_mean': mean_t,
            'lift (%)': lift_pct,
            'p_value': p_value,
            'significant': significant,
            'control_n': int(n_c),
            'treatment_n': int(n_t),
        }

    def run_analysis(self, conversion_metric=None, continuous_metrics=None, alpha=0.05):
        """
        Runs analysis for the specified conversion and continuous metrics.
        """
        results = {}
        if conversion_metric:
            res = self.analyze_conversion_rate(conversion_metric, alpha)
            if res:
                results[res['metric']] = res

        if continuous_metrics:
            for metric_col in continuous_metrics:
                res = self.analyze_continuous_metric(metric_col, alpha)
                if res:
                    results[res['metric']] = res
        return results