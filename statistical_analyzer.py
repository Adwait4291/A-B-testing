# statistical_analyzer.py
"""Performs statistical analysis for A/B testing."""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.weightstats import ttest_ind
import config # Import config to use its variables in the example block
import os
import json
import traceback


class ABAnalyzer:
    """
    Analyzes A/B test data using Z-test for proportions and T-test for means.
    """
    def __init__(self, df, group_col='group', control_val='Control', treatment_val='Treatment'):
        """
        Initializes the analyzer.

        Args:
            df (pd.DataFrame): The A/B test data. Must contain group_col.
            group_col (str): Name of the column indicating the group.
            control_val (str): Value in group_col representing the Control group.
            treatment_val (str): Value in group_col representing the Treatment group.

        Raises:
            ValueError: If df is None, empty, or group_col is not found, or control/treatment vals missing.
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None.")
        if group_col not in df.columns:
             raise ValueError(f"Group column '{group_col}' not found in DataFrame columns: {df.columns.tolist()}")

        self.df = df.copy() # Work on a copy
        self.group_col = group_col
        # Convert control/treatment identifiers to string for consistent comparison
        self.control_val = str(control_val)
        self.treatment_val = str(treatment_val)

        # Ensure group column is string type for filtering, handle potential errors
        try:
             self.df[self.group_col] = self.df[self.group_col].astype(str)
        except Exception as e:
             raise ValueError(f"Could not convert group column '{group_col}' to string type: {e}")

        # Filter dataframes for control and treatment groups
        self.control_df = self.df[self.df[self.group_col] == self.control_val]
        self.treatment_df = self.df[self.df[self.group_col] == self.treatment_val]

        print(f"Analyzer Initialized: Found {len(self.control_df)} in Control ('{self.control_val}') and {len(self.treatment_df)} in Treatment ('{self.treatment_val}').")
        if self.control_df.empty:
            print(f"Warning: No data found for the Control group ('{self.control_val}'). Analysis might fail.")
            # raise ValueError(f"No data found for Control group ('{self.control_val}')") # Option: raise error
        if self.treatment_df.empty:
            print(f"Warning: No data found for the Treatment group ('{self.treatment_val}'). Analysis might fail.")
            # raise ValueError(f"No data found for Treatment group ('{self.treatment_val}')") # Option: raise error


    def analyze_conversion_rate(self, conversion_col, alpha=0.05):
        """
        Analyzes conversion rates using Z-test for proportions.
        Assumes conversion_col contains boolean or 0/1 numeric data.

        Args:
            conversion_col (str): Column containing user-level binary flags
                                  (True/False or 1/0 for conversion).
            alpha (float): Significance level (default 0.05).

        Returns:
            dict or None: Analysis results or None if analysis fails.
        """
        metric_name = f"{conversion_col} Rate"
        # --- Basic Checks ---
        if self.control_df.empty or self.treatment_df.empty:
            print(f"Warning [analyze_conversion_rate]: Control or Treatment group empty for '{metric_name}'.")
            return None
        if conversion_col not in self.df.columns:
            print(f"Warning [analyze_conversion_rate]: Conversion column '{conversion_col}' not found.")
            return None

        # Convert boolean column to 0/1 if necessary
        if pd.api.types.is_bool_dtype(self.df[conversion_col]):
            control_conv_data = self.control_df[conversion_col].astype(int)
            treatment_conv_data = self.treatment_df[conversion_col].astype(int)
        elif pd.api.types.is_numeric_dtype(self.df[conversion_col]):
            # Check if it's binary 0/1
            unique_vals = self.df[conversion_col].dropna().unique()
            if not np.all(np.isin(unique_vals, [0, 1])):
                print(f"Warning [analyze_conversion_rate]: Numeric column '{conversion_col}' is not binary (0/1). Cannot calculate rate directly.")
                return None
            control_conv_data = self.control_df[conversion_col]
            treatment_conv_data = self.treatment_df[conversion_col]
        else:
             print(f"Warning [analyze_conversion_rate]: Column '{conversion_col}' is not boolean or binary numeric.")
             return None

        # --- Calculate Successes and Totals ---
        try:
            control_successes = control_conv_data.sum()
            control_total = len(control_conv_data) # Total users in control group
            treatment_successes = treatment_conv_data.sum()
            treatment_total = len(treatment_conv_data) # Total users in treatment group

        except Exception as e:
             print(f"Error during conversion data aggregation for '{metric_name}': {e}")
             traceback.print_exc()
             return None

        # --- Check Denominators ---
        if control_total <= 0 or treatment_total <= 0:
            print(f"Warning [analyze_conversion_rate]: Zero denominator for '{metric_name}' (C: {control_total}, T: {treatment_total}).")
            return None

        # --- Calculate Rates ---
        rate_c = control_successes / control_total
        rate_t = treatment_successes / treatment_total

        # --- Perform Z-test ---
        count = np.array([treatment_successes, control_successes]) # Order: Treatment, Control
        nobs = np.array([treatment_total, control_total])

        z_stat, p_value = np.nan, np.nan # Initialize
        if rate_c == rate_t or (rate_c in [0, 1] and rate_t in [0, 1] and rate_c == rate_t):
             p_value = 1.0
             z_stat = 0.0
        else:
            try:
                z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
            except Exception as e:
                print(f"Error during Z-test calculation for '{metric_name}': {e}")
                traceback.print_exc()
                return None # Cannot proceed without test result

        # --- Calculate CI for Difference (Treatment Rate - Control Rate) ---
        # Uses statsmodels `proportion_confint` for difference CI is complex, calculate manually
        diff = rate_t - rate_c
        pooled_prop = (control_successes + treatment_successes) / (control_total + treatment_total)

        se_diff = 0.0
        if pooled_prop > 0 and pooled_prop < 1 and control_total > 0 and treatment_total > 0 :
            se_diff = np.sqrt(pooled_prop * (1 - pooled_prop) * (1 / control_total + 1 / treatment_total))

        ci_lower, ci_upper = diff, diff # Default if SE is zero
        if se_diff > 0:
            z_crit = stats.norm.ppf(1 - alpha / 2)
            ci_lower = diff - z_crit * se_diff
            ci_upper = diff + z_crit * se_diff

        # --- Calculate Relative Lift (%) ---
        lift_pct = np.nan
        if rate_c == 0:
            if diff > 0: lift_pct = np.inf
            elif diff == 0: lift_pct = 0.0
        elif rate_c > 0:
            lift_pct = (diff / rate_c) * 100

        # --- Significance Check ---
        significant = bool(pd.notna(p_value) and (p_value < alpha))

        # --- Return Results ---
        return {
            'metric': metric_name,
            'test_type': 'Z-test (Proportions)',
            'control_group': self.control_val,
            'treatment_group': self.treatment_val,
            'control_n': int(control_total),
            'treatment_n': int(treatment_total),
            'control_conv': int(control_successes),
            'treatment_conv': int(treatment_successes),
            'control_rate': rate_c,
            'treatment_rate': rate_t,
            'absolute_diff': diff,
            'lift (%)': lift_pct,
            'p_value': p_value,
            'z_stat': z_stat,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'alpha': alpha,
            'significant': significant,
            'error': None
        }


    def analyze_continuous_metric(self, metric_col, alpha=0.05, use_welch=True):
        """
        Analyzes continuous metrics using T-test (Welch's T-test by default).

        Args:
            metric_col (str): Name of the column containing the continuous metric.
            alpha (float): Significance level (default 0.05).
            use_welch (bool): If True (default), performs Welch's T-test (unequal variances).
                              If False, performs Student's T-test (assumes equal variances).

        Returns:
            dict or None: Analysis results or None if analysis fails.
        """
        # --- Basic Checks ---
        if self.control_df.empty or self.treatment_df.empty:
            print(f"Warning [analyze_continuous_metric]: Control or Treatment group empty for '{metric_col}'.")
            return None
        if metric_col not in self.df.columns:
             print(f"Warning [analyze_continuous_metric]: Metric column '{metric_col}' not found.")
             return None
        if not pd.api.types.is_numeric_dtype(self.df[metric_col]):
             print(f"Warning [analyze_continuous_metric]: Metric column '{metric_col}' is not numeric.")
             return None

        # --- Prepare Data (Drop NaNs) ---
        control_data = self.control_df[metric_col].dropna()
        treatment_data = self.treatment_df[metric_col].dropna()
        n_c, n_t = len(control_data), len(treatment_data)

        # Check for sufficient data points (need >= 2 for variance calculation)
        if n_c < 2 or n_t < 2:
            print(f"Warning [analyze_continuous_metric]: Insufficient non-NaN data for '{metric_col}' (C: {n_c}, T: {n_t}). Need >= 2 per group.")
            return None

        # --- Calculate Basic Stats ---
        mean_c, mean_t = control_data.mean(), treatment_data.mean()
        std_c, std_t = control_data.std(ddof=1), treatment_data.std(ddof=1) # Sample std dev

        # --- Perform T-test ---
        t_stat, p_value, dof = np.nan, np.nan, np.nan # Initialize
        test_name = "Welch's T-test" if use_welch else "Student's T-test"
        try:
            variance_type = 'unequal' if use_welch else 'pooled'
            t_stat, p_value, dof = ttest_ind(treatment_data, control_data,
                                             alternative='two-sided', usevar=variance_type)
        except Exception as e:
            print(f"Error during T-test calculation for '{metric_col}': {e}")
            traceback.print_exc()
            return None # Cannot proceed

        # --- Calculate CI for Difference in Means (Treatment Mean - Control Mean) ---
        diff = mean_t - mean_c
        se_diff = np.nan
        if use_welch:
            # SE for Welch's T-test (handle potential zero std dev or N)
            if n_c > 0 and n_t > 0:
                se_diff = np.sqrt(max(0, std_t**2) / n_t + max(0, std_c**2) / n_c) # Use max(0, var)
        else:
            # SE for Student's T-test (pooled variance)
            if n_c > 1 and n_t > 1 : # Ensure DoF > 0
                pooled_var = ((n_t - 1) * max(0, std_t**2) + (n_c - 1) * max(0, std_c**2)) / (n_t + n_c - 2)
                se_diff = np.sqrt(pooled_var * (1 / n_t + 1 / n_c))

        ci_lower, ci_upper = diff, diff # Default if SE is zero or invalid
        if pd.notna(se_diff) and se_diff > 0 and pd.notna(dof) and dof > 0:
            try:
                t_crit = stats.t.ppf(1 - alpha / 2, df=dof)
                ci_lower = diff - t_crit * se_diff
                ci_upper = diff + t_crit * se_diff
            except Exception as e:
                 print(f"Error calculating CI bounds for '{metric_col}': {e}")
                 ci_lower, ci_upper = np.nan, np.nan

        # --- Calculate Relative Lift (%) ---
        lift_pct = np.nan
        if mean_c == 0:
            if diff > 0: lift_pct = np.inf
            elif diff == 0: lift_pct = 0.0
        elif mean_c != 0: # Avoid division by zero
            lift_pct = (diff / mean_c) * 100

        # --- Significance Check ---
        significant = bool(pd.notna(p_value) and (p_value < alpha))

        # --- Return Results ---
        return {
            'metric': metric_col, # Use original column name as metric identifier
            'test_type': test_name,
            'control_group': self.control_val,
            'treatment_group': self.treatment_val,
            'control_n': n_c,
            'treatment_n': n_t,
            'control_mean': mean_c,
            'treatment_mean': mean_t,
            'control_std': std_c,
            'treatment_std': std_t,
            'absolute_diff': diff,
            'lift (%)': lift_pct,
            'p_value': p_value,
            't_stat': t_stat,
            'dof': dof,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'alpha': alpha,
            'significant': significant,
            'error': None
        }


    def run_analysis(self, conversion_metric=None, continuous_metrics=None, alpha=0.05):
        """
        Runs analysis for the specified conversion and continuous metrics.

        Args:
            conversion_metric (str): Name of the single binary/boolean conversion column.
            continuous_metrics (list): List of continuous metric column names.
            alpha (float): Significance level.

        Returns:
            dict: Results keyed by metric name/type. Only includes successful analyses.
        """
        results = {}

        # --- Analyze Conversion Metric ---
        if conversion_metric:
            if not isinstance(conversion_metric, str):
                print("Warning [run_analysis]: conversion_metric should be a string column name.")
            else:
                print(f"\n--- Analyzing Conversion Metric: {conversion_metric} ---")
                res = self.analyze_conversion_rate(conversion_metric, alpha)
                if res:
                    results[res['metric']] = res # Use metric name from result dict as key
                else:
                    print(f"--- Analysis FAILED for conversion metric: {conversion_metric} ---")

        # --- Analyze Continuous Metrics ---
        if continuous_metrics:
            if not isinstance(continuous_metrics, list):
                print("Warning [run_analysis]: continuous_metrics should be a list.")
                continuous_metrics = [] # Treat as empty

            for metric_col in continuous_metrics:
                if isinstance(metric_col, str):
                    print(f"\n--- Analyzing Continuous Metric: {metric_col} ---")
                    res = self.analyze_continuous_metric(metric_col, alpha, use_welch=True)
                    if res:
                        results[res['metric']] = res # Use original metric column name as key
                    else:
                         print(f"--- Analysis FAILED for continuous metric: {metric_col} ---")
                else:
                    print(f"Warning [run_analysis]: Skipping invalid continuous metric format: {metric_col}. Expecting string.")
                    continue

        return results


# Example usage block - Updated for marketing_AB.csv
if __name__ == "__main__":
    print("\n" + "="*30)
    print("Running A/B Analyzer Example with Marketing Data")
    print("="*30)

    try:
        # --- Load Config ---
        if not all(hasattr(config, attr) for attr in ['INPUT_DATA_FILE', 'GROUP_COLUMN', 'CONTROL_GROUP_VALUE', 'TREATMENT_GROUP_VALUE', 'CONVERSION_COLUMN', 'CONTINUOUS_METRICS', 'ALPHA']):
             raise ValueError("Config error: One or more required variables missing in config.py")

        file_path = config.INPUT_DATA_FILE
        print(f"\nAttempting to load data from: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The specified data file was not found: {file_path}.")

        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # --- Validate Data against Config ---
        required_cols = [config.GROUP_COLUMN, config.CONVERSION_COLUMN] + config.CONTINUOUS_METRICS
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns specified in config from CSV: {missing_cols}")

        expected_groups = [str(config.CONTROL_GROUP_VALUE), str(config.TREATMENT_GROUP_VALUE)]
        actual_groups = df[config.GROUP_COLUMN].astype(str).unique()
        if not all(group in actual_groups for group in expected_groups):
             print(f"Warning: Expected groups {expected_groups} not found in '{config.GROUP_COLUMN}'. Found: {actual_groups}. Check config.")

        print("\nInitializing ABAnalyzer...")
        analyzer = ABAnalyzer(df,
                              group_col=config.GROUP_COLUMN,
                              control_val=config.CONTROL_GROUP_VALUE,
                              treatment_val=config.TREATMENT_GROUP_VALUE)
        print("ABAnalyzer initialized.")

        # --- Define Metrics based on Config ---
        conversion_metric_to_run = config.CONVERSION_COLUMN
        continuous_metrics_to_run = config.CONTINUOUS_METRICS

        print(f"\nMetrics to analyze:")
        print(f"  Conversion: '{conversion_metric_to_run}'")
        print(f"  Continuous: {continuous_metrics_to_run if continuous_metrics_to_run else 'None'}")

        print("\nRunning analysis...")
        analysis_results = analyzer.run_analysis(
            conversion_metric=conversion_metric_to_run,
            continuous_metrics=continuous_metrics_to_run,
            alpha=config.ALPHA
        )
        print("\nAnalysis complete.")

        print("\n--- Analysis Results ---")
        if analysis_results:
             print(json.dumps(analysis_results, indent=4, default=str))
        else:
             print("No results generated. Check warnings above.")
        print("--- End of Results ---")

    # --- Error Handling ---
    except FileNotFoundError as e:
        print(f"\n--- ERROR: File Not Found ---")
        print(e)
    except KeyError as e:
         print(f"\n--- ERROR: Column Not Found (KeyError) ---")
         print(f"Column {e} was likely expected but not found in the DataFrame.")
    except ValueError as e:
         print(f"\n--- ERROR: Data/Config Problem (ValueError) ---")
         print(e)
    except ImportError as e:
        print(f"\n--- ERROR: Missing Library (ImportError) ---")
        print(e)
        print("Suggestion: Ensure all required libraries (pandas, numpy, scipy, statsmodels) are installed.")
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        traceback.print_exc()