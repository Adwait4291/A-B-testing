# dashboard.py
"""
Ultra-Simplified Streamlit dashboard for A/B test analysis.
Hardcoded parameters, no config file needed, no sidebar.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import traceback

# --- Import Custom Analyzer ---
# Make sure 'statistical_analyzer.py' is in the same folder
try:
    from statistical_analyzer import SimpleABAnalyzer
except ImportError as e:
    st.error(f"Error importing analyzer: {e}. Make sure 'statistical_analyzer.py' is in the same folder.")
    st.stop()

# --- Hardcoded Configuration ---
# Define your settings directly here
INPUT_DATA_FILE = 'finance_ab.csv'  # The data file name
GROUP_COLUMN = 'Version'            # Column identifying the groups
CONTROL_VALUE = 'A'                 # Value for Control group
TREATMENT_VALUE = 'B'               # Value for Treatment group
CONVERSION_METRIC = 'ApplicationCompleted' # Binary conversion column
CONTINUOUS_METRICS_TO_RUN = []      # List of continuous metrics to analyze (e.g., ['SessionDuration_seconds']) - Empty means none
ALPHA = 0.05                        # Significance level
DASHBOARD_TITLE = 'A/B Test Results'
# --- End Hardcoded Configuration ---


# --- Page Configuration ---
st.set_page_config(
    page_title=DASHBOARD_TITLE,
    page_icon="📊",
    layout="wide"
)

# --- Helper Functions ---
@st.cache_data # Cache data loading
def load_data(file_path):
    """Loads data from the specified file path."""
    if not os.path.exists(file_path):
        st.error(f"Data file not found: {file_path}")
        st.info(f"Please ensure '{file_path}' is in the same directory as this script.")
        return None
    try:
        df = pd.read_csv(file_path)
        st.success(f"Data loaded successfully from '{os.path.basename(file_path)}'!")
        return df
    except Exception as e:
        st.error(f"Error loading data from '{file_path}': {e}")
        return None

def display_simplified_metrics(results):
    """Displays key metrics using st.metric."""
    if not isinstance(results, dict) or not results:
        st.warning("No analysis results to display.")
        return

    st.subheader("Key Metric Performance")
    num_metrics = len(results)
    if num_metrics == 0:
        st.info("Analysis run, but no valid metric results generated.")
        return

    cols = st.columns(num_metrics)
    col_idx = 0

    for metric, data in results.items():
         if not isinstance(data, dict): continue

         with cols[col_idx]:
             is_rate = 'Rate' in metric
             control_val = data.get('control_rate') if is_rate else data.get('control_mean')
             treat_val = data.get('treatment_rate') if is_rate else data.get('treatment_mean')
             lift = data.get('lift (%)')
             pval = data.get('p_value')
             significant = data.get('significant', False)

             # Format values
             control_fmt = f"{control_val:.2%}" if is_rate and pd.notna(control_val) else f"{control_val:.2f}" if pd.notna(control_val) else "N/A"
             treat_fmt = f"{treat_val:.2%}" if is_rate and pd.notna(treat_val) else f"{treat_val:.2f}" if pd.notna(treat_val) else "N/A"
             lift_fmt = f"{lift:+.2f}%" if isinstance(lift, (int, float)) and pd.notna(lift) and np.isfinite(lift) else "N/A"
             pval_fmt = f"(p={pval:.3f})" if pd.notna(pval) and pval >= 0.001 else "(p<0.001)" if pd.notna(pval) else "(p=N/A)"

             delta_color = "off"
             if significant:
                if isinstance(lift, (int, float)) and pd.notna(lift) and np.isfinite(lift):
                    delta_color = "normal" if lift > 0 else "inverse"


             st.metric(
                 label=metric,
                 value=treat_fmt,
                 delta=f"{lift_fmt} {pval_fmt}",
                 delta_color=delta_color,
                 help=f"Control: {control_fmt} | N(ctrl): {data.get('control_n','N/A')}, N(treat): {data.get('treatment_n','N/A')}"
             )
         col_idx += 1


def plot_comparison(df_display, metric_name, result_data):
     """Generates a comparison plot using Plotly Express."""
     st.subheader(f"Comparison: {metric_name}")

     is_rate = 'Rate' in metric_name
     control_val = result_data.get('control_rate') if is_rate else result_data.get('control_mean')
     treat_val = result_data.get('treatment_rate') if is_rate else result_data.get('treatment_mean')
     y_label = "Rate" if is_rate else "Mean Value"
     y_format = ".1%" if is_rate else ".2f"
     original_metric_col = metric_name.replace(' Rate','') if is_rate else metric_name


     # Data for Bar Chart (Means/Rates)
     plot_df = pd.DataFrame({
         'Group': [CONTROL_VALUE, TREATMENT_VALUE],
         'Value': [control_val if pd.notna(control_val) else 0, treat_val if pd.notna(treat_val) else 0]
     })

     fig_bar = px.bar(plot_df, x='Group', y='Value', color='Group',
                      text='Value', title=f"{metric_name} Comparison",
                      color_discrete_map={CONTROL_VALUE: 'skyblue', TREATMENT_VALUE: 'lightcoral'},
                      labels={'Value': y_label})
     fig_bar.update_traces(texttemplate=f'%{{text:{y_format}}}', textposition='outside')
     fig_bar.update_layout(yaxis_tickformat=y_format, showlegend=False, title_x=0.5)

     st.plotly_chart(fig_bar, use_container_width=True)

     # Optional: Box plot for continuous metrics if data available
     if not is_rate and original_metric_col in df_display.columns and not df_display[original_metric_col].isnull().all():
          try:
            fig_box = px.box(df_display.dropna(subset=[original_metric_col]),
                            x=GROUP_COLUMN, y=original_metric_col, color=GROUP_COLUMN,
                            title=f"{metric_name} Distribution", points="outliers",
                            color_discrete_map={CONTROL_VALUE: 'skyblue', TREATMENT_VALUE: 'lightcoral'},
                            labels={original_metric_col: y_label})
            fig_box.update_layout(showlegend=False, title_x=0.5)
            st.plotly_chart(fig_box, use_container_width=True)
          except Exception as box_e:
              st.warning(f"Could not generate box plot for {metric_name}: {box_e}")


# --- Main Application ---
st.title(f"📊 {DASHBOARD_TITLE}")

# --- Display Fixed Configuration ---
st.markdown("---")
st.subheader("Experiment Setup")
col1, col2, col3 = st.columns(3)
col1.info(f"Data File: `{os.path.basename(INPUT_DATA_FILE)}`")
col2.info(f"Groups: `{CONTROL_VALUE}` (Control) vs `{TREATMENT_VALUE}` (Treatment)")
col3.info(f"Alpha: `{ALPHA}`")

st.markdown(f"**Conversion Metric:** `{CONVERSION_METRIC}`")
if CONTINUOUS_METRICS_TO_RUN:
    st.markdown(f"**Continuous Metrics:** `{'`, `'.join(CONTINUOUS_METRICS_TO_RUN)}`")
else:
    st.markdown("**Continuous Metrics:** None")
st.markdown("---")


# --- Load Data ---
df_loaded = load_data(INPUT_DATA_FILE)

if df_loaded is None:
    st.stop() # Stop execution if data loading failed


# --- Analysis Execution ---
analysis_results = {}
run_analysis_button = st.button("Run Analysis", key='run_button', type="primary")

if run_analysis_button:
    try:
        with st.spinner("Running analysis..."):
            # Validate columns exist before analysis
            required_cols = [GROUP_COLUMN, CONVERSION_METRIC] + CONTINUOUS_METRICS_TO_RUN
            missing_cols = [col for col in required_cols if col not in df_loaded.columns]
            if missing_cols:
                 st.error(f"Missing required columns in data file: {', '.join(missing_cols)}")
                 st.stop()

            analyzer = SimpleABAnalyzer(df_loaded, GROUP_COLUMN, CONTROL_VALUE, TREATMENT_VALUE)

            analysis_results = analyzer.run_analysis(
                conversion_metric=CONVERSION_METRIC,
                continuous_metrics=CONTINUOUS_METRICS_TO_RUN, # Use the hardcoded list
                alpha=ALPHA
            )
            st.session_state.analysis_results = analysis_results # Store results
            if not analysis_results:
                 st.warning("Analysis completed but produced no results.")
            else:
                 # Use success message with rerun
                 st.success("Analysis Complete!")
                 # --- CORRECTED LINE ---
                 st.rerun() # Rerun to display results below the button
                 # --- END CORRECTION ---

    except ValueError as ve:
        st.error(f"Analysis Setup Error: {ve}")
    except Exception as e:
        st.error(f"An unexpected error occurred during analysis: {e}")
        traceback.print_exc()

# --- Display Results ---
# Check if results exist in session state (from previous button click)
if 'analysis_results' in st.session_state and st.session_state.analysis_results:
    results_to_display = st.session_state.analysis_results
    display_simplified_metrics(results_to_display)

    st.divider()
    st.header("📈 Detailed Metric Analysis")

    # Check if there are results to show in tabs
    if results_to_display:
        # Create tabs for each analyzed metric
        metric_tabs = st.tabs(list(results_to_display.keys()))
        tab_index = 0
        for metric_name, result_data in results_to_display.items():
            with metric_tabs[tab_index]:
                plot_comparison(df_loaded, metric_name, result_data)
            tab_index += 1
    else:
        st.info("No detailed metric results to display.")


# --- Show Data Sample ---
with st.expander("View Raw Data Sample (First 100 Rows)"):
    st.dataframe(df_loaded.head(100))