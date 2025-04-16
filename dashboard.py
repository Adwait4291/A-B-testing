# dashboard.py
"""
Streamlit dashboard for interactive A/B test analysis and reporting.
Adapted for marketing_AB.csv dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime
import os
import tempfile
import traceback
from pandas.api.types import is_datetime64_any_dtype as is_datetime

# --- Import Custom Modules & Config ---
# Note: data_simulator is no longer needed for the main workflow
from statistical_analyzer import ABAnalyzer
from pdf_generator import generate_pdf_report

# Import config variables safely
try:
    import config
    INPUT_DATA_FILE = getattr(config, 'INPUT_DATA_FILE', 'marketing_AB.csv')
    REPORT_FILENAME_BASE = os.path.splitext(getattr(config, 'REPORT_FILENAME', 'marketing_ab_test_report.pdf'))[0]
    DEFAULT_ALPHA = getattr(config, 'ALPHA', 0.05)
    DEFAULT_GROUP_COL = getattr(config, 'GROUP_COLUMN', 'test group')
    DEFAULT_CONTROL_VAL = getattr(config, 'CONTROL_GROUP_VALUE', 'psa')
    DEFAULT_TREATMENT_VAL = getattr(config, 'TREATMENT_GROUP_VALUE', 'ad')
    DEFAULT_CONVERSION_COL = getattr(config, 'CONVERSION_COLUMN', 'converted')
    DEFAULT_CONTINUOUS_METRICS = getattr(config, 'CONTINUOUS_METRICS', ['total ads', 'most ads hour'])
    DEFAULT_SEGMENT_COLS = getattr(config, 'SEGMENTATION_COLUMNS', ['most ads day'])
except ImportError:
    st.error("Could not import `config.py`. Using default settings.")
    INPUT_DATA_FILE = 'marketing_AB.csv'
    REPORT_FILENAME_BASE = 'marketing_ab_test_report'
    DEFAULT_ALPHA = 0.05
    DEFAULT_GROUP_COL = 'test group'
    DEFAULT_CONTROL_VAL = 'psa'
    DEFAULT_TREATMENT_VAL = 'ad'
    DEFAULT_CONVERSION_COL = 'converted'
    DEFAULT_CONTINUOUS_METRICS = ['total ads', 'most ads hour']
    DEFAULT_SEGMENT_COLS = ['most ads day']


# --- Page Configuration ---
st.set_page_config(
    page_title="Marketing A/B Test Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---

@st.cache_data # Cache data loading
def load_data(file_path):
    """Loads data from the specified file path."""
    if not os.path.exists(file_path):
         st.error(f"Data file not found at specified path: {file_path}")
         st.error("Please ensure 'marketing_AB.csv' exists in the same directory as config.py, or update INPUT_DATA_FILE in config.py.")
         return None
    try:
        df = pd.read_csv(file_path)
        st.success(f"Data loaded successfully from '{file_path}'!")
        # Basic preprocessing: convert boolean 'converted' to int 0/1 if needed by analyzer later
        if DEFAULT_CONVERSION_COL in df.columns and pd.api.types.is_bool_dtype(df[DEFAULT_CONVERSION_COL]):
             df[DEFAULT_CONVERSION_COL] = df[DEFAULT_CONVERSION_COL].astype(int)
             print(f"Converted boolean column '{DEFAULT_CONVERSION_COL}' to int.")
        return df
    except Exception as e:
        st.error(f"Error loading data from '{file_path}': {e}")
        st.exception(e)
        return None

def display_metrics(results, primary_kpi=None):
    """Displays key metrics using st.metric."""
    # (Code is identical to the previous corrected version - keeping it for completeness)
    if not isinstance(results, dict) or not results:
        st.warning("No analysis results to display.")
        return

    st.subheader("Key Metric Performance")
    valid_results = {k: v for k, v in results.items() if isinstance(v, dict)}
    num_metrics = len(valid_results)
    if num_metrics == 0:
        st.info("Analysis complete, but no valid metric results generated.")
        return

    cols = st.columns(min(num_metrics, 5)) # Allow up to 5 cols

    col_idx = 0
    for metric, data in valid_results.items():
        current_col = cols[col_idx % len(cols)]
        with current_col:
            is_primary = metric == primary_kpi
            title = f"*{metric}*" if is_primary else metric

            lift = data.get('lift (%)')
            lift_str = f"{lift:+.2f}%" if isinstance(lift, (int, float)) and pd.notna(lift) and np.isfinite(lift) else "N/A"

            pval = data.get('p_value')
            pval_fmt = ""
            if pval is not None and pd.notna(pval):
                if pval < 0.001: pval_fmt = "(p<0.001)"
                else: pval_fmt = f"(p={pval:.3f})"

            is_significant = data.get('significant', False)
            delta_color = "off"
            if is_significant:
                if isinstance(lift, (int, float)) and pd.notna(lift) and np.isfinite(lift):
                    delta_color = "normal" if lift > 0 else "inverse"
                    if any(term in metric.lower() for term in ['bounce', 'error', 'dropoff', 'cost']): # Add other "lower is better" terms
                        delta_color = "inverse" if lift > 0 else "normal"

            main_value_fmt = "N/A"
            control_value_fmt = "N/A"
            if 'treatment_rate' in data and pd.notna(data.get('treatment_rate')):
                main_value_fmt = f"{data['treatment_rate']:.2%}"
                if pd.notna(data.get('control_rate')):
                     control_value_fmt = f"{data['control_rate']:.2%}"
            elif 'treatment_mean' in data and pd.notna(data.get('treatment_mean')):
                 main_value_fmt = f"{data['treatment_mean']:.2f}"
                 if pd.notna(data.get('control_mean')):
                      control_value_fmt = f"{data['control_mean']:.2f}"

            ci_lower = data.get('ci_lower')
            ci_upper = data.get('ci_upper')
            ci_fmt = "[N/A, N/A]"
            if pd.notna(ci_lower) and pd.notna(ci_upper):
                 ci_fmt = f"[{ci_lower:+.3f}, {ci_upper:+.3f}]"

            st.metric(
                label=title,
                value=main_value_fmt,
                delta=f"{lift_str} lift {pval_fmt}",
                delta_color=delta_color,
                help=f"Control: {control_value_fmt} | 95% CI for Diff: {ci_fmt}"
            )
        col_idx += 1

def plot_metric_comparison(df_filtered, metric_display_name, metric_col, group_col, control_name, treatment_name, analysis_result):
    """Generates bar/box plots for metric comparison."""
    if not isinstance(analysis_result, dict):
         st.warning(f"No valid analysis result provided for {metric_display_name}.")
         return

    st.subheader(f"Comparison: {metric_display_name}")
    test_type = analysis_result.get('test_type', '')
    is_conversion = 'proportion' in test_type.lower()
    is_continuous = 't-test' in test_type.lower()

    col1, col2 = st.columns([1, 2]) # Ratio for stats | plot

    # --- Column 1: Summary Stats ---
    with col1:
        control_label = f"{control_name} { 'Rate' if is_conversion else 'Mean'}"
        treatment_label = f"{treatment_name} { 'Rate' if is_conversion else 'Mean'}"

        control_val_fmt = "N/A"
        treatment_val_fmt = "N/A"

        if is_conversion and pd.notna(analysis_result.get('control_rate')):
            control_val_fmt = f"{analysis_result['control_rate']:.2%}"
            if pd.notna(analysis_result.get('treatment_rate')):
                 treatment_val_fmt = f"{analysis_result['treatment_rate']:.2%}"
        elif is_continuous and pd.notna(analysis_result.get('control_mean')):
            control_val_fmt = f"{analysis_result['control_mean']:.2f}"
            if pd.notna(analysis_result.get('treatment_mean')):
                 treatment_val_fmt = f"{analysis_result['treatment_mean']:.2f}"

        st.write(f"**{control_label}:** {control_val_fmt}")
        st.write(f"**{treatment_label}:** {treatment_val_fmt}")

        lift = analysis_result.get('lift (%)')
        lift_str = "N/A"
        if isinstance(lift, (int, float)) and pd.notna(lift) and np.isfinite(lift):
             lift_str = f"{lift:+.2f}%"
        st.write(f"**Lift (%):** {lift_str}")

        pval = analysis_result.get('p_value')
        pval_str = "N/A"
        if pval is not None and pd.notna(pval): pval_str = f"{pval:.3f}" if pval >=0.001 else "<0.001"
        st.write(f"**P-value:** {pval_str}")

        sig = analysis_result.get('significant')
        sig_str = 'N/A' if sig is None else ('✅ Yes' if sig else '❌ No')
        st.write(f"**Significant (alpha={analysis_result.get('alpha', DEFAULT_ALPHA)}):** {sig_str}")

        ci_lower = analysis_result.get('ci_lower')
        ci_upper = analysis_result.get('ci_upper')
        ci_fmt = "[N/A, N/A]"
        if pd.notna(ci_lower) and pd.notna(ci_upper):
             ci_fmt = f"[{ci_lower:+.3f}, {ci_upper:+.3f}]"
        st.write(f"**95% CI for Diff:** {ci_fmt}")


    # --- Column 2: Plot ---
    with col2:
        control_name_str = str(control_name)
        treatment_name_str = str(treatment_name)
        color_map = {control_name_str: 'skyblue', treatment_name_str: 'lightcoral'}

        try:
            if is_conversion:
                # Bar chart for rates
                data_to_plot = pd.DataFrame({
                    'Group': [control_name_str, treatment_name_str],
                    'Rate': [analysis_result.get('control_rate', 0), analysis_result.get('treatment_rate', 0)]
                })
                fig = px.bar(data_to_plot, x='Group', y='Rate', color='Group', text='Rate', color_discrete_map=color_map)
                fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                fig.update_layout(yaxis_tickformat=".1%", showlegend=False, title_text=None)
                st.plotly_chart(fig, use_container_width=True)

            elif is_continuous:
                # Box plot for continuous data distribution
                if metric_col not in df_filtered.columns:
                     st.warning(f"Could not find column '{metric_col}' in filtered data for box plot.")
                else:
                    df_plot = df_filtered[[group_col, metric_col]].copy()
                    df_plot[group_col] = df_plot[group_col].astype(str) # Ensure group is string for plotting
                    fig = px.box(df_plot.dropna(subset=[metric_col]),
                                 x=group_col, y=metric_col, color=group_col,
                                 title=f"{metric_display_name} Distribution", points="outliers",
                                 color_discrete_map=color_map)
                    fig.update_layout(showlegend=False, title_text=None)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                 st.write("Cannot determine plot type (conversion or continuous).")

        except Exception as e:
            st.error(f"Error plotting comparison for {metric_display_name}: {e}")
            st.exception(e)

# --- Main Application ---
st.title(f"📊 {DASHBOARD_TITLE}")
st.markdown(f"""
Analyze A/B test results from `{os.path.basename(INPUT_DATA_FILE)}`.
Configure the experiment parameters in the sidebar.
""")

# --- Session State Initialization ---
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = {}
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'df_loaded' not in st.session_state: st.session_state.df_loaded = pd.DataFrame() # Raw loaded data
if 'df_filtered' not in st.session_state: st.session_state.df_filtered = pd.DataFrame() # Data after filtering

# --- Load Data ---
# Load data only once and store in session state
if not st.session_state.data_loaded:
    df_loaded_data = load_data(INPUT_DATA_FILE)
    if df_loaded_data is not None:
        st.session_state.df_loaded = df_loaded_data
        st.session_state.df_filtered = df_loaded_data.copy() # Initialize filtered df
        st.session_state.data_loaded = True
    else:
        # Stop execution if data loading fails
        st.stop()


# --- Sidebar for Controls ---
with st.sidebar:
    st.header("⚙️ Controls")

    if st.session_state.data_loaded:
        df_processed = st.session_state.df_loaded # Start with the raw loaded data for controls

        # 1. Experiment Setup
        st.subheader("1. Experiment Setup")

        # --- Group Column and Values ---
        st.info(f"Using Group Column: `{DEFAULT_GROUP_COL}`")
        st.info(f"Control Group: `{DEFAULT_CONTROL_VAL}`, Treatment Group: `{DEFAULT_TREATMENT_VAL}`")
        # Use defaults from config directly for this version
        group_col_selected = DEFAULT_GROUP_COL
        control_val_selected = DEFAULT_CONTROL_VAL
        treatment_val_selected = DEFAULT_TREATMENT_VAL
        # Add a check if default values exist in the column
        if group_col_selected in df_processed.columns:
            actual_groups = df_processed[group_col_selected].astype(str).unique()
            if str(control_val_selected) not in actual_groups or str(treatment_val_selected) not in actual_groups:
                st.warning(f"Configured control/treatment values ('{control_val_selected}', '{treatment_val_selected}') not found in column '{group_col_selected}'. Found: {actual_groups}. Please check config.py or data.")
                # Invalidate selections if defaults don't match data
                control_val_selected, treatment_val_selected = None, None
        else:
             st.error(f"Configured group column '{group_col_selected}' not found in the data.")
             group_col_selected = None # Invalidate

        # 2. Define Metrics
        st.subheader("2. Define Metrics")
        # --- Conversion Metric (Fixed based on config) ---
        st.info(f"Analyzing Conversion Metric: `{DEFAULT_CONVERSION_COL}`")
        if DEFAULT_CONVERSION_COL not in df_processed.columns:
            st.error(f"Configured conversion column '{DEFAULT_CONVERSION_COL}' not found!")
            conv_metrics_defined = []
            primary_kpi_selected = None
        else:
            # Define the structure for the analyzer (user-level conversion)
            conv_metrics_defined = [(DEFAULT_CONVERSION_COL, None)]
            primary_kpi_selected = f"{DEFAULT_CONVERSION_COL} Rate" # Auto-set primary KPI

        # --- Continuous Metrics (Selectable based on config) ---
        available_continuous = [m for m in DEFAULT_CONTINUOUS_METRICS if m in df_processed.columns]
        if not available_continuous:
             st.info("No continuous metrics found/configured to analyze.")
             cont_metrics_selected = []
        else:
             cont_metrics_selected = st.multiselect(
                 "Select Continuous Metrics to Analyze",
                 options=available_continuous,
                 default=available_continuous, # Default to analyzing all found
                 key='continuous_metrics_selector'
             )

        # --- Primary KPI (Auto-set based on conversion col) ---
        st.subheader("3. Primary KPI")
        if primary_kpi_selected:
             st.info(f"Primary KPI set to: `{primary_kpi_selected}`")
        else:
             st.warning("Conversion column not found, cannot set Primary KPI.")


        # 4. Filters (Optional) - Based on SEGMENTATION_COLUMNS from config
        st.subheader("4. Filters (Optional)")
        available_segment_cols = [c for c in DEFAULT_SEGMENT_COLS if c in df_processed.columns]
        filters_applied = {}
        df_filtered_intermediate = df_processed.copy() # Start filtering from original loaded data

        if not available_segment_cols:
            st.info("No segmentation columns found/configured for filtering.")
        else:
            for col in available_segment_cols:
                 options = sorted(df_processed[col].astype(str).unique())
                 # Use session state keys for multiselect to remember choices across reruns
                 filter_key = f"filter_{col}"
                 selected_options = st.multiselect(
                     f"Filter by {col.replace('_',' ').title()}",
                     options=options,
                     default=options, # Default to all selected
                     key=filter_key
                 )
                 if selected_options != options: # Apply filter only if not all options are selected
                      filters_applied[col] = selected_options
                      df_filtered_intermediate = df_filtered_intermediate[df_filtered_intermediate[col].astype(str).isin(selected_options)]

        # Update the filtered dataframe in session state
        st.session_state.df_filtered = df_filtered_intermediate
        if filters_applied:
             st.info(f"Filtered Data: {len(st.session_state.df_filtered)} rows remaining.")
        else:
             st.info("No filters applied.")


    else: # No data loaded
        st.warning("Load data to enable controls.")
        # Reset state variables
        control_val_selected, treatment_val_selected = None, None
        group_col_selected = None
        metric_col_selected = None
        primary_kpi_selected = None
        conv_metrics_defined, cont_metrics_selected = [], []


# --- Main Panel ---
# Determine if setup is complete based on sidebar selections and data
analysis_ready = (
    st.session_state.data_loaded
    and not st.session_state.df_filtered.empty # Check filtered data
    and group_col_selected is not None
    and control_val_selected is not None # Check if invalidated
    and treatment_val_selected is not None # Check if invalidated
    and DEFAULT_CONVERSION_COL in st.session_state.df_filtered.columns # Ensure conversion col exists after filter
)

if analysis_ready:
    st.header("📊 Analysis Results")
    # --- Run Analysis Button ---
    if st.button("Run Analysis", key='run_button'):
        st.session_state.analysis_results = {} # Clear previous results
        try:
            with st.spinner("Running statistical analysis..."):
                if st.session_state.df_filtered.empty:
                     raise ValueError("Filtered data is empty. Cannot run analysis.")

                # Instantiate analyzer with selected parameters
                analyzer = ABAnalyzer(
                    st.session_state.df_filtered,
                    group_col=group_col_selected,
                    control_val=control_val_selected,
                    treatment_val=treatment_val_selected
                )
                # Run analysis with defined metrics
                results = analyzer.run_analysis(
                    conversion_metric=DEFAULT_CONVERSION_COL, # Pass the single conversion col name
                    continuous_metrics=cont_metrics_selected, # Pass list selected by user
                    alpha=DEFAULT_ALPHA
                )
                st.session_state.analysis_results = results # Store results
                if not results:
                    st.warning("Analysis completed but produced no results. Check data and group definitions.")
                else:
                    st.success("Analysis Complete!")

        except ValueError as ve:
            st.error(f"Analysis Configuration Error: {ve}. Check setup and filters.")
        except KeyError as ke:
            st.error(f"Data Error: Column {ke} not found. Check data or config.")
        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {e}")
            st.exception(e)

    # --- Display results if they exist in session state ---
    if isinstance(st.session_state.analysis_results, dict) and st.session_state.analysis_results:
         results_to_display = st.session_state.analysis_results

         # Display st.metric widgets
         display_metrics(results_to_display, primary_kpi_selected)

         # Display Detailed Metric Comparisons in Tabs
         st.header("📈 Detailed Metric Analysis")
         valid_results_detail = {k:v for k,v in results_to_display.items() if isinstance(v, dict)}
         if valid_results_detail:
              # Create tabs using the metric names from the results keys
              metric_tabs = st.tabs(list(valid_results_detail.keys()))
              tab_index = 0
              for metric_name, result_data in valid_results_detail.items():
                  with metric_tabs[tab_index]:
                      # Determine the original column name (metric name might have " Rate" suffix)
                      original_metric_col = metric_name.replace(' Rate', '') if metric_name.endswith(' Rate') else metric_name
                      plot_metric_comparison(
                            df_filtered=st.session_state.df_filtered,
                            metric_display_name=metric_name,
                            metric_col = original_metric_col, # Pass original col name for box plot
                            group_col=group_col_selected,
                            control_name=control_val_selected,
                            treatment_name=treatment_val_selected,
                            analysis_result=result_data
                        )
                  tab_index += 1
         else:
             st.info("No detailed metric results to display.")


         # --- PDF Report Generation ---
         st.header("📄 Generate Report")
         report_name_input = st.text_input("Report Filename Prefix", REPORT_FILENAME_BASE)
         if st.button("Generate PDF Report", key='pdf_button'):
             with st.spinner("Generating PDF report..."):
                 # --- Temporary File Workaround ---
                 temp_pdf_filename = None
                 try:
                     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_f:
                         temp_pdf_filename = temp_f.name

                     print(f"Attempting PDF generation to temp file: {temp_pdf_filename}")
                     success = generate_pdf_report(
                         filename=temp_pdf_filename, # Pass string filename
                         experiment_name=report_name_input.replace("_", " "),
                         run_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         analysis_results=results_to_display,
                         raw_data=None, # Funnel chart columns likely don't exist
                         primary_kpi=primary_kpi_selected,
                         control_group_name=str(control_val_selected),
                         treatment_group_name=str(treatment_val_selected)
                     )

                     if success:
                         print(f"PDF generation successful to temp file: {temp_pdf_filename}")
                         with open(temp_pdf_filename, "rb") as f_read:
                             pdf_buffer = BytesIO(f_read.read())
                         pdf_buffer.seek(0)
                         st.success("PDF Report Ready for Download!")
                         st.download_button(
                             label="Download PDF Report",
                             data=pdf_buffer,
                             file_name=f"{report_name_input}.pdf",
                             mime="application/pdf",
                             key='pdf_download_button'
                         )
                     else:
                          st.error("PDF generation function returned failure. Check console logs.")

                 except Exception as pdf_e:
                     st.error(f"Failed to generate PDF: {pdf_e}")
                     st.exception(pdf_e)
                     traceback.print_exc()
                 finally:
                     if temp_pdf_filename and os.path.exists(temp_pdf_filename):
                         try:
                             print(f"Cleaning up temp file: {temp_pdf_filename}")
                             os.remove(temp_pdf_filename)
                         except Exception as cleanup_e:
                              st.warning(f"Could not remove temporary file {temp_pdf_filename}: {cleanup_e}")


elif st.session_state.data_loaded: # Data loaded but setup incomplete
     st.info("Please complete the experiment setup (Group Column, Control/Treatment) in the sidebar.")
else: # No data loaded yet
     st.info("Loading data...") # Initial state before data is confirmed loaded


# --- Optional: Display Raw Data Sample ---
if st.session_state.data_loaded and not st.session_state.df_filtered.empty:
    with st.expander("View Filtered Data Sample (First 100 Rows)"):
        st.dataframe(st.session_state.df_filtered.head(100))