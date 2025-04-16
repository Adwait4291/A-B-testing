# dashboard.py
"""
Streamlit dashboard for interactive A/B test analysis and reporting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime
import os # To check if simulated data exists

# Import our custom modules
from data_simulator import simulate_ab_data
from statistical_analyzer import ABAnalyzer
from pdf_generator import generate_pdf_report
from config import COUNTRIES, CHANNELS, DEVICES # Import lists for filters

# --- Page Configuration ---
st.set_page_config(
    page_title="A/B Testing Automation Framework",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
@st.cache_data # Cache the data loading/simulation
def load_data(uploaded_file=None):
    """Loads data from uploaded file or simulates new data."""
    if uploaded_file is not None:
        try:
            # Determine file type and load
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                 df = pd.read_excel(uploaded_file)
            else:
                 st.error("Unsupported file format. Please upload CSV or Excel.")
                 return None
            st.success("Uploaded data loaded successfully!")
            return df
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            return None
    else:
        # Option to generate sample data if none exists
        if not os.path.exists("simulated_ab_data.csv"):
             st.warning("No uploaded file. Generating sample data...")
             df = simulate_ab_data()
             df.to_csv("simulated_ab_data.csv", index=False)
             st.success("Sample data generated and saved!")
             return df
        else:
            try:
                 df = pd.read_csv("simulated_ab_data.csv")
                 st.info("Loaded existing sample data (simulated_ab_data.csv). Upload a file to analyze your own data.")
                 return df
            except Exception as e:
                 st.error(f"Error loading sample data: {e}")
                 return None


def display_metrics(results, primary_kpi=None):
    """Displays key metrics using st.metric."""
    if not results:
        st.warning("No analysis results to display.")
        return

    st.subheader("Key Metric Performance")

    # Determine number of columns based on number of results
    num_metrics = len(results)
    cols = st.columns(min(num_metrics, 4)) # Max 4 columns for better layout

    col_idx = 0
    for metric, data in results.items():
        current_col = cols[col_idx % len(cols)]
        with current_col:
            is_primary = metric == primary_kpi
            title = f"*{metric}*" if is_primary else metric

            lift = data.get('lift (%)', 'N/A')
            lift_str = f"{lift:.2f}%" if isinstance(lift, (int, float)) and pd.notna(lift) else str(lift)

            pval = data.get('p_value')
            pval_fmt = f" (p={pval:.3f})" if pval is not None and pd.notna(pval) else ""
            if pval is not None and pval < 0.001:
                 pval_fmt = " (p<0.001)"

            is_significant = data.get('significant', False)
            delta_color = "off" # Default color for st.metric
            if is_significant:
                # Check lift direction for color
                if isinstance(lift, (int, float)) and lift > 0:
                    delta_color = "normal" # Green for positive lift
                elif isinstance(lift, (int, float)) and lift < 0:
                     # For bounce rate, negative lift is good, otherwise bad
                     if "bounce" in metric.lower():
                          delta_color = "normal" # Green for lower bounce
                     else:
                          delta_color = "inverse" # Red for negative lift
                else: # Significant but zero/NA lift? Rare.
                     delta_color = "off"
            elif pval is not None: # Not significant
                 delta_color = "off" # Grey


            st.metric(
                label=title,
                value=f"{data['treatment_rate']:.2%}" if 'rate' in data else f"{data['treatment_mean']:.2f}",
                delta=f"{lift_str} vs Control{pval_fmt}",
                delta_color=delta_color,
                help=f"Control: {data['control_rate']:.2%} | CI: [{data['ci_lower']:.3f}, {data['ci_upper']:.3f}]" if 'rate' in data else f"Control: {data['control_mean']:.2f} | CI: [{data['ci_lower']:.3f}, {data['ci_upper']:.3f}]"
            )
        col_idx += 1


def plot_funnel(df_filtered, group_col, control_val, treatment_val):
    """Generates and displays a Plotly funnel chart."""
    control_df = df_filtered[df_filtered[group_col] == control_val]
    treatment_df = df_filtered[df_filtered[group_col] == treatment_val]

    steps = ['views', 'clicks', 'applications_started', 'applications_completed']
    control_counts = [control_df[step].sum() for step in steps]
    treatment_counts = [treatment_df[step].sum() for step in steps]
    step_names = [s.replace('_', ' ').title() for s in steps]

    fig = go.Figure()
    fig.add_trace(go.Funnel(
        name=control_val, y=step_names, x=control_counts,
        textinfo="value+percent initial", marker={"color": "skyblue"}
    ))
    fig.add_trace(go.Funnel(
        name=treatment_val, y=step_names, x=treatment_counts,
        textinfo="value+percent initial", marker={"color": "lightcoral"}
    ))
    fig.update_layout(title="Acquisition Funnel Comparison", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def plot_metric_comparison(df_filtered, metric, group_col, control_val, treatment_val, analysis_result):
    """Generates bar/box plots for metric comparison."""
    st.subheader(f"Comparison: {metric}")
    is_conversion = 'rate' in analysis_result # Check if it was analyzed as conversion/rate
    is_continuous = 'mean' in analysis_result # Check if it was analyzed as continuous/mean

    col1, col2 = st.columns([1, 2]) # Adjust column ratios as needed

    with col1: # Summary Stats
        st.write(f"**Control { 'Rate' if is_conversion else 'Mean'}:** {analysis_result['control_rate']:.2%}" if is_conversion else f"{analysis_result['control_mean']:.2f}")
        st.write(f"**Treatment { 'Rate' if is_conversion else 'Mean'}:** {analysis_result['treatment_rate']:.2%}" if is_conversion else f"{analysis_result['treatment_mean']:.2f}")
        lift = analysis_result.get('lift (%)', 'N/A')
        lift_str = f"{lift:.2f}%" if isinstance(lift, (int, float)) else str(lift)
        st.write(f"**Lift:** {lift_str}")
        st.write(f"**P-value:** {analysis_result['p_value']:.3f}" if analysis_result['p_value'] else "N/A")
        st.write(f"**Significant:** {'✅ Yes' if analysis_result['significant'] else '❌ No'}")


    with col2: # Plot
        if is_conversion:
             # Bar chart for rates
             data_to_plot = pd.DataFrame({
                 'Group': [control_val, treatment_val],
                 'Rate': [analysis_result['control_rate'], analysis_result['treatment_rate']]
             })
             fig = px.bar(data_to_plot, x='Group', y='Rate', color='Group',
                          title=f"{metric} Comparison", text='Rate',
                          color_discrete_map={control_val: 'skyblue', treatment_val: 'lightcoral'})
             fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
             fig.update_layout(yaxis_tickformat=".1%")
             st.plotly_chart(fig, use_container_width=True)

        elif is_continuous:
             # Box plot for continuous data distribution
             fig = px.box(df_filtered.dropna(subset=[metric]), x=group_col, y=metric, color=group_col,
                          title=f"{metric} Distribution by Group", points="outliers",
                          color_discrete_map={control_val: 'skyblue', treatment_val: 'lightcoral'})
             st.plotly_chart(fig, use_container_width=True)


# --- Main Application ---
st.title("📊 A/B Testing Automation Framework")
st.markdown("""
Welcome to the A/B testing dashboard. Upload your experiment data (CSV/Excel) or use the generated sample data
to analyze performance, visualize results, and generate reports.
""")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("⚙️ Controls")

    # 1. Data Upload
    st.subheader("1. Load Data")
    uploaded_file = st.file_uploader("Upload A/B Test Data (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
    df = load_data(uploaded_file)

    # --- Data dependent controls ---
    if df is not None and not df.empty:
        st.success(f"Loaded DataFrame with {df.shape[0]} rows and {df.shape[1]} columns.")

        # 2. Experiment Setup
        st.subheader("2. Experiment Setup")
        # Dynamically find potential group columns
        potential_group_cols = [col for col in df.columns if df[col].nunique() == 2] # Guess columns with 2 unique values
        group_col_default_ix = potential_group_cols.index('group') if 'group' in potential_group_cols else 0
        group_col = st.selectbox("Select Group Column", df.columns, index=group_col_default_ix)

        # Get unique values for selected group column
        if group_col and df[group_col].nunique() >= 2:
             group_values = df[group_col].unique()
             control_val = st.selectbox("Select Control Group Value", group_values, index=0)
             treatment_val = st.selectbox("Select Treatment Group Value", group_values, index=1)
        else:
             st.warning("Selected group column needs at least two unique values.")
             control_val, treatment_val = None, None

        # Define Metrics (allow selection from columns)
        st.subheader("3. Define Metrics")
        default_metrics = ['views', 'clicks', 'applications_started', 'applications_completed', 'time_on_page_seconds', 'bounced']
        available_cols = df.columns.tolist()
        # Separate potential conversion (binary/count) and continuous metrics
        potential_conversion_cols = [col for col in available_cols if df[col].dropna().isin([0, 1]).all() or df[col].dtype in ['int64', 'int32']]
        potential_continuous_cols = [col for col in available_cols if df[col].dtype in ['float64', 'int64', 'float32', 'int32'] and col not in potential_conversion_cols and df[col].nunique() > 2]

        # --- Conversion Metrics Definition ---
        st.markdown("**Conversion Funnel/Rates** (Numerator / Denominator)")
        conv_metrics_to_analyze = []
        # Pre-populate common funnel steps if columns exist
        if {'views', 'clicks', 'applications_started', 'applications_completed'}.issubset(df.columns):
             conv_metrics_to_analyze.append(('clicks', 'views'))
             conv_metrics_to_analyze.append(('applications_started', 'clicks'))
             conv_metrics_to_analyze.append(('applications_completed', 'applications_started'))
             conv_metrics_to_analyze.append(('applications_completed', 'views')) # Overall
        if 'bounced' in df.columns:
             conv_metrics_to_analyze.append(('bounced', None)) # Treat bounce as user-level rate

        # Allow adding custom conversion metrics
        num_custom_conv = st.number_input("Add Custom Conversion Metrics", min_value=0, max_value=5, value=0, step=1)
        for i in range(num_custom_conv):
            cols_conv = st.columns(2)
            num = cols_conv[0].selectbox(f"Numerator {i+1}", potential_conversion_cols, key=f"num_{i}")
            den = cols_conv[1].selectbox(f"Denominator {i+1} (or None for user-level)", [None] + potential_conversion_cols, key=f"den_{i}")
            if num: # Ensure numerator is selected
                conv_metrics_to_analyze.append((num, den))

        # Display selected conversion metrics (read-only)
        st.write("Metrics to Analyze (Conversion):")
        st.json([f"{n}{' / ' + d if d else ' Rate'}" for n,d in conv_metrics_to_analyze])

        # --- Continuous Metrics Definition ---
        st.markdown("**Continuous Metrics**")
        default_continuous = ['time_on_page_seconds'] if 'time_on_page_seconds' in potential_continuous_cols else []
        cont_metrics_to_analyze = st.multiselect(
             "Select Continuous Metrics",
             potential_continuous_cols,
             default=default_continuous
             )

        # --- Primary KPI Selection ---
        st.subheader("4. Primary KPI")
        all_metric_names = [f"{n}{' / ' + d if d else ' Rate'}" for n,d in conv_metrics_to_analyze] + cont_metrics_to_analyze
        # Try to find a default primary KPI
        default_primary_kpi = None
        if 'applications_completed / views' in all_metric_names: default_primary_kpi = 'applications_completed / views'
        elif 'clicks / views' in all_metric_names: default_primary_kpi = 'clicks / views'
        elif all_metric_names: default_primary_kpi = all_metric_names[0]

        primary_kpi = st.selectbox("Select Primary KPI for Report", all_metric_names, index=(all_metric_names.index(default_primary_kpi) if default_primary_kpi in all_metric_names else 0))


        # 4. Filters
        st.subheader("5. Filters")
        selected_countries = st.multiselect("Filter by Country", options=df['country'].unique(), default=df['country'].unique())
        selected_channels = st.multiselect("Filter by Channel", options=df['channel'].unique(), default=df['channel'].unique())
        selected_devices = st.multiselect("Filter by Device", options=df['device'].unique(), default=df['device'].unique())

        # Filter the DataFrame
        df_filtered = df[
            df['country'].isin(selected_countries) &
            df['channel'].isin(selected_channels) &
            df['device'].isin(selected_devices)
        ]
        st.info(f"Analyzing {len(df_filtered)} rows after filtering.")

    else: # No data loaded or error
        st.warning("Upload data or use sample data to enable analysis controls.")
        df_filtered = pd.DataFrame() # Empty dataframe
        control_val, treatment_val = None, None
        conv_metrics_to_analyze, cont_metrics_to_analyze = [], []
        primary_kpi = None

# --- Main Panel for Results ---
if df_filtered is not None and not df_filtered.empty and control_val and treatment_val:
    try:
        # --- Run Analysis ---
        analyzer = ABAnalyzer(df_filtered, group_col, control_val, treatment_val)
        results = analyzer.run_analysis(
            conversion_metrics=conv_metrics_to_analyze,
            continuous_metrics=cont_metrics_to_analyze # Analyzer handles 'bounced' internally now
        )

        # --- Display Results ---
        st.header("📊 Analysis Results")

        # Display st.metric widgets
        display_metrics(results, primary_kpi)

        # Display Funnel Chart
        st.header(" M Funnel Visualization")
        plot_funnel(df_filtered, group_col, control_val, treatment_val)

        # Display Individual Metric Comparisons
        st.header("🔎 Detailed Metric Analysis")
        if results:
             # Create tabs for each metric result for cleaner layout
             metric_tabs = st.tabs(list(results.keys()))
             for i, metric_name in enumerate(results.keys()):
                 with metric_tabs[i]:
                      plot_metric_comparison(df_filtered, metric_name, group_col, control_val, treatment_val, results[metric_name])

             # Show Raw Results Table (optional)
             with st.expander("View Full Analysis Results Table"):
                 results_df = pd.DataFrame.from_dict(results, orient='index').reset_index(drop=True)
                 # Format for display
                 results_df_display = results_df.copy()
                 for col in ['control_rate', 'treatment_rate']:
                      if col in results_df_display.columns:
                           results_df_display[col] = results_df_display[col].map('{:.2%}'.format, na_action='ignore')
                 for col in ['control_mean', 'treatment_mean', 'absolute_diff', 'ci_lower', 'ci_upper', 'control_std', 'treatment_std']:
                      if col in results_df_display.columns:
                           results_df_display[col] = results_df_display[col].map('{:.2f}'.format, na_action='ignore')
                 if 'lift (%)' in results_df_display.columns:
                     results_df_display['lift (%)'] = results_df_display['lift (%)'].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) and pd.notna(x) else x)
                 if 'p_value' in results_df_display.columns:
                     results_df_display['p_value'] = results_df_display['p_value'].map('{:.3f}'.format, na_action='ignore')

                 st.dataframe(results_df_display)


             # --- PDF Report Generation ---
             st.header("📄 Generate Report")
             report_name = st.text_input("Report Filename (e.g., Homepage_Test_Results)", "AB_Test_Report")
             report_button = st.button("Generate PDF Report")

             if report_button:
                 with st.spinner("Generating PDF report..."):
                     pdf_buffer = BytesIO()
                     exp_name = report_name.replace("_", " ") # Basic experiment name from filename
                     generate_pdf_report(
                         filename=pdf_buffer, # Write to buffer
                         experiment_name=exp_name,
                         run_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         analysis_results=results,
                         raw_data=df_filtered,
                         primary_kpi=primary_kpi,
                         # Recommendation is auto-calculated in generate_pdf_report
                     )
                     pdf_buffer.seek(0)
                     st.success("PDF Report Generated!")
                     st.download_button(
                         label="Download PDF Report",
                         data=pdf_buffer,
                         file_name=f"{report_name}.pdf",
                         mime="application/pdf"
                     )

        else:
             st.warning("Analysis did not produce any results. Check data filters or metric definitions.")

    except ValueError as ve:
        st.error(f"Analysis Error: {ve}. Please check experiment setup (group column, control/treatment values).")
    except KeyError as ke:
        st.error(f"Data Error: Column {ke} not found in the data after filtering. Please check metric definitions and filters.")
    except Exception as e:
        st.error(f"An unexpected error occurred during analysis or visualization: {e}")
        st.exception(e) # Show full traceback for debugging

elif not uploaded_file and df is None:
     st.info("Upload a file or ensure 'simulated_ab_data.csv' exists to start the analysis.")
elif df_filtered.empty and (selected_countries or selected_channels or selected_devices):
     st.warning("No data matches the selected filters. Try adjusting the filters in the sidebar.")
elif not control_val or not treatment_val:
      st.warning("Please select valid Control and Treatment group values in the sidebar.")


# --- Optional: Display Raw Data ---
if df is not None and not df.empty:
    with st.expander("View Loaded Data Sample"):
        st.dataframe(df.head(100)) # Show first 100 rows