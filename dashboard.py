# dashboard.py
"""
Simplified Streamlit dashboard for A/B test analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import traceback

# --- Import Custom Modules & Config ---
try:
    from statistical_analyzer import SimpleABAnalyzer # Import simplified analyzer
    import config # Import simplified config
    INPUT_DATA_FILE = config.INPUT_DATA_FILE
    GROUP_COLUMN = config.GROUP_COLUMN
    CONTROL_VALUE = config.CONTROL_GROUP_VALUE
    TREATMENT_VALUE = config.TREATMENT_GROUP_VALUE
    CONVERSION_METRIC = config.CONVERSION_COLUMN
    CONTINUOUS_METRICS = config.CONTINUOUS_METRICS
    ALPHA = config.ALPHA
    DASHBOARD_TITLE = config.DASHBOARD_TITLE
except ImportError as e:
    st.error(f"Error importing modules: {e}. Make sure 'config.py' and 'statistical_analyzer.py' are in the same folder.")
    st.stop()
except AttributeError as e:
    st.error(f"Error accessing configuration in 'config.py': {e}. Make sure all required variables are defined.")
    st.stop()


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
        st.info("Please ensure 'finance_ab.csv' is in the same directory as the script, or update INPUT_DATA_FILE in config.py.")
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
    cols = st.columns(len(results))
    col_idx = 0

    for metric, data in results.items():
         if not isinstance(data, dict): continue # Skip if result format is wrong

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
     if not is_rate and original_metric_col in df_display.columns:
          fig_box = px.box(df_display.dropna(subset=[original_metric_col]),
                           x=GROUP_COLUMN, y=original_metric_col, color=GROUP_COLUMN,
                           title=f"{metric_name} Distribution", points="outliers",
                           color_discrete_map={CONTROL_VALUE: 'skyblue', TREATMENT_VALUE: 'lightcoral'},
                           labels={original_metric_col: y_label})
          fig_box.update_layout(showlegend=False, title_x=0.5)
          st.plotly_chart(fig_box, use_container_width=True)


# --- Main Application ---
st.title(f"📊 {DASHBOARD_TITLE}")
st.markdown(f"Analyze A/B test results from `{