# pdf_generator.py
"""Generates a PDF report for A/B test results using ReportLab."""

import io
import os
import traceback
from datetime import datetime

# --- ReportLab Imports ---
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib import colors
from reportlab.lib.units import inch

# --- Data & Plotting Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go # Keep for potential future funnel

# --- Configuration ---
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import config # Import config for example usage
import json # For example usage printing


# --- Helper Functions ---

def _create_summary_table(results_df, primary_kpi=None):
    """Creates a formatted summary table (ReportLab Table object) for the PDF."""
    styles = getSampleStyleSheet()
    normal_style = styles['Normal']
    normal_style.alignment = TA_CENTER

    # Define headers for the output table
    headers = ['Metric', 'Control', 'Treatment', 'Lift (%)', 'P-value', 'Significant']
    data = [headers]

    # Basic styles for the table
    table_style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]

    row_idx = 1
    if results_df is None or results_df.empty:
        return Table([["No summary data available."]], style=[('GRID', (0,0), (-1,-1), 1, colors.black)])

    for index, row in results_df.iterrows():
        try:
            metric = str(row.get('metric', f'Unknown Row {index}'))
            # Check if the metric is a rate (ends with ' Rate') or continuous
            is_rate_metric = metric.endswith(' Rate')
            is_primary = primary_kpi is not None and metric == primary_kpi

            # Determine metric type (rate or mean) and format values
            control_val_str, treat_val_str = 'N/A', 'N/A'
            if is_rate_metric and 'control_rate' in row and pd.notna(row['control_rate']):
                control_val_str = f"{row['control_rate']:.2%}"
                treat_val_str = f"{row['treatment_rate']:.2%}" if pd.notna(row.get('treatment_rate')) else 'N/A'
            elif not is_rate_metric and 'control_mean' in row and pd.notna(row['control_mean']):
                control_val_str = f"{row['control_mean']:.2f}"
                treat_val_str = f"{row['treatment_mean']:.2f}" if pd.notna(row.get('treatment_mean')) else 'N/A'
            # Fallback if keys don't match expected pattern (shouldn't happen with controlled results dict)
            elif 'control_rate' in row and pd.notna(row['control_rate']): # Check rate again as fallback
                control_val_str = f"{row['control_rate']:.2%}"
                treat_val_str = f"{row['treatment_rate']:.2%}" if pd.notna(row.get('treatment_rate')) else 'N/A'
            elif 'control_mean' in row and pd.notna(row['control_mean']): # Check mean again
                 control_val_str = f"{row['control_mean']:.2f}"
                 treat_val_str = f"{row['treatment_mean']:.2f}" if pd.notna(row.get('treatment_mean')) else 'N/A'


            # Format Lift
            lift = row.get('lift (%)')
            lift_str = "N/A"
            if isinstance(lift, (int, float)) and pd.notna(lift) and np.isfinite(lift):
                lift_str = f"{lift:+.2f}%"

            # Format P-value
            pval = row.get('p_value')
            pval_str = 'N/A'
            if pval is not None and pd.notna(pval):
                if pval < 0.001: pval_str = "<0.001"
                else: pval_str = f"{pval:.3f}"

            # Format Significance
            sig = row.get('significant', False)
            sig_str = 'Yes' if sig else 'No'
            sig_color = colors.lightgreen if sig else colors.pink

            # Use Paragraph for metric name for wrapping
            metric_paragraph = Paragraph(metric, styles['Normal'])
            metric_paragraph.style.fontSize = 8
            metric_paragraph.style.alignment = TA_LEFT

            data.append([
                metric_paragraph, control_val_str, treat_val_str,
                lift_str, pval_str, sig_str
            ])

            # Apply specific styles
            table_style_cmds.append(('BACKGROUND', (len(headers)-1, row_idx), (len(headers)-1, row_idx), sig_color))
            table_style_cmds.append(('ALIGN', (0, row_idx), (0, row_idx), 'LEFT'))
            if is_primary:
                 table_style_cmds.append(('FONTNAME', (0, row_idx), (-1, row_idx), 'Helvetica-Bold'))

            row_idx += 1

        except KeyError as e:
             print(f"KeyError processing row {index} for PDF summary table: {e}. Row data: {row.to_dict()}")
             continue
        except Exception as e:
            print(f"General error processing row {index} for PDF summary table: {e}. Row data: {row.to_dict()}")
            traceback.print_exc()
            continue

    # --- Create and style the table ---
    if len(data) > 1:
        col_widths = [2.5*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.8*inch, 0.7*inch]
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle(table_style_cmds))
        return table
    else:
        return Table([["No valid metric data found in results."]], style=[('GRID', (0,0), (-1,-1), 1, colors.black)])


def _create_comparison_plot(metric_row_series):
    """Creates a comparison plot (ReportLab Image) for a metric (bar or box)."""
    fig, ax = plt.subplots(figsize=(4, 2)) # Keep small size for PDF
    styles = getSampleStyleSheet()
    error_style = styles['Italic']
    error_style.textColor = colors.red

    try:
        metric_name = str(metric_row_series.get('metric', 'Unknown Metric'))
        test_type = metric_row_series.get('test_type', '')
        is_rate = 'proportion' in test_type.lower()
        is_continuous = 't-test' in test_type.lower()

        control_group_name = str(metric_row_series.get('control_group', 'Control'))
        treatment_group_name = str(metric_row_series.get('treatment_group', 'Treatment'))
        bar_colors = {control_group_name: "skyblue", treatment_group_name: "lightcoral"}
        plot_x_names = [control_group_name, treatment_group_name]

        plot_title = f"{metric_name}"
        plot_ylabel = "Value"

        # Plotting logic depends on metric type
        if is_rate:
            control_val = metric_row_series.get('control_rate', np.nan)
            treat_val = metric_row_series.get('treatment_rate', np.nan)
            plot_y = [0 if pd.isna(control_val) else control_val, 0 if pd.isna(treat_val) else treat_val]
            plot_ylabel = "Rate"

            sns.barplot(x=plot_x_names, y=plot_y, ax=ax, palette=[bar_colors[control_group_name], bar_colors[treatment_group_name]])
            ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.1%}'.format))
            labels = [f"{v:.1%}" if pd.notna(v) else "N/A" for v in plot_y]
            if ax.containers:
                 ax.bar_label(ax.containers[0], labels=labels, label_type='edge', fontsize=7, padding=2)

        elif is_continuous:
             # For box plot, we ideally need the original data.
             # Since we only have summary stats in `metric_row_series`,
             # we can only plot the means +/- std deviation as a bar chart with error bars.
             control_mean = metric_row_series.get('control_mean', np.nan)
             treat_mean = metric_row_series.get('treatment_mean', np.nan)
             control_std = metric_row_series.get('control_std', 0) # Default std dev to 0 if missing
             treat_std = metric_row_series.get('treatment_std', 0)
             plot_ylabel = "Mean Value"

             means = [0 if pd.isna(control_mean) else control_mean, 0 if pd.isna(treat_mean) else treat_mean]
             stds = [0 if pd.isna(control_std) else control_std, 0 if pd.isna(treat_std) else treat_std]

             ax.bar(plot_x_names, means, yerr=stds, capsize=4,
                    color=[bar_colors[control_group_name], bar_colors[treatment_group_name]],
                    error_kw=dict(ecolor='gray', lw=1))
             ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
             # Labels for mean values on bars
             labels = [f"{v:.2f}" if pd.notna(v) else "N/A" for v in means]
             # Add labels manually since bar_label doesn't work easily with error bars in basic matplotlib
             for i, (mean_val, label_text) in enumerate(zip(means, labels)):
                  ax.text(i, mean_val + 0.01 * max(means) if max(means) > 0 else 0.01 , label_text, ha='center', va='bottom', fontsize=7)

        else: # Unknown type
             return Paragraph(f"<i>Cannot generate plot: Unknown test type for {metric_name}</i>", error_style)

        # Common formatting
        ax.set_title(plot_title, fontsize=9, pad=5)
        ax.set_ylabel(plot_ylabel, fontsize=7)
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout(pad=0.5)

        # Save plot to buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150)
        plt.close(fig) # IMPORTANT: Close the figure
        img_buffer.seek(0)

        # Return ReportLab Image object
        return Image(img_buffer, width=2.5*inch, height=1.25*inch)

    except Exception as e:
        print(f"Error creating comparison plot for {metric_row_series.get('metric', 'Unknown Metric')}: {e}")
        traceback.print_exc()
        plt.close(fig) # Ensure figure is closed on error
        return Paragraph(f"<i>Error generating plot for {metric_row_series.get('metric', 'N/A')}</i>", error_style)


def _create_funnel_chart(df, group_col='group', control_val='Control', treatment_val='Treatment'):
    """
    Creates a Plotly funnel chart IF relevant columns exist (e.g., views, clicks).
    Checks for columns and returns message if missing. Requires 'kaleido'.
    """
    styles = getSampleStyleSheet()
    error_style = styles['Italic']
    error_style.textColor = colors.red

    # Define expected funnel steps
    # !! These steps need to exist as columns in the input `df` !!
    # !! The marketing_AB.csv dataset does NOT contain these by default !!
    steps_cols = ['views', 'clicks', 'conversions'] # Example steps
    missing_cols = [step for step in steps_cols if step not in df.columns]

    if missing_cols:
        print(f"Info [_create_funnel_chart]: Cannot generate funnel chart. Missing required columns: {missing_cols}")
        return Paragraph(f"<i>Funnel chart cannot be generated (Missing columns: {', '.join(missing_cols)}).</i>", styles['Italic'])

    # --- Proceed only if columns exist ---
    if df is None or df.empty: return None # Should not happen if cols exist, but check anyway
    if group_col not in df.columns: return None

    control_val_str = str(control_val)
    treatment_val_str = str(treatment_val)

    df_funnel = df.copy()
    df_funnel[group_col] = df_funnel[group_col].astype(str)
    control_df = df_funnel[df_funnel[group_col] == control_val_str]
    treatment_df = df_funnel[df_funnel[group_col] == treatment_val_str]

    if control_df.empty or treatment_df.empty:
        return Paragraph(f"<i>Funnel chart: No data for one or both groups ('{control_val_str}', '{treatment_val_str}').</i>", styles['Italic'])

    try:
        control_counts = [control_df[step].sum() for step in steps_cols]
        treatment_counts = [treatment_df[step].sum() for step in steps_cols]
        step_names = [s.replace('_', ' ').title() for s in steps_cols]

        if sum(control_counts) == 0 and sum(treatment_counts) == 0:
            return Paragraph("<i>Funnel counts are all zero.</i>", styles['Italic'])

        fig = go.Figure()
        fig.add_trace(go.Funnel(name=control_val_str, y=step_names, x=control_counts, textinfo="value+percent previous", marker={"color": "skyblue"}))
        fig.add_trace(go.Funnel(name=treatment_val_str, y=step_names, x=treatment_counts, textinfo="value+percent previous", marker={"color": "lightcoral"}))
        fig.update_layout(title_text="Funnel Comparison", title_x=0.5, title_font_size=12, margin=dict(t=40, b=10, l=10, r=10), legend_title_text="Group", height=250, font_size=9)

        img_buffer = io.BytesIO()
        fig.write_image(img_buffer, format='png', scale=2, engine='kaleido') # Requires 'kaleido'
        img_buffer.seek(0)
        return Image(img_buffer, width=5.5*inch, height=2.75*inch)

    except ValueError as ve:
         if "requires the kaleido package" in str(ve) or "Could not find kaleido" in str(ve):
              print("ERROR generating funnel chart: `kaleido` package not found or not working.")
              return Paragraph("<i>Error: Funnel chart requires 'kaleido'. Install with `pip install kaleido`.</i>", error_style)
         else: return Paragraph(f"<i>ValueError generating funnel chart: {ve}</i>", error_style)
    except Exception as e:
        print(f"Error generating Plotly funnel image: {e}")
        traceback.print_exc()
        return Paragraph(f"<i>Error generating funnel chart image. Check logs.</i>", error_style)


# --- Main PDF Generation Function ---

def generate_pdf_report(
    filename="ab_test_report.pdf",
    experiment_name="Unnamed Experiment",
    run_date=None,
    analysis_results=None, # Dict from ABAnalyzer.run_analysis()
    raw_data=None, # Optional: DataFrame for funnel chart (check columns exist first!)
    primary_kpi=None,
    control_group_name='Control', # Use defaults consistent with analysis
    treatment_group_name='Treatment'
    ):
    """Generates a PDF report summarizing the A/B test results."""

    print(f"\nAttempting to generate PDF report: {filename}")

    if run_date is None: run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if analysis_results is None or not isinstance(analysis_results, dict) or not analysis_results:
        print("Error: No analysis results (dict) provided for PDF generation. Aborting.")
        # Optionally create a PDF saying "No results"
        # For now, return failure
        return False

    # Use filename (string or buffer) directly with SimpleDocTemplate
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    story = []

    # --- Title and Metadata ---
    story.append(Paragraph(f"A/B Test Report: {experiment_name}", styles['h1']))
    story.append(Spacer(1, 0.1*inch))
    metadata_style = styles['Normal']
    metadata_style.fontSize = 9
    story.append(Paragraph(f"<i>Analysis Run Date: {run_date}</i>", metadata_style))
    story.append(Paragraph(f"<i>Control Group: '{control_group_name}', Treatment Group: '{treatment_group_name}'</i>", metadata_style))
    story.append(Spacer(1, 0.2*inch))

    # --- Convert results dict to DataFrame ---
    try:
        results_df = pd.DataFrame.from_dict(analysis_results, orient='index')
        if 'metric' not in results_df.columns:
             results_df.index.name = 'metric_orig_key'
             results_df = results_df.reset_index().rename(columns={'metric_orig_key': 'metric'})

        # Sort results: primary KPI first, then by significance, then p-value
        if primary_kpi and primary_kpi in results_df['metric'].values:
             results_df['is_primary'] = (results_df['metric'] == primary_kpi)
             results_df = results_df.sort_values(by=['is_primary', 'significant', 'p_value'], ascending=[False, False, True]).drop(columns=['is_primary'])
        else:
             # Sort by significance then p-value if primary KPI not specified or found
             results_df = results_df.sort_values(by=['significant', 'p_value'], ascending=[False, True])

        print(f"Analysis results converted to DataFrame with {len(results_df)} metrics for PDF.")

    except Exception as e:
         print(f"Error converting analysis results dict to DataFrame for PDF: {e}")
         traceback.print_exc()
         # Build a minimal PDF with error message
         story = [Paragraph("<b>Error:</b> Could not process analysis results for PDF.", styles['h2'])]
         try: doc.build(story)
         except Exception: pass # Avoid error loop if build fails here too
         return False # Indicate failure

    # --- Executive Summary ---
    story.append(Paragraph("Executive Summary", styles['h2']))
    summary_text = f"The A/B test '{experiment_name}' compared '{treatment_group_name}' against '{control_group_name}'. "
    recommendation = "Analyze Further / No Clear Winner"

    primary_result_series = None
    if primary_kpi and primary_kpi in results_df['metric'].values:
        primary_result_series = results_df[results_df['metric'] == primary_kpi].iloc[0]
    else:
         summary_text += f"Primary KPI '{primary_kpi}' not specified or not found in results. "

    if primary_result_series is not None:
        lift_prim = primary_result_series.get('lift (%)')
        pval_prim = primary_result_series.get('p_value')
        sig_prim = primary_result_series.get('significant', False)

        lift_prim_str = "N/A"
        if isinstance(lift_prim, (int, float)) and pd.notna(lift_prim) and np.isfinite(lift_prim):
             lift_prim_str = f"{lift_prim:+.2f}%"

        pval_prim_str = "N/A"
        if pval_prim is not None and pd.notna(pval_prim):
             pval_prim_str = "<0.001" if pval_prim < 0.001 else f"{pval_prim:.3f}"

        summary_text += f"The primary KPI, <b>'{primary_kpi}'</b>, showed a relative lift of <b>{lift_prim_str}</b> for the Treatment group. "
        if sig_prim:
            summary_text += f"This result is <b>statistically significant</b> (p={pval_prim_str}). "
            if isinstance(lift_prim, (int, float)) and pd.notna(lift_prim) and np.isfinite(lift_prim):
                if lift_prim > 1.0: recommendation = "Recommend Rollout" # Example threshold
                elif lift_prim > -1.0: recommendation = "Monitor / Iterate (Slight Change)"
                else: recommendation = "Recommend Abort (Negative Impact)"
            else: recommendation = "Analyze Further (Significance/Lift Unclear)"
        else:
            recommendation = "Iterate / Monitor (Primary KPI Not Significant)"
            summary_text += f"This result is <b>not statistically significant</b> (p={pval_prim_str}). "
    else: # Base recommendation on secondary metrics if primary absent
        significant_results = results_df[results_df['significant'] == True]
        if not significant_results.empty:
            positive_lifts = significant_results['lift (%)'].apply(lambda x: isinstance(x, (int, float)) and pd.notna(x) and np.isfinite(x) and x > 0)
            if positive_lifts.any(): recommendation = "Consider Rollout (Positive Secondary KPIs)"
            else: recommendation = "Abort / Iterate (No Positive Wins)"
        else: recommendation = "Iterate / Abort (No Significant Wins)"

    # Mention other significant results
    significant_others = results_df[(results_df['significant'] == True) & (results_df['metric'] != primary_kpi)]
    if not significant_others.empty:
        summary_text += "Other statistically significant changes observed: "
        sig_list = []
        for i, r in significant_others.iterrows():
             lift_val = r.get('lift (%)')
             lift_fmt = f"{lift_val:+.2f}%" if isinstance(lift_val, (int, float)) and pd.notna(lift_val) and np.isfinite(lift_val) else "N/A"
             sig_list.append(f"'{r.get('metric','Unknown')}' ({lift_fmt})")
        summary_text += ", ".join(sig_list) + ". "

    summary_text += f"<br/><br/><b>Overall Recommendation: {recommendation}</b>"
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # --- Funnel Chart (Conditional based on raw_data and columns) ---
    # Check if raw_data is provided AND necessary columns exist
    funnel_cols_exist = False
    if raw_data is not None and not raw_data.empty:
         # Define expected cols here again or pass from config/params
         expected_funnel_cols = ['views', 'clicks', 'conversions'] # Example
         if all(col in raw_data.columns for col in expected_funnel_cols):
              funnel_cols_exist = True

    if funnel_cols_exist:
        story.append(Paragraph("Funnel Performance", styles['h2']))
        funnel_img_or_msg = _create_funnel_chart(
            raw_data, group_col=config.GROUP_COLUMN, # Use config group col
            control_val=control_group_name, treatment_val=treatment_group_name
        )
        if funnel_img_or_msg: story.append(funnel_img_or_msg)
        story.append(Spacer(1, 0.1*inch))
    else:
         print("Skipping funnel chart in PDF: Raw data not provided or required columns missing.")


    # --- KPI Summary Table ---
    story.append(Paragraph("Key Performance Indicators (KPI) Summary", styles['h2']))
    story.append(Spacer(1, 0.1*inch))
    summary_table_obj = _create_summary_table(results_df, primary_kpi)
    if summary_table_obj: story.append(summary_table_obj)
    story.append(Spacer(1, 0.2*inch))

    # --- Individual KPI Plots ---
    story.append(Paragraph("KPI Details", styles['h2']))
    story.append(Spacer(1, 0.1*inch))
    if results_df is not None and not results_df.empty:
        plot_style = styles['Normal']
        plot_style.fontSize = 8
        for index, row_data in results_df.iterrows():
            plot_title = Paragraph(f"<b>{row_data.get('metric', 'Unknown Metric')}</b>", styles['h3'])
            story.append(plot_title)

            plot_img_or_msg = _create_comparison_plot(row_data) # Pass the row Series
            story.append(plot_img_or_msg)

            # Add CI and N info textually below plot if plot was successful
            if isinstance(plot_img_or_msg, Image):
                alpha = row_data.get('alpha', 0.05)
                ci_lower = row_data.get('ci_lower')
                ci_upper = row_data.get('ci_upper')
                ci_text = f"CI ({100*(1-alpha):.0f}%) for Difference: [N/A, N/A]"
                if pd.notna(ci_lower) and pd.notna(ci_upper):
                     prec = 3 if 'mean' in row_data.get('test_type', '').lower() else 4
                     ci_text = f"CI ({100*(1-alpha):.0f}%) for Diff (Treat - Ctrl): [{ci_lower:+.{prec}f}, {ci_upper:+.{prec}f}]"

                n_c = row_data.get('control_n', 'N/A')
                n_t = row_data.get('treatment_n', 'N/A')
                n_text = f"N ({control_group_name}/{treatment_group_name}): {n_c} / {n_t}"

                story.append(Paragraph(ci_text, plot_style))
                story.append(Paragraph(n_text, plot_style))

            story.append(Spacer(1, 0.15*inch))
    else:
        story.append(Paragraph("No metric details to display.", styles['Normal']))


    # --- Build PDF ---
    print("Building PDF document...")
    try:
        doc.build(story)
        # If filename was a buffer, the caller handles it. If it was a string, file is saved.
        print(f"Report build process complete for: {filename}")
        return True # Indicate success
    except Exception as e:
        print(f"FATAL ERROR building PDF: {e}")
        traceback.print_exc()
        return False # Indicate failure


# Example Usage - Updated for marketing_AB.csv via config
if __name__ == "__main__":
    print("\n" + "="*30)
    print("Running PDF Generator Example with Marketing Data")
    print("="*30)

    try:
        # --- Prerequisites ---
        if not all(hasattr(config, attr) for attr in ['INPUT_DATA_FILE', 'GROUP_COLUMN', 'CONTROL_GROUP_VALUE', 'TREATMENT_GROUP_VALUE', 'CONVERSION_COLUMN', 'CONTINUOUS_METRICS', 'ALPHA', 'REPORT_FILENAME']):
             raise ValueError("Config error: One or more required config variables missing.")

        # --- Load data using config path ---
        data_file_path = config.INPUT_DATA_FILE
        print(f"Loading data from: {data_file_path}")
        if not os.path.exists(data_file_path):
             raise FileNotFoundError(f"Data file not found: {data_file_path}.")
        df = pd.read_csv(data_file_path)
        print(f"Data loaded: {len(df)} rows.")

        # --- Run Analysis ---
        print("Running statistical analysis...")
        from statistical_analyzer import ABAnalyzer # Import here for example scope
        analyzer = ABAnalyzer(df,
                              group_col=config.GROUP_COLUMN,
                              control_val=config.CONTROL_GROUP_VALUE,
                              treatment_val=config.TREATMENT_GROUP_VALUE)

        results = analyzer.run_analysis(
            conversion_metric=config.CONVERSION_COLUMN,
            continuous_metrics=config.CONTINUOUS_METRICS,
            alpha=config.ALPHA
        )

        if not results:
             print("Analysis returned no results. PDF cannot be generated with details.")
        else:
             print("Analysis complete. Proceeding with PDF generation...")
             # --- Define PDF Parameters ---
             report_filename_example = config.REPORT_FILENAME.replace(".pdf", "_Example.pdf")
             primary_kpi_name = f"{config.CONVERSION_COLUMN} Rate" # Construct expected metric name
             if primary_kpi_name not in results:
                  print(f"Warning: Primary KPI '{primary_kpi_name}' not in results. Check metric naming.")
                  primary_kpi_name = next(iter(results), None) # Fallback

             # --- Call PDF Generation ---
             success = generate_pdf_report(
                 filename=report_filename_example, # Pass string filename for example
                 experiment_name=config.REPORT_TITLE.replace(" Report", ""), # Use title from config
                 run_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 analysis_results=results,
                 raw_data=df, # Pass raw data if funnel cols were added to CSV
                 primary_kpi=primary_kpi_name,
                 control_group_name=config.CONTROL_GROUP_VALUE,
                 treatment_group_name=config.TREATMENT_GROUP_VALUE
             )
             if success: print(f"\nPDF generation successful: {report_filename_example}")
             else: print(f"\nPDF generation FAILED for {report_filename_example}.")

    # --- Error Handling ---
    except FileNotFoundError as e: print(f"\n--- ERROR: File Not Found ---\n{e}")
    except ImportError as e: print(f"\n--- ERROR: Missing Library ---\n{e}\nSuggestion: pip install reportlab pandas numpy scipy statsmodels matplotlib seaborn plotly kaleido")
    except KeyError as e: print(f"\n--- ERROR: Column Not Found (KeyError) ---\nColumn {e} not found.")
    except ValueError as e: print(f"\n--- ERROR: Data/Config Problem (ValueError) ---\n{e}")
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred in PDF Example ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        traceback.print_exc()