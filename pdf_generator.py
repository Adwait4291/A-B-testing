# pdf_generator.py
"""Generates a PDF report for A/B test results using ReportLab."""

import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib import colors
from reportlab.lib.units import inch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go # For funnel chart
import numpy as np

# Configure matplotlib/seaborn for non-interactive backend suitable for reportlab
import matplotlib
matplotlib.use('Agg')

def _create_summary_table(results_df, primary_kpi=None):
    """Creates a formatted summary table for the PDF."""
    # Select and format columns
    cols_to_show = ['Metric', 'Control Rate/Mean', 'Treatment Rate/Mean', 'Lift (%)', 'P-value', 'Significant']
    data = [['Metric', 'Control', 'Treatment', 'Lift (%)', 'P-value', 'Significant']]

    # Style setup
    header_style = ('BACKGROUND', (0, 0), (-1, 0), colors.grey)
    table_style = [header_style,
                   ('GRID', (0, 0), (-1, -1), 1, colors.black),
                   ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                   ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                   ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')]

    row_idx = 1
    for index, row in results_df.iterrows():
        metric = row['metric']
        is_primary = primary_kpi is not None and metric == primary_kpi
        font_name = 'Helvetica-Bold' if is_primary else 'Helvetica'

        # Format numbers nicely
        control_val = f"{row['control_rate']:.2%}" if 'rate' in row and pd.notna(row['control_rate']) else f"{row['control_mean']:.2f}" if 'mean' in row and pd.notna(row['control_mean']) else 'N/A'
        treat_val = f"{row['treatment_rate']:.2%}" if 'rate' in row and pd.notna(row['treatment_rate']) else f"{row['treatment_mean']:.2f}" if 'mean' in row and pd.notna(row['treatment_mean']) else 'N/A'

        lift = row.get('lift (%)', 'N/A')
        lift_str = f"{lift:.2f}%" if isinstance(lift, (int, float)) and pd.notna(lift) else str(lift)

        pval = row['p_value']
        pval_str = f"{pval:.3f}" if pd.notna(pval) else 'N/A'
        if pd.notna(pval) and pval < 0.001:
            pval_str = "<0.001"

        sig = 'Yes' if row['significant'] else 'No'
        sig_color = colors.lightgreen if row['significant'] else colors.pink

        data.append([
            Paragraph(metric, getSampleStyleSheet()['Normal']), # Wrap long metric names
            control_val,
            treat_val,
            lift_str,
            pval_str,
            sig
        ])

        # Apply significance color and bold primary KPI
        table_style.append(('BACKGROUND', (5, row_idx), (5, row_idx), sig_color))
        if is_primary:
             table_style.append(('FONTNAME', (0, row_idx), (0, row_idx), 'Helvetica-Bold'))


        row_idx += 1

    # Adjust column widths (example widths, adjust as needed)
    col_widths = [2.5*inch, 1*inch, 1*inch, 1*inch, 0.8*inch, 0.7*inch]

    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle(table_style))
    return table

def _create_comparison_plot(results_df, metric_row):
    """Creates a bar plot comparing control vs treatment for a metric."""
    fig, ax = plt.subplots(figsize=(6, 3))

    metric_type = 'rate' if 'rate' in metric_row else 'mean'
    control_val = metric_row[f'control_{metric_type}']
    treat_val = metric_row[f'treatment_{metric_type}']
    metric_name = metric_row['metric']
    is_rate = metric_type == 'rate'

    sns.barplot(x=['Control', 'Treatment'], y=[control_val, treat_val], ax=ax, palette="viridis")

    ax.set_title(f"{metric_name} Comparison")
    ax.set_ylabel("Rate" if is_rate else "Mean Value")
    if is_rate:
        ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.1%}'.format)) # Format y-axis as percentage

    # Add value labels
    for container in ax.containers:
        labels = [f"{v:.2%}" if is_rate else f"{v:.2f}" for v in container.datavalues]
        ax.bar_label(container, labels=labels, label_type='edge')

    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150)
    plt.close(fig)
    img_buffer.seek(0)
    return Image(img_buffer, width=4*inch, height=2*inch) # Adjust size as needed

def _create_funnel_chart(df, group_col='group', control_val='Control', treatment_val='Treatment'):
    """Creates a funnel chart comparing the two groups."""
    control_df = df[df[group_col] == control_val]
    treatment_df = df[df[group_col] == treatment_val]

    steps = ['views', 'clicks', 'applications_started', 'applications_completed']
    control_counts = [control_df[step].sum() for step in steps]
    treatment_counts = [treatment_df[step].sum() for step in steps]

    fig = go.Figure()

    fig.add_trace(go.Funnel(
        name = control_val,
        y = [s.replace('_', ' ').title() for s in steps],
        x = control_counts,
        textinfo = "value+percent initial"))

    fig.add_trace(go.Funnel(
        name = treatment_val,
        y = [s.replace('_', ' ').title() for s in steps],
        x = treatment_counts,
        textinfo = "value+percent initial"))

    fig.update_layout(
        title="Acquisition Funnel Comparison",
        funnelmode="stack" # Or 'group'
    )

    img_buffer = io.BytesIO()
    fig.write_image(img_buffer, format='png', scale=2) # Increase scale for better resolution
    img_buffer.seek(0)
    return Image(img_buffer, width=6*inch, height=3.5*inch) # Adjust size

def generate_pdf_report(
    filename="ab_test_report.pdf",
    experiment_name="Unnamed Experiment",
    run_date="N/A",
    analysis_results=None,
    raw_data=None, # Pass the filtered DataFrame for funnel chart
    primary_kpi=None, # e.g., 'Overall Conversion Rate (from Views)'
    recommendation="Analyze Further" # Default recommendation
    ):
    """
    Generates a PDF report summarizing the A/B test results.

    Args:
        filename (str): Path to save the PDF file.
        experiment_name (str): Name of the experiment.
        run_date (str): Date the analysis was run.
        analysis_results (dict): Dictionary of results from ABAnalyzer.
        raw_data (pd.DataFrame): Filtered data used for analysis (needed for funnel).
        primary_kpi (str): Name of the primary KPI to highlight.
        recommendation (str): Suggested action ('Rollout', 'Iterate', 'Abort', etc.)
    """
    if analysis_results is None or not analysis_results:
        print("No analysis results provided for PDF generation.")
        return

    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    story = []

    # --- Title and Metadata ---
    story.append(Paragraph(f"A/B Test Report: {experiment_name}", styles['h1']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Analysis Run Date: {run_date}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))

    # --- Executive Summary ---
    story.append(Paragraph("Executive Summary", styles['h2']))
    summary_text = f"This report details the results of the '{experiment_name}' A/B test. "
    # Convert results dict to DataFrame for easier handling
    results_df = pd.DataFrame.from_dict(analysis_results, orient='index').reset_index(drop=True)
    results_df.rename(columns={'metric': 'Metric'}, inplace=True) # Ensure consistent naming

    significant_results = results_df[results_df['significant'] == True]
    primary_result = results_df[results_df['Metric'] == primary_kpi].iloc[0] if primary_kpi and not results_df[results_df['Metric'] == primary_kpi].empty else None

    if primary_result is not None:
         lift_prim = primary_result.get('lift (%)', 0)
         lift_prim_str = f"{lift_prim:.2f}%" if isinstance(lift_prim, (int, float)) else "N/A"
         pval_prim = primary_result['p_value']
         pval_prim_str = f"{pval_prim:.3f}" if pval_prim else "N/A"
         sig_prim = primary_result['significant']

         summary_text += f"The primary KPI, <b>'{primary_kpi}'</b>, showed a lift of <b>{lift_prim_str}</b> for the Treatment group. "
         if sig_prim:
             summary_text += f"This result is <b>statistically significant</b> (p={pval_prim_str}). "
             # Determine Recommendation based on primary KPI significance and lift
             if lift_prim > 0:
                 recommendation = "Rollout"
             elif lift_prim < 0:
                  recommendation = "Abort"
             else: # Significant but zero lift? Unlikely but possible
                  recommendation = "Iterate" # Or Abort
         else:
             recommendation = "Iterate / Monitor" # If primary not significant
             summary_text += f"This result is <b>not statistically significant</b> (p={pval_prim_str}). "

    else:
        summary_text += "No primary KPI was specified or found. "
        # Basic recommendation if no primary KPI
        if not significant_results.empty:
             positive_sig = significant_results[significant_results['lift (%)'] > 0]
             if not positive_sig.empty: recommendation = "Consider Rollout (Positive Secondary KPIs)"
             else: recommendation = "Abort / Iterate (Negative/No Positive Secondary KPIs)"
        else:
             recommendation = "Iterate / Abort (No Significant Wins)"


    if not significant_results.empty and primary_result is None: # Mention other significant results if no primary
        summary_text += "Other statistically significant changes were observed in: "
        summary_text += ", ".join([f"'{r['Metric']}' ({r['lift (%)']:.2f}%)" for i, r in significant_results.iterrows()]) + ". "

    summary_text += f"<br/><br/><b>Recommendation: {recommendation}</b>"
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))


    # --- Funnel Chart (if data provided) ---
    if raw_data is not None and not raw_data.empty:
        story.append(Paragraph("Funnel Performance", styles['h2']))
        try:
            funnel_img = _create_funnel_chart(raw_data, group_col='group', control_val='Control', treatment_val='Treatment')
            story.append(funnel_img)
            story.append(Spacer(1, 0.2*inch))
        except Exception as e:
            print(f"Error creating funnel chart: {e}")
            story.append(Paragraph(f"<i>Error creating funnel chart: {e}</i>", styles['Italic']))


    # --- KPI Summary Table ---
    story.append(Paragraph("Key Performance Indicators (KPI) Summary", styles['h2']))
    story.append(Spacer(1, 0.1*inch))
    summary_table = _create_summary_table(results_df, primary_kpi)
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))

    # --- Individual KPI Plots ---
    story.append(Paragraph("KPI Details", styles['h2']))
    for index, row_data in results_df.iterrows():
         story.append(Spacer(1, 0.1*inch))
         try:
             plot_img = _create_comparison_plot(results_df, row_data)
             story.append(plot_img)
         except Exception as e:
             print(f"Error creating plot for {row_data['Metric']}: {e}")
             story.append(Paragraph(f"<i>Error creating plot for {row_data['Metric']}: {e}</i>", styles['Italic']))
         # Add CI and N info textually
         ci_text = f"CI ({100*(1-row_data['alpha']):.0f}%) for Difference: [{row_data['ci_lower']:.3f}, {row_data['ci_upper']:.3f}]"
         n_text = f"N (Control/Treatment): {row_data['control_n']}/{row_data['treatment_n']}"
         story.append(Paragraph(ci_text, styles['Normal']))
         story.append(Paragraph(n_text, styles['Normal']))
         story.append(Spacer(1, 0.2*inch))


    # --- Build PDF ---
    try:
        doc.build(story)
        print(f"Report successfully generated: {filename}")
    except Exception as e:
        print(f"Error building PDF: {e}")


# Example Usage (requires results from analyzer)
if __name__ == "__main__":
    try:
        # Load data and results (assuming they exist from previous steps)
        df = pd.read_csv("simulated_ab_data.csv")
        analyzer = ABAnalyzer(df)
        results = analyzer.run_analysis(
            conversion_metrics=[
                ('clicks', 'views'),
                ('applications_started', 'clicks'),
                ('applications_completed', 'applications_started'),
                ('applications_completed', 'views')
            ],
            continuous_metrics=['time_on_page_seconds', 'bounced']
        )
        from datetime import datetime
        generate_pdf_report(
            filename="Test_Report_Example.pdf",
            experiment_name="Homepage Button Color Test",
            run_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            analysis_results=results,
            raw_data=df, # Pass the full dataframe for the funnel chart
            primary_kpi="applications_completed / views", # Example primary KPI
            # Recommendation will be auto-determined or use default
        )
    except FileNotFoundError:
        print("Error: simulated_ab_data.csv not found. Run data_simulator.py first.")
    except ImportError:
         print("Error: Required libraries (reportlab, matplotlib, seaborn, plotly) not installed.")
    except Exception as e:
        print(f"An error occurred during PDF generation example: {e}")