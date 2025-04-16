# A/B Testing Analysis Framework

## Overview & Purpose

This project provides a Python-based framework for the rigorous analysis of A/B test results. Its core purpose is to move beyond simple metric comparisons and apply sound statistical methods to determine if observed differences between a Control group ('A') and a Treatment group ('B') are statistically significant and practically meaningful.

It automates the workflow from data loading and statistical testing to interactive visualization, enabling data-driven decisions grounded in reliable evidence. Built for analysts and teams who need to quickly understand the true impact of their experiments.

## Core Concepts Applied

This framework implements key A/B testing principles:

1.  **Hypothesis Testing:** Formally testing if an observed effect (e.g., difference in conversion rates) is likely real or due to random chance.
2.  **Control vs. Treatment:** Comparing the performance of a new version ('B') against a baseline ('A').
3.  **Key Metrics:** Focusing on quantifiable outcomes:
    * **Conversion Rate:** Proportion of users completing a binary action (e.g., `ApplicationCompleted` - True/False). Analyzed using appropriate proportion tests.
    * **Continuous Metrics:** Average value of numerical measurements (e.g., `SessionDuration_seconds`). Analyzed using tests comparing means.
4.  **Statistical Significance:** Using p-values derived from statistical tests compared against a significance level (alpha, typically 0.05) to determine the likelihood that an observed difference is not just random noise.
5.  **Lift:** Quantifying the percentage change in a metric for the treatment group relative to the control, indicating the magnitude of the impact.

## Features & Methodology

* **Data Loading:** Leverages Pandas for efficient loading of A/B test data from CSV files (like `finance_ab.csv`).
* **Statistical Analysis Engine (`statistical_analyzer.py` [cite: 1]):**
    * **Z-test for Proportions:** Correctly applied for comparing binary conversion rates between the two groups. This is the standard test for this type of data.
        ```python
        # Example snippet from statistical_analyzer.py [cite: 1]
        from statsmodels.stats.proportion import proportions_ztest
        # ... inside analyze_conversion_rate method ...
        count = np.array([treatment_successes, control_successes])
        nobs = np.array([treatment_total, control_total])
        z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
        ```
    * **Welch's T-test (Default):** Used for comparing the means of continuous metrics. **Rationale:** Welch's T-test is chosen over the standard Student's T-test because it does not assume equal variances between the groups, making it more robust for real-world data where variances often differ.
        ```python
        # Example snippet from statistical_analyzer.py [cite: 1]
        from statsmodels.stats.weightstats import ttest_ind
        # ... inside analyze_continuous_metric method ...
        t_stat, p_value, _ = ttest_ind(treatment_data, control_data,
                                      alternative='two-sided', usevar='unequal') # 'unequal' implies Welch's T-test
        ```
    * **Robust Calculations:** Includes checks for edge cases (e.g., zero variance, empty groups) to ensure stability[cite: 1].
* **KPI Calculation:** Computes essential metrics: rates/means, absolute difference, percentage lift, p-values, and significance flags.
* **Interactive Dashboard (`dashboard.py`):**
    * **Technology:** Built with Streamlit for rapid development of interactive web UIs from Python. Plotly is used for rich, interactive charting capabilities.
    * **Clear Results Display:** Presents key summary statistics upfront using `st.metric` for quick insights, highlighting significant results visually.
        ```python
        # Example snippet from dashboard.py
        st.metric(
            label=metric,
            value=treat_fmt, # Formatted treatment value
            delta=f"{lift_fmt} {pval_fmt}", # Lift and p-value
            delta_color=delta_color, # 'normal' or 'inverse' if significant
            help=f"Control: {control_fmt} | N(ctrl): {n_control}, N(treat): {n_treatment}"
        )
        ```
    * **Detailed Exploration:** Uses tabs and plots (bar charts, box plots) for deeper dives into each metric's performance and distribution across groups.

## Dataset

The analysis uses the `finance_ab.csv` dataset, which includes:

* `UserID`: Unique identifier.
* `Version`: Group assignment ('A' or 'B').
* `ApplicationCompleted`: Primary binary conversion metric.
* *(Other demographic/behavioral columns)*

## How It Works: Analysis Flow

1.  **Load:** The Streamlit app (`dashboard.py`) loads data from `finance_ab.csv` using Pandas.
2.  **Configure:** Experiment parameters (group column, metrics, alpha) are identified (currently hardcoded in `dashboard.py`).
3.  **Analyze:** An instance of `SimpleABAnalyzer` [cite: 1] is created and its `run_analysis` method is called. This performs the Z-tests and T-tests as described above.
4.  **Visualize:** `dashboard.py` takes the analysis results and renders them using Streamlit components (`st.metric`, `st.tabs`, etc.) and Plotly charts.

## Setup and Usage

1.  **Clone Repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt [cite: 2]
    ```
4.  **Data File:** Ensure `finance_ab.csv` is in the project directory.
5.  **Run Dashboard:**
    ```bash
    streamlit run dashboard.py
    ```
6.  Interact with the dashboard and click "Run Analysis".

## Configuration Notes

While `dashboard.py` currently uses hardcoded analysis parameters for simplicity, the `config.py` file provides a template demonstrating how configuration could be externalized for greater flexibility (e.g., easily changing target metrics or data files without modifying the main script).

## Key Files

* `finance_ab.csv`: Sample data.
* `dashboard.py`: Main application script (UI and orchestration).
* `statistical_analyzer.py`[cite: 1]: Core statistical computation class.
* `requirements.txt`[cite: 2]: Project dependencies.

## Core Libraries Used [cite: 2]

* **Pandas:** Data manipulation.
* **Statsmodels:** Statistical tests (Z-test, T-test).
* **Plotly:** Interactive visualizations.
* **Streamlit:** Web application framework.
* **NumPy:** Numerical computation foundation.

## Contributing

Contributions aimed at enhancing the framework's capabilities or robustness are welcome via pull requests or issue reports.

## License

*(Optional: Specify license, e.g., MIT License)*
