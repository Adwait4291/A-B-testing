# A/B Testing Analysis Framework

## Overview

This project provides a framework for analyzing the results of A/B tests. A/B testing is a method of comparing two versions (A and B) of something (like a webpage, app feature, or marketing email) to determine which one performs better based on a specific metric (e.g., conversion rate, click-through rate, session duration).

This framework takes user data from an A/B test, performs statistical analysis to compare the performance of the two versions (Control 'A' vs. Treatment 'B'), and presents the results in an interactive web dashboard.

## Core Concepts of A/B Testing Applied Here

1.  **Hypothesis Testing:** The fundamental goal is to test a hypothesis. For example, "Does version B lead to a significantly higher application completion rate than version A?".
2.  **Control vs. Treatment:** Users are typically divided into two groups:
    * **Control (Group 'A' in this project):** Shown the original version.
    * **Treatment (Group 'B' in this project):** Shown the new version being tested.
3.  **Key Metrics:** Performance is measured using specific metrics. This project focuses on:
    * **Conversion Rate:** Calculated from a binary outcome (e.g., `ApplicationCompleted` - True/False). It measures the proportion of users who took the desired action.
    * **Continuous Metrics:** Numerical measurements (e.g., `SessionDuration_seconds`, though currently disabled in the dashboard config) where averages can be compared.
4.  **Statistical Significance:** It's not enough for metrics to be different; we need to know if the difference is likely real or just due to random chance. This is determined using statistical tests (Z-test, T-test) and a pre-defined significance level (alpha, typically 5%). A low p-value (less than alpha) suggests the difference is statistically significant.
5.  **Lift:** This quantifies the percentage improvement (or decline) of the treatment group's metric compared to the control group's metric.

## Why This Project?

This project was built to:

* **Automate A/B Test Analysis:** Provide a reusable tool to quickly analyze standard A/B test results without manual calculations.
* **Apply Statistical Rigor:** Ensure that conclusions drawn from the test are statistically sound by using appropriate tests (Z-test for proportions, T-test for means).
* **Visualize Results:** Make the results easy to understand for stakeholders through an interactive dashboard with clear metrics and charts.
* **Demonstrate A/B Testing Workflow:** Serve as a practical example of implementing an A/B testing analysis pipeline.

## Features

* Loads A/B test data from a CSV file.
* Performs statistical analysis:
    * **Z-test for Proportions:** Compares conversion rates between groups.
    * **Welch's T-test for Means (default):** Compares the average of continuous metrics between groups.
* Calculates key performance indicators like conversion rates, means, lift, and p-values.
* Determines statistical significance based on the chosen alpha level.
* Provides an interactive web dashboard built with Streamlit to visualize results.
* Includes plots (bar charts, box plots) for easy comparison using Plotly.

## Dataset

The analysis uses the `finance_ab.csv` dataset[cite: 1]. Key columns used in the analysis include:

* `UserID`: Unique identifier for each user.
* `Version`: Identifies the group ('A' for Control, 'B' for Treatment).
* `ApplicationCompleted`: The primary binary conversion metric (True/False).
* *(Other columns like `Age`, `IncomeBracket`, `SessionDuration_seconds` are present but may or may not be used depending on the configuration)*.

## How It Works

1.  **Data Loading:** The `dashboard.py` script loads the `finance_ab.csv` data using pandas.
2.  **Configuration:** Key parameters like the group column (`Version`), control/treatment values ('A'/'B'), conversion metric (`ApplicationCompleted`), continuous metrics to analyze, and the significance level (`ALPHA`) are defined (currently hardcoded in `dashboard.py`).
3.  **Analysis:** The `SimpleABAnalyzer` class in `statistical_analyzer.py` is used:
    * It separates the data into control and treatment groups.
    * It calculates the conversion rate for the specified `conversion_metric` and performs a Z-test to compare the two groups.
    * For each specified `continuous_metric`, it calculates the mean for each group and performs a T-test.
    * P-values are calculated to determine if observed differences are statistically significant.
    * Lift (percentage change) is calculated.
4.  **Visualization:** The `dashboard.py` script uses Streamlit to create a web interface:
    * Displays key metrics (rates/means, lift, p-value) using `st.metric`.
    * Shows comparison plots (bar charts for rates/means, box plots for distributions) using Plotly.
    * Allows users to view a sample of the raw data.

## Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Ensure `finance_ab.csv` is in the same directory as `dashboard.py`.**
5.  **Run the Streamlit dashboard:**
    ```bash
    streamlit run dashboard.py
    ```
6.  Click the "Run Analysis" button in the dashboard to see the results.

## Configuration

While the `dashboard.py` currently uses hardcoded parameters, the `config.py` file exists and demonstrates a pattern for external configuration. Key parameters include:

* `INPUT_DATA_FILE`: Path to the data file.
* `GROUP_COLUMN`: Column name distinguishing groups (e.g., 'Version').
* `CONTROL_GROUP_VALUE`: Value identifying the control group (e.g., 'A').
* `TREATMENT_GROUP_VALUE`: Value identifying the treatment group (e.g., 'B').
* `CONVERSION_COLUMN`: Column name for the binary conversion metric.
* `CONTINUOUS_METRICS`: List of column names for continuous metrics analysis.
* `ALPHA`: Significance level for statistical tests (e.g., 0.05).

## Files

* `finance_ab.csv`[cite: 1]: The raw data for the A/B test.
* `dashboard.py`: The Streamlit application script that runs the analysis and displays the dashboard.
* `statistical_analyzer.py`: Contains the `SimpleABAnalyzer` class responsible for performing the statistical tests.
* `config.py`: (Currently unused by dashboard) Defines configuration parameters.
* `requirements.txt`[cite: 2]: Lists the necessary Python libraries.
* `README.md`: This file.

## Libraries Used [cite: 2]

* Pandas: Data manipulation and loading.
* NumPy: Numerical operations.
* SciPy: Statistical functions (used within statsmodels).
* Statsmodels: Statistical tests (Z-test, T-test).
* Plotly: Creating interactive plots.
* Streamlit: Building the web dashboard.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

(Optional: Add license information here, e.g., MIT License)
