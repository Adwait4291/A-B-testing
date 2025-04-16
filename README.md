# A/B Testing Analysis Framework

## Project Introduction

This repository contains a framework designed for the analysis of A/B testing results. A/B testing is a methodology used to compare two variants (denoted 'A' and 'B') of a component, such as a webpage or application feature, to ascertain which variant yields superior performance concerning a defined metric.

This framework facilitates the ingestion of user data from an A/B test, executes statistical analysis to compare the performance differential between the control ('A') and treatment ('B') groups, and presents the findings via an interactive web-based dashboard.

## Background: A/B Testing Principles

The analysis performed by this framework adheres to standard A/B testing principles:

1.  **Hypothesis Testing:** The core objective is to evaluate a hypothesis, typically regarding the relative effectiveness of the treatment variant compared to the control (e.g., "Variant B results in a statistically significant increase in the application completion rate relative to Variant A").
2.  **Group Allocation:** User cohorts are segregated into distinct groups:
    * **Control Group (Variant 'A'):** Exposed to the baseline version.
    * **Treatment Group (Variant 'B'):** Exposed to the modified version under evaluation.
3.  **Metric Definition:** Performance is quantified using specific metrics. This project primarily utilizes:
    * **Conversion Rate:** Derived from a binary outcome variable (e.g., `ApplicationCompleted` [cite: 2]), representing the proportion of users completing a target action.
    * **Continuous Metrics:** Numerical variables (e.g., `SessionDuration_seconds`[cite: 2], although currently unconfigured for analysis in the dashboard) for which mean values are compared between groups.
4.  **Statistical Significance:** Analysis aims to determine if observed differences in metrics are statistically significant or likely attributable to random variation. Significance is assessed using statistical tests (Z-test, T-test) against a predetermined significance level (alpha). A p-value below alpha indicates statistical significance.
5.  **Performance Lift:** Quantifies the relative percentage change in the metric for the treatment group compared to the control group.

## Project Objectives

The development of this framework serves several purposes:

* To provide an automated and reusable tool for efficient analysis of A/B test data.
* To ensure statistical validity in the comparison of variants using appropriate methodologies (Z-test for proportions, T-test for means).
* To facilitate comprehension of results through clear data visualization within an interactive dashboard.
* To offer a practical implementation example of an A/B testing analysis workflow.

## Core Features

* Data ingestion from CSV-formatted A/B test results.
* Implementation of statistical comparison tests:
    * **Two-Proportion Z-test:** For comparing binary conversion rates.
    * **Welch's T-test (default):** For comparing means of continuous variables.
* Calculation of standard A/B testing metrics including conversion rates, means, percentage lift, and p-values.
* Assessment of statistical significance based on a defined alpha level.
* Generation of an interactive web dashboard using Streamlit for results visualization.
* Integration of Plotly for generating comparative plots (bar charts, box plots).

## Dataset Description

The analysis utilizes the `finance_ab.csv` dataset[cite: 2]. Relevant columns include:

* `UserID`: Unique user identifier.
* `Version`: Group assignment ('A' or 'B').
* `ApplicationCompleted`: Binary conversion outcome[cite: 2].
* Additional demographic or behavioral columns may be present but usage depends on configuration.

## Methodology

1.  **Data Loading:** The `dashboard.py` script utilizes the Pandas library to load the dataset specified in `finance_ab.csv`.
2.  **Configuration:** Parameters defining the experiment (group column, control/treatment identifiers, metric columns, alpha level) are specified (currently hardcoded within `dashboard.py`). The `config.py` file provides a template for external configuration.
3.  **Statistical Analysis:** The `SimpleABAnalyzer` class (`statistical_analyzer.py`) performs the core analysis:
    * Data segmentation into control and treatment datasets based on the `Version` column.
    * Calculation and comparison of conversion rates using the Z-test.
    * Calculation and comparison of means for specified continuous metrics using the T-test.
    * Computation of p-values and determination of statistical significance relative to the configured alpha.
    * Calculation of the percentage lift for each metric.
4.  **Results Visualization:** The `dashboard.py` script employs Streamlit to render the analysis results:
    * Summarized metrics are displayed using `st.metric`.
    * Comparative visualizations are generated using Plotly.
    * A data preview option is provided.

## Installation and Execution

1.  **Clone Repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Environment Setup (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Linux/macOS
    # venv\Scripts\activate  # For Windows
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Data File:** Ensure the `finance_ab.csv` file is located within the project's root directory[cite: 2].
5.  **Launch Dashboard:**
    ```bash
    streamlit run dashboard.py
    ```
6.  Initiate the analysis by activating the "Run Analysis" control within the dashboard interface.

## Configuration Parameters

Key parameters for configuring the analysis (defined in `config.py` or hardcoded in `dashboard.py`) include:

* `INPUT_DATA_FILE`: Path to the input data file.
* `GROUP_COLUMN`: Name of the column containing group assignments.
* `CONTROL_GROUP_VALUE`: Identifier for the control group.
* `TREATMENT_GROUP_VALUE`: Identifier for the treatment group.
* `CONVERSION_COLUMN`: Name of the binary conversion metric column.
* `CONTINUOUS_METRICS`: A list containing names of continuous metric columns for analysis.
* `ALPHA`: The significance level threshold for hypothesis testing.

## Repository Contents

* `finance_ab.csv`: Sample A/B test dataset[cite: 2].
* `dashboard.py`: Streamlit application script.
* `statistical_analyzer.py`: Module containing the `SimpleABAnalyzer` class.
* `config.py`: Configuration file template.
* `requirements.txt`: List of Python package dependencies[cite: 1].
* `README.md`: This document.

## Utilized Libraries

* Pandas
* NumPy
* SciPy
* Statsmodels
* Plotly
* Streamlit

## Contributions

Contributions to enhance this framework are welcome. Please refer to standard GitHub practices for pull requests or issue reporting.

## License

(Optional: Specify project license, e.g., MIT License)
