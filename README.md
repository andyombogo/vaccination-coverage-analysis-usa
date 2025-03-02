# Vaccination Coverage Analysis Among Pregnant Women in the US

This project analyzes vaccination coverage among pregnant women in the United States using data from the [CDC's Pregnancy Vaccination Coverage dataset](https://data.cdc.gov/Pregnancy-Vaccination/Vaccination-Coverage-among-Pregnant-Women/h7pm-wmjc/about_data). The analysis is performed using PySpark for data processing, exploratory data analysis (EDA), and machine learning.

## Project Overview

The goal of this project is to:
1. Load and clean the vaccination coverage data.
2. Perform exploratory data analysis (EDA) to understand the data and identify patterns.
3. Demonstrate machine learning techniques to predict vaccination coverage.

## Data Source

The data used in this project is sourced from the CDC's Pregnancy Vaccination Coverage dataset, which can be found [here](https://data.cdc.gov/Pregnancy-Vaccination/Vaccination-Coverage-among-Pregnant-Women/h7pm-wmjc/about_data).

## Technologies Used

- **PySpark**: For data processing and analysis.
- **Matplotlib**: For data visualization.
- **Seaborn**: For enhanced data visualization.
- **Machine Learning**: Using PySpark's MLlib for classification tasks.

## Project Structure

- `project.py`: The main script that performs data loading, cleaning, EDA, and machine learning.
- `requirements.txt`: Lists the Python dependencies required for the project.
- `Procfile`: Specifies the command to run the application on Heroku.
- `runtime.txt`: Specifies the Python runtime version.
- `.gitignore`: Specifies files and directories to be ignored by Git.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/andyombogo/vaccination-coverage-analysis-usa.git
    cd vaccination-coverage-analysis-usa
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the main script:
    ```sh
    python project.py
    ```

2. The script will perform the following tasks:
    - Initialize a Spark session.
    - Load the vaccination coverage data from a CSV file.
    - Clean and preprocess the data.
    - Perform exploratory data analysis (EDA).
    - Plot various visualizations to understand the data.
    - Demonstrate machine learning techniques to predict vaccination coverage.

## Deployment

This project can be deployed to Heroku using the provided [Procfile](http://_vscodecontentref_/0) and [heroku.yml](http://_vscodecontentref_/1) for container-based deployment.

1. Create a new Heroku app:
    ```sh
    heroku create
    ```

2. Push the code to Heroku:
    ```sh
    git push heroku main
    ```

3. Open the app in your browser:
    ```sh
    heroku open
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- The data used in this project is provided by the CDC.
- Special thanks to the open-source community for providing the tools and libraries used in this project.
