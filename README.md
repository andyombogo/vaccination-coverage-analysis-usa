# Vaccination Coverage Dashboard for Pregnant Women in the US

This repository now ships with a deployment-ready Streamlit dashboard for exploring maternal vaccination coverage in the United States, backed by the [CDC Pregnancy Vaccination Coverage dataset](https://data.cdc.gov/Pregnancy-Vaccination/Vaccination-Coverage-among-Pregnant-Women/h7pm-wmjc/about_data).

## What was improved

- Added a dedicated web dashboard in `dashboard.py` with filters, KPIs, trend charts, geography comparisons, and CSV export.
- Kept the existing Spark scripts for offline analysis instead of forcing heavy PySpark startup in production.
- Replaced the stale Heroku-oriented deployment path with a Render Blueprint in `render.yaml`.
- Updated the Docker image so it starts a real web service that binds to `PORT`.
- Split lightweight deployment dependencies from optional analysis dependencies.

## Why the previous deployment failed

The old deployment files started `python app.py` or `python project.py`. Those scripts run batch analysis logic, but they do not start an HTTP server or bind to the platform-assigned `PORT`. On platforms such as Heroku or Render, that causes the service to fail health checks or time out during port detection.

This repo now deploys the Streamlit dashboard instead:

```sh
streamlit run dashboard.py --server.address 0.0.0.0 --server.port $PORT
```

## Project layout

- `dashboard.py`: Lightweight Streamlit dashboard intended for deployment.
- `app.py`: Spark-based command-line summary script.
- `project.py`: Extended Spark EDA and model-training workflow for local experimentation.
- `requirements.txt`: Minimal dependencies for the deployed dashboard.
- `requirements-analysis.txt`: Optional heavier dependencies for Spark analysis.
- `render.yaml`: Render Blueprint for one-click cloud deployment.
- `Dockerfile`: Containerized deployment option for Docker-compatible platforms.

## Local quick start

1. Clone the repository:

   ```sh
   git clone https://github.com/andyombogo/vaccination-coverage-analysis-usa.git
   cd vaccination-coverage-analysis-usa
   ```

2. Create and activate a virtual environment:

   ```sh
   python -m venv .venv
   .venv\Scripts\activate
   ```

   On macOS or Linux:

   ```sh
   source .venv/bin/activate
   ```

3. Install dashboard dependencies:

   ```sh
   pip install -r requirements.txt
   ```

4. Run the dashboard:

   ```sh
   streamlit run dashboard.py
   ```

5. Open `http://localhost:8501`.

## Optional Spark analysis workflow

If you want the offline PySpark scripts as well:

```sh
pip install -r requirements-analysis.txt
python app.py
python project.py
```

## Deploy on Render

1. Push this repository to GitHub.
2. In Render, choose `New +` and then `Blueprint`.
3. Connect the GitHub repository and select this project.
4. Render will detect `render.yaml` and create the web service automatically.
5. After the build completes, open the generated `.onrender.com` URL.

Render uses this configuration:

```yaml
services:
  - type: web
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run dashboard.py --server.address 0.0.0.0 --server.port $PORT
```

## Deploy with Docker

For any Docker-friendly platform:

```sh
docker build -t vaccination-coverage-dashboard .
docker run -p 8501:8501 vaccination-coverage-dashboard
```

## Data source

- Source: CDC Pregnancy Vaccination Coverage dataset
- File included in repo: `vaccination_data.csv`

## Contributing

Issues and pull requests are welcome. For collaboration or questions, contact `andyombogo@gmail.com`.
