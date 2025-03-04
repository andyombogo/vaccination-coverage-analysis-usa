import logging
import os
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, mean
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_spark():
    """Initialize Spark session."""
    return SparkSession.builder.appName("VaccinationAnalysis").getOrCreate()

def load_data(spark):
    """Load dataset from CSV file."""
    file_path = "vaccination_data.csv"  # Ensure the file is in the root directory
    logger.info(f"Loading data from {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    try:
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        logger.info("Data loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def clean_data(df):
    """Clean and preprocess the data."""
    df = df.withColumn("Estimate (%)", regexp_replace("Estimate (%)", "[^0-9.]", "").cast("float"))
    df = df.withColumn("Sample Size", col("Sample Size").cast("int"))
    df = df.dropna()
    logger.info("Data cleaned successfully")
    return df

def calculate_average_vaccination(df):
    """Calculate mean vaccination rate."""
    avg_vaccination = df.select(mean("Estimate (%)")).collect()[0][0]
    logger.info(f"Average vaccination rate calculated: {avg_vaccination}")
    return avg_vaccination

def prepare_data_for_ml(df):
    """Prepare data for machine learning."""
    df = StringIndexer(inputCol="Vaccine", outputCol="Vaccine_Index").fit(df).transform(df)
    df = StringIndexer(inputCol="Geography", outputCol="Geography_Index").fit(df).transform(df)
    df = df.withColumn("Estimate_Int", col("Estimate (%)").cast("int"))
    assembler = VectorAssembler(inputCols=["Vaccine_Index", "Geography_Index", "Sample Size"], outputCol="features")
    df = assembler.transform(df)
    logger.info("Data prepared for machine learning")
    return df

def train_and_evaluate_model(df):
    """Train and evaluate the machine learning model."""
    train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)
    classifier = RandomForestClassifier(featuresCol="features", labelCol="Estimate_Int", predictionCol="prediction", maxBins=60)
    pipeline = Pipeline(stages=[classifier])
    model = pipeline.fit(train_data)
    model.write().overwrite().save("vaccination_model")
    logger.info("Model training complete")
    return "Model trained and saved successfully!"

# Streamlit app
st.title("Vaccination Coverage Analysis Among Pregnant Women in the US")

# Initialize Spark session
spark = initialize_spark()

# Load data
df = load_data(spark)
if df is None:
    st.error("Failed to load data. Check file path and permissions.")
else:
    # Clean data
    df = clean_data(df)

    # Display data
    st.write("### Data Preview")
    st.write(df.toPandas().head())

    # Calculate and display average vaccination rate
    avg_vaccination = calculate_average_vaccination(df)
    st.write(f"### Average Vaccination Rate: {avg_vaccination:.2f}%")

    # Plot vaccination coverage distribution
    st.write("### Vaccination Coverage Distribution")
    vaccination_rates = [row["Estimate (%)"] for row in df.select("Estimate (%)").collect()]
    plt.figure(figsize=(10, 6))
    sns.histplot(vaccination_rates, kde=True)
    plt.title("Distribution of Vaccination Coverage (%)")
    plt.xlabel("Vaccination Rate (%)")
    plt.ylabel("Frequency")
    st.pyplot(plt)

    # Train and evaluate model
    if st.button("Train Model"):
        df_ml = prepare_data_for_ml(df)
        result = train_and_evaluate_model(df_ml)
        st.success(result)

