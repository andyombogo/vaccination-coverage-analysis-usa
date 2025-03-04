import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan, regexp_replace, mean
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_spark(app_name="VaccinationAnalysis"):
    """Initialize Spark session."""
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_data(spark, file_path):
    """Load dataset from CSV file."""
    try:
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        logger.info("Data loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def clean_data(df):
    """Clean and preprocess the data."""
    df = df.withColumn("Estimate (%)", regexp_replace("Estimate (%)", "[^0-9.]", "").cast("float"))
    df = df.withColumn("Sample Size", col("Sample Size").cast("int"))
    df = df.dropna()
    logger.info("Data cleaned successfully")
    return df

def exploratory_data_analysis(df):
    """Perform exploratory data analysis."""
    for col_name in ["Vaccine", "Geography Type", "Dimension Type"]:
        df.select(col_name).distinct().show()

    vaccination_rates = [row["Estimate (%)"] for row in df.select("Estimate (%)").collect()]
    sns.histplot(vaccination_rates, kde=True)
    plt.title("Distribution of Vaccination Coverage (%)")
    plt.xlabel("Vaccination Rate (%)")
    plt.ylabel("Frequency")
    plt.show()

    geo_avg = df.groupBy("Geography").agg(mean("Estimate (%)").alias("avg_vaccination"))
    geo_avg_df = geo_avg.toPandas()
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Geography", y="avg_vaccination", data=geo_avg_df.head(20))
    plt.xticks(rotation=90)
    plt.title("Average Vaccination Coverage by State")
    plt.show()

def plot_vaccination_coverage_over_time(df):
    """Plot vaccination coverage over time."""
    # Assuming there is a 'Survey Year/Influenza Season' column in the dataset
    time_avg = df.groupBy("Survey Year/Influenza Season").agg(mean("Estimate (%)").alias("avg_vaccination"))
    time_avg_df = time_avg.toPandas()
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="Survey Year/Influenza Season", y="avg_vaccination", data=time_avg_df)
    plt.title("Average Vaccination Coverage Over Time")
    plt.xlabel("Survey Year/Influenza Season")
    plt.ylabel("Average Vaccination Coverage (%)")
    plt.show()

def plot_vaccination_coverage_by_vaccine(df):
    """Plot vaccination coverage by vaccine type."""
    vaccine_avg = df.groupBy("Vaccine", "Survey Year/Influenza Season").agg(mean("Estimate (%)").alias("avg_vaccination"))
    vaccine_avg_df = vaccine_avg.toPandas()
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="Survey Year/Influenza Season", y="avg_vaccination", hue="Vaccine", data=vaccine_avg_df)
    plt.title("Average Vaccination Coverage by Vaccine Type Over Time")
    plt.xlabel("Survey Year/Influenza Season")
    plt.ylabel("Average Vaccination Coverage (%)")
    plt.legend(title="Vaccine Type")
    plt.show()

def plot_vaccination_coverage_by_geography(df):
    """Plot vaccination coverage by geography."""
    geo_avg = df.groupBy("Geography", "Survey Year/Influenza Season").agg(mean("Estimate (%)").alias("avg_vaccination"))
    geo_avg_df = geo_avg.toPandas()
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="Survey Year/Influenza Season", y="avg_vaccination", hue="Geography", data=geo_avg_df)
    plt.title("Average Vaccination Coverage by Geography Over Time")
    plt.xlabel("Survey Year/Influenza Season")
    plt.ylabel("Average Vaccination Coverage (%)")
    plt.legend(title="Geography", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

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
    predictions = model.transform(test_data)
    predictions.select("Estimate (%)", "Estimate_Int", "prediction", "features").show(5)
    model.write().overwrite().save("vaccination_model")
    logger.info("Pipeline training and evaluation complete.")

def main():
    """Main function to run the analysis."""
    spark = initialize_spark()
    file_path = "vaccination_data.csv"
    df = load_data(spark, file_path)
    df.printSchema()
    df.show(5)
    logger.info(f"Total Rows: {df.count()}, Total Columns: {len(df.columns)}")
    df.describe().show()
    df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()
    df = clean_data(df)
    df.printSchema()
    logger.info(f"Cleaned Data: {df.count()} rows")
    exploratory_data_analysis(df)
    
    # Additional EDA
    plot_vaccination_coverage_over_time(df)
    plot_vaccination_coverage_by_vaccine(df)
    plot_vaccination_coverage_by_geography(df)
    
    df = prepare_data_for_ml(df)
    train_and_evaluate_model(df)

if __name__ == "__main__":
    main()
