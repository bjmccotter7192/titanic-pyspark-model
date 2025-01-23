import logging
import time
from typing import List, Tuple
from contextlib import contextmanager

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import matplotlib.pyplot as plt

from utils import generate_sample_dataset, generate_extra_cols

# Constants
JAVA_SECURITY_OPTION = "-Djava.security.manager=allow"
DRIVER_MEMORY = "10g"
TRAIN_TEST_SPLIT_RATIO = [0.7, 0.3]
NUM_EXPERIMENTS = 30

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@contextmanager
def create_spark_session(app_name: str = "Random Data") -> SparkSession:
    """Create a SparkSession with error handling."""
    try:
        spark = (SparkSession.builder
                .appName(app_name)
                .config("spark.driver.memory", DRIVER_MEMORY)
                .config("spark.driver.extraJavaOptions", JAVA_SECURITY_OPTION)
                .getOrCreate())
        yield spark
    finally:
        if 'spark' in locals():
            spark.stop()

def load_and_preprocess_data(spark: SparkSession) -> DataFrame:
    """Load and preprocess the dataset."""
    try:
        generate_sample_dataset()
        logger.info("Reading data from random_data.csv")
        df = spark.read.csv("random_data.csv", inferSchema=True, header=True)
        
        selected_columns = [
            "Survived", "Pclass", "Name", "Sex", "Age", 
            "SibSp", "Parch", "Fare", "Cabin", "Embarked"
        ]
        df = df.select(selected_columns)
        df = df.na.drop()
        return generate_extra_cols(df)
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

def create_pipeline() -> Pipeline:
    """Create the ML pipeline."""
    sex_indexer = StringIndexer(inputCol="Sex", outputCol="SexIndex")
    sex_encoder = OneHotEncoder(inputCol="SexIndex", outputCol="SexVec")
    
    feature_cols = ["Pclass", "SexVec", "Age", "SibSp", "Parch", "Fare"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    log_reg = LogisticRegression(featuresCol="features", labelCol="Survived")
    
    return Pipeline(stages=[sex_indexer, sex_encoder, assembler, log_reg])

def evaluate_model(results: DataFrame) -> float:
    """Evaluate the model using ROC AUC."""
    evaluator = BinaryClassificationEvaluator(
        rawPredictionCol="prediction",
        labelCol="Survived"
    )
    return evaluator.evaluate(results)

def run_experiment(spark: SparkSession) -> float:
    """Run a single experiment and return the ROC AUC score."""
    try:
        df = load_and_preprocess_data(spark)
        pipeline = create_pipeline()
        
        logger.info("Splitting data for training and testing")
        train_data, test_data = df.randomSplit(TRAIN_TEST_SPLIT_RATIO)
        
        logger.info("Training model")
        model = pipeline.fit(train_data)
        
        logger.info("Transforming test data")
        results = model.transform(test_data)
        
        roc_auc = evaluate_model(results)
        logger.info(f"ROC AUC score: {roc_auc:.4f}")
        
        return roc_auc
    except Exception as e:
        logger.error(f"Error in experiment: {str(e)}")
        raise

def plot_results(scores: List[float], date: str) -> None:
    """Plot and save the ROC AUC scores."""
    try:
        plt.figure(figsize=(8, 6))
        plt.plot(scores, marker="o", linestyle="-", color="b")
        plt.title("ROC AUC Scores")
        plt.xlabel("Experiment Number")
        plt.ylabel("AUC Score")
        plt.grid(True)
        plt.savefig(f"plots/roc_auc_plot_{date}.png")
    except Exception as e:
        logger.error(f"Error in plotting results: {str(e)}")
        raise

if __name__ == "__main__":
    start_time = time.time()
    date = time.strftime("%Y-%m-%d-%H-%M-%S")
    roc_scores = []

    with create_spark_session() as spark:
        for i in range(1, NUM_EXPERIMENTS + 1):
            logger.info(f"Running experiment {i}")
            score = run_experiment(spark)
            roc_scores.append(score)

    plot_results(roc_scores, date)

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"The total time taken is {total_time:.2f} seconds")
