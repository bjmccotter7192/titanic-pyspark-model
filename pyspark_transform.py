import time

# Starting the Spark Session
from pyspark.sql import SparkSession
from utils import generate_sample_dataset, generate_extra_cols

# Importing the required libraries
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, Imputer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col

import matplotlib.pyplot as plt

import resource

# Starting the Spark Session
print("Starting the Spark Session")
spark = SparkSession.builder.appName("Random Data").config("spark.driver.memory", "10g").getOrCreate()


# Setting the start time
start_time = time.time()

def run_experiment(spark=spark):
    # generate_sample_dataset()

    # print("Reading the data from random_data.csv")
    # df = spark.read.csv("random_data.csv", inferSchema=True, header=True)

    print("Reading the data from Kaggle")
    df = spark.read.csv("tested.xls", inferSchema=True, header=True)

    print("Starting Feature Engineering")
    rm_columns = df.select(
        ["Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]
    )

    # Drops the data having null values
    result = rm_columns.na.drop()

    # Generating extra columns
    result = generate_extra_cols(result)

    # Converting the Sex Column
    sexIdx = StringIndexer(inputCol="Sex", outputCol="SexIndex")
    sexEncode = OneHotEncoder(inputCol="SexIndex", outputCol="SexVec")

    # Vectorizing the data into a new column "features"
    # which will be our input/features class
    assembler = VectorAssembler(
        inputCols=["Pclass", "SexVec", "Age", "SibSp", "Parch", "Fare"],
        outputCol="features",
    )

    log_reg = LogisticRegression(featuresCol="features", labelCol="Survived")

    # Creating the pipeline
    pipeline = Pipeline(
        stages=[sexIdx, sexEncode, assembler, log_reg]
    )

    print("Splitting the data for training and testing")
    train_data, test_data = result.randomSplit([0.7, 0.3])

    print("Fit the Model (Logistic Regression)")
    fit_model = pipeline.fit(train_data)

    print("Transform the test data")
    results = fit_model.transform(test_data)

    print("Run BinaryClassificationEvaluator on the results")
    res = BinaryClassificationEvaluator(
        rawPredictionCol="prediction", labelCol="Survived"
    )

    # Evaluating the AUC on results
    roc_auc = res.evaluate(results)

    print(f"The ROC_AUC score is {roc_auc}")

    return roc_auc


roc_scores = []

for i in range(1, 10):
    print(f"Running experiment {i}")
    score = run_experiment(spark=spark)
    roc_scores.append(score)

plt.figure(figsize=(8, 6))
plt.plot(roc_scores, marker="o", linestyle="-", color="b")
plt.title("ROC AUC Scores")
plt.xlabel("Experiment Number")
plt.ylabel("AUC Score")
plt.grid(True)
plt.savefig("roc_auc_plot.png")

spark.stop()

# Setting the end time
end_time = time.time()

# Calculating the total time taken
total_time = end_time - start_time
print(f"The total time taken is {total_time} seconds")

MAX_RAM = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

print(f"Max RAM used: {MAX_RAM} MB")
