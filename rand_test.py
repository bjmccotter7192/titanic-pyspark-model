import random
from pyspark.sql import SparkSession
from utils import generate_sample_dataset
from pyspark.sql.functions import col, when, rand

# Starting the Spark Session
print("Starting the Spark Session")
spark = SparkSession.builder.appName("Random Data").config("spark.driver.memory", "10g").getOrCreate()

# generate_sample_dataset()

# print("Reading the data from random_data.csv")
# df = spark.read.csv("random_data.csv", inferSchema=True, header=True)

# print("Reading the data from Kaggle")
df = spark.read.csv("tested.xls", inferSchema=True, header=True)

# Define a list of possible replacement values
replacement_values = ["A", "B", "C", "D", "E"]

# Replace NULL values in the "Age" column with random values from the list
df_randomized = df.withColumn(
    "Cabin",
    when(col("Cabin").isNull(), replacement_values[int(random.randint(0, 4))]).otherwise(col("Cabin"))
)

# Show the DataFrame with NULL values replaced with random values
df_randomized.show(10)

spark.stop()