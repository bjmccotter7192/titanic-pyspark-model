# Titanic Pyspark Model

This repository is home to the code used for exploration and testing around Pyspark.

Data that was used is just fake data that resembles the traditional Titanic example.

Here is a Kaggle that I kind of started with and spiraled a little away from: [Kaggle Logistic Regression with Python](https://www.kaggle.com/code/mnassrib/titanic-logistic-regression-with-python/notebook)

## How does it work?

This model is used with a Logistic Regression model from Pyspark and it will consume data from Kaggle with a schema of:

```bash
Schema of the data
root
 |-- PassengerId: integer (nullable = true)
 |-- Survived: integer (nullable = true)
 |-- Pclass: integer (nullable = true)
 |-- Name: string (nullable = true)
 |-- Sex: string (nullable = true)
 |-- Age: double (nullable = true)
 |-- SibSp: integer (nullable = true)
 |-- Parch: integer (nullable = true)
 |-- Ticket: string (nullable = true)
 |-- Fare: double (nullable = true)
 |-- Cabin: string (nullable = true)
 |-- Embarked: string (nullable = true)
```

After reading in the data, the pyspark transformations get created (StringIndexer, OneHotEncoder), the VectorAssembler is "assembled" and finally create the LogisticRegression like so:

```python
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
```

Once the pipeline is created we need to fit the model on our training data and finally test the accuracy with our test data.

This will produce a plot in PNG format that you can see the AOC curve! These plots can be found in the `plots` directory.

## How to Run Locally

1. Create Python virtual environment (however you like, just requires pip)
2. `pip install -r requirements.txt`
3. `python pyspark_transform.py`

## Contributing

This is a work in progress and will continually be changing so feel free to contribute/comment where you see fit.
