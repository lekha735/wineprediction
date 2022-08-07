import os
import sys
import quinn
import requests
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler as VAB, Normalizer, StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

from pyspark.context import SparkContext
from pyspark.sql import SparkSession

sparkContext = SparkContext('local')
sparkReader = SparkSession(sparkContext)


def getSpark():
    return SparkSession.builder.appName("wine-quality-prediction-project").getOrCreate()


input_file1 = os.path.join(os.path.dirname(__file__), "TrainingDataset.csv")
input_file2 = os.path.join(os.path.dirname(__file__), "ValidationDataset.csv")

trainingSet = sparkReader.read.format('csv').options(header='true', inferSchema='true', sep=';').load(input_file1)
testingSet = sparkReader.read.format('csv').options(header='true', inferSchema='true', sep=';').load(input_file2)

def remove_quotes(s):
    return s.replace('"', '')

trainingSet = quinn.with_columns_renamed(remove_quotes)(trainingSet)
trainingSet = trainingSet.withColumnRenamed('quality', 'label')

testingSet = quinn.with_columns_renamed(remove_quotes)(testingSet)
testingSet = testingSet.withColumnRenamed('quality', 'label')

feat_cols = trainingSet.columns[:-1]
assembler = VAB(inputCols=feat_cols, outputCol="op_features")

scaler = Normalizer(inputCol="op_features", outputCol="features")

logreg = LogisticRegression()

pipeline1 = Pipeline(stages=[assembler, scaler, logreg])

parametergrid = ParamGridBuilder().build()

evaluate = MulticlassClassificationEvaluator(metricName="f1")

crossval = CrossValidator(estimator=pipeline1,
                         estimatorParamMaps=parametergrid,
                         evaluator=evaluate,
                         numFolds=3
                        )

valueModel = crossval.fit(trainingSet)
print("Scored F1: ", evaluate.evaluate(valueModel.transform(testingSet)))
