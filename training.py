import os
import sys
from pyspark.ml.feature import VectorAssembler as VAB
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.sql import SparkSession


def getSpark():
    return SparkSession.builder.appName("wine-quality-prediction-project").getOrCreate()


# args to pass dataset
system_input = sys.argv[1]


# #Input path to the training dataset
system_input = os.path.join(os.path.dirname(__file__), system_input)

# Importing training data
train_data = getSpark().read.csv(system_input, header='true',
                                       inferSchema='true', sep=';')

# parsing the data
feature_columns = train_data.columns[:-1]
constructor = VAB(inputCols=feature_columns, outputCol="features")
trans_dataset = constructor.transform(train_data)

# train the model using logistic regression
organised_pipeline = LogisticRegression(
    featuresCol="features", labelCol='""""quality"""""')
model = organised_pipeline.fit(trans_dataset)

# train the model using randomforest classifier
classifier_pipeline = RandomForestClassifier(
    featuresCol="features", labelCol='""""quality"""""')
forestModel = classifier_pipeline.fit(trans_dataset)

# Save training model
model.write().overwrite().save('calculated_model')
forestModel.write().overwrite().save('calculated_modelfm')

# Evaluation using logistic regression
finalSummary = model.summary
fm1 = finalSummary.weightedFMeasure()

# Evaluation using ForestClassifier
finalSummary2 = forestModel.summary
fm2 = finalSummary2.weightedFMeasure()

print("F-measure recorded using ForestClassifier is: %s"
      % fm1)
print("F-measure recorded using Logistic regression is: %s"
      % fm2)

getSpark().stop()

