from pyspark.sql.functions import split, explode
from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("SelectExample").getOrCreate()

data = [("hello world",), ("spark is fun",)]
df = spark.createDataFrame(data, ["sentence"])

df.select(explode(split(df.sentence, " ")).alias("word")).show()


