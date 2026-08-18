from pyspark.sql import SparkSession
from pyspark.sql.functions import split, explode

spark = SparkSession.builder.appName("SelectExample").getOrCreate()

data = [("hello world",), ("spark is fun",)]
df = spark.createDataFrame(data, ["sentence"])

df.show()

df_split = df.withColumn("words", split(df.sentence, " "))
df_split.show(truncate=False)

df_exploded = df_split.select(explode(df_split.words).alias("word"))
df_exploded.show()

