from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TestStream").getOrCreate()

# Read from socket
lines = spark.readStream.format("socket").option("host", "localhost").option("port", 9999).load()

# Simple transformation
lines = lines.selectExpr("value as message")

# Write to console
query = lines.writeStream.outputMode("append").format("console").start()

query.awaitTermination()