from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, current_timestamp

# Create SparkSession
spark = SparkSession.builder \
    .appName("StreamToStreamJoin") \
    .config("spark.hadoop.io.native.lib.available", "false") \
    .getOrCreate()

# Stream 1: User actions
actions_stream = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load() \
    .withColumnRenamed("value", "action") \
    .withColumn("timestamp", current_timestamp())

# Stream 2: User purchases
purchases_stream = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9998) \
    .load() \
    .withColumnRenamed("value", "purchase") \
    .withColumn("timestamp", current_timestamp())

# Add watermarks to both streams
actions_stream = actions_stream.withWatermark("timestamp", "10 minutes")
purchases_stream = purchases_stream.withWatermark("timestamp", "10 minutes")

# Alias the streams for join
actions_alias = actions_stream.alias("actions")
purchases_alias = purchases_stream.alias("purchases")

# Perform the stream-to-stream join
joined_stream = actions_alias.join(
    purchases_alias,
    expr("""
        actions.action = purchases.purchase AND
        actions.timestamp BETWEEN purchases.timestamp - interval 5 minutes AND purchases.timestamp + interval 5 minutes
    """)
)

# Write the joined output to the console
query = joined_stream.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
    .option("checkpointLocation", "C:/tmp/checkpoint") \
    .start()

query.awaitTermination()