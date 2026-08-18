from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, split, current_timestamp

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("StreamToStreamJoin_Filtered") \
    .getOrCreate()

# Stream 1: User Activity Events
activity_stream = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9995) \
    .load()
activity_stream = activity_stream.select(
    split(col("value"), ",")[0].alias("ticket_id"),
    split(col("value"), ",")[1].alias("status"),
    current_timestamp().alias("timestamp")  # Using current time for simplicity
)
filtered_activity_stream = activity_stream.filter(col("status") == "resolved")
purchase_stream = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9994) \
    .load()
purchase_stream = purchase_stream.select(
    split(col("value"), ",")[0].alias("ticket_id"),
    split(col("value"), ",")[1].alias("refund_amount"),
    current_timestamp().alias("timestamp")  # Using current time for simplicity
)
filtered_activity_stream = filtered_activity_stream.withWatermark("timestamp", "15 minutes")
purchase_stream = purchase_stream.withWatermark("timestamp", "15 minutes")
joined_stream = filtered_activity_stream.alias("activity").join(
    purchase_stream.alias("purchase"),
    (col("activity.ticket_id") == col("purchase.ticket_id")) &
    (col("activity.timestamp") >= col("purchase.timestamp") - expr("INTERVAL 10 MINUTES")) &
    (col("activity.timestamp") <= col("purchase.timestamp") + expr("INTERVAL 10 MINUTES"))
)

# Write output to console
query = joined_stream.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
    .start()

query.awaitTermination()
