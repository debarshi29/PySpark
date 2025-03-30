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
    .option("port", 9999) \
    .load()

# Parsing the activity stream (CSV-like format: user_id, event_type, timestamp)
activity_stream = activity_stream.select(
    split(col("value"), ",")[0].alias("user_id"),
    split(col("value"), ",")[1].alias("event_type"),
    current_timestamp().alias("timestamp")  # Using current time for simplicity
)

# Filter only "add_to_cart" events
filtered_activity_stream = activity_stream.filter(col("event_type") == "add_to_cart")

# Stream 2: User Purchase Events
purchase_stream = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9998) \
    .load()

# Parsing the purchase stream (CSV-like format: user_id, item, timestamp)
purchase_stream = purchase_stream.select(
    split(col("value"), ",")[0].alias("user_id"),
    split(col("value"), ",")[1].alias("item"),
    current_timestamp().alias("timestamp")  # Using current time for simplicity
)

# Apply watermarks
filtered_activity_stream = filtered_activity_stream.withWatermark("timestamp", "12 minutes")
purchase_stream = purchase_stream.withWatermark("timestamp", "12 minutes")

# Join the filtered activity stream with the purchase stream
joined_stream = filtered_activity_stream.alias("activity").join(
    purchase_stream.alias("purchase"),
    (col("activity.user_id") == col("purchase.user_id")) &
    (col("activity.timestamp") >= col("purchase.timestamp") - expr("INTERVAL 7 MINUTES")) &
    (col("activity.timestamp") <= col("purchase.timestamp") + expr("INTERVAL 7 MINUTES"))
)

# Write output to console
query = joined_stream.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
    .start()

query.awaitTermination()

