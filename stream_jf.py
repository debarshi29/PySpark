from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Create Spark session
spark = SparkSession.builder \
    .appName("Streaming Event Join and Filter") \
    .getOrCreate()

# Define schemas
event_schema = "event_id STRING, participant_id STRING, event_type STRING, timestamp STRING"
participant_schema = "participant_id STRING, name STRING, category STRING"

# Read streaming data from events
events_stream = spark.readStream \
    .schema(event_schema) \
    .csv("input/events/")

# Read static participant data
participants_static = spark.read \
    .schema(participant_schema) \
    .csv("input/participants/participants.csv")

# Filter events for specific types (e.g., learning-related)
learning_events = ["Workshop", "Seminar"]
filtered_stream = events_stream.filter(col("event_type").isin(learning_events))

# Join filtered stream with participants
joined_stream = filtered_stream.join(participants_static, "participant_id", "left_outer")

# Write the result to console
query = joined_stream.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
    .start()

# Wait for termination
query.awaitTermination()

# Stop the Spark session
spark.stop()