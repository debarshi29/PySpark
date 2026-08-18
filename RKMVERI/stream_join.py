from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName("Streaming Event Join") \
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

# Join streaming events with static participants
joined_stream = events_stream.join(participants_static, "participant_id", "left_outer")

# Write the joined stream to console
query = joined_stream.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
    .start()

# Wait for termination
query.awaitTermination()

# Stop the Spark session
spark.stop()