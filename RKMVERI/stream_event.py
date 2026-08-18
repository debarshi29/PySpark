from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName("Basic Event Streaming") \
    .getOrCreate()

# Define schema for events
event_schema = "event_id STRING, participant_id STRING, event_type STRING, timestamp STRING"

# Read streaming data from events directory
events_stream = spark.readStream \
    .schema(event_schema) \
    .csv("input/events/")

# Display all incoming events
query = events_stream.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

# Wait for termination
query.awaitTermination()

# Stop the Spark session
spark.stop()