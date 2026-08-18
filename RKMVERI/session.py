from pyspark.sql import SparkSession
from pyspark.sql.functions import session_window, split, explode, current_timestamp

# Create SparkSession
spark = SparkSession.builder \
    .appName("SessionWindowingExample") \
    .getOrCreate()

# Read streaming data from a socket source
lines = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# Add a timestamp column to the streaming data
lines_with_timestamp = lines.withColumn("timestamp", current_timestamp())

# Split lines into words
words = lines_with_timestamp.select(
    explode(split(lines_with_timestamp.value, " ")).alias("word"),
    "timestamp"
)

# Apply session windowing with a gap duration of 10 seconds
session_windowed_counts = words.groupBy(
    session_window("timestamp", "10 seconds"),  # Session window with a 10-second gap
    "word"  # Group by each word
).count()

# Output the results to the console
query = session_windowed_counts.writeStream \
    .outputMode("complete") \
    .format("console") \
    .option("truncate", "false") \
    .start()

query.awaitTermination()

