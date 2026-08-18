from pyspark.sql import SparkSession

# Create a Spark Session
spark = SparkSession.builder.appName("WordCount").getOrCreate()
sc = spark.sparkContext  # Get SparkContext

# Create an RDD
rdd = sc.parallelize(["hello world", "hello Spark", "hello Hadoop"])

# Function to split lines into words
def split_words(line):
    return line.split(" ")

# Function to create key-value pairs (word, 1)
def map_word_to_tuple(word):
    return (word, 1)

# Function to sum word counts
def sum_counts(a, b):
    return a + b

# Apply transformations without lambda functions
words_rdd = rdd.flatMap(split_words)  # Split sentences into words
word_pairs_rdd = words_rdd.map(map_word_to_tuple)  # Map words to (word, 1)
word_counts_rdd = word_pairs_rdd.reduceByKey(sum_counts)  # Aggregate counts

# Collect and print results
print(word_counts_rdd.collect())

# Stop Spark session
spark.stop()

