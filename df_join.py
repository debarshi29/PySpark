from pyspark.sql import SparkSession
from pyspark.sql.functions import split, explode

# Step 1: Create Spark Session
spark = SparkSession.builder.appName("JoinExample").getOrCreate()

# Step 2: Create the first DataFrame
data = [("hello world",), ("spark is fun",)]
df = spark.createDataFrame(data, ["sentence"])

print("Original DataFrame:")
df.show()

# Step 3: Split sentences into lists of words
df_split = df.withColumn("words", split(df.sentence, " "))
print("DataFrame after splitting:")
df_split.show(truncate=False)

# Step 4: Explode the words into separate rows
df_exploded = df_split.select(explode(df_split.words).alias("word"))
print("DataFrame after exploding:")
df_exploded.show()

# Step 5: Create a second DataFrame with additional word information
word_info_data = [("hello", "greeting"), ("world", "noun"), ("spark", "technology"), ("fun", "adjective"), ("RAM", "person")]
df_word_info = spark.createDataFrame(word_info_data, ["word", "category"])

print("Word Information DataFrame:")
df_word_info.show()

# Step 6: Perform an inner join on the "word" column
df_joined = df_exploded.join(df_word_info, on="word", how="inner")

print("Joined DataFrame:")
df_joined.show()

