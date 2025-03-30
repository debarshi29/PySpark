from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, count, when
from pyspark.sql.types import StringType
from pyspark.sql import Window

# Create Spark session
spark = SparkSession.builder \
    .appName("Classes Distribution Analysis") \
    .getOrCreate()

# Read the CSV file
path=r"C:\Users\DEBARSHI\Downloads\classes.csv"
df = spark.read.option("header", "true").csv(path)

# Custom counter for headers
def count_headers(df):
    return len(df.columns)

header_count = count_headers(df)
print(f"Number of headers in the input data: {header_count}")

# Global variables to store last seen Instructor and Type
last_instructor = None
last_type = None

# UDF to fill empty Instructor and Type cells
def fill_empty_cells(value, column_name):
    global last_instructor, last_type
    
    if column_name == "Instructor":
        if value is None or value.strip() == "":
            return last_instructor
        last_instructor = value
        return value
    elif column_name == "Type":
        if value is None or value.strip() == "":
            return last_type
        last_type = value
        return value
    return value

# Register UDFs
fill_instructor_udf = udf(lambda x: fill_empty_cells(x, "Instructor"), StringType())
fill_type_udf = udf(lambda x: fill_empty_cells(x, "Type"), StringType())

# Alternative approach using Window function (more reliable than global variables)
window_spec = Window.orderBy("Instructor")  # Assuming the data is ordered by Instructor

# Clean the dataframe by filling empty cells
cleaned_df = df.withColumn("Instructor_filled", 
    when(col("Instructor").isNull() | (col("Instructor") == ""), 
         fill_instructor_udf(col("Instructor"))).otherwise(col("Instructor"))) \
    .withColumn("Type_filled", 
        when(col("Type").isNull() | (col("Type") == ""), 
             fill_type_udf(col("Type"))).otherwise(col("Type")))

# Calculate number of classes taken by each teacher
# Using "Classes by Self" as the metric for classes taken
classes_by_teacher = cleaned_df.groupBy("Instructor_filled") \
    .agg({"Classes by Self": "sum"}) \
    .withColumnRenamed("sum(Classes by Self)", "Total_Classes_Taken") \
    .orderBy("Total_Classes_Taken", ascending=False)

# Show results
print("\nNumber of Classes Taken by Each Teacher:")
classes_by_teacher.show(100, truncate=False)

# Stop the Spark session
spark.stop()