from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

# Create Spark session
spark = SparkSession.builder \
    .appName("Employee Distribution Analysis") \
    .getOrCreate()

# Read the CSV file
# Assuming the data is saved as 'teachers.csv'
path=r"C:\Users\DEBARSHI\Downloads\teachers.csv"
df = spark.read.option("header", "true").csv(path)

# 1. Calculate number of employees by designation
designation_dist = df.groupBy("Designation") \
    .agg(count("*").alias("EmployeeCount")) \
    .orderBy("EmployeeCount", ascending=False)

# 2. Calculate number of employees by department
department_dist = df.groupBy("Dept") \
    .agg(count("*").alias("EmployeeCount")) \
    .orderBy("EmployeeCount", ascending=False)

# Show results
print("Distribution by Designation:")
designation_dist.show(100, truncate=False)

print("\nDistribution by Department:")
department_dist.show(100, truncate=False)

# Stop the Spark session
spark.stop()