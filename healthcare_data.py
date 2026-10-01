
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, sum, when, regexp_extract, concat_ws, trim, datediff
)

spark = SparkSession.builder \
    .appName("Healthcare") \
    .getOrCreate()

df = spark.read.csv(
    "healthcare_data.csv",
    header=True,
    inferSchema=True
)

print("First 5 Rows:")
df.show(5)

print("Schema:")
df.printSchema()

print("Selected Columns:")
df.select("patient_id", "age", "gender", "bill_amount").show(5)

print("Patients above 50:")
df.filter(col("age") > 50).show(5)

print("Description:")
df.describe().show()

print("Total Rows:", df.count())
print("Total Columns:", len(df.columns))

print("Missing Values:")
df.select([
    sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in df.columns
]).show()

print("Duplicate Rows:", df.count() - df.dropDuplicates().count())

print("\nDATA CLEANING PROCESS\n")

df = df.withColumn(
    "age",
    when((col("age") < 0) | (col("age") > 120), None)
    .otherwise(col("age"))
)

df = df.withColumn(
    "bill_amount",
    when(col("bill_amount") < 0, None)
    .otherwise(col("bill_amount"))
)

df = df.fillna("Unknown", subset=[
    "first_name", "last_name", "gender", "phone",
    "email", "department", "diagnosis", "treatment", "status"
])

age_median = df.approxQuantile("age", [0.5], 0.01)[0]
bill_median = df.approxQuantile("bill_amount", [0.5], 0.01)[0]

df = df.fillna({
    "age": age_median,
    "bill_amount": bill_median
})

df = df.withColumn("age", col("age").cast("integer"))
df = df.withColumn("bill_amount", col("bill_amount").cast("double"))
df = df.withColumn("admission_date", col("admission_date").cast("date"))
df = df.withColumn("discharge_date", col("discharge_date").cast("date"))

df = df.dropDuplicates()

print("Duplicate Rows:", df.count() - df.dropDuplicates().count())
print("Columns Retained:", df.columns)

print("Missing Values After Cleaning:")
df.select([
    sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in df.columns
]).show()

print("Final Rows:", df.count())
print("Final Columns:", len(df.columns))

df = df.withColumn(
    "patient_number",
    regexp_extract(col("patient_id"), r"(\d+)$", 1).cast("int")
)

df = df.orderBy("patient_number").drop("patient_number")

df.show(5, truncate=False)
df.printSchema()

df.toPandas().to_csv("cleaned_healthcare_data.csv", index=False)

print("\nDATA TRANSFORMATION PROCESS\n")

df = spark.read.csv(
    "cleaned_healthcare_data.csv",
    header=True,
    inferSchema=True
)

df = df.withColumn(
    "age_group",
    when(col("age").isNull(), "Unknown")
    .when(col("age") < 18, "Child")
    .when(col("age") < 40, "Young Adult")
    .when(col("age") < 60, "Middle Aged")
    .otherwise("Senior")
)

df = df.withColumn(
    "full_name",
    concat_ws(" ", trim(col("first_name")), trim(col("last_name")))
)

df = df.withColumn(
    "stay_days",
    when(
        col("admission_date").isNotNull() &
        col("discharge_date").isNotNull() &
        (col("discharge_date") >= col("admission_date")),
        datediff(col("discharge_date"), col("admission_date"))
    ).otherwise(None).cast("integer")
)

df = df.withColumn(
    "bill_category",
    when(col("bill_amount").isNull(), "Unknown")
    .when(col("bill_amount") < 50000, "Low")
    .when(col("bill_amount") < 150000, "Medium")
    .otherwise("High")
)

df = df.withColumn(
    "patient_status",
    when(col("status") == "Admitted", "Currently Admitted")
    .when(col("status") == "Discharged", "Completed")
    .when(col("status") == "Under Treatment", "Treatment Ongoing")
    .otherwise("Unknown")
)

df.show(10, truncate=False)
df.printSchema()

print("Total Rows:", df.count())
print("Total Columns:", len(df.columns))

print("Age Group Validation:")
df.groupBy("age_group").count().show()

print("Bill Category Validation:")
df.groupBy("bill_category").count().show()

print("Patient Status Validation:")
df.groupBy("patient_status").count().show()

print("Invalid Stay Days:")
df.filter(col("stay_days") < 0).show()

print("Missing Values in Derived Columns:")
df.select([
    sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in ["age_group", "full_name", "stay_days",
              "bill_category", "patient_status"]
]).show()

print("Duplicate Rows:", df.count() - df.dropDuplicates().count())

df.toPandas().to_csv("healthcare_transformed.csv", index=False)

spark.stop()