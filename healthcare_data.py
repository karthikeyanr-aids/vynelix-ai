from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, sum, when, regexp_extract, concat_ws, trim, datediff, avg
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
    )
)
df = df.withColumn(
    "date_status",
    when(
        col("admission_date").isNull() |
        col("discharge_date").isNull(),
        "Missing Date"
    )
    .when(
        col("discharge_date") < col("admission_date"),
        "Invalid Date"
    )
    .otherwise("Valid Date")
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

print("Date Status Details:")
df.select(
    "patient_id",
    "admission_date",
    "discharge_date",
    "stay_days",
    "date_status"
).filter(
    col("date_status") != "Valid Date"
).show(20, truncate=False)

print("Invalid Date Records:")
df.filter(
    col("admission_date").isNotNull() &
    col("discharge_date").isNotNull() &
    (col("discharge_date") < col("admission_date"))
).select(
    "patient_id", "admission_date", "discharge_date"
).show()

print("Missing Values in Derived Columns:")
df.select([
    sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in ["age_group", "full_name", "stay_days",
              "bill_category", "patient_status","date_status"]
]).show()

print("Duplicate Rows:", df.count() - df.dropDuplicates().count())

df.toPandas().to_csv("healthcare_transformed.csv", index=False)

print("\nDATA ANALYSIS\n")

print("1. Patient Count by Department:")
df.groupBy("department") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()

print("2. Patient Count by Gender:")
df.groupBy("gender") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()

print("3. Patient Count by Age Group:")
df.groupBy("age_group") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()

print("4. Total Bill by Department:")
df.groupBy("department") \
    .sum("bill_amount") \
    .orderBy(col("sum(bill_amount)").desc()) \
    .show()

print("5. Average Bill by Department:")
df.groupBy("department") \
    .avg("bill_amount") \
    .orderBy(col("avg(bill_amount)").desc()) \
    .show()

print("6. Minimum Bill by Department:")
df.groupBy("department") \
    .min("bill_amount") \
    .orderBy(col("min(bill_amount)").asc()) \
    .show()

print("7. Maximum Bill by Department:")
df.groupBy("department") \
    .max("bill_amount") \
    .orderBy(col("max(bill_amount)").desc()) \
    .show()

print("8. Patient Count by Status:")
df.groupBy("status") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()

print("9. Average Hospital Stay by Department:")
df.groupBy("department") \
    .avg("stay_days") \
    .orderBy(col("avg(stay_days)").desc()) \
    .show()

print("10. High Bill Patients:")
df.filter(
    col("bill_amount") >= 150000
).select(
    "patient_id",
    "department",
    "bill_amount",
    "status"
).orderBy(
    col("bill_amount").desc()
).show(20)

print("11. Patients Above 60:")
df.filter(
    col("age") >= 60
).select(
    "patient_id",
    "age",
    "department",
    "diagnosis"
).orderBy(
    col("age").desc()
).show(20)

print("12. Date Status:")
df.groupBy("date_status") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()

print("13. Before and After Cleaning:")

original_df = spark.read.csv(
    "healthcare_data.csv",
    header=True,
    inferSchema=True
)

print("Rows Before Cleaning:", original_df.count())
print("Rows After Cleaning:", df.count())

print(
    "Duplicates Before Cleaning:",
    original_df.count() - original_df.dropDuplicates().count()
)

print(
    "Duplicates After Cleaning:",
    df.count() - df.dropDuplicates().count()
)

print("Average Age Before Cleaning:")
original_df.select(
    avg("age").alias("average_age")
).show()

print("Average Age After Cleaning:")
df.select(
    avg("age").alias("average_age")
).show()

print("Average Bill Before Cleaning:")
original_df.select(
    avg("bill_amount").alias("average_bill")
).show()

print("Average Bill After Cleaning:")
df.select(
    avg("bill_amount").alias("average_bill")
).show()

df.toPandas().to_csv(
    "healthcare_final_processed.csv",
    index=False
)

print("Final dataset saved successfully")
spark.stop()