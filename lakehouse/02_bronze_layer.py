# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Configuration and Setup
# Bronze Layer Configuration
# This notebook implements the Bronze layer for customer analytics data
# using Auto Loader for incremental ingestion from the landing zone

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
from datetime import datetime
import uuid

# Configuration
CATALOG = "workspace"
SCHEMA = "retail"
VOLUME = "databrics_customer_analytics"
LANDING_BASE_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/landing"
CHECKPOINT_BASE_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/checkpoints/bronze"

# Generate batch ID for this run
BATCH_ID = str(uuid.uuid4())

print(f"Bronze Layer Ingestion")
print(f"Batch ID: {BATCH_ID}")
print(f"Landing Path: {LANDING_BASE_PATH}")
print(f"Target Schema: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# DBTITLE 1,Helper Function - Bronze Ingestion
def ingest_to_bronze(source_system, source_folder, file_format, target_table_name):
    """
    Ingest data from landing to bronze using Auto Loader.
    """
    source_path = f"{LANDING_BASE_PATH}/{source_folder}"
    checkpoint_path = f"{CHECKPOINT_BASE_PATH}/{source_folder}"
    
    print(f"Ingesting {source_system.upper()} from {source_path}")
    
    # Read with Auto Loader for incremental ingestion
    reader = (spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", file_format)
        .option("cloudFiles.inferColumnTypes", "true")
        .option("cloudFiles.schemaLocation", checkpoint_path)
    )
    
    # CSV-specific options
    if file_format == "csv":
        reader = reader.option("header", "true")
    
    df_raw = reader.load(source_path)
    
    # Add metadata columns for traceability
    # Using select() to add all columns at once (avoids chained withColumn anti-pattern)
    df_bronze = df_raw.select(
        "*",
        F.current_timestamp().alias("_ingestion_timestamp"),
        F.lit(source_system).alias("_source_system"),
        F.col("_metadata.file_path").alias("_source_file"),
        F.lit(BATCH_ID).alias("_batch_id")
    )
    
    # Write to bronze table
    query = (df_bronze.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", checkpoint_path)
        .trigger(availableNow=True)
        .toTable(target_table_name)
    )
    
    query.awaitTermination()
    return target_table_name

# COMMAND ----------

# DBTITLE 1,Bronze - CRM Customer
# Ingest CRM customer data
# Source: CSV files from CRM system
ingest_to_bronze(
    source_system="crm",
    source_folder="crm",
    file_format="csv",
    target_table_name=f"{CATALOG}.{SCHEMA}.bronze_crm_customer"
)

# COMMAND ----------

# DBTITLE 1,Bronze - Products
# Ingest product reference data
# Source: Parquet files from product catalog
ingest_to_bronze(
    source_system="products",
    source_folder="products",
    file_format="parquet",
    target_table_name=f"{CATALOG}.{SCHEMA}.bronze_products"
)

# COMMAND ----------

# DBTITLE 1,Bronze - Customer Service
# Ingest customer service interactions
# Source: CSV files from customer service system
ingest_to_bronze(
    source_system="customer_service",
    source_folder="customer_service",
    file_format="csv",
    target_table_name=f"{CATALOG}.{SCHEMA}.bronze_customer_service_interactions"
)

# COMMAND ----------

# DBTITLE 1,Bronze - Ecommerce Transactions
# Ingest ecommerce transactions
# Source: JSON files from ecommerce platform
ingest_to_bronze(
    source_system="ecommerce",
    source_folder="ecommerce",
    file_format="json",
    target_table_name=f"{CATALOG}.{SCHEMA}.bronze_ecommerce_transactions"
)

# COMMAND ----------

# DBTITLE 1,Bronze - Loyalty Events
# Ingest loyalty program events
# Source: JSON files from loyalty system
ingest_to_bronze(
    source_system="loyalty",
    source_folder="loyalty",
    file_format="json",
    target_table_name=f"{CATALOG}.{SCHEMA}.bronze_loyalty_events"
)

# COMMAND ----------

# DBTITLE 1,Bronze - Marketing Events
# Ingest marketing events
# Source: Parquet files from marketing platform
ingest_to_bronze(
    source_system="marketing",
    source_folder="marketing",
    file_format="parquet",
    target_table_name=f"{CATALOG}.{SCHEMA}.bronze_marketing_events"
)

# COMMAND ----------

# DBTITLE 1,Bronze - POS Transactions
# Ingest point-of-sale transactions
# Source: Parquet files from POS system
ingest_to_bronze(
    source_system="pos",
    source_folder="pos",
    file_format="parquet",
    target_table_name=f"{CATALOG}.{SCHEMA}.bronze_pos_transactions"
)

# COMMAND ----------

# DBTITLE 1,Validation - Bronze Tables
# Validation of all Bronze tables
# Per Validation Contract: schema, sample data, row counts, and data quality checks

bronze_tables = [
    f"{CATALOG}.{SCHEMA}.bronze_crm_customer",
    f"{CATALOG}.{SCHEMA}.bronze_products",
    f"{CATALOG}.{SCHEMA}.bronze_customer_service_interactions",
    f"{CATALOG}.{SCHEMA}.bronze_ecommerce_transactions",
    f"{CATALOG}.{SCHEMA}.bronze_loyalty_events",
    f"{CATALOG}.{SCHEMA}.bronze_marketing_events",
    f"{CATALOG}.{SCHEMA}.bronze_pos_transactions"
]

validation_results = []

for table_name in bronze_tables:
    print(f"\n{'='*80}")
    print(f"VALIDATING: {table_name}")
    print(f"{'='*80}")
    
    df = spark.table(table_name)
    
    # 1. Schema validation
    print("\n1. SCHEMA:")
    df.printSchema()
    
    # 2. Row count
    row_count = df.count()
    print(f"\n2. ROW COUNT: {row_count:,}")
    
    # 3. Metadata column validation
    print("\n3. METADATA COLUMNS:")
    metadata_check = df.select(
        F.count("*").alias("total_rows"),
        F.count("_ingestion_timestamp").alias("has_ingestion_ts"),
        F.count("_source_system").alias("has_source_system"),
        F.count("_source_file").alias("has_source_file"),
        F.count("_batch_id").alias("has_batch_id")
    ).collect()[0]
    
    print(f"  Total rows: {metadata_check['total_rows']:,}")
    print(f"  With _ingestion_timestamp: {metadata_check['has_ingestion_ts']:,}")
    print(f"  With _source_system: {metadata_check['has_source_system']:,}")
    print(f"  With _source_file: {metadata_check['has_source_file']:,}")
    print(f"  With _batch_id: {metadata_check['has_batch_id']:,}")
    
    # 4. Sample data (5 rows)
    print("\n4. SAMPLE DATA (5 rows):")
    display(df.limit(5))
    
    validation_results.append({
        "table": table_name,
        "row_count": row_count,
        "metadata_complete": metadata_check['total_rows'] == metadata_check['has_batch_id']
    })

print(f"\n{'='*80}")
print("VALIDATION SUMMARY")
print(f"{'='*80}")
for result in validation_results:
    status = "✓ PASS" if result['metadata_complete'] else "✗ FAIL"
    print(f"{status} - {result['table']}: {result['row_count']:,} rows")

# COMMAND ----------

# DBTITLE 1,Anti-Pattern Analysis Summary
# MAGIC %md
# MAGIC ## Spark Anti-Pattern Analysis & Resolution
# MAGIC
# MAGIC ### Contract Compliance Review
# MAGIC
# MAGIC **✓ COMPLIANT:**
# MAGIC * No `collect()` for data processing (only for validation metrics)
# MAGIC * No `toPandas()` usage
# MAGIC * No driver-side data processing
# MAGIC * No Python UDFs
# MAGIC * No Python loops over DataFrame rows
# MAGIC * No unnecessary `cache()` or `persist()`
# MAGIC * No unnecessary `repartition()` or `coalesce()`
# MAGIC * Auto Loader handles incremental processing efficiently
# MAGIC
# MAGIC **✓ FIXED:**
# MAGIC * **Chained `.withColumn()` calls** → Replaced with single `.select()` operation
# MAGIC   * **Before:** 4 chained `.withColumn()` calls created deeply nested execution plan
# MAGIC   * **After:** Single `.select("*", col1, col2, col3, col4)` adds all metadata columns at once
# MAGIC   * **Benefit:** Simplified execution plan, better optimizer performance
# MAGIC
# MAGIC ### Validation Approach
# MAGIC * Uses `collect()[0]` for aggregated metrics only (acceptable per contract)
# MAGIC * Multiple table scans for validation are intentional and expected
# MAGIC * Validation is separated from production ingestion code