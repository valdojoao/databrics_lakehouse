# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Silver Layer - Customer 360 Foundation
# MAGIC %md
# MAGIC # Silver Layer - Customer 360 Foundation
# MAGIC
# MAGIC This notebook creates the cleaned and integrated customer data layer:
# MAGIC - **Customer Identity Resolution**: Unify customer IDs across source systems
# MAGIC - **Standardized Customer Master**: Create consistent customer entity
# MAGIC - **Product Reference Data**: Clean product catalog with standardized categories and brands
# MAGIC - **Clean Behavioral Tables**: Purchases, Service, Loyalty, Marketing
# MAGIC
# MAGIC **Source**: Bronze tables in `workspace.retail` schema  
# MAGIC **Target**: Silver tables in `workspace.retail` schema  
# MAGIC **Approach**: PySpark DataFrame API with distributed processing

# COMMAND ----------

# DBTITLE 1,Import libraries and setup
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import IntegerType, StringType, DateType, DoubleType, TimestampType

# Unity Catalog configuration
CATALOG = "workspace"
SCHEMA = "retail"
VOLUME = "databrics_customer_analytics"

print(f"Configuration loaded")
print(f"Source: {CATALOG}.{SCHEMA}.bronze_*")
print(f"Target: {CATALOG}.{SCHEMA}.silver_*")

# COMMAND ----------

# DBTITLE 1,Load bronze tables
# Load bronze tables - filter only necessary columns early per contract
# Each table retains source system metadata for traceability

brn_crm = spark.table(f"{CATALOG}.{SCHEMA}.bronze_crm_customer").select(
    F.col("customer_id").cast(IntegerType()).alias("source_customer_id"),
    F.lit("crm").alias("source_system"),
    "first_name", "last_name", "email", "country", "region",
    "customer_since", "customer_status", "age_band", "preferred_channel",
    "_ingestion_timestamp"
)

brn_ecom = spark.table(f"{CATALOG}.{SCHEMA}.bronze_ecommerce_transactions").select(
    F.col("customer_id").cast(IntegerType()).alias("source_customer_id"),
    F.lit("ecommerce").alias("source_system"),
    "transaction_id", "transaction_date", "channel", "product_id",
    "quantity", "total_amount", "unit_price", "device_type",
    "_ingestion_timestamp"
)

brn_pos = spark.table(f"{CATALOG}.{SCHEMA}.bronze_pos_transactions").select(
    F.col("customer_id").cast(IntegerType()).alias("source_customer_id"),
    F.lit("pos").alias("source_system"),
    "transaction_id", "transaction_date", "store_id", "product_id",
    "quantity", "unit_price", "total_amount", "payment_method",
    "_ingestion_timestamp"
)

brn_loyalty = spark.table(f"{CATALOG}.{SCHEMA}.bronze_loyalty_events").select(
    F.col("customer_id").cast(IntegerType()).alias("source_customer_id"),
    F.lit("loyalty").alias("source_system"),
    "loyalty_id", "event_date", "loyalty_event_type", "loyalty_tier",
    "points_earned", "points_redeemed",
    "_ingestion_timestamp"
)

brn_service = spark.table(f"{CATALOG}.{SCHEMA}.bronze_customer_service_interactions").select(
    F.col("customer_id").cast(IntegerType()).alias("source_customer_id"),
    F.lit("customer_service").alias("source_system"),
    "interaction_id", "interaction_date", "interaction_type", "channel",
    "issue_category", "resolution_status", "satisfaction_score",
    "_ingestion_timestamp"
)

brn_marketing = spark.table(f"{CATALOG}.{SCHEMA}.bronze_marketing_events").select(
    F.col("customer_id").cast(IntegerType()).alias("source_customer_id"),
    F.lit("marketing").alias("source_system"),
    "campaign_id", "campaign_date", "channel", "campaign_type",
    "sent", "opened", "clicked", "converted",
    "_ingestion_timestamp"
)

brn_products = spark.table(f"{CATALOG}.{SCHEMA}.bronze_products").select(
    "product_id",
    "product_name",
    "product_category",
    "brand",
    "_ingestion_timestamp"
)

print("Bronze tables loaded with standardized customer_id casting")
print(f"Products loaded: {brn_products.count():,} records")

# COMMAND ----------

# DBTITLE 1,Create customer identity mapping
# SILVER_CUSTOMER_IDENTITY: Map source system customer IDs to unified customer_id
# Strategy: Use dense_rank to create consistent unified IDs across all sources
# Assumption: Same source_customer_id across systems represents same customer

# Performance: Single union followed by one distinct() to avoid multiple shuffles
# This approach scans each bronze table once and performs a single deduplication
all_source_ids = (
    brn_crm.select("source_customer_id", "source_system")
    .unionByName(brn_ecom.select("source_customer_id", "source_system"))
    .unionByName(brn_pos.select("source_customer_id", "source_system"))
    .unionByName(brn_loyalty.select("source_customer_id", "source_system"))
    .unionByName(brn_service.select("source_customer_id", "source_system"))
    .unionByName(brn_marketing.select("source_customer_id", "source_system"))
    .distinct()  # Single distinct operation at the end
)

# Assign unified customer_id using dense_rank over source_customer_id
# This creates consistent mapping where same source_customer_id gets same unified_id
# Partitioning: Window function requires shuffle but only over distinct source IDs
window_spec = Window.orderBy("source_customer_id")

silver_customer_identity = all_source_ids.withColumn(
    "customer_id",
    F.dense_rank().over(window_spec)
).select(
    "customer_id",
    "source_system",
    "source_customer_id"
)

# Validation: Check identity mapping
print("=== CUSTOMER IDENTITY MAPPING ===")
silver_customer_identity.printSchema()
print(f"Total identity mappings: {silver_customer_identity.count():,}")
print(f"Unique unified customer_ids: {silver_customer_identity.select('customer_id').distinct().count():,}")
print(f"Unique source customer_ids: {silver_customer_identity.select('source_customer_id').distinct().count():,}")
print("\nSample mappings by source system:")
display(silver_customer_identity.groupBy("source_system").count().orderBy("source_system"))
print("\nSample identity records:")
display(silver_customer_identity.orderBy("customer_id").limit(10))

# COMMAND ----------

# DBTITLE 1,Create silver customer master
# SILVER_CUSTOMER: Create unified customer master from CRM
# Business Rule: CRM is authoritative source for customer attributes
# Join with identity mapping and derive customer_segment from activity
#
# JOIN STRATEGY (Contract Section A.6):
# - Identity mapping join: INNER join (expect all CRM customers in identity map)
# - Loyalty tier join: LEFT join (not all customers have loyalty activity)
# - Join type selection: Spark optimizer will choose broadcast vs sort-merge based on table sizes
# - AQE enabled (default) will adapt join strategy at runtime based on actual data sizes

# Join CRM with identity mapping to get unified customer_id
silver_customer_base = (
    brn_crm
    .join(
        silver_customer_identity.filter(F.col("source_system") == "crm"),
        brn_crm.source_customer_id == silver_customer_identity.source_customer_id,
        "inner"
    )
    .select(
        silver_customer_identity.customer_id,
        F.upper(F.col("country")).alias("country"),  # Standardize to uppercase
        F.initcap(F.col("region")).alias("region"),  # Standardize to title case
        F.col("customer_since").cast(DateType()),
        F.lower(F.trim(F.col("customer_status"))).alias("customer_status"),  # Standardize to lowercase
        F.lower(F.trim(F.col("preferred_channel"))).alias("preferred_channel"),  # Standardize
        F.col("age_band")
    )
)

# Add loyalty_tier from most recent loyalty event
latest_loyalty = (
    brn_loyalty
    .withColumn("event_date_parsed", F.to_date(F.col("event_date")))
    .withColumn(
        "row_num",
        F.row_number().over(
            Window.partitionBy("source_customer_id").orderBy(F.desc("event_date_parsed"))
        )
    )
    .filter(F.col("row_num") == 1)
    .select(
        "source_customer_id",
        F.lower(F.col("loyalty_tier")).alias("loyalty_tier")
    )
)

# Join with loyalty tier
silver_customer = (
    silver_customer_base
    .join(
        silver_customer_identity.filter(F.col("source_system") == "crm"),
        "customer_id",
        "inner"
    )
    .join(
        latest_loyalty,
        silver_customer_identity.source_customer_id == latest_loyalty.source_customer_id,
        "left"
    )
    .select(
        silver_customer_base["*"],
        F.coalesce(latest_loyalty.loyalty_tier, F.lit("none")).alias("loyalty_tier")
    )
)

# Derive customer_segment based on customer_since (simple segmentation)
# New: < 1 year, Growing: 1-3 years, Established: > 3 years
silver_customer = silver_customer.withColumn(
    "customer_segment",
    F.when(
        F.datediff(F.current_date(), F.col("customer_since")) < 365,
        F.lit("new")
    ).when(
        F.datediff(F.current_date(), F.col("customer_since")) < 1095,
        F.lit("growing")
    ).otherwise(F.lit("established"))
)

# Validation
print("=== SILVER CUSTOMER ===")
silver_customer.printSchema()
print(f"Total customers: {silver_customer.count():,}")
print(f"\nCustomers by status:")
display(silver_customer.groupBy("customer_status").count().orderBy("customer_status"))
print(f"\nCustomers by segment:")
display(silver_customer.groupBy("customer_segment").count().orderBy("customer_segment"))
print(f"\nCustomers by loyalty tier:")
display(silver_customer.groupBy("loyalty_tier").count().orderBy("loyalty_tier"))
print("\nSample records:")
display(silver_customer.limit(5))

# COMMAND ----------

# DBTITLE 1,Create silver purchase table
# SILVER_PURCHASE: Combine e-commerce and POS transactions into unified purchase table
# Standardize: dates, channels, revenue calculations
#
# JOIN STRATEGY:
# - Identity mapping joins: INNER (expect all transactions have customer mapping)
# - Each source joined separately before union to maintain data quality per source
# - Filtered identity map by source_system enables partition pruning on identity table

# Process e-commerce transactions
ecom_purchases = (
    brn_ecom
    .join(
        silver_customer_identity.filter(F.col("source_system") == "ecommerce"),
        brn_ecom.source_customer_id == silver_customer_identity.source_customer_id,
        "inner"
    )
    .select(
        silver_customer_identity.customer_id,
        F.col("transaction_id"),
        F.to_date(F.col("transaction_date")).alias("transaction_date"),  # Standardize to date
        F.lower(F.trim(F.col("channel"))).alias("channel"),  # Standardize channel
        F.lit(None).cast(StringType()).alias("store_id"),  # No store for e-commerce
        F.col("product_id"),
        F.col("quantity").cast(IntegerType()),
        F.col("total_amount").cast(DoubleType()).alias("revenue")
    )
)

# Process POS transactions
pos_purchases = (
    brn_pos
    .join(
        silver_customer_identity.filter(F.col("source_system") == "pos"),
        brn_pos.source_customer_id == silver_customer_identity.source_customer_id,
        "inner"
    )
    .select(
        silver_customer_identity.customer_id,
        F.col("transaction_id"),
        F.col("transaction_date").cast(DateType()),
        F.lit("store").alias("channel"),  # POS = store channel
        F.col("store_id"),
        F.col("product_id"),
        F.col("quantity").cast(IntegerType()),
        F.col("total_amount").cast(DoubleType()).alias("revenue")
    )
)

# Union e-commerce and POS purchases
silver_purchase = ecom_purchases.unionByName(pos_purchases)

# Validation
print("=== SILVER PURCHASE ===")
silver_purchase.printSchema()
print(f"Total purchases: {silver_purchase.count():,}")
print(f"E-commerce purchases: {ecom_purchases.count():,}")
print(f"POS purchases: {pos_purchases.count():,}")
print(f"Unique customers with purchases: {silver_purchase.select('customer_id').distinct().count():,}")
print(f"\nPurchases by channel:")
display(silver_purchase.groupBy("channel").count().orderBy("channel"))
print(f"\nRevenue summary:")
display(silver_purchase.select(
    F.sum("revenue").alias("total_revenue"),
    F.avg("revenue").alias("avg_revenue"),
    F.min("revenue").alias("min_revenue"),
    F.max("revenue").alias("max_revenue")
))
print("\nSample purchase records:")
display(silver_purchase.orderBy(F.desc("transaction_date")).limit(5))

# COMMAND ----------

# DBTITLE 1,Create silver products table
# SILVER_PRODUCTS: Clean product reference/dimension data
# Standardize: product names, categories, brands
# No customer ID join needed - this is reference data

silver_products = (
    brn_products
    .select(
        F.col("product_id"),
        F.trim(F.col("product_name")).alias("product_name"),
        F.lower(F.trim(F.col("product_category"))).alias("product_category"),
        F.lower(F.trim(F.col("brand"))).alias("brand")
    )
    .distinct()  # Ensure unique products
)

# Validation
print("=== SILVER PRODUCTS ===")
silver_products.printSchema()
print(f"Total products: {silver_products.count():,}")
print(f"\nProducts by category:")
display(silver_products.groupBy("product_category").count().orderBy(F.desc("count")))
print(f"\nTop brands:")
display(silver_products.groupBy("brand").count().orderBy(F.desc("count")).limit(10))
print(f"\nProducts with missing brand:")
print(f"  {silver_products.filter(F.col('brand').isNull()).count():,}")
print("\nSample product records:")
display(silver_products.limit(10))

# COMMAND ----------

# DBTITLE 1,Create silver customer service table
# SILVER_CUSTOMER_SERVICE: Clean customer service interactions
# Standardize: channels, interaction types, statuses
#
# JOIN STRATEGY: INNER join with filtered identity map (expect all interactions have customer)

silver_customer_service = (
    brn_service
    .join(
        silver_customer_identity.filter(F.col("source_system") == "customer_service"),
        brn_service.source_customer_id == silver_customer_identity.source_customer_id,
        "inner"
    )
    .select(
        silver_customer_identity.customer_id,
        F.col("interaction_date").cast(DateType()),
        F.lower(F.trim(F.col("interaction_type"))).alias("interaction_type"),
        F.lower(F.trim(F.col("issue_category"))).alias("issue_category"),
        F.lower(F.trim(F.col("resolution_status"))).alias("resolution_status"),
        F.col("satisfaction_score").cast(IntegerType())
    )
)

# Validation
print("=== SILVER CUSTOMER SERVICE ===")
silver_customer_service.printSchema()
print(f"Total interactions: {silver_customer_service.count():,}")
print(f"Unique customers with service interactions: {silver_customer_service.select('customer_id').distinct().count():,}")
print(f"\nInteractions by type:")
display(silver_customer_service.groupBy("interaction_type").count().orderBy("interaction_type"))
print(f"\nInteractions by resolution status:")
display(silver_customer_service.groupBy("resolution_status").count().orderBy("resolution_status"))
print(f"\nSatisfaction score distribution:")
display(silver_customer_service.groupBy("satisfaction_score").count().orderBy("satisfaction_score"))
print("\nSample service records:")
display(silver_customer_service.orderBy(F.desc("interaction_date")).limit(5))

# COMMAND ----------

# DBTITLE 1,Create silver loyalty activity table
# SILVER_LOYALTY_ACTIVITY: Clean loyalty events
# Standardize: dates, event types, tiers
#
# JOIN STRATEGY: INNER join with filtered identity map (expect all events have customer)

silver_loyalty_activity = (
    brn_loyalty
    .join(
        silver_customer_identity.filter(F.col("source_system") == "loyalty"),
        brn_loyalty.source_customer_id == silver_customer_identity.source_customer_id,
        "inner"
    )
    .select(
        silver_customer_identity.customer_id,
        F.to_date(F.col("event_date")).alias("event_date"),  # Standardize to date
        F.lower(F.trim(F.col("loyalty_event_type"))).alias("loyalty_event_type"),
        F.col("points_earned").cast(IntegerType()),
        F.col("points_redeemed").cast(IntegerType()),
        F.lower(F.trim(F.col("loyalty_tier"))).alias("loyalty_tier")
    )
)

# Validation
print("=== SILVER LOYALTY ACTIVITY ===")
silver_loyalty_activity.printSchema()
print(f"Total loyalty events: {silver_loyalty_activity.count():,}")
print(f"Unique customers with loyalty activity: {silver_loyalty_activity.select('customer_id').distinct().count():,}")
print(f"\nEvents by type:")
display(silver_loyalty_activity.groupBy("loyalty_event_type").count().orderBy("loyalty_event_type"))
print(f"\nPoints summary:")
display(silver_loyalty_activity.select(
    F.sum("points_earned").alias("total_points_earned"),
    F.sum("points_redeemed").alias("total_points_redeemed"),
    F.avg("points_earned").alias("avg_points_earned"),
    F.avg("points_redeemed").alias("avg_points_redeemed")
))
print("\nSample loyalty records:")
display(silver_loyalty_activity.orderBy(F.desc("event_date")).limit(5))

# COMMAND ----------

# DBTITLE 1,Create silver marketing engagement table
# SILVER_MARKETING_ENGAGEMENT: Clean marketing campaign events
# Standardize: dates, channels, campaign types
#
# JOIN STRATEGY: INNER join with filtered identity map (expect all campaigns have customer)

silver_marketing_engagement = (
    brn_marketing
    .join(
        silver_customer_identity.filter(F.col("source_system") == "marketing"),
        brn_marketing.source_customer_id == silver_customer_identity.source_customer_id,
        "inner"
    )
    .select(
        silver_customer_identity.customer_id,
        F.col("campaign_id"),
        F.col("campaign_date").cast(DateType()),
        F.lower(F.trim(F.col("channel"))).alias("channel"),
        F.lower(F.trim(F.col("campaign_type"))).alias("campaign_type"),
        F.col("sent").cast(IntegerType()),
        F.col("opened").cast(IntegerType()),
        F.col("clicked").cast(IntegerType()),
        F.col("converted").cast(IntegerType())
    )
)

# Validation
print("=== SILVER MARKETING ENGAGEMENT ===")
silver_marketing_engagement.printSchema()
print(f"Total marketing events: {silver_marketing_engagement.count():,}")
print(f"Unique customers with marketing engagement: {silver_marketing_engagement.select('customer_id').distinct().count():,}")
print(f"\nEngagement by channel:")
display(silver_marketing_engagement.groupBy("channel").count().orderBy("channel"))
print(f"\nEngagement by campaign type:")
display(silver_marketing_engagement.groupBy("campaign_type").count().orderBy("campaign_type"))
print(f"\nEngagement funnel metrics:")
display(silver_marketing_engagement.select(
    F.sum("sent").alias("total_sent"),
    F.sum("opened").alias("total_opened"),
    F.sum("clicked").alias("total_clicked"),
    F.sum("converted").alias("total_converted"),
    (F.sum("opened") / F.sum("sent") * 100).alias("open_rate_pct"),
    (F.sum("clicked") / F.sum("opened") * 100).alias("click_rate_pct"),
    (F.sum("converted") / F.sum("clicked") * 100).alias("conversion_rate_pct")
))
print("\nSample marketing records:")
display(silver_marketing_engagement.orderBy(F.desc("campaign_date")).limit(5))

# COMMAND ----------

# DBTITLE 1,Pre-write validation and reconciliation
# PRE-WRITE VALIDATION: Verify data quality before persisting silver tables
# Contract Requirement: Validation must occur before writes, not after

print("=" * 80)
print("PRE-WRITE VALIDATION - CONTRACT COMPLIANCE")
print("=" * 80)

# ============================================================================
# 1. ROW COUNT RECONCILIATION
# ============================================================================
print("\n1. ROW COUNT RECONCILIATION")
print("-" * 80)

# Count distinct source customers from bronze
bronze_source_counts = {
    "crm": brn_crm.select("source_customer_id").distinct().count(),
    "ecommerce": brn_ecom.select("source_customer_id").distinct().count(),
    "pos": brn_pos.select("source_customer_id").distinct().count(),
    "loyalty": brn_loyalty.select("source_customer_id").distinct().count(),
    "customer_service": brn_service.select("source_customer_id").distinct().count(),
    "marketing": brn_marketing.select("source_customer_id").distinct().count()
}

total_bronze_sources = sum(bronze_source_counts.values())
identity_map_count = silver_customer_identity.count()
unified_customer_count = silver_customer_identity.select("customer_id").distinct().count()
silver_customer_count = silver_customer.count()

print(f"Bronze source customer IDs (total across systems): {total_bronze_sources:,}")
for system, count in bronze_source_counts.items():
    print(f"  - {system}: {count:,}")
print(f"Identity mappings created: {identity_map_count:,}")
print(f"Unified customer IDs: {unified_customer_count:,}")
print(f"Silver customer master records: {silver_customer_count:,}")

# Validate identity mapping completeness
assert identity_map_count == total_bronze_sources, \
    f"Identity map count mismatch: {identity_map_count} != {total_bronze_sources}"
print("✓ Identity mapping row count matches bronze sources")

# Validate customer master matches CRM count (CRM is authoritative)
assert silver_customer_count == bronze_source_counts["crm"], \
    f"Customer master count mismatch with CRM: {silver_customer_count} != {bronze_source_counts['crm']}"
print("✓ Customer master row count matches CRM (authoritative source)")

# ============================================================================
# 2. JOIN MATCH/UNMATCH VALIDATION
# ============================================================================
print("\n2. JOIN VALIDATION - Verify all joins completed successfully")
print("-" * 80)

# Check for unmatched records in customer master join
brn_crm_with_source = brn_crm.withColumn("_bronze_crm_key", F.col("source_customer_id"))
unmatched_customers = brn_crm_with_source.join(
    silver_customer_identity.filter(F.col("source_system") == "crm"),
    brn_crm_with_source.source_customer_id == silver_customer_identity.source_customer_id,
    "left_anti"
).count()

print(f"Unmatched customers in CRM → Identity join: {unmatched_customers:,}")
assert unmatched_customers == 0, f"Found {unmatched_customers} unmatched CRM customers"
print("✓ All CRM customers successfully joined to identity mapping")

# Check behavioral table joins
ecom_unmatched = brn_ecom.join(
    silver_customer_identity.filter(F.col("source_system") == "ecommerce"),
    brn_ecom.source_customer_id == silver_customer_identity.source_customer_id,
    "left_anti"
).count()

pos_unmatched = brn_pos.join(
    silver_customer_identity.filter(F.col("source_system") == "pos"),
    brn_pos.source_customer_id == silver_customer_identity.source_customer_id,
    "left_anti"
).count()

print(f"Unmatched e-commerce transactions: {ecom_unmatched:,}")
print(f"Unmatched POS transactions: {pos_unmatched:,}")
assert ecom_unmatched == 0, f"Found {ecom_unmatched} unmatched e-commerce transactions"
assert pos_unmatched == 0, f"Found {pos_unmatched} unmatched POS transactions"
print("✓ All behavioral events successfully joined to identity mapping")

# ============================================================================
# 3. NULL VALUE VALIDATION FOR CRITICAL COLUMNS
# ============================================================================
print("\n3. NULL VALUE VALIDATION - Critical columns")
print("-" * 80)

# Customer master critical columns
customer_nulls = silver_customer.select(
    F.sum(F.when(F.col("customer_id").isNull(), 1).otherwise(0)).alias("null_customer_id"),
    F.sum(F.when(F.col("country").isNull(), 1).otherwise(0)).alias("null_country"),
    F.sum(F.when(F.col("customer_since").isNull(), 1).otherwise(0)).alias("null_customer_since"),
    F.sum(F.when(F.col("customer_status").isNull(), 1).otherwise(0)).alias("null_status"),
    F.sum(F.when(F.col("customer_segment").isNull(), 1).otherwise(0)).alias("null_segment")
).collect()[0]

print("Customer master null counts:")
for field in customer_nulls.asDict():
    null_count = customer_nulls[field]
    print(f"  {field}: {null_count:,}")
    if "customer_id" in field or "customer_since" in field or "segment" in field:
        assert null_count == 0, f"Critical column {field} has {null_count} nulls"

print("✓ No nulls in critical customer master columns")

# Purchase table critical columns
purchase_nulls = silver_purchase.select(
    F.sum(F.when(F.col("customer_id").isNull(), 1).otherwise(0)).alias("null_customer_id"),
    F.sum(F.when(F.col("transaction_id").isNull(), 1).otherwise(0)).alias("null_transaction_id"),
    F.sum(F.when(F.col("revenue").isNull(), 1).otherwise(0)).alias("null_revenue")
).collect()[0]

print("\nPurchase table null counts:")
for field in purchase_nulls.asDict():
    null_count = purchase_nulls[field]
    print(f"  {field}: {null_count:,}")
    assert null_count == 0, f"Critical column {field} has {null_count} nulls"

print("✓ No nulls in critical purchase columns")

# ============================================================================
# 4. DUPLICATE VALIDATION
# ============================================================================
print("\n4. DUPLICATE VALIDATION")
print("-" * 80)

# Check for duplicate customer_ids in customer master (should be grain: one row per customer)
customer_dups = silver_customer.groupBy("customer_id").count().filter(F.col("count") > 1)
customer_dup_count = customer_dups.count()

print(f"Duplicate customer_ids in customer master: {customer_dup_count:,}")
assert customer_dup_count == 0, f"Found {customer_dup_count} duplicate customers"
print("✓ No duplicate customer_ids in customer master (correct grain maintained)")

# Check for duplicate identity mappings (source_system + source_customer_id should be unique)
identity_dups = silver_customer_identity.groupBy("source_system", "source_customer_id").count().filter(F.col("count") > 1)
identity_dup_count = identity_dups.count()

print(f"Duplicate (source_system, source_customer_id) in identity map: {identity_dup_count:,}")
assert identity_dup_count == 0, f"Found {identity_dup_count} duplicate identity mappings"
print("✓ No duplicate identity mappings (correct grain maintained)")

# ============================================================================
# 5. SCHEMA VALIDATION
# ============================================================================
print("\n5. SCHEMA VALIDATION")
print("-" * 80)

# Verify expected columns exist
expected_customer_cols = {"customer_id", "country", "region", "customer_since", 
                          "customer_status", "preferred_channel", "age_band", 
                          "loyalty_tier", "customer_segment"}
actual_customer_cols = set(silver_customer.columns)
missing_cols = expected_customer_cols - actual_customer_cols

print(f"Customer master columns: {len(actual_customer_cols)} (expected {len(expected_customer_cols)})")
assert len(missing_cols) == 0, f"Missing columns in customer master: {missing_cols}"
print("✓ All expected columns present in customer master")

expected_purchase_cols = {"customer_id", "transaction_id", "transaction_date", 
                          "channel", "store_id", "product_id", "quantity", "revenue"}
actual_purchase_cols = set(silver_purchase.columns)
missing_purchase_cols = expected_purchase_cols - actual_purchase_cols

print(f"Purchase table columns: {len(actual_purchase_cols)} (expected {len(expected_purchase_cols)})")
assert len(missing_purchase_cols) == 0, f"Missing columns in purchase: {missing_purchase_cols}"
print("✓ All expected columns present in purchase table")

# ============================================================================
# 6. DATA QUALITY METRICS
# ============================================================================
print("\n6. DATA QUALITY METRICS")
print("-" * 80)

# Revenue validation
revenue_stats = silver_purchase.select(
    F.sum("revenue").alias("total_revenue"),
    F.avg("revenue").alias("avg_revenue"),
    F.min("revenue").alias("min_revenue"),
    F.max("revenue").alias("max_revenue"),
    F.sum(F.when(F.col("revenue") < 0, 1).otherwise(0)).alias("negative_revenue_count")
).collect()[0]

print(f"Total revenue: ${revenue_stats['total_revenue']:,.2f}")
print(f"Average revenue: ${revenue_stats['avg_revenue']:,.2f}")
print(f"Revenue range: ${revenue_stats['min_revenue']:,.2f} to ${revenue_stats['max_revenue']:,.2f}")
print(f"Negative revenue records: {revenue_stats['negative_revenue_count']:,}")

if revenue_stats['negative_revenue_count'] > 0:
    print("⚠ Warning: Found negative revenue values - may need investigation")
else:
    print("✓ No negative revenue values detected")

# Customer distribution validation
customer_status_dist = silver_customer.groupBy("customer_status").count().orderBy(F.desc("count"))
print("\nCustomer status distribution:")
customer_status_dist.show(truncate=False)

print("\n" + "=" * 80)
print("PRE-WRITE VALIDATION COMPLETE - ALL CHECKS PASSED")
print("=" * 80)
print("Ready to persist silver tables to catalog.\n")

# COMMAND ----------

# DBTITLE 1,Write silver tables to catalog
# Write all silver tables to db_analytics schema
# 
# WRITE STRATEGY DECISION (Contract Section A.8):
# Using full-table overwrite (mode="overwrite") for initial load because:
# 1. This is the initial silver layer creation - no existing data to merge
# 2. Bronze layer is full snapshot without incremental markers (no _change_type or watermark)
# 3. Customer identity resolution requires full cross-system view to maintain consistency
# 4. Idempotency: Re-running produces identical results given same bronze input
# 
# INCREMENTAL PROCESSING PLAN:
# Once bronze layer implements incremental ingestion with change tracking:
# - Switch to MERGE for customer master (key: customer_id)
# - Use append mode for behavioral tables with deduplication
# - Add processing_date watermark column for incremental filtering
# 
# PERFORMANCE:
# - No partitioning specified (using Spark defaults) - appropriate for initial load
# - Each write is independent, no dependencies between tables
# - Table sizes are manageable for full refresh in initial deployment

print("Writing silver tables to catalog...\n")

# Write customer identity
silver_customer_identity.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.silver_customer_identity")
print("✓ silver_customer_identity written")

# Write customer master
silver_customer.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.silver_customer")
print("✓ silver_customer written")

# Write products (reference data)
silver_products.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.silver_products")
print("✓ silver_products written")

# Write purchase
silver_purchase.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.silver_purchase")
print("✓ silver_purchase written")

# Write customer service
silver_customer_service.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.silver_customer_service")
print("✓ silver_customer_service written")

# Write loyalty activity
silver_loyalty_activity.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.silver_loyalty_activity")
print("✓ silver_loyalty_activity written")

# Write marketing engagement
silver_marketing_engagement.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.silver_marketing_engagement")
print("✓ silver_marketing_engagement written")

print("\nAll silver tables written successfully!")

# COMMAND ----------

# DBTITLE 1,Final validation - verify silver tables
# Final validation: Verify all silver tables exist and have data
print("=== FINAL SILVER LAYER VALIDATION ===")

silver_tables = spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").filter("tableName LIKE 'silver_%'")
print(f"\nSilver tables created:")
display(silver_tables)

# Row count summary
print("\nRow counts:")
table_stats = [
    ("silver_customer_identity", spark.table(f"{CATALOG}.{SCHEMA}.silver_customer_identity").count()),
    ("silver_customer", spark.table(f"{CATALOG}.{SCHEMA}.silver_customer").count()),
    ("silver_products", spark.table(f"{CATALOG}.{SCHEMA}.silver_products").count()),
    ("silver_purchase", spark.table(f"{CATALOG}.{SCHEMA}.silver_purchase").count()),
    ("silver_customer_service", spark.table(f"{CATALOG}.{SCHEMA}.silver_customer_service").count()),
    ("silver_loyalty_activity", spark.table(f"{CATALOG}.{SCHEMA}.silver_loyalty_activity").count()),
    ("silver_marketing_engagement", spark.table(f"{CATALOG}.{SCHEMA}.silver_marketing_engagement").count())
]

from pyspark.sql import Row
stats_df = spark.createDataFrame([Row(table=t, row_count=c) for t, c in table_stats])
display(stats_df)

print("\n" + "="*60)
print("SILVER LAYER COMPLETE")
print("="*60)
print("Customer 360 foundation created successfully!")
print("- Identity mapping across 6 source systems")
print("- Unified customer master with standardized attributes")
print("- Product reference data with standardized categories and brands")
print("- Clean behavioral tables ready for analytics")