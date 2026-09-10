# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Import libraries and set parameters
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Define schema and table names following silver naming convention
catalog = "workspace"
schema = "retail"

# Get the actual max transaction date from the data (avoid using current_date for historical data)
# This ensures metrics are calculated relative to the actual data end date, not today
max_date_df = spark.table(f"{catalog}.{schema}.silver_purchase").agg(
    F.max("transaction_date").alias("max_date")
)
max_purchase_date = max_date_df.first()["max_date"]
reference_date = F.lit(max_purchase_date)

print(f"Building gold.customer_360 table in {catalog}.{schema}")
print(f"Reference date (max transaction date): {max_purchase_date}")
print(f"Using data-driven reference date instead of current_date() to ensure accurate time-based metrics")

# COMMAND ----------

# DBTITLE 1,Load and prepare base customer data
# Load base customer profile data
# Contract: Select only required columns to minimize data movement
base_customer = spark.table(f"{catalog}.{schema}.silver_customer").select(
    "customer_id",
    "country",
    "region",
    "customer_since",
    "customer_status",
    "customer_segment",
    "loyalty_tier"
)

print(f"Base customer records: {base_customer.count():,}")
print("\nBase customer schema:")
base_customer.printSchema()
display(base_customer.limit(5))

# COMMAND ----------

# DBTITLE 1,Calculate value and purchase metrics
# Load purchase data and calculate value metrics
# Contract: Filter early, select only required columns
purchases = spark.table(f"{catalog}.{schema}.silver_purchase").select(
    "customer_id",
    "transaction_id",
    "transaction_date",
    "revenue"
)

# Calculate date boundaries for time-based metrics
# Using Spark-native date functions for distributed processing
date_12m_ago = F.date_sub(reference_date, 365)
date_90d_ago = F.date_sub(reference_date, 90)
date_60d_ago = F.date_sub(reference_date, 60)
date_30d_ago = F.date_sub(reference_date, 30)

# Calculate value metrics: revenue, transactions, AOV, recency, frequency
# Contract: Use Spark-native aggregations for distributed processing
value_metrics = purchases.groupBy("customer_id").agg(
    # Value metrics - all time
    F.sum("revenue").alias("total_revenue"),
    F.count("transaction_id").alias("number_of_transactions"),
    F.avg("revenue").alias("average_order_value"),
    
    # Recency metrics
    F.max("transaction_date").alias("last_purchase_date"),
    F.datediff(reference_date, F.max("transaction_date")).alias("days_since_last_purchase"),
    
    # Value metrics - last 12 months
    F.sum(F.when(F.col("transaction_date") >= date_12m_ago, F.col("revenue")).otherwise(0)).alias("revenue_last_12_months"),
    F.count(F.when(F.col("transaction_date") >= date_12m_ago, F.col("transaction_id"))).alias("number_of_transactions_last_12_months"),
    
    # Frequency metrics - different time windows
    F.count(F.when(F.col("transaction_date") >= date_12m_ago, F.col("transaction_id"))).alias("purchases_last_12_months"),
    F.count(F.when(F.col("transaction_date") >= date_90d_ago, F.col("transaction_id"))).alias("purchases_last_90_days"),
    F.count(F.when(F.col("transaction_date") >= date_60d_ago, F.col("transaction_id"))).alias("purchases_last_60_days"),
    F.count(F.when(F.col("transaction_date") >= date_30d_ago, F.col("transaction_id"))).alias("purchases_last_30_days")
)

print(f"Value metrics calculated for {value_metrics.count():,} customers")
print("\nValue metrics schema:")
value_metrics.printSchema()
display(value_metrics.limit(5))

# COMMAND ----------

# DBTITLE 1,Calculate marketing engagement metrics
# Load marketing engagement data and calculate metrics
# Contract: Select only required columns
marketing = spark.table(f"{catalog}.{schema}.silver_marketing_engagement").select(
    "customer_id",
    "sent",
    "opened",
    "clicked"
)

# Calculate engagement metrics using Spark-native aggregations
engagement_metrics = marketing.groupBy("customer_id").agg(
    F.sum("sent").alias("marketing_emails_sent"),
    F.sum("opened").alias("marketing_emails_opened"),
    F.sum("clicked").alias("marketing_emails_clicked")
).withColumn(
    # Calculate engagement rate: (opened + clicked) / sent
    # Handle division by zero with nullif or when clause
    "marketing_engagement_rate",
    F.when(
        F.col("marketing_emails_sent") > 0,
        (F.col("marketing_emails_opened") + F.col("marketing_emails_clicked")) / F.col("marketing_emails_sent")
    ).otherwise(0.0)
)

print(f"Engagement metrics calculated for {engagement_metrics.count():,} customers")
print("\nEngagement metrics schema:")
engagement_metrics.printSchema()
display(engagement_metrics.limit(5))

# COMMAND ----------

# DBTITLE 1,Calculate customer service metrics
# Load customer service data and calculate metrics
# Contract: Select only required columns, filter early
service = spark.table(f"{catalog}.{schema}.silver_customer_service").select(
    "customer_id",
    "interaction_date",
    "resolution_status",
    "satisfaction_score"
)

# Calculate service metrics using Spark-native aggregations
service_metrics = service.groupBy("customer_id").agg(
    F.count("*").alias("service_interactions"),
    F.count(F.when(F.col("interaction_date") >= date_90d_ago, 1)).alias("service_interactions_last_90_days"),
    F.avg("satisfaction_score").alias("average_satisfaction_score"),
    F.sum(F.when(F.col("resolution_status") != "resolved", 1).otherwise(0)).alias("unresolved_interactions")
)

print(f"Service metrics calculated for {service_metrics.count():,} customers")
print("\nService metrics schema:")
service_metrics.printSchema()
display(service_metrics.limit(5))

# COMMAND ----------

# DBTITLE 1,Calculate loyalty metrics
# Load loyalty activity data and calculate metrics
# Contract: Select only required columns
loyalty = spark.table(f"{catalog}.{schema}.silver_loyalty_activity").select(
    "customer_id",
    "event_date",
    "loyalty_tier",
    "points_earned",
    "points_redeemed"
)

# Calculate loyalty metrics using Spark-native aggregations
loyalty_metrics = loyalty.groupBy("customer_id").agg(
    F.sum("points_earned").alias("loyalty_points_earned"),
    F.sum("points_redeemed").alias("loyalty_points_redeemed"),
    # Get most recent loyalty tier
    F.last("loyalty_tier").alias("current_loyalty_tier"),
    # Count loyalty events in last 90 days
    F.count(F.when(F.col("event_date") >= date_90d_ago, 1)).alias("loyalty_activity_last_90_days")
)

print(f"Loyalty metrics calculated for {loyalty_metrics.count():,} customers")
print("\nLoyalty metrics schema:")
loyalty_metrics.printSchema()
display(loyalty_metrics.limit(5))

# COMMAND ----------

# DBTITLE 1,Calculate behavioral trend metrics
# Calculate behavioral trend metrics for churn prediction
# Compare current periods vs previous periods to detect changes

# Define additional date boundaries for trend analysis
date_60d_prev_start = F.date_sub(reference_date, 120)  # Previous 60d period
date_60d_prev_end = date_60d_ago
date_30d_prev_start = F.date_sub(reference_date, 60)   # Previous 30d period
date_30d_prev_end = date_30d_ago

# Calculate trend metrics from purchase data
# Contract: Use Spark-native aggregations, avoid Python loops
trend_metrics = purchases.groupBy("customer_id").agg(
    # Purchase frequency trends (30-day comparison)
    F.count(F.when(
        (F.col("transaction_date") >= date_30d_ago) & (F.col("transaction_date") < reference_date),
        F.col("transaction_id")
    )).alias("purchase_frequency_30d"),
    F.count(F.when(
        (F.col("transaction_date") >= date_30d_prev_start) & (F.col("transaction_date") < date_30d_prev_end),
        F.col("transaction_id")
    )).alias("purchase_frequency_previous_30d"),
    
    # Revenue trends (60-day comparison)
    F.sum(F.when(
        (F.col("transaction_date") >= date_60d_ago) & (F.col("transaction_date") < reference_date),
        F.col("revenue")
    ).otherwise(0)).alias("revenue_60d"),
    F.sum(F.when(
        (F.col("transaction_date") >= date_60d_prev_start) & (F.col("transaction_date") < date_60d_prev_end),
        F.col("revenue")
    ).otherwise(0)).alias("revenue_previous_60d"),
    
    # Average basket trends (30-day comparison)
    F.avg(F.when(
        (F.col("transaction_date") >= date_30d_ago) & (F.col("transaction_date") < reference_date),
        F.col("revenue")
    )).alias("avg_basket_30d"),
    F.avg(F.when(
        (F.col("transaction_date") >= date_30d_prev_start) & (F.col("transaction_date") < date_30d_prev_end),
        F.col("revenue")
    )).alias("avg_basket_previous_30d")
).withColumn(
    # Calculate percentage changes with null/zero handling
    "purchase_frequency_change_pct",
    F.when(
        F.col("purchase_frequency_previous_30d") > 0,
        ((F.col("purchase_frequency_30d") - F.col("purchase_frequency_previous_30d")) / F.col("purchase_frequency_previous_30d")) * 100
    ).otherwise(F.lit(None))
).withColumn(
    "revenue_change_pct",
    F.when(
        F.col("revenue_previous_60d") > 0,
        ((F.col("revenue_60d") - F.col("revenue_previous_60d")) / F.col("revenue_previous_60d")) * 100
    ).otherwise(F.lit(None))
).withColumn(
    "avg_basket_change_pct",
    F.when(
        F.col("avg_basket_previous_30d").isNotNull() & (F.col("avg_basket_previous_30d") > 0),
        ((F.col("avg_basket_30d") - F.col("avg_basket_previous_30d")) / F.col("avg_basket_previous_30d")) * 100
    ).otherwise(F.lit(None))
)

print(f"Behavioral trend metrics calculated for {trend_metrics.count():,} customers")
print("\nTrend metrics schema:")
trend_metrics.printSchema()
display(trend_metrics.limit(5))

# COMMAND ----------

# DBTITLE 1,Build customer_360 table by joining all metrics
# Build the complete customer_360 table
# Contract: Use left joins to preserve all customers
# Assumption: All customers in base_customer should appear in final table,
# even if they have no transactions, marketing, service, or loyalty activity

customer_360 = base_customer \
    .join(value_metrics, "customer_id", "left") \
    .join(engagement_metrics, "customer_id", "left") \
    .join(service_metrics, "customer_id", "left") \
    .join(loyalty_metrics, "customer_id", "left") \
    .join(trend_metrics, "customer_id", "left")

# Fill nulls with appropriate defaults for cleaner analytics
# Contract: Handle nulls explicitly for better data quality
customer_360_clean = customer_360.fillna({
    # Value metrics
    "total_revenue": 0.0,
    "revenue_last_12_months": 0.0,
    "average_order_value": 0.0,
    "number_of_transactions": 0,
    "number_of_transactions_last_12_months": 0,
    
    # Frequency metrics
    "purchases_last_30_days": 0,
    "purchases_last_60_days": 0,
    "purchases_last_90_days": 0,
    "purchases_last_12_months": 0,
    
    # Engagement metrics
    "marketing_emails_sent": 0,
    "marketing_emails_opened": 0,
    "marketing_emails_clicked": 0,
    "marketing_engagement_rate": 0.0,
    
    # Service metrics
    "service_interactions": 0,
    "service_interactions_last_90_days": 0,
    "unresolved_interactions": 0,
    
    # Loyalty metrics
    "loyalty_points_earned": 0,
    "loyalty_points_redeemed": 0,
    "loyalty_activity_last_90_days": 0,
    
    # Trend metrics - keep as null if no data (indicates no activity)
    "purchase_frequency_30d": 0,
    "purchase_frequency_previous_30d": 0,
    "revenue_60d": 0.0,
    "revenue_previous_60d": 0.0
})

print(f"\ncustomer_360 table built with {customer_360_clean.count():,} customers")
print("\nFinal schema:")
customer_360_clean.printSchema()

# COMMAND ----------

# DBTITLE 1,Write customer_360 to gold layer
# Write the customer_360 table to the gold layer
# Contract: Following silver naming convention - gold_customer_360
# Write strategy: Using CREATE OR REPLACE for full refresh
# Assumption: This is a daily or periodic full refresh of customer metrics
# For production, consider incremental/MERGE strategy if performance becomes an issue

target_table = f"{catalog}.{schema}.gold_customer_360"

print(f"Writing to {target_table}...")

customer_360_clean.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(target_table)

print(f"✓ Successfully wrote {target_table}")

# Verify the write
result_count = spark.table(target_table).count()
print(f"\nVerification: Table contains {result_count:,} records")

# COMMAND ----------

# DBTITLE 1,VALIDATION: Schema and sample data
# ============================================
# VALIDATION CONTRACT EXECUTION
# ============================================

print("=" * 60)
print("VALIDATION: Schema and Sample Data")
print("=" * 60)

# Load the final table for validation
gold_table = spark.table(target_table)

# 1. Schema validation
print("\n1. SCHEMA VALIDATION")
print("-" * 60)
gold_table.printSchema()

# 2. Sample data - show representative 5 rows
print("\n2. SAMPLE DATA (5 rows)")
print("-" * 60)
display(gold_table.limit(5))

# 3. Column count verification
expected_min_columns = 40  # Based on requirements
actual_columns = len(gold_table.columns)
print(f"\n3. COLUMN COUNT: {actual_columns} columns (expected minimum: {expected_min_columns})")
if actual_columns >= expected_min_columns:
    print("✓ PASS: Sufficient columns present")
else:
    print(f"⚠ WARNING: Only {actual_columns} columns found, expected at least {expected_min_columns}")

# COMMAND ----------

# DBTITLE 1,VALIDATION: Row count reconciliation
print("\n" + "=" * 60)
print("VALIDATION: Row Count Reconciliation")
print("=" * 60)

# Count records in each source table
base_count = spark.table(f"{catalog}.{schema}.silver_customer").count()
gold_count = gold_table.count()

print(f"\nBase customer count (silver_customer): {base_count:,}")
print(f"Gold customer_360 count:                {gold_count:,}")
print(f"Difference:                             {gold_count - base_count:,}")

# Validation: Gold should have same count as base (1:1 relationship)
if gold_count == base_count:
    print("✓ PASS: Row count matches base customer table (1:1 relationship maintained)")
else:
    print(f"⚠ WARNING: Row count mismatch. Expected {base_count:,}, got {gold_count:,}")

# Check for duplicates
print("\nDuplicate check:")
duplicate_count = gold_table.groupBy("customer_id").count().filter("count > 1").count()
if duplicate_count == 0:
    print("✓ PASS: No duplicate customer_ids found")
else:
    print(f"⚠ FAIL: Found {duplicate_count} duplicate customer_ids")

# COMMAND ----------

# DBTITLE 1,VALIDATION: Data quality checks
print("\n" + "=" * 60)
print("VALIDATION: Data Quality Checks")
print("=" * 60)

# Check for nulls in critical columns (customer_id should never be null)
print("\n1. NULL CHECKS FOR CRITICAL COLUMNS")
print("-" * 60)
critical_cols = ["customer_id", "country", "region", "customer_status"]

for col in critical_cols:
    null_count = gold_table.filter(F.col(col).isNull()).count()
    if null_count == 0:
        print(f"✓ {col}: 0 nulls")
    else:
        print(f"⚠ {col}: {null_count:,} nulls found")

# Check metric value distributions
print("\n2. METRIC VALUE DISTRIBUTIONS")
print("-" * 60)

metric_stats = gold_table.select(
    F.count("*").alias("total_customers"),
    F.sum(F.when(F.col("total_revenue") > 0, 1).otherwise(0)).alias("customers_with_revenue"),
    F.sum(F.when(F.col("number_of_transactions") > 0, 1).otherwise(0)).alias("customers_with_transactions"),
    F.sum(F.when(F.col("marketing_emails_sent") > 0, 1).otherwise(0)).alias("customers_with_marketing"),
    F.sum(F.when(F.col("service_interactions") > 0, 1).otherwise(0)).alias("customers_with_service"),
    F.sum(F.when(F.col("loyalty_points_earned") > 0, 1).otherwise(0)).alias("customers_with_loyalty")
).collect()[0]

print(f"Total customers: {metric_stats['total_customers']:,}")
print(f"Customers with revenue: {metric_stats['customers_with_revenue']:,} ({metric_stats['customers_with_revenue']/metric_stats['total_customers']*100:.1f}%)")
print(f"Customers with transactions: {metric_stats['customers_with_transactions']:,} ({metric_stats['customers_with_transactions']/metric_stats['total_customers']*100:.1f}%)")
print(f"Customers with marketing activity: {metric_stats['customers_with_marketing']:,} ({metric_stats['customers_with_marketing']/metric_stats['total_customers']*100:.1f}%)")
print(f"Customers with service interactions: {metric_stats['customers_with_service']:,} ({metric_stats['customers_with_service']/metric_stats['total_customers']*100:.1f}%)")
print(f"Customers with loyalty activity: {metric_stats['customers_with_loyalty']:,} ({metric_stats['customers_with_loyalty']/metric_stats['total_customers']*100:.1f}%)")

# COMMAND ----------

# DBTITLE 1,VALIDATION: Business logic validation
print("\n" + "=" * 60)
print("VALIDATION: Business Logic Validation")
print("=" * 60)

# 1. Check that 12-month metrics <= all-time metrics
print("\n1. LOGICAL CONSISTENCY CHECKS")
print("-" * 60)

logic_check = gold_table.select(
    F.sum(F.when(F.col("revenue_last_12_months") > F.col("total_revenue"), 1).otherwise(0)).alias("revenue_12m_exceeds_total"),
    F.sum(F.when(F.col("number_of_transactions_last_12_months") > F.col("number_of_transactions"), 1).otherwise(0)).alias("trans_12m_exceeds_total"),
    F.sum(F.when((F.col("average_order_value") > 0) & (F.col("number_of_transactions") == 0), 1).otherwise(0)).alias("aov_without_transactions")
).collect()[0]

if logic_check['revenue_12m_exceeds_total'] == 0:
    print("✓ PASS: All 12-month revenue <= total revenue")
else:
    print(f"⚠ FAIL: {logic_check['revenue_12m_exceeds_total']} customers have 12-month revenue > total revenue")

if logic_check['trans_12m_exceeds_total'] == 0:
    print("✓ PASS: All 12-month transactions <= total transactions")
else:
    print(f"⚠ FAIL: {logic_check['trans_12m_exceeds_total']} customers have 12-month transactions > total transactions")

if logic_check['aov_without_transactions'] == 0:
    print("✓ PASS: No customers with AOV but zero transactions")
else:
    print(f"⚠ WARNING: {logic_check['aov_without_transactions']} customers have AOV without transactions")

# 2. Check recency metrics
print("\n2. RECENCY METRICS VALIDATION")
print("-" * 60)

recency_stats = gold_table.filter(F.col("last_purchase_date").isNotNull()).select(
    F.count("*").alias("customers_with_purchases"),
    F.avg("days_since_last_purchase").alias("avg_days_since_purchase"),
    F.min("days_since_last_purchase").alias("min_days_since_purchase"),
    F.max("days_since_last_purchase").alias("max_days_since_purchase")
).collect()[0]

print(f"Customers with purchase history: {recency_stats['customers_with_purchases']:,}")
print(f"Average days since last purchase: {recency_stats['avg_days_since_purchase']:.1f}")
print(f"Min days since last purchase: {recency_stats['min_days_since_purchase']}")
print(f"Max days since last purchase: {recency_stats['max_days_since_purchase']}")

# COMMAND ----------

# DBTITLE 1,VALIDATION: Summary statistics and final checks
print("\n" + "=" * 60)
print("VALIDATION: Summary Statistics")
print("=" * 60)

# Key summary statistics for the customer_360 table
print("\nKEY METRICS SUMMARY:")
print("-" * 60)

summary = gold_table.select(
    F.avg("total_revenue").alias("avg_revenue"),
    F.max("total_revenue").alias("max_revenue"),
    F.avg("number_of_transactions").alias("avg_transactions"),
    F.avg("average_order_value").alias("avg_order_value"),
    F.avg("marketing_engagement_rate").alias("avg_engagement_rate"),
    F.avg("average_satisfaction_score").alias("avg_satisfaction"),
    F.avg("purchases_last_30_days").alias("avg_purchases_30d"),
    F.avg("purchase_frequency_change_pct").alias("avg_freq_change_pct")
).collect()[0]

print(f"Average revenue per customer: ${summary['avg_revenue']:.2f}")
print(f"Maximum revenue: ${summary['max_revenue']:.2f}")
print(f"Average transactions per customer: {summary['avg_transactions']:.2f}")
print(f"Average order value: ${summary['avg_order_value']:.2f}")
print(f"Average marketing engagement rate: {summary['avg_engagement_rate']:.2%}")
print(f"Average satisfaction score: {summary['avg_satisfaction']:.2f}")
print(f"Average purchases (last 30 days): {summary['avg_purchases_30d']:.2f}")

if summary['avg_freq_change_pct'] is not None:
    print(f"Average frequency change: {summary['avg_freq_change_pct']:.1f}%")

# Distribution by customer segment and status
print("\nCUSTOMER DISTRIBUTION:")
print("-" * 60)
print("\nBy Status:")
display(gold_table.groupBy("customer_status").count().orderBy(F.desc("count")))

print("\nBy Segment:")
display(gold_table.groupBy("customer_segment").count().orderBy(F.desc("count")))

print("\nBy Loyalty Tier:")
display(gold_table.groupBy("loyalty_tier").count().orderBy(F.desc("count")))

print("\n" + "=" * 60)
print("✓ VALIDATION COMPLETE")
print("=" * 60)


# DBTITLE 1,Calculate daily purchase activity
# Calculate daily purchase activity
# Contract: Select only required columns, use Spark-native aggregations
# Note: Reloading source data with date columns for daily aggregation

print("Building customer_activity_daily table...")
print("\n1. Loading source data and aggregating daily purchase activity...")

# Reload purchases with date for daily aggregation (already in memory from earlier cells)
# Daily purchase metrics
daily_purchases = purchases.groupBy("customer_id", "transaction_date").agg(
    F.count("transaction_id").alias("daily_transactions"),
    F.sum("revenue").alias("daily_revenue"),
    F.avg("revenue").alias("daily_avg_basket")
).withColumnRenamed("transaction_date", "activity_date")

print("  ✓ Daily purchase metrics aggregated")

# COMMAND ----------

# DBTITLE 1,Calculate daily marketing, service, and loyalty activity
# Calculate daily marketing engagement
# Reload data with date columns for daily aggregation
print("\n2. Loading and aggregating daily marketing engagement...")

marketing_daily_data = spark.table(f"{catalog}.{schema}.silver_marketing_engagement").select(
    "customer_id", "campaign_date", "sent", "opened", "clicked"
)

daily_marketing = marketing_daily_data.withColumnRenamed("campaign_date", "activity_date") \
    .groupBy("customer_id", "activity_date").agg(
        F.sum("sent").alias("daily_emails_sent"),
        F.sum("opened").alias("daily_emails_opened"),
        F.sum("clicked").alias("daily_emails_clicked")
    )

print("  ✓ Daily marketing metrics aggregated")

# Calculate daily service interactions
# Service data already loaded with interaction_date in cell 5
print("\n3. Aggregating daily service interactions...")

daily_service = service.withColumnRenamed("interaction_date", "activity_date") \
    .groupBy("customer_id", "activity_date").agg(
        F.count("*").alias("daily_service_interactions"),
        F.avg("satisfaction_score").alias("daily_avg_satisfaction"),
        F.sum(F.when(F.col("resolution_status") != "resolved", 1).otherwise(0)).alias("daily_unresolved")
    )

print("  ✓ Daily service metrics aggregated")

# Calculate daily loyalty activity
# Loyalty data already loaded with event_date in cell 6
print("\n4. Aggregating daily loyalty activity...")

daily_loyalty = loyalty.withColumnRenamed("event_date", "activity_date") \
    .groupBy("customer_id", "activity_date").agg(
        F.sum("points_earned").alias("daily_points_earned"),
        F.sum("points_redeemed").alias("daily_points_redeemed"),
        F.count("*").alias("daily_loyalty_events")
    )

print("  ✓ Daily loyalty metrics aggregated")

# COMMAND ----------

# DBTITLE 1,Combine daily activities and add rolling windows
# Combine all daily activities
print("\n5. Combining all daily activities...")

# Full outer join to capture all activity dates
# Contract: Use broadcast joins where appropriate (Spark will handle this via AQE)
customer_daily = daily_purchases \
    .join(daily_marketing, ["customer_id", "activity_date"], "full") \
    .join(daily_service, ["customer_id", "activity_date"], "full") \
    .join(daily_loyalty, ["customer_id", "activity_date"], "full")

# Fill nulls for days with no activity in specific channels
customer_daily_clean = customer_daily.fillna({
    "daily_transactions": 0,
    "daily_revenue": 0.0,
    "daily_avg_basket": 0.0,
    "daily_emails_sent": 0,
    "daily_emails_opened": 0,
    "daily_emails_clicked": 0,
    "daily_service_interactions": 0,
    "daily_unresolved": 0,
    "daily_points_earned": 0,
    "daily_points_redeemed": 0,
    "daily_loyalty_events": 0
})

print("  ✓ Daily activities combined")

# Add rolling window metrics for ML features
print("\n6. Calculating rolling window metrics...")

# Define window specifications for rolling calculations
# Contract: Use Spark-native window functions for distributed processing
window_7d = Window.partitionBy("customer_id").orderBy("activity_date").rowsBetween(-6, 0)
window_30d = Window.partitionBy("customer_id").orderBy("activity_date").rowsBetween(-29, 0)

# Add rolling metrics
customer_daily_with_windows = customer_daily_clean \
    .withColumn("rolling_7d_transactions", F.sum("daily_transactions").over(window_7d)) \
    .withColumn("rolling_7d_revenue", F.sum("daily_revenue").over(window_7d)) \
    .withColumn("rolling_30d_transactions", F.sum("daily_transactions").over(window_30d)) \
    .withColumn("rolling_30d_revenue", F.sum("daily_revenue").over(window_30d)) \
    .withColumn("rolling_7d_emails_opened", F.sum("daily_emails_opened").over(window_7d)) \
    .withColumn("rolling_30d_emails_opened", F.sum("daily_emails_opened").over(window_30d))

print("  ✓ Rolling window metrics calculated")

# Add cumulative metrics (running totals)
print("\n7. Calculating cumulative metrics...")

window_cumulative = Window.partitionBy("customer_id").orderBy("activity_date").rowsBetween(Window.unboundedPreceding, 0)

customer_daily_final = customer_daily_with_windows \
    .withColumn("cumulative_transactions", F.sum("daily_transactions").over(window_cumulative)) \
    .withColumn("cumulative_revenue", F.sum("daily_revenue").over(window_cumulative)) \
    .withColumn("cumulative_points_earned", F.sum("daily_points_earned").over(window_cumulative)) \
    .withColumn("cumulative_points_redeemed", F.sum("daily_points_redeemed").over(window_cumulative))

print("  ✓ Cumulative metrics calculated")
print("\nFinal daily activity schema:")
customer_daily_final.printSchema()

# COMMAND ----------

# DBTITLE 1,Write gold_customer_activity_daily table
# Write the customer_activity_daily table
# Contract: Following silver naming convention - gold_customer_activity_daily
# Write strategy: Full refresh (overwrite)
# Performance note: This table can be large; consider partitioning by activity_date in production

target_table_daily = f"{catalog}.{schema}.gold_customer_activity_daily"

print(f"\nWriting to {target_table_daily}...")
print("Note: This may take longer due to table size and window calculations")

customer_daily_final.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(target_table_daily)

print(f"✓ Successfully wrote {target_table_daily}")

# Verify the write
result_count_daily = spark.table(target_table_daily).count()
print(f"\nVerification: Table contains {result_count_daily:,} records (customer-day combinations)")

# COMMAND ----------

# DBTITLE 1,VALIDATION: Daily activity table checks
print("\n" + "=" * 60)
print("VALIDATION: Daily Activity Table")
print("=" * 60)

daily_table = spark.table(target_table_daily)

# 1. Schema validation
print("\n1. SCHEMA VALIDATION")
print("-" * 60)
daily_table.printSchema()

# 2. Sample data
print("\n2. SAMPLE DATA (5 rows)")
print("-" * 60)
display(daily_table.orderBy(F.desc("activity_date"), F.desc("daily_revenue")).limit(5))

# 3. Date range check
print("\n3. DATE RANGE VALIDATION")
print("-" * 60)

date_stats = daily_table.select(
    F.min("activity_date").alias("earliest_date"),
    F.max("activity_date").alias("latest_date"),
    F.countDistinct("activity_date").alias("distinct_dates"),
    F.countDistinct("customer_id").alias("distinct_customers")
).collect()[0]

print(f"Earliest date: {date_stats['earliest_date']}")
print(f"Latest date: {date_stats['latest_date']}")
print(f"Distinct dates: {date_stats['distinct_dates']:,}")
print(f"Distinct customers: {date_stats['distinct_customers']:,}")

# 4. Data quality checks
print("\n4. DATA QUALITY CHECKS")
print("-" * 60)

quality_stats = daily_table.select(
    F.count("*").alias("total_rows"),
    F.sum(F.when(F.col("daily_transactions") > 0, 1).otherwise(0)).alias("days_with_purchases"),
    F.sum(F.when(F.col("daily_emails_sent") > 0, 1).otherwise(0)).alias("days_with_marketing"),
    F.sum(F.when(F.col("daily_service_interactions") > 0, 1).otherwise(0)).alias("days_with_service"),
    F.sum(F.when(F.col("daily_loyalty_events") > 0, 1).otherwise(0)).alias("days_with_loyalty")
).collect()[0]

print(f"Total customer-day records: {quality_stats['total_rows']:,}")
print(f"Days with purchases: {quality_stats['days_with_purchases']:,} ({quality_stats['days_with_purchases']/quality_stats['total_rows']*100:.1f}%)")
print(f"Days with marketing: {quality_stats['days_with_marketing']:,} ({quality_stats['days_with_marketing']/quality_stats['total_rows']*100:.1f}%)")
print(f"Days with service: {quality_stats['days_with_service']:,} ({quality_stats['days_with_service']/quality_stats['total_rows']*100:.1f}%)")
print(f"Days with loyalty: {quality_stats['days_with_loyalty']:,} ({quality_stats['days_with_loyalty']/quality_stats['total_rows']*100:.1f}%)")

# 5. Rolling window validation
print("\n5. ROLLING WINDOW VALIDATION")
print("-" * 60)

window_check = daily_table.filter(
    (F.col("rolling_7d_revenue") < 0) | 
    (F.col("rolling_30d_revenue") < 0) |
    (F.col("cumulative_revenue") < 0)
).count()

if window_check == 0:
    print("✓ PASS: All rolling window and cumulative metrics are non-negative")
else:
    print(f"⚠ WARNING: Found {window_check} rows with negative rolling/cumulative values")

print("\n" + "=" * 60)
print("✓ DAILY ACTIVITY TABLE VALIDATION COMPLETE")
print("=" * 60)


# DBTITLE 1,OPTIMIZED: Load and prepare base customer data
# Load base customer profile data
# Contract: Select only required columns to minimize data movement
# OPTIMIZATION: Removed unnecessary .count() call to avoid extra scan
base_customer = spark.table(f"{catalog}.{schema}.silver_customer").select(
    "customer_id",
    "country",
    "region",
    "customer_since",
    "customer_status",
    "customer_segment",
    "loyalty_tier"
)

print("Base customer data loaded")
print("\nBase customer schema:")
base_customer.printSchema()
display(base_customer.limit(5))

# COMMAND ----------

# DBTITLE 1,OPTIMIZED: Calculate value and purchase metrics
# Load purchase data and calculate value metrics
# Contract: Filter early, select only required columns
# OPTIMIZATION: Removed unnecessary .count() call to avoid extra scan
purchases = spark.table(f"{catalog}.{schema}.silver_purchase").select(
    "customer_id",
    "transaction_id",
    "transaction_date",
    "revenue"
)

# Calculate date boundaries for time-based metrics
# Using Spark-native date functions for distributed processing
date_12m_ago = F.date_sub(reference_date, 365)
date_90d_ago = F.date_sub(reference_date, 90)
date_60d_ago = F.date_sub(reference_date, 60)
date_30d_ago = F.date_sub(reference_date, 30)

# Calculate value metrics: revenue, transactions, AOV, recency, frequency
# Contract: Use Spark-native aggregations for distributed processing
value_metrics = purchases.groupBy("customer_id").agg(
    # Value metrics - all time
    F.sum("revenue").alias("total_revenue"),
    F.count("transaction_id").alias("number_of_transactions"),
    F.avg("revenue").alias("average_order_value"),
    
    # Recency metrics
    F.max("transaction_date").alias("last_purchase_date"),
    F.datediff(reference_date, F.max("transaction_date")).alias("days_since_last_purchase"),
    
    # Value metrics - last 12 months
    F.sum(F.when(F.col("transaction_date") >= date_12m_ago, F.col("revenue")).otherwise(0)).alias("revenue_last_12_months"),
    F.count(F.when(F.col("transaction_date") >= date_12m_ago, F.col("transaction_id"))).alias("number_of_transactions_last_12_months"),
    
    # Frequency metrics - different time windows
    F.count(F.when(F.col("transaction_date") >= date_12m_ago, F.col("transaction_id"))).alias("purchases_last_12_months"),
    F.count(F.when(F.col("transaction_date") >= date_90d_ago, F.col("transaction_id"))).alias("purchases_last_90_days"),
    F.count(F.when(F.col("transaction_date") >= date_60d_ago, F.col("transaction_id"))).alias("purchases_last_60_days"),
    F.count(F.when(F.col("transaction_date") >= date_30d_ago, F.col("transaction_id"))).alias("purchases_last_30_days")
)

print("Value metrics calculated")
print("\nValue metrics schema:")
value_metrics.printSchema()
display(value_metrics.limit(5))

# COMMAND ----------

# DBTITLE 1,OPTIMIZED: Calculate all remaining metrics and build customer_360
# OPTIMIZED VERSION: Calculate all metrics and build final table in one pass
# Removes 5 unnecessary .count() operations (anti-pattern elimination)

print("Calculating all customer metrics...")

# Marketing engagement metrics
marketing = spark.table(f"{catalog}.{schema}.silver_marketing_engagement").select(
    "customer_id", "sent", "opened", "clicked"
)

engagement_metrics = marketing.groupBy("customer_id").agg(
    F.sum("sent").alias("marketing_emails_sent"),
    F.sum("opened").alias("marketing_emails_opened"),
    F.sum("clicked").alias("marketing_emails_clicked")
).withColumn(
    "marketing_engagement_rate",
    F.when(
        F.col("marketing_emails_sent") > 0,
        (F.col("marketing_emails_opened") + F.col("marketing_emails_clicked")) / F.col("marketing_emails_sent")
    ).otherwise(0.0)
)

print("  ✓ Marketing engagement metrics calculated")

# Customer service metrics
service = spark.table(f"{catalog}.{schema}.silver_customer_service").select(
    "customer_id", "interaction_date", "resolution_status", "satisfaction_score"
)

service_metrics = service.groupBy("customer_id").agg(
    F.count("*").alias("service_interactions"),
    F.count(F.when(F.col("interaction_date") >= date_90d_ago, 1)).alias("service_interactions_last_90_days"),
    F.avg("satisfaction_score").alias("average_satisfaction_score"),
    F.sum(F.when(F.col("resolution_status") != "resolved", 1).otherwise(0)).alias("unresolved_interactions")
)

print("  ✓ Customer service metrics calculated")

# Loyalty metrics
loyalty = spark.table(f"{catalog}.{schema}.silver_loyalty_activity").select(
    "customer_id", "event_date", "loyalty_tier", "points_earned", "points_redeemed"
)

loyalty_metrics = loyalty.groupBy("customer_id").agg(
    F.sum("points_earned").alias("loyalty_points_earned"),
    F.sum("points_redeemed").alias("loyalty_points_redeemed"),
    F.last("loyalty_tier").alias("current_loyalty_tier"),
    F.count(F.when(F.col("event_date") >= date_90d_ago, 1)).alias("loyalty_activity_last_90_days")
)

print("  ✓ Loyalty metrics calculated")

# Behavioral trend metrics
date_60d_prev_start = F.date_sub(reference_date, 120)
date_60d_prev_end = date_60d_ago
date_30d_prev_start = F.date_sub(reference_date, 60)
date_30d_prev_end = date_30d_ago

trend_metrics = purchases.groupBy("customer_id").agg(
    F.count(F.when(
        (F.col("transaction_date") >= date_30d_ago) & (F.col("transaction_date") < reference_date),
        F.col("transaction_id")
    )).alias("purchase_frequency_30d"),
    F.count(F.when(
        (F.col("transaction_date") >= date_30d_prev_start) & (F.col("transaction_date") < date_30d_prev_end),
        F.col("transaction_id")
    )).alias("purchase_frequency_previous_30d"),
    F.sum(F.when(
        (F.col("transaction_date") >= date_60d_ago) & (F.col("transaction_date") < reference_date),
        F.col("revenue")
    ).otherwise(0)).alias("revenue_60d"),
    F.sum(F.when(
        (F.col("transaction_date") >= date_60d_prev_start) & (F.col("transaction_date") < date_60d_prev_end),
        F.col("revenue")
    ).otherwise(0)).alias("revenue_previous_60d"),
    F.avg(F.when(
        (F.col("transaction_date") >= date_30d_ago) & (F.col("transaction_date") < reference_date),
        F.col("revenue")
    )).alias("avg_basket_30d"),
    F.avg(F.when(
        (F.col("transaction_date") >= date_30d_prev_start) & (F.col("transaction_date") < date_30d_prev_end),
        F.col("revenue")
    )).alias("avg_basket_previous_30d")
).withColumn(
    "purchase_frequency_change_pct",
    F.when(
        F.col("purchase_frequency_previous_30d") > 0,
        ((F.col("purchase_frequency_30d") - F.col("purchase_frequency_previous_30d")) / F.col("purchase_frequency_previous_30d")) * 100
    ).otherwise(F.lit(None))
).withColumn(
    "revenue_change_pct",
    F.when(
        F.col("revenue_previous_60d") > 0,
        ((F.col("revenue_60d") - F.col("revenue_previous_60d")) / F.col("revenue_previous_60d")) * 100
    ).otherwise(F.lit(None))
).withColumn(
    "avg_basket_change_pct",
    F.when(
        F.col("avg_basket_previous_30d").isNotNull() & (F.col("avg_basket_previous_30d") > 0),
        ((F.col("avg_basket_30d") - F.col("avg_basket_previous_30d")) / F.col("avg_basket_previous_30d")) * 100
    ).otherwise(F.lit(None))
)

print("  ✓ Behavioral trend metrics calculated")

# Build the complete customer_360 table
print("\nBuilding customer_360 table...")

customer_360 = base_customer \
    .join(value_metrics, "customer_id", "left") \
    .join(engagement_metrics, "customer_id", "left") \
    .join(service_metrics, "customer_id", "left") \
    .join(loyalty_metrics, "customer_id", "left") \
    .join(trend_metrics, "customer_id", "left")

# Fill nulls with appropriate defaults
customer_360_clean = customer_360.fillna({
    "total_revenue": 0.0,
    "revenue_last_12_months": 0.0,
    "average_order_value": 0.0,
    "number_of_transactions": 0,
    "number_of_transactions_last_12_months": 0,
    "purchases_last_30_days": 0,
    "purchases_last_60_days": 0,
    "purchases_last_90_days": 0,
    "purchases_last_12_months": 0,
    "marketing_emails_sent": 0,
    "marketing_emails_opened": 0,
    "marketing_emails_clicked": 0,
    "marketing_engagement_rate": 0.0,
    "service_interactions": 0,
    "service_interactions_last_90_days": 0,
    "unresolved_interactions": 0,
    "loyalty_points_earned": 0,
    "loyalty_points_redeemed": 0,
    "loyalty_activity_last_90_days": 0,
    "purchase_frequency_30d": 0,
    "purchase_frequency_previous_30d": 0,
    "revenue_60d": 0.0,
    "revenue_previous_60d": 0.0
})

print("  ✓ Customer_360 table built (no intermediate scans performed)")
print("\nFinal schema:")
customer_360_clean.printSchema()
