# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Setup and Configuration
# Configuration for synthetic data generation
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime, timedelta
import random

# Unity Catalog configuration
CATALOG = "workspace"
SCHEMA = "retail"
VOLUME = "databrics_customer_analytics"

# Data generation parameters
NUM_CUSTOMERS = 20_000            #5_000_000
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2026, 8, 31)  # 24 months of data
TOTAL_DAYS = (END_DATE - START_DATE).days  # Calculate actual days between START_DATE and END_DATE

# European locations
LOCATIONS = {
    "Italy": ["Milan", "Florence", "Genoa"],
    "France": ["Paris"],
    "Spain": ["Madrid"],
    "Portugal": ["Lisbon"]
}

print(f"Configuration loaded")
print(f"Target: {CATALOG}.{SCHEMA}.{VOLUME}")
print(f"Customer count: {NUM_CUSTOMERS:,}")
print(f"Date range: {START_DATE.date()} to {END_DATE.date()}")

# COMMAND ----------

# DBTITLE 1,Create Unity Catalog Volume
# MAGIC %sql
# MAGIC -- Create catalog if not exists
# MAGIC CREATE CATALOG IF NOT EXISTS workspace;
# MAGIC
# MAGIC -- Create schema if not exists
# MAGIC CREATE SCHEMA IF NOT EXISTS workspace.retail;
# MAGIC
# MAGIC -- Create volume if not exists
# MAGIC CREATE VOLUME IF NOT EXISTS workspace.retail.databrics_customer_analytics;

# COMMAND ----------

# DBTITLE 1,Generate CRM Base Data
# Generate CRM data with realistic patterns and data quality issues
# Using Spark-native operations for distributed generation

# Generate customer base using Spark range for distributed processing
customer_df = spark.range(NUM_CUSTOMERS).toDF("customer_id")

# Add realistic attributes using Spark-native functions
crm_df = (customer_df
    # Generate names with some missing values (2% missing)
    .withColumn("first_name", 
        F.when(F.rand() < 0.98, 
            F.array_join(F.array(
                F.element_at(F.array(F.lit("Marco"), F.lit("Giovanni"), F.lit("Luca"), F.lit("Pierre"), F.lit("Jean"), 
                    F.lit("Carlos"), F.lit("Miguel"), F.lit("João"), F.lit("Pedro"), F.lit("Maria")), 
                    (F.rand() * 10).cast("int") + 1)
            ), ""))
    )
    .withColumn("last_name",
        F.when(F.rand() < 0.97,  # 3% missing
            F.array_join(F.array(
                F.element_at(F.array(F.lit("Rossi"), F.lit("Ferrari"), F.lit("Bianchi"), F.lit("Dubois"), F.lit("Martin"), 
                    F.lit("Garcia"), F.lit("Rodriguez"), F.lit("Silva"), F.lit("Santos"), F.lit("Costa")), 
                    (F.rand() * 10).cast("int") + 1)
            ), ""))
    )
    # Generate email with customer_id for join key variations
    .withColumn("email", 
        F.when(F.rand() < 0.95,  # 5% missing emails
            F.concat(
                F.lower(F.col("first_name")), 
                F.lit("."),
                F.lower(F.col("last_name")),
                F.lit("@"),
                F.when(F.rand() < 0.5, F.lit("gmail.com")).otherwise(F.lit("email.com"))
            )
        )
    )
    # Assign countries and regions
    .withColumn("country_code", 
        F.when(F.rand() < 0.40, F.lit("IT"))  # Italy 40%
         .when(F.rand() < 0.65, F.lit("FR"))  # France 25%
         .when(F.rand() < 0.85, F.lit("ES"))  # Spain 20%
         .otherwise(F.lit("PT"))               # Portugal 15%
    )
    # Add inconsistent country naming (data quality issue)
    .withColumn("country",
        F.when(F.col("country_code") == "IT", 
            F.when(F.rand() < 0.8, F.lit("Italy")).otherwise(F.lit("ITALY")))
         .when(F.col("country_code") == "FR", 
            F.when(F.rand() < 0.8, F.lit("France")).otherwise(F.lit("france")))
         .when(F.col("country_code") == "ES", F.lit("Spain"))
         .otherwise(F.lit("Portugal"))
    )
    # Assign regions based on country
    .withColumn("region",
        F.when(F.col("country_code") == "IT",
            F.when(F.rand() < 0.5, F.lit("Milan"))
             .when(F.rand() < 0.75, F.lit("Florence"))
             .otherwise(F.lit("Genoa")))
         .when(F.col("country_code") == "FR", F.lit("Paris"))
         .when(F.col("country_code") == "ES", F.lit("Madrid"))
         .otherwise(F.lit("Lisbon"))
    )
    # Customer since date (varied across 2+ years before start date)
    .withColumn("days_before_start", (F.rand() * 730).cast("int"))  # 0-730 days before start (intentionally using 730 for historical dates)
    .withColumn("customer_since", 
        F.date_sub(F.lit(START_DATE.date()), F.col("days_before_start")))
    # Customer status with some variations
    .withColumn("customer_status",
        F.when(F.rand() < 0.80, F.lit("active"))
         .when(F.rand() < 0.90, F.lit("ACTIVE"))  # Inconsistent casing
         .otherwise(F.lit("inactive"))
    )
    # Age band distribution
    .withColumn("age_band",
        F.when(F.rand() < 0.15, F.lit("18-24"))
         .when(F.rand() < 0.35, F.lit("25-34"))
         .when(F.rand() < 0.55, F.lit("35-44"))
         .when(F.rand() < 0.75, F.lit("45-54"))
         .when(F.rand() < 0.90, F.lit("55-64"))
         .otherwise(F.lit("65+"))
    )
    # Preferred channel with some missing values
    .withColumn("preferred_channel",
        F.when(F.rand() < 0.05, F.lit(None))  # 5% missing
         .when(F.rand() < 0.40, F.lit("online"))
         .when(F.rand() < 0.70, F.lit("store"))
         .otherwise(F.lit("mobile"))
    )
    .drop("days_before_start", "country_code")
)

# Store count to avoid repeated scans (contract: avoid repeated scans)
crm_count = crm_df.count()
print(f"CRM data generated: {crm_count:,} customers")
print("\nSchema:")
crm_df.printSchema()
print("\nSample (5 rows):")
display(crm_df.limit(5))

# COMMAND ----------

# DBTITLE 1,Generate Product Reference Data
# Generate Product dimension table
# Reference data for all products (PROD1-PROD500)

# Create 500 products
product_base = spark.range(1, 501).toDF("product_num")

products_df = (product_base
    # Format product_id to match transaction data
    .withColumn("product_id", F.concat(F.lit("PROD"), F.col("product_num").cast("string")))
    
    # Assign product categories with distribution
    .withColumn("product_category",
        F.when(F.col("product_num") <= 125, F.lit("Electronics"))           # 25%
         .when(F.col("product_num") <= 200, F.lit("Fashion & Apparel"))     # 15%
         .when(F.col("product_num") <= 275, F.lit("Home & Garden"))         # 15%
         .when(F.col("product_num") <= 350, F.lit("Beauty & Personal Care")) # 15%
         .when(F.col("product_num") <= 400, F.lit("Sports & Outdoors"))     # 10%
         .when(F.col("product_num") <= 440, F.lit("Food & Beverages"))      # 8%
         .when(F.col("product_num") <= 470, F.lit("Books & Media"))         # 6%
         .otherwise(F.lit("Health & Wellness"))                              # 6%
    )
    
    # Generate category-appropriate product names
    .withColumn("product_name",
        F.when(F.col("product_category") == "Electronics",
            F.element_at(F.array(
                F.lit("Samsung Smart TV"), F.lit("Apple iPhone"), F.lit("Sony Headphones"),
                F.lit("Dell Laptop"), F.lit("HP Printer"), F.lit("LG Monitor"),
                F.lit("Canon Camera"), F.lit("Apple iPad"), F.lit("Samsung Galaxy"),
                F.lit("Bluetooth Speaker"), F.lit("Wireless Mouse"), F.lit("Gaming Keyboard")
            ), (F.rand() * 12).cast("int") + 1))
         .when(F.col("product_category") == "Fashion & Apparel",
            F.element_at(F.array(
                F.lit("Nike Sneakers"), F.lit("Adidas Running Shoes"), F.lit("Levi's Jeans"),
                F.lit("Zara Dress"), F.lit("H&M T-Shirt"), F.lit("Gucci Handbag"),
                F.lit("Tommy Hilfiger Jacket"), F.lit("Puma Tracksuit"), F.lit("Ray-Ban Sunglasses"),
                F.lit("Calvin Klein Underwear"), F.lit("North Face Jacket"), F.lit("Converse Sneakers")
            ), (F.rand() * 12).cast("int") + 1))
         .when(F.col("product_category") == "Home & Garden",
            F.element_at(F.array(
                F.lit("IKEA Sofa"), F.lit("Dyson Vacuum"), F.lit("Philips Air Fryer"),
                F.lit("Garden Tool Set"), F.lit("LED Lamp"), F.lit("Kitchen Mixer"),
                F.lit("Coffee Maker"), F.lit("Bed Sheets Set"), F.lit("Wall Clock"),
                F.lit("Plant Pot"), F.lit("Storage Box"), F.lit("Curtains")
            ), (F.rand() * 12).cast("int") + 1))
         .when(F.col("product_category") == "Beauty & Personal Care",
            F.element_at(F.array(
                F.lit("L'Oréal Shampoo"), F.lit("Nivea Cream"), F.lit("Gillette Razor"),
                F.lit("Dove Soap"), F.lit("Oral-B Toothbrush"), F.lit("Maybelline Mascara"),
                F.lit("Lancôme Perfume"), F.lit("Neutrogena Moisturizer"), F.lit("Colgate Toothpaste"),
                F.lit("Head & Shoulders"), F.lit("Pantene Conditioner"), F.lit("Garnier Face Mask")
            ), (F.rand() * 12).cast("int") + 1))
         .when(F.col("product_category") == "Sports & Outdoors",
            F.element_at(F.array(
                F.lit("Yoga Mat"), F.lit("Dumbbell Set"), F.lit("Tennis Racket"),
                F.lit("Football"), F.lit("Bicycle Helmet"), F.lit("Running Belt"),
                F.lit("Camping Tent"), F.lit("Hiking Backpack"), F.lit("Water Bottle"),
                F.lit("Fitness Tracker"), F.lit("Swim Goggles"), F.lit("Golf Clubs")
            ), (F.rand() * 12).cast("int") + 1))
         .when(F.col("product_category") == "Food & Beverages",
            F.element_at(F.array(
                F.lit("Lavazza Coffee"), F.lit("Barilla Pasta"), F.lit("Ferrero Chocolate"),
                F.lit("San Pellegrino Water"), F.lit("Parmigiano Cheese"), F.lit("Nutella Spread"),
                F.lit("Peroni Beer"), F.lit("Olive Oil"), F.lit("Balsamic Vinegar"),
                F.lit("Italian Wine"), F.lit("Espresso Beans"), F.lit("Pasta Sauce")
            ), (F.rand() * 12).cast("int") + 1))
         .when(F.col("product_category") == "Books & Media",
            F.element_at(F.array(
                F.lit("Fiction Novel"), F.lit("Cookbook"), F.lit("Travel Guide"),
                F.lit("Biography"), F.lit("Business Book"), F.lit("Children's Book"),
                F.lit("Magazine Subscription"), F.lit("Art Book"), F.lit("History Book"),
                F.lit("Self-Help Book"), F.lit("Comic Book"), F.lit("Poetry Collection")
            ), (F.rand() * 12).cast("int") + 1))
         .otherwise(  # Health & Wellness
            F.element_at(F.array(
                F.lit("Vitamin C Tablets"), F.lit("Protein Powder"), F.lit("Omega-3 Supplements"),
                F.lit("Multivitamins"), F.lit("Probiotics"), F.lit("Herbal Tea"),
                F.lit("Essential Oils"), F.lit("Massage Oil"), F.lit("Sleep Aid"),
                F.lit("Energy Bars"), F.lit("Zinc Supplements"), F.lit("Green Tea Extract")
            ), (F.rand() * 12).cast("int") + 1))
    )
    
    # Assign brands by category
    .withColumn("brand",
        F.when(F.col("product_category") == "Electronics",
            F.when(F.rand() < 0.25, F.lit("Samsung"))
             .when(F.rand() < 0.45, F.lit("Apple"))
             .when(F.rand() < 0.60, F.lit("Sony"))
             .when(F.rand() < 0.75, F.lit("LG"))
             .when(F.rand() < 0.85, F.lit("HP"))
             .otherwise(F.lit("Dell")))
         .when(F.col("product_category") == "Fashion & Apparel",
            F.when(F.rand() < 0.20, F.lit("Nike"))
             .when(F.rand() < 0.40, F.lit("Adidas"))
             .when(F.rand() < 0.55, F.lit("Zara"))
             .when(F.rand() < 0.70, F.lit("H&M"))
             .when(F.rand() < 0.85, F.lit("Levi's"))
             .otherwise(F.lit("Puma")))
         .when(F.col("product_category") == "Home & Garden",
            F.when(F.rand() < 0.30, F.lit("IKEA"))
             .when(F.rand() < 0.50, F.lit("Philips"))
             .when(F.rand() < 0.70, F.lit("Dyson"))
             .otherwise(F.lit("Bosch")))
         .when(F.col("product_category") == "Beauty & Personal Care",
            F.when(F.rand() < 0.20, F.lit("L'Oréal"))
             .when(F.rand() < 0.40, F.lit("Nivea"))
             .when(F.rand() < 0.55, F.lit("Dove"))
             .when(F.rand() < 0.70, F.lit("Gillette"))
             .otherwise(F.lit("Neutrogena")))
         .when(F.col("product_category") == "Sports & Outdoors",
            F.when(F.rand() < 0.35, F.lit("Nike"))
             .when(F.rand() < 0.60, F.lit("Adidas"))
             .when(F.rand() < 0.80, F.lit("Decathlon"))
             .otherwise(F.lit("Under Armour")))
         .when(F.col("product_category") == "Food & Beverages",
            F.when(F.rand() < 0.25, F.lit("Barilla"))
             .when(F.rand() < 0.45, F.lit("Lavazza"))
             .when(F.rand() < 0.65, F.lit("Ferrero"))
             .otherwise(F.lit("Nestlé")))
         .when(F.col("product_category") == "Books & Media",
            F.when(F.rand() < 0.40, F.lit("Penguin Books"))
             .when(F.rand() < 0.70, F.lit("HarperCollins"))
             .otherwise(F.lit("Random House")))
         .otherwise(  # Health & Wellness
            F.when(F.rand() < 0.35, F.lit("Nature's Way"))
             .when(F.rand() < 0.65, F.lit("Centrum"))
             .otherwise(F.lit("GNC")))
    )
    
    # Add some data quality issues (2% missing brand names)
    .withColumn("brand",
        F.when(F.rand() < 0.02, F.lit(None))
         .otherwise(F.col("brand")))
    
    # Add inconsistent brand casing (data quality issue)
    .withColumn("brand",
        F.when((F.col("brand").isNotNull()) & (F.rand() < 0.10),
            F.upper(F.col("brand")))  # 10% uppercase
         .otherwise(F.col("brand")))
    
    .select("product_id", "product_name", "product_category", "brand")
)

# Store count
product_count = products_df.count()
print(f"Product reference data generated: {product_count:,} products")
print("\nSchema:")
products_df.printSchema()
print("\nSample (10 rows):")
display(products_df.limit(10))
print("\nCategory distribution:")
display(products_df.groupBy("product_category").count().orderBy(F.desc("count")))

# COMMAND ----------

# DBTITLE 1,Generate E-commerce Transactions
# Generate E-commerce transactions with varying customer activity patterns
# Transactions distributed over 24 months

# Customer behavior segments (using Spark-native operations)
customers_segmented = (crm_df
    .select("customer_id")
    .withColumn("segment_rand", F.rand())
    # Define customer activity patterns
    .withColumn("customer_segment",
        F.when(F.col("segment_rand") < 0.15, F.lit("frequent"))      # 15% frequent
         .when(F.col("segment_rand") < 0.35, F.lit("moderate"))      # 20% moderate
         .when(F.col("segment_rand") < 0.65, F.lit("occasional"))    # 30% occasional  
         .when(F.col("segment_rand") < 0.85, F.lit("declining"))     # 20% declining
         .otherwise(F.lit("inactive"))                                 # 15% mostly inactive
    )
    # Assign transaction frequency based on segment
    .withColumn("avg_transactions_per_month",
        F.when(F.col("customer_segment") == "frequent", (F.rand() * 20 + 15).cast("int"))      # 15-35 per month
         .when(F.col("customer_segment") == "moderate", (F.rand() * 8 + 4).cast("int"))        # 4-12 per month
         .when(F.col("customer_segment") == "occasional", (F.rand() * 4 + 1).cast("int"))      # 1-5 per month
         .when(F.col("customer_segment") == "declining", (F.rand() * 3 + 1).cast("int"))       # 1-4 per month (will decline over time)
         .otherwise((F.rand() * 2).cast("int"))                                                  # 0-2 per month
    )
    .drop("segment_rand")
)

# Generate transaction counts per customer (24 months average)
total_months = 24
ecommerce_base = (customers_segmented
    .withColumn("total_transactions", 
        (F.col("avg_transactions_per_month") * total_months * F.rand()).cast("int"))
    .filter(F.col("total_transactions") > 0)  # Only customers with at least 1 transaction
)

# Explode to create individual transaction rows
ecommerce_transactions = (ecommerce_base
    .withColumn("transaction_num", F.expr("explode(sequence(1, total_transactions))"))
    .withColumn("transaction_id", 
        F.concat(F.lit("EC"), F.col("customer_id").cast("string"), F.lit("-"), F.col("transaction_num").cast("string")))
    # Generate transaction dates across 24 months
    .withColumn("days_offset", (F.rand() * TOTAL_DAYS).cast("int"))  # Random day within date range
    .withColumn("transaction_date", F.date_add(F.lit(START_DATE.date()), F.col("days_offset")))
    # Add declining pattern for declining segment (fewer transactions in later months)
    .withColumn("transaction_date",
        F.when((F.col("customer_segment") == "declining") & (F.rand() < 0.7),
            F.date_add(F.lit(START_DATE.date()), (F.rand() * 365).cast("int")))  # Concentrate in first year
         .otherwise(F.col("transaction_date"))
    )
    # Product and pricing
    .withColumn("product_id", F.concat(F.lit("PROD"), ((F.rand() * 500) + 1).cast("int").cast("string")))
    .withColumn("quantity", ((F.rand() * 5) + 1).cast("int"))
    .withColumn("unit_price", F.round((F.rand() * 200 + 10), 2))  # €10-€210
    .withColumn("total_amount", F.round(F.col("quantity") * F.col("unit_price"), 2))
    # Channel and device with inconsistent values (data quality issue)
    .withColumn("channel",
        F.when(F.rand() < 0.33, F.lit("web"))
         .when(F.rand() < 0.67, F.lit("WEB"))      # Inconsistent casing
         .otherwise(F.lit("mobile"))
    )
    .withColumn("device_type",
        F.when(F.rand() < 0.40, F.lit("desktop"))
         .when(F.rand() < 0.70, F.lit("mobile"))
         .when(F.rand() < 0.95, F.lit("tablet"))
         .otherwise(F.lit(None))  # Some missing values
    )
    # Session ID
    .withColumn("session_id", 
        F.concat(F.lit("SES"), F.col("customer_id").cast("string"), F.lit("-"), 
                 F.date_format(F.col("transaction_date"), "yyyyMMdd"), F.lit("-"),
                 ((F.rand() * 100) + 1).cast("int").cast("string")))
    .select("customer_id", "transaction_id", "transaction_date", "product_id", 
            "quantity", "unit_price", "total_amount", "channel", "device_type", "session_id")
)

# Add duplicates (2% duplicate records as data quality issue)
# More efficient: sample directly without materializing intermediate flag column
ecommerce_df = ecommerce_transactions.union(
    ecommerce_transactions.sample(withReplacement=False, fraction=0.02)
)

# Store count to avoid repeated scans (contract: avoid repeated scans)
ecommerce_count = ecommerce_df.count()
print(f"E-commerce transactions generated: {ecommerce_count:,} records")
print("\nSchema:")
ecommerce_df.printSchema()
print("\nSample (5 rows):")
display(ecommerce_df.limit(5))

# COMMAND ----------

# DBTITLE 1,Generate POS Transactions
# Generate POS (Point of Sale) transactions for in-store purchases
# Similar pattern to e-commerce but for physical stores

# Different customer sample for POS (some customers shop both online and in-store)
pos_customers = (crm_df
    .select("customer_id", "region")
    .withColumn("shops_in_store", F.when(F.rand() < 0.60, 1).otherwise(0))  # 60% shop in stores
    .filter(F.col("shops_in_store") == 1)
    .withColumn("segment_rand", F.rand())
    .withColumn("customer_segment",
        F.when(F.col("segment_rand") < 0.20, F.lit("frequent"))  
         .when(F.col("segment_rand") < 0.50, F.lit("moderate"))  
         .otherwise(F.lit("occasional"))
    )
    .withColumn("avg_transactions_per_month",
        F.when(F.col("customer_segment") == "frequent", (F.rand() * 10 + 5).cast("int"))   
         .when(F.col("customer_segment") == "moderate", (F.rand() * 6 + 2).cast("int"))   
         .otherwise((F.rand() * 3 + 1).cast("int"))  
    )
    .drop("segment_rand", "shops_in_store")
)

# Generate POS transactions
pos_base = (pos_customers
    .withColumn("total_transactions", 
        (F.col("avg_transactions_per_month") * 24 * F.rand()).cast("int"))
    .filter(F.col("total_transactions") > 0)
)

pos_df = (pos_base
    .withColumn("transaction_num", F.expr("explode(sequence(1, total_transactions))"))
    .withColumn("transaction_id", 
        F.concat(F.lit("POS"), F.col("customer_id").cast("string"), F.lit("-"), F.col("transaction_num").cast("string")))
    .withColumn("days_offset", (F.rand() * TOTAL_DAYS).cast("int"))
    .withColumn("transaction_date", F.date_add(F.lit(START_DATE.date()), F.col("days_offset")))
    # Store ID based on region
    .withColumn("store_id",
        F.concat(F.lit("ST-"), 
                 F.substring(F.col("region"), 1, 3),
                 F.lit("-"),
                 ((F.rand() * 5) + 1).cast("int").cast("string"))  # 5 stores per region
    )
    .withColumn("product_id", F.concat(F.lit("PROD"), ((F.rand() * 500) + 1).cast("int").cast("string")))
    .withColumn("quantity", ((F.rand() * 4) + 1).cast("int"))
    .withColumn("unit_price", F.round((F.rand() * 150 + 5), 2))  
    .withColumn("total_amount", F.round(F.col("quantity") * F.col("unit_price"), 2))
    # Payment method with variations
    .withColumn("payment_method",
        F.when(F.rand() < 0.40, F.lit("card"))
         .when(F.rand() < 0.70, F.lit("Card"))  # Inconsistent casing
         .when(F.rand() < 0.90, F.lit("cash"))
         .otherwise(F.lit("mobile_pay"))
    )
    .select("customer_id", "transaction_id", "transaction_date", "store_id", 
            "product_id", "quantity", "unit_price", "total_amount", "payment_method")
)

# Store count to avoid repeated scans (contract: avoid repeated scans)
pos_count = pos_df.count()
print(f"POS transactions generated: {pos_count:,} records")
print("\nSchema:")
pos_df.printSchema()
print("\nSample (5 rows):")
display(pos_df.limit(5))

# COMMAND ----------

# DBTITLE 1,Generate Customer Service Interactions
# Generate customer service interactions
# Higher interaction rates for declining customers

# Select customers who interact with customer service (correlation with issues)
cs_customers = (crm_df
    .select("customer_id")
    .withColumn("has_interactions", F.when(F.rand() < 0.35, 1).otherwise(0))  # 35% contact support
    .filter(F.col("has_interactions") == 1)
    .withColumn("interaction_frequency",
        F.when(F.rand() < 0.60, F.lit("low"))       # 1-2 interactions
         .when(F.rand() < 0.90, F.lit("medium"))    # 3-6 interactions
         .otherwise(F.lit("high"))                   # 7-15 interactions (problem customers)
    )
    .withColumn("num_interactions",
        F.when(F.col("interaction_frequency") == "low", ((F.rand() * 2) + 1).cast("int"))
         .when(F.col("interaction_frequency") == "medium", ((F.rand() * 4) + 3).cast("int"))
         .otherwise(((F.rand() * 9) + 7).cast("int"))
    )
    .drop("has_interactions")
)

customer_service_df = (cs_customers
    .withColumn("interaction_num", F.expr("explode(sequence(1, num_interactions))"))
    .withColumn("interaction_id", 
        F.concat(F.lit("CS"), F.col("customer_id").cast("string"), F.lit("-"), F.col("interaction_num").cast("string")))
    .withColumn("days_offset", (F.rand() * TOTAL_DAYS).cast("int"))
    .withColumn("interaction_date", F.date_add(F.lit(START_DATE.date()), F.col("days_offset")))
    # Interaction type
    .withColumn("interaction_type",
        F.when(F.rand() < 0.35, F.lit("call"))
         .when(F.rand() < 0.70, F.lit("email"))
         .when(F.rand() < 0.90, F.lit("chat"))
         .otherwise(F.lit("in-store"))
    )
    # Channel with inconsistent values
    .withColumn("channel",
        F.when(F.rand() < 0.40, F.lit("phone"))
         .when(F.rand() < 0.50, F.lit("Phone"))  # Inconsistent casing
         .when(F.rand() < 0.80, F.lit("online"))
         .otherwise(F.lit("store"))
    )
    # Issue category
    .withColumn("issue_category",
        F.when(F.rand() < 0.25, F.lit("delivery"))
         .when(F.rand() < 0.50, F.lit("product_quality"))
         .when(F.rand() < 0.70, F.lit("returns"))
         .when(F.rand() < 0.85, F.lit("payment"))
         .otherwise(F.lit("general_inquiry"))
    )
    # Resolution status - some unresolved (data quality indicator)
    .withColumn("resolution_status",
        F.when(F.rand() < 0.75, F.lit("resolved"))
         .when(F.rand() < 0.90, F.lit("pending"))
         .otherwise(F.lit("escalated"))
    )
    # Satisfaction score (1-5, some missing)
    .withColumn("satisfaction_score",
        F.when(F.rand() < 0.85,  # 15% missing scores
            F.when(F.col("resolution_status") == "resolved", ((F.rand() * 2) + 3).cast("int"))  # 3-5 for resolved
             .otherwise(((F.rand() * 3) + 1).cast("int"))  # 1-4 for unresolved
        )
    )
    .select("customer_id", "interaction_id", "interaction_date", "interaction_type", 
            "channel", "issue_category", "resolution_status", "satisfaction_score")
)

# Store count to avoid repeated scans (contract: avoid repeated scans)
cs_count = customer_service_df.count()
print(f"Customer service interactions generated: {cs_count:,} records")
print("\nSchema:")
customer_service_df.printSchema()
print("\nSample (5 rows):")
display(customer_service_df.limit(5))

# COMMAND ----------

# DBTITLE 1,Generate Loyalty Events
# Generate loyalty program events
# Not all customers are in the loyalty program

# Select loyalty program members (60% of customers)
loyalty_customers = (crm_df
    .select("customer_id")
    .withColumn("in_loyalty_program", F.when(F.rand() < 0.60, 1).otherwise(0))
    .filter(F.col("in_loyalty_program") == 1)
    .withColumn("loyalty_id", F.concat(F.lit("LOY"), F.col("customer_id").cast("string")))
    # Assign loyalty tier
    .withColumn("loyalty_tier",
        F.when(F.rand() < 0.60, F.lit("bronze"))
         .when(F.rand() < 0.90, F.lit("silver"))
         .otherwise(F.lit("gold"))
    )
    # Activity level in loyalty program
    .withColumn("events_per_year",
        F.when(F.col("loyalty_tier") == "gold", ((F.rand() * 50) + 20).cast("int"))    # 20-70 events
         .when(F.col("loyalty_tier") == "silver", ((F.rand() * 30) + 10).cast("int"))  # 10-40 events
         .otherwise(((F.rand() * 15) + 5).cast("int"))                                   # 5-20 events
    )
    .withColumn("total_events", (F.col("events_per_year") * 2).cast("int"))  # 24 months
    .drop("in_loyalty_program", "events_per_year")
)

loyalty_df = (loyalty_customers
    .withColumn("event_num", F.expr("explode(sequence(1, total_events))"))
    .withColumn("days_offset", (F.rand() * TOTAL_DAYS).cast("int"))
    .withColumn("event_date", F.date_add(F.lit(START_DATE.date()), F.col("days_offset")))
    # Loyalty event types
    .withColumn("loyalty_event_type",
        F.when(F.rand() < 0.50, F.lit("points_earned"))
         .when(F.rand() < 0.75, F.lit("points_redeemed"))
         .when(F.rand() < 0.90, F.lit("tier_upgrade"))
         .otherwise(F.lit("bonus_points"))
    )
    # Points earned/redeemed based on event type
    .withColumn("points_earned",
        F.when(F.col("loyalty_event_type") == "points_earned", ((F.rand() * 500) + 10).cast("int"))
         .when(F.col("loyalty_event_type") == "bonus_points", ((F.rand() * 1000) + 100).cast("int"))
         .otherwise(0)
    )
    .withColumn("points_redeemed",
        F.when(F.col("loyalty_event_type") == "points_redeemed", ((F.rand() * 300) + 50).cast("int"))
         .otherwise(0)
    )
    .select("customer_id", "loyalty_id", "event_date", "loyalty_event_type", 
            "points_earned", "points_redeemed", "loyalty_tier")
)

# Store count to avoid repeated scans (contract: avoid repeated scans)
loyalty_count = loyalty_df.count()
print(f"Loyalty events generated: {loyalty_count:,} records")
print("\nSchema:")
loyalty_df.printSchema()
print("\nSample (5 rows):")
display(loyalty_df.limit(5))

# COMMAND ----------

# DBTITLE 1,Generate Marketing Campaign Data
# Generate marketing campaign engagement data
# Shows declining engagement patterns for some customers

# All customers receive marketing communications
marketing_base = (crm_df
    .select("customer_id")
    .withColumn("engagement_profile",
        F.when(F.rand() < 0.20, F.lit("highly_engaged"))      # 20% highly engaged
         .when(F.rand() < 0.50, F.lit("moderately_engaged"))  # 30% moderately engaged
         .when(F.rand() < 0.80, F.lit("low_engagement"))      # 30% low engagement
         .otherwise(F.lit("disengaged"))                       # 20% disengaged
    )
    # Campaign frequency (campaigns per month)
    .withColumn("campaigns_per_month",
        ((F.rand() * 6) + 2).cast("int")  # 2-8 campaigns per month
    )
    .withColumn("total_campaigns", (F.col("campaigns_per_month") * 24).cast("int"))
)

marketing_df = (marketing_base
    .withColumn("campaign_num", F.expr("explode(sequence(1, total_campaigns))"))
    # Generate campaign IDs (reused across customers)
    .withColumn("campaign_id", 
        F.concat(F.lit("CAMP"), ((F.rand() * 200) + 1).cast("int").cast("string")))
    .withColumn("days_offset", (F.rand() * TOTAL_DAYS).cast("int"))
    .withColumn("campaign_date", F.date_add(F.lit(START_DATE.date()), F.col("days_offset")))
    # Channel variations
    .withColumn("channel",
        F.when(F.rand() < 0.50, F.lit("email"))
         .when(F.rand() < 0.75, F.lit("Email"))  # Inconsistent casing
         .when(F.rand() < 0.90, F.lit("sms"))
         .otherwise(F.lit("push"))
    )
    # Campaign type
    .withColumn("campaign_type",
        F.when(F.rand() < 0.30, F.lit("promotional"))
         .when(F.rand() < 0.60, F.lit("seasonal"))
         .when(F.rand() < 0.85, F.lit("product_launch"))
         .otherwise(F.lit("loyalty_bonus"))
    )
    # All campaigns are sent
    .withColumn("sent", F.lit(1))
    # Opened based on engagement profile
    .withColumn("opened",
        F.when(F.col("engagement_profile") == "highly_engaged", 
            F.when(F.rand() < 0.80, 1).otherwise(0))
         .when(F.col("engagement_profile") == "moderately_engaged", 
            F.when(F.rand() < 0.50, 1).otherwise(0))
         .when(F.col("engagement_profile") == "low_engagement", 
            F.when(F.rand() < 0.20, 1).otherwise(0))
         .otherwise(F.when(F.rand() < 0.05, 1).otherwise(0))  # Disengaged
    )
    # Clicked (only if opened)
    .withColumn("clicked",
        F.when(F.col("opened") == 1,
            F.when(F.col("engagement_profile") == "highly_engaged", 
                F.when(F.rand() < 0.60, 1).otherwise(0))
             .when(F.col("engagement_profile") == "moderately_engaged", 
                F.when(F.rand() < 0.30, 1).otherwise(0))
             .otherwise(F.when(F.rand() < 0.10, 1).otherwise(0))
        ).otherwise(0)
    )
    # Converted (only if clicked)
    .withColumn("converted",
        F.when(F.col("clicked") == 1,
            F.when(F.rand() < 0.20, 1).otherwise(0)  # 20% conversion on click
        ).otherwise(0)
    )
    .select("customer_id", "campaign_id", "campaign_date", "channel", "campaign_type",
            "sent", "opened", "clicked", "converted")
)

# Store count to avoid repeated scans (contract: avoid repeated scans)
marketing_count = marketing_df.count()
print(f"Marketing campaign records generated: {marketing_count:,} records")
print("\nSchema:")
marketing_df.printSchema()
print("\nSample (5 rows):")
display(marketing_df.limit(5))

# COMMAND ----------

# DBTITLE 1,Save Data to Landing Layer
# Save all datasets to Unity Catalog volume in landing layer
# Using different formats to simulate fragmented source systems

# Base path for landing layer
base_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/landing"

print("Saving data to landing layer...")
print(f"Base path: {base_path}")

# CRM - Save as CSV (with different date format)
crm_path = f"{base_path}/crm"
print(f"\nSaving CRM data to {crm_path} (CSV format)...")
(crm_df
    .write
    .mode("overwrite")
    .option("header", "true")
    .csv(crm_path)
)
print(f"CRM data saved successfully")

# Products - Save as Parquet (reference data)
products_path = f"{base_path}/products"
print(f"\nSaving Products data to {products_path} (Parquet format)...")
(products_df
    .write
    .mode("overwrite")
    .parquet(products_path)
)
print(f"Products data saved successfully")

# E-commerce - Save as JSON (multi-line)
ecommerce_path = f"{base_path}/ecommerce"
print(f"\nSaving E-commerce data to {ecommerce_path} (JSON format)...")
(ecommerce_df
    .write
    .mode("overwrite")
    .json(ecommerce_path)
)
print(f"E-commerce data saved successfully")

# POS - Save as Parquet
pos_path = f"{base_path}/pos"
print(f"\nSaving POS data to {pos_path} (Parquet format)...")
(pos_df
    .write
    .mode("overwrite")
    .parquet(pos_path)
)
print(f"POS data saved successfully")

# Customer Service - Save as CSV
cs_path = f"{base_path}/customer_service"
print(f"\nSaving Customer Service data to {cs_path} (CSV format)...")
(customer_service_df
    .write
    .mode("overwrite")
    .option("header", "true")
    .csv(cs_path)
)
print(f"Customer Service data saved successfully")

# Loyalty - Save as JSON
loyalty_path = f"{base_path}/loyalty"
print(f"\nSaving Loyalty data to {loyalty_path} (JSON format)...")
(loyalty_df
    .write
    .mode("overwrite")
    .json(loyalty_path)
)
print(f"Loyalty data saved successfully")

# Marketing - Save as Parquet
marketing_path = f"{base_path}/marketing"
print(f"\nSaving Marketing data to {marketing_path} (Parquet format)...")
(marketing_df
    .write
    .mode("overwrite")
    .parquet(marketing_path)
)
print(f"Marketing data saved successfully")

print("\n" + "="*60)
print("All data saved to landing layer successfully!")
print("="*60)

# COMMAND ----------

# DBTITLE 1,Data Generation Summary
# Summary of generated data
import pyspark.sql.functions as F

print("DATA GENERATION SUMMARY")
print("="*60)
print(f"\nBase Configuration:")
print(f"  - Catalog: {CATALOG}")
print(f"  - Schema: {SCHEMA}")
print(f"  - Volume: {VOLUME}")
print(f"  - Date Range: {START_DATE.date()} to {END_DATE.date()}")
print(f"  - Number of Customers: {NUM_CUSTOMERS:,}")

# Use stored counts to avoid repeated scans (contract: avoid repeated scans)
print(f"\nDatasets Generated:")
print(f"  1. CRM: {crm_count:,} customer records (CSV format)")
print(f"  2. Products: {product_count:,} product records (Parquet format)")
print(f"  3. E-commerce: {ecommerce_count:,} transactions (JSON format)")
print(f"  4. POS: {pos_count:,} transactions (Parquet format)")
print(f"  5. Customer Service: {cs_count:,} interactions (CSV format)")
print(f"  6. Loyalty: {loyalty_count:,} events (JSON format)")
print(f"  7. Marketing: {marketing_count:,} campaign records (Parquet format)")

total_records = (crm_count + product_count + ecommerce_count + pos_count + 
                 cs_count + loyalty_count + marketing_count)

print(f"\nTotal Records Generated: {total_records:,}")

print(f"\nData Quality Issues Included:")
print(f"  - Missing values in multiple fields")
print(f"  - Inconsistent casing (e.g., 'active' vs 'ACTIVE')")
print(f"  - Duplicate records (~2% in e-commerce)")
print(f"  - Different date formats across sources")
print(f"  - Inconsistent naming conventions")

print(f"\nBehavioral Patterns Included:")
print(f"  - Frequent, moderate, and occasional customers")
print(f"  - High-value and low-value customers")
print(f"  - Declining customer activity patterns")
print(f"  - Increasing time between purchases")
print(f"  - Declining marketing engagement")
print(f"  - Increasing customer service interactions")

print(f"\nLanding Layer Structure:")
print(f"  {base_path}/")
print(f"    ├── crm/         (CSV)")
print(f"    ├── products/    (Parquet)")
print(f"    ├── ecommerce/   (JSON)")
print(f"    ├── pos/         (Parquet)")
print(f"    ├── customer_service/ (CSV)")
print(f"    ├── loyalty/     (JSON)")
print(f"    └── marketing/   (Parquet)")

print("\n" + "="*60)
print("Data ingestion complete! Ready for Bronze layer processing.")
print("="*60)