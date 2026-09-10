# -*- coding: utf-8 -*-
import os
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from databricks import sql
from databricks.sdk.core import Config
import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ============================================
# LOGGING
# ============================================
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')

# Ensure environment variable is set correctly
assert os.getenv('DATABRICKS_WAREHOUSE_ID'), "DATABRICKS_WAREHOUSE_ID must be set in app.yaml."

# Databricks config
cfg = Config()

# ============================================
# DATA SOURCE CONFIGURATION
# ============================================
CATALOG = "workspace"
SCHEMA = "retail"
VOLUME = "databrics_customer_analytics"

# Table references - update these to match your actual table names
CUSTOMER_360_TABLE = f"{CATALOG}.{SCHEMA}.gold_customer_360"
RISK_PREDICTIONS_TABLE = f"{CATALOG}.{SCHEMA}.risk_predictions"
INTERVENTION_TABLE = f"{CATALOG}.{SCHEMA}.intervention_recommendations"
CUSTOMER_ACTIVITY_DAILY_TABLE = f"{CATALOG}.{SCHEMA}.gold_customer_activity_daily"
CUSTOMER_STATE_HISTORY_TABLE = f"{CATALOG}.{SCHEMA}.customer_state_history"
BRONZE_CRM_CUSTOMER_TABLE = f"{CATALOG}.{SCHEMA}.bronze_crm_customer"
SILVER_PURCHASE_TABLE = f"{CATALOG}.{SCHEMA}.silver_purchase"

# ============================================
# DATABASE CONNECTION (Thread-Local Connection Pool)
# ============================================
# Each Gradio worker thread gets its own cached connection. Connections are
# reused across queries within the same thread+token, eliminating repeated
# TCP/TLS handshakes. When the token changes (different user on same thread),
# the old connection is closed and a new one is opened.

_thread_local = threading.local()


def _get_connection(user_token: str):
    """Return a thread-local cached connection, creating one if needed."""
    if not hasattr(_thread_local, 'conn'):
        _thread_local.conn = None
        _thread_local.token = None

    # Reuse existing connection if token matches
    if _thread_local.conn is not None and _thread_local.token == user_token:
        return _thread_local.conn

    # Close stale connection from a different token
    if _thread_local.conn is not None:
        try:
            _thread_local.conn.close()
        except Exception:
            pass

    # Open new connection
    if user_token == "service_principal":
        _thread_local.conn = sql.connect(
            server_hostname=cfg.host,
            http_path=f"/sql/1.0/warehouses/{cfg.warehouse_id}",
            credentials_provider=lambda: cfg.authenticate,
            _connection_timeout=60
        )
    else:
        _thread_local.conn = sql.connect(
            server_hostname=cfg.host,
            http_path=f"/sql/1.0/warehouses/{cfg.warehouse_id}",
            access_token=user_token,
            _connection_timeout=60
        )
    _thread_local.token = user_token
    return _thread_local.conn


def _execute_query(query: str, user_token: str) -> pd.DataFrame:
    """Execute a query using the thread-local connection pool."""
    try:
        conn = _get_connection(user_token)
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            result = cursor.fetchall_arrow().to_pandas()
            return result if result is not None else pd.DataFrame()
        finally:
            cursor.close()
    except Exception as e:
        logger.warning("Query error: %s", e)
        # Reset thread-local connection on error (it may be stale)
        if hasattr(_thread_local, 'conn') and _thread_local.conn is not None:
            try:
                _thread_local.conn.close()
            except Exception:
                pass
            _thread_local.conn = None
            _thread_local.token = None
        raise


def sql_query_with_service_principal(query: str) -> pd.DataFrame:
    """Execute a SQL query as the service principal."""
    try:
        return _execute_query(query, "service_principal")
    except Exception as e:
        logger.error("Service principal query failed: %s", e)
        return pd.DataFrame()


def sql_query_with_user_token(query: str, user_token: str) -> pd.DataFrame:
    """Execute a SQL query with user credentials (OBO), falling back to service principal."""
    try:
        return _execute_query(query, user_token)
    except Exception as e:
        logger.warning("OBO query failed, falling back to service principal: %s", e)
        try:
            return sql_query_with_service_principal(query)
        except Exception as e2:
            logger.error("Service principal query also failed: %s", e2)
            return pd.DataFrame()

def get_user_token(request: gr.Request):
    """Extract user token from request headers."""
    user_token = request.headers.get("X-Forwarded-Access-Token", None)
    return user_token if user_token else "service_principal"

# ============================================
# WAREHOUSE CONFIGURATION & PREWARM
# ============================================

def _configure_warehouse_auto_suspend():
    """Update SQL warehouse auto-suspend timeout from 10 min to 30 min."""
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        warehouse_id = cfg.warehouse_id
        
        # Get current warehouse config
        warehouse = w.warehouses.get(warehouse_id)
        
        # Update auto_stop_mins to 30 if it's currently 10
        if warehouse.auto_stop_mins == 10:
            w.warehouses.edit(
                id=warehouse_id,
                auto_stop_mins=30
            )
            logger.info("Updated warehouse %s auto-suspend from 10 min to 30 min", warehouse_id)
        else:
            logger.info("Warehouse %s auto-suspend already set to %d min", warehouse_id, warehouse.auto_stop_mins)
    except Exception as e:
        logger.warning("Could not update warehouse auto-suspend timeout: %s", e)

def _prewarm_warehouse_and_genie():
    """Prewarm SQL warehouse and Genie by querying multiple tables at module initialization.
    
    This runs BEFORE the Gradio UI loads, ensuring:
    1. SQL warehouse is fully warmed with real table queries
    2. Genie space is ready
    3. Table metadata is cached
    
    Fires 3 lightweight queries in parallel against the main Genie tables.
    """
    logger.info("Starting warehouse and Genie prewarm...")
    
    # Configure warehouse auto-suspend timeout (30 min)
    _configure_warehouse_auto_suspend()
    
    # Define lightweight prewarm queries for each main table
    prewarm_queries = [
        f"SELECT 1 FROM {CUSTOMER_360_TABLE} LIMIT 1",
        f"SELECT 1 FROM {RISK_PREDICTIONS_TABLE} LIMIT 1",
        f"SELECT 1 FROM {INTERVENTION_TABLE} LIMIT 1"
    ]
    
    def _run_prewarm_query(query):
        """Execute a single prewarm query, ignore errors."""
        try:
            sql_query_with_service_principal(query)
            logger.debug("Prewarm query succeeded: %s", query[:50])
        except Exception as e:
            logger.debug("Prewarm query failed (ignored): %s", e)
    
    # Run all prewarm queries in parallel for faster warmup
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_run_prewarm_query, q) for q in prewarm_queries]
        # Wait for all to complete (with timeout)
        for future in futures:
            try:
                future.result(timeout=30)
            except Exception:
                pass  # Best-effort, ignore failures
    
    logger.info("Warehouse and Genie prewarm completed")

# Execute prewarm at module initialization time (before Gradio starts)
_prewarm_warehouse_and_genie()

# ============================================
# GENIE SHORTCUT QUERIES (Optimized for Performance)
# ============================================

SHORTCUT_QUERIES = {
    "How many customers are there?": f"""
        SELECT COUNT(DISTINCT customer_id) as total_customers
        FROM {CUSTOMER_360_TABLE}
    """,
    
    "What is the total revenue?": f"""
        SELECT 
            ROUND(SUM(total_revenue), 2) as total_revenue,
            ROUND(SUM(revenue_last_12_months), 2) as revenue_12m
        FROM {CUSTOMER_360_TABLE}
    """,
    
    "What is the average revenue per customer?": f"""
        SELECT 
            ROUND(AVG(total_revenue), 2) as avg_revenue_per_customer,
            ROUND(AVG(revenue_last_12_months), 2) as avg_revenue_12m
        FROM {CUSTOMER_360_TABLE}
    """,
    
    "How many active vs inactive customers?": f"""
        SELECT 
            customer_status,
            COUNT(*) as customer_count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as percentage
        FROM {CUSTOMER_360_TABLE}
        GROUP BY customer_status
        ORDER BY customer_count DESC
    """,
    
    "Revenue by country?": f"""
        SELECT 
            country,
            COUNT(DISTINCT customer_id) as customers,
            ROUND(SUM(total_revenue), 2) as total_revenue,
            ROUND(AVG(total_revenue), 2) as avg_revenue_per_customer
        FROM {CUSTOMER_360_TABLE}
        GROUP BY country
        ORDER BY total_revenue DESC
    """,
    
    "Top 10 customers by revenue?": f"""
        SELECT 
            customer_id,
            customer_status as status,
            country,
            ROUND(total_revenue, 2) as total_revenue,
            number_of_transactions as transactions,
            ROUND(average_order_value, 2) as avg_order_value
        FROM {CUSTOMER_360_TABLE}
        ORDER BY total_revenue DESC
        LIMIT 10
    """,
    
    "What is the average satisfaction score?": f"""
        SELECT 
            ROUND(AVG(average_satisfaction_score), 2) as avg_satisfaction_score,
            COUNT(*) as customers_with_score,
            ROUND(MIN(average_satisfaction_score), 2) as min_score,
            ROUND(MAX(average_satisfaction_score), 2) as max_score
        FROM {CUSTOMER_360_TABLE}
        WHERE average_satisfaction_score IS NOT NULL
    """,
    
    "How many customers are at high churn risk?": f"""
        SELECT 
            risk_tier,
            COUNT(*) as customer_count,
            ROUND(AVG(churn_probability * 100), 1) as avg_churn_probability
        FROM {RISK_PREDICTIONS_TABLE}
        WHERE risk_tier IN ('high', 'critical')
        GROUP BY risk_tier
        ORDER BY 
            CASE risk_tier
                WHEN 'critical' THEN 1
                WHEN 'high' THEN 2
            END
    """,
    
    "Which customers are at high churn risk?": f"""
        SELECT 
            CAST(customer_id AS STRING) as customer_id,
            risk_tier,
            ROUND(churn_probability * 100, 1) as churn_probability_pct,
            predicted_days_to_churn as days_to_churn,
            semaphore_state
        FROM {RISK_PREDICTIONS_TABLE}
        WHERE risk_tier IN ('high', 'critical')
        ORDER BY churn_probability DESC, predicted_days_to_churn ASC
        LIMIT 20
    """,
    
    "What interventions are recommended for critical customers?": f"""
        SELECT 
            ir.customer_id,
            ir.recommended_action,
            ir.recommended_channel,
            ir.action_priority,
            rp.risk_tier,
            ROUND(rp.churn_probability * 100, 1) as churn_probability_pct
        FROM {INTERVENTION_TABLE} ir
        INNER JOIN {RISK_PREDICTIONS_TABLE} rp
            ON CAST(rp.customer_id AS STRING) = ir.customer_id
        WHERE rp.risk_tier = 'critical'
        ORDER BY ir.action_priority ASC
        LIMIT 20
    """,
    
    "What is the average order value?": f"""
        SELECT 
            ROUND(AVG(average_order_value), 2) as avg_order_value,
            ROUND(MIN(average_order_value), 2) as min_order_value,
            ROUND(MAX(average_order_value), 2) as max_order_value,
            COUNT(*) as total_customers
        FROM {CUSTOMER_360_TABLE}
    """
}

def dataframe_to_markdown(df):
    """Convert DataFrame to markdown table without requiring tabulate library."""
    if df.empty:
        return "_No data_"

    columns = df.columns.tolist()
    header = "| " + " | ".join(str(col) for col in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"

    # Use vectorized string conversion instead of iterrows() for ~10x speedup
    rows = df.astype(str).agg(lambda r: "| " + " | ".join(r) + " |", axis=1).tolist()
    return "\n".join([header, separator] + rows)

def run_shortcut_query(user_token, question):
    """Execute a pre-defined shortcut SQL query and format results as markdown table."""
    if question not in SHORTCUT_QUERIES:
        return "⚠️ Query not found in shortcuts"
    
    try:
        sql = SHORTCUT_QUERIES[question]
        df = sql_query_with_user_token(sql, user_token)
        
        if df.empty:
            return f"**{question}**\n\n✅ Query executed successfully but returned no data."
        
        # Format as markdown table using custom formatter (no tabulate dependency)
        result = f"**✅ {question}**\n\n"
        result += dataframe_to_markdown(df)
        result += f"\n\n<details><summary>📄 View SQL Query</summary>\n\n```sql\n{sql.strip()}\n```\n</details>"
        
        return result
    
    except Exception as e:
        error_msg = str(e)
        return f"**❌ Error: {question}**\n\n```\n{error_msg}\n```\n\n<details><summary>📄 SQL Query</summary>\n\n```sql\n{SHORTCUT_QUERIES[question].strip()}\n```\n</details>"

# ============================================
# DATA QUERIES
# ============================================

def get_executive_overview(user_token: str, country_filter: str = "All", status_filter: str = "All"):
    """Get executive KPIs with filters."""
    where_clause = "WHERE 1=1"
    if country_filter != "All":
        where_clause += f" AND country = '{country_filter}'"
    if status_filter != "All":
        where_clause += f" AND customer_status = '{status_filter}'"
    
    query = f"""
    SELECT 
        COUNT(DISTINCT customer_id) as total_customers,
        COUNT(DISTINCT CASE WHEN customer_status = 'active' THEN customer_id END) as active_customers,
        COALESCE(SUM(total_revenue), 0) as total_revenue,
        COALESCE(SUM(revenue_last_12_months), 0) as revenue_12m,
        SUM(number_of_transactions) as total_transactions,
        COALESCE(SUM(number_of_transactions_last_12_months), 0) as transactions_12m,
        COALESCE(AVG(average_order_value), 0) as avg_order_value,
        COUNT(DISTINCT CASE WHEN revenue_change_pct < -15 THEN customer_id END) as declining_customers
    FROM {CUSTOMER_360_TABLE}
    {where_clause}
    """
    return sql_query_with_user_token(query, user_token)

def get_top_customers_by_revenue(user_token: str, country_filter: str = "All", status_filter: str = "All", limit: int = 10):
    """Get top N customers by total revenue with filters."""
    where_clause = "WHERE 1=1"
    if country_filter != "All":
        where_clause += f" AND country = '{country_filter}'"
    if status_filter != "All":
        where_clause += f" AND customer_status = '{status_filter}'"
    
    query = f"""
    SELECT 
        customer_id,
        country,
        customer_status as status,
        ROUND(total_revenue, 2) as total_revenue,
        ROUND(revenue_last_12_months, 2) as revenue_12m,
        number_of_transactions as transactions,
        ROUND(average_order_value, 2) as avg_order_value
    FROM {CUSTOMER_360_TABLE}
    {where_clause}
    ORDER BY total_revenue DESC
    LIMIT {limit}
    """
    return sql_query_with_user_token(query, user_token)


def build_filter_clause(segment=None, status=None, loyalty=None, 
                       revenue_min=None, revenue_max=None,
                       aov_min=None, aov_max=None,
                       trans_min=None, trans_max=None,
                       days_since_min=None, days_since_max=None,
                       purch30_min=None, purch30_max=None,
                       purch60_min=None, purch60_max=None,
                       purch90_min=None, purch90_max=None,
                       engagement_min=None, engagement_max=None):
    """Build SQL WHERE clause from filter parameters."""
    try:
        logger.debug("Filter values: segment=%s, status=%s, loyalty=%s, revenue_min=%s, revenue_max=%s",
                     segment, status, loyalty, revenue_min, revenue_max)
        
        conditions = ["1=1"]
        
        # Categorical filters
        if segment and segment != "All":
            conditions.append(f"customer_segment = '{segment}'")
        
        if status and status != "All":
            conditions.append(f"customer_status = '{status}'")
        
        # Handle loyalty filter (multiselect - can be list, string, or None)
        if loyalty:
            # Convert to list if needed
            if isinstance(loyalty, str):
                loyalty = [loyalty]
            
            # Filter out "All" and check if there are actual values
            loyalty_filtered = [v for v in loyalty if v != "All"]
            
            if loyalty_filtered:
                loyalty_values = "', '".join(loyalty_filtered)
                conditions.append(f"loyalty_tier IN ('{loyalty_values}')")
        
        # Range filters
        if revenue_min is not None:
            conditions.append(f"total_revenue >= {revenue_min}")
        if revenue_max is not None:
            conditions.append(f"total_revenue <= {revenue_max}")
            
        if aov_min is not None:
            conditions.append(f"average_order_value >= {aov_min}")
        if aov_max is not None:
            conditions.append(f"average_order_value <= {aov_max}")
            
        if trans_min is not None:
            conditions.append(f"number_of_transactions >= {trans_min}")
        if trans_max is not None:
            conditions.append(f"number_of_transactions <= {trans_max}")
            
        if days_since_min is not None:
            conditions.append(f"days_since_last_purchase >= {days_since_min}")
        if days_since_max is not None:
            conditions.append(f"days_since_last_purchase <= {days_since_max}")
            
        if purch30_min is not None:
            conditions.append(f"purchases_last_30_days >= {purch30_min}")
        if purch30_max is not None:
            conditions.append(f"purchases_last_30_days <= {purch30_max}")
            
        if purch60_min is not None:
            conditions.append(f"purchases_last_60_days >= {purch60_min}")
        if purch60_max is not None:
            conditions.append(f"purchases_last_60_days <= {purch60_max}")
            
        if purch90_min is not None:
            conditions.append(f"purchases_last_90_days >= {purch90_min}")
        if purch90_max is not None:
            conditions.append(f"purchases_last_90_days <= {purch90_max}")
            
        if engagement_min is not None:
            conditions.append(f"marketing_engagement_rate >= {engagement_min}")
        if engagement_max is not None:
            conditions.append(f"marketing_engagement_rate <= {engagement_max}")
        
        where_clause = " AND ".join(conditions)
        logger.debug("Generated WHERE clause: %s", where_clause)
        return where_clause
    except Exception as e:
        logger.error("Error in build_filter_clause: %s", e, exc_info=True)
        return "1=1"  # Return default filter on error



def get_customer_list(user_token: str, page: int = 0, page_size: int = 50, 
                      search: str = "", sort_by: str = "total_revenue", sort_desc: bool = True,
                      filter_clause: str = "1=1"):
    """Get paginated customer list with search, sorting, and filters."""
    offset = page * page_size
    sort_order = "DESC" if sort_desc else "ASC"
    
    # Build WHERE clause using the filter_clause parameter
    where_clause = f"WHERE {filter_clause}"
    if search:
        where_clause += f" AND CAST(customer_id AS STRING) LIKE '%{search}%'"

    logger.debug("Final WHERE clause in get_customer_list: %s", where_clause)
    
    query = f"""
    SELECT 
        customer_id,
        -- customer_segment as segment,  -- SEGMENT COLUMN - Commented out (can be re-enabled in future)
        customer_status as status,
        loyalty_tier,
        ROUND(total_revenue, 2) as revenue,
        ROUND(average_order_value, 2) as aov,
        number_of_transactions as transactions,
        days_since_last_purchase,
        purchases_last_30_days as purchases_30d,
        purchases_last_60_days as purchases_60d,
        purchases_last_90_days as purchases_90d,
        ROUND(marketing_engagement_rate, 2) as engagement
    FROM {CUSTOMER_360_TABLE}
    {where_clause}
    ORDER BY {sort_by} {sort_order}
    LIMIT {page_size} OFFSET {offset}
    """
    return sql_query_with_user_token(query, user_token)

def get_customer_360(user_token: str, customer_id: str):
    """Get detailed profile for a single customer."""
    query = f"""
    SELECT 
        customer_id, customer_status, customer_segment, loyalty_tier,
        country, region, customer_since,
        total_revenue, revenue_last_12_months, average_order_value,
        number_of_transactions, number_of_transactions_last_12_months,
        last_purchase_date, days_since_last_purchase,
        purchases_last_30_days, purchases_last_60_days, purchases_last_90_days,
        marketing_engagement_rate, average_satisfaction_score, unresolved_interactions,
        revenue_change_pct, purchase_frequency_change_pct,
        loyalty_points_earned, loyalty_points_redeemed
    FROM {CUSTOMER_360_TABLE}
    WHERE customer_id = '{customer_id}'
    """
    return sql_query_with_user_token(query, user_token)

def get_customer_activity_trend(user_token: str, customer_id: str, days: int = 90):
    """Get customer activity time series."""
    query = f"""
    WITH max_date AS (
        SELECT MAX(activity_date) as latest_date
        FROM {CUSTOMER_ACTIVITY_DAILY_TABLE}
        WHERE customer_id = '{customer_id}'
    )
    SELECT 
        activity_date,
        daily_revenue,
        daily_transactions,
        rolling_7d_revenue,
        rolling_30d_revenue,
        cumulative_revenue
    FROM {CUSTOMER_ACTIVITY_DAILY_TABLE}
    WHERE customer_id = '{customer_id}'
    AND activity_date >= (SELECT DATE_SUB(latest_date, {days}) FROM max_date)
    ORDER BY activity_date
    """
    return sql_query_with_user_token(query, user_token)

def get_revenue_trends(user_token: str, days: int = 90):
    """Get aggregated revenue trends."""
    query = f"""
    WITH max_date AS (
        SELECT MAX(activity_date) as latest_date
        FROM {CUSTOMER_ACTIVITY_DAILY_TABLE}
    )
    SELECT 
        activity_date,
        SUM(daily_revenue) as total_revenue,
        COUNT(DISTINCT customer_id) as active_customers,
        SUM(daily_transactions) as total_transactions
    FROM {CUSTOMER_ACTIVITY_DAILY_TABLE}
    WHERE activity_date >= (SELECT DATE_SUB(latest_date, {days}) FROM max_date)
    GROUP BY activity_date
    ORDER BY activity_date
    """
    return sql_query_with_user_token(query, user_token)

def get_at_risk_customers(user_token: str):
    """Identify customers with risk signals."""
    query = """
    SELECT 
        customer_id,
        customer_status as status,
        ROUND(total_revenue, 2) as revenue,
        days_since_last_purchase,
        ROUND(purchase_frequency_change_pct, 1) as freq_change_pct,
        ROUND(revenue_change_pct, 1) as revenue_change_pct,
        ROUND(marketing_engagement_rate, 2) as engagement,
        unresolved_interactions,
        CASE 
            WHEN revenue_change_pct < -30 OR purchase_frequency_change_pct < -30 OR days_since_last_purchase > 90 THEN 'High'
            WHEN revenue_change_pct < -15 OR purchase_frequency_change_pct < -10 OR days_since_last_purchase > 60 THEN 'Medium'
            ELSE 'Low'
        END as risk_level
    FROM {CUSTOMER_360_TABLE}
    WHERE 
        (revenue_change_pct < -15 
         OR purchase_frequency_change_pct < -10 
         OR days_since_last_purchase > 60 
         OR marketing_engagement_rate < 3.0 
         OR unresolved_interactions > 0)
    ORDER BY 
        CASE 
            WHEN revenue_change_pct < -30 OR purchase_frequency_change_pct < -30 OR days_since_last_purchase > 90 THEN 1
            WHEN revenue_change_pct < -15 OR purchase_frequency_change_pct < -10 OR days_since_last_purchase > 60 THEN 2
            ELSE 3
        END,
        revenue_change_pct
    """
    return sql_query_with_user_token(query, user_token)

# ============================================
# RISK PREDICTIVE QUERY FUNCTIONS
# ============================================

def get_risk_predictions(user_token: str, risk_tier: str = "All", limit: int = 1000):
    """Get churn predictions from ML model."""
    where_clause = "WHERE 1=1"
    if risk_tier != "All":
        where_clause += f" AND risk_tier = '{risk_tier}'"
    
    query = f"""
    SELECT 
        customer_id,
        ROUND(churn_probability * 100, 1) as churn_probability_pct,
        CAST(predicted_days_to_churn AS INT) as days_to_churn,
        ROUND(uplift_score, 3) as uplift_score,
        semaphore_state,
        risk_tier,
        is_high_value,
        priority_rank
    FROM {RISK_PREDICTIONS_TABLE}
    {where_clause}
    ORDER BY priority_rank
    LIMIT {limit}
    """
    return sql_query_with_user_token(query, user_token)

def get_semaphore_distribution(user_token: str):
    """Get count of customers in each semaphore state."""
    query = """
    SELECT 
        semaphore_state,
        COUNT(*) as customer_count,
        ROUND(AVG(churn_probability * 100), 1) as avg_churn_prob,
        ROUND(AVG(predicted_days_to_churn), 0) as avg_days_to_churn
    FROM {RISK_PREDICTIONS_TABLE}
    GROUP BY semaphore_state
    ORDER BY 
        CASE semaphore_state
            WHEN '🟢 GREEN' THEN 1
            WHEN '🟡 YELLOW' THEN 2
            WHEN '🟠 ORANGE' THEN 3
            WHEN '🔴 RED' THEN 4
        END
    """
    return sql_query_with_user_token(query, user_token)

def get_intervention_recommendations(user_token: str, action_filter: str = "All", limit: int = 100):
    """Get AI-recommended interventions."""
    where_clause = "WHERE 1=1"
    if action_filter != "All":
        where_clause += f" AND recommended_action = '{action_filter}'"
    
    query = f"""
    SELECT 
        customer_id,
        recommended_action,
        action_priority,
        ROUND(expected_success_probability * 100, 1) as success_probability_pct,
        recommended_channel,
        recommended_timing,
        business_justification
    FROM {INTERVENTION_TABLE}
    {where_clause}
    ORDER BY action_priority
    LIMIT {limit}
    """
    return sql_query_with_user_token(query, user_token)

def get_state_transitions(user_token: str, limit: int = 100):
    """Get recent state transitions for monitoring."""
    query = f"""
    SELECT 
        customer_id,
        previous_state,
        current_state,
        transition_date,
        ROUND(churn_probability * 100, 1) as churn_prob_pct,
        CAST(days_to_churn AS INT) as days_to_churn,
        trigger_event
    FROM {CUSTOMER_STATE_HISTORY_TABLE}
    WHERE previous_state IS NOT NULL
    ORDER BY transition_date DESC
    LIMIT {limit}
    """
    return sql_query_with_user_token(query, user_token)

def get_merged_risk_predictions(user_token: str, risk_tier: str = "All", action_filter: str = "All", limit: int = 100):
    """Get merged risk predictions and intervention recommendations."""
    where_clause = "WHERE 1=1"
    if risk_tier != "All":
        where_clause += f" AND rp.risk_tier = '{risk_tier}'"
    if action_filter != "All":
        where_clause += f" AND ir.recommended_action = '{action_filter}'"
    
    query = f"""
    SELECT 
        rp.customer_id,
        rp.risk_tier,
        ROUND(rp.churn_probability * 100, 2) as churn_probability_pct,
        CAST(rp.predicted_days_to_churn AS INT) as days_to_churn,
        ROUND(rp.uplift_score * 100, 2) as uplift_score,
        rp.priority_rank,
        ir.recommended_action,
        ir.recommended_channel,
        ir.recommended_timing,
        ir.business_justification
    FROM {RISK_PREDICTIONS_TABLE} rp
    LEFT JOIN {INTERVENTION_TABLE} ir
        ON rp.customer_id = ir.customer_id
    {where_clause}
    ORDER BY rp.priority_rank
    LIMIT {limit}
    """
    df = sql_query_with_user_token(query, user_token)
    if not df.empty:
        # Rename columns for display
        df = df.rename(columns={
            'customer_id': 'customer id',
            'risk_tier': 'Risk Tier',
            'churn_probability_pct': 'churn probability',
            'days_to_churn': 'days to churn',
            'uplift_score': 'uplift score %',
            'priority_rank': 'priority rank',
            'recommended_action': 'recommended action',
            'recommended_channel': 'recommended channel',
            'recommended_timing': 'recommended timing',
            'business_justification': 'business justification'
        })
        # Ensure uplift score displays exactly 2 decimal places
        if 'uplift score %' in df.columns:
            df['uplift score %'] = df['uplift score %'].apply(lambda x: round(float(x), 2) if pd.notna(x) else x)
    return df

def get_demographic_performance(user_token: str, country_filter: str = "All", status_filter: str = "All"):
    """Get customer performance by age band with filters."""
    where_clause = "WHERE crm.age_band IS NOT NULL"
    if country_filter != "All":
        where_clause += f" AND c360.country = '{country_filter}'"
    if status_filter != "All":
        where_clause += f" AND c360.customer_status = '{status_filter}'"
    
    query = f"""
    SELECT 
        crm.age_band,
        c360.customer_status,
        COUNT(*) as customer_count,
        ROUND(SUM(c360.total_revenue), 2) as total_revenue,
        ROUND(AVG(c360.total_revenue), 2) as avg_revenue,
        ROUND(AVG(c360.average_order_value), 2) as avg_basket,
        ROUND(AVG(c360.purchases_last_90_days), 1) as avg_purchases_90d,
        ROUND(AVG(c360.marketing_engagement_rate), 2) as avg_engagement,
        ROUND(AVG(c360.average_satisfaction_score), 2) as avg_satisfaction
    FROM {CUSTOMER_360_TABLE} c360
    LEFT JOIN {BRONZE_CRM_CUSTOMER_TABLE} crm
        ON c360.customer_id = crm.customer_id
    {where_clause}
    GROUP BY crm.age_band, c360.customer_status
    ORDER BY crm.age_band, c360.customer_status
    """
    return sql_query_with_user_token(query, user_token)

def get_geographic_performance(user_token: str, age_filter: str = "All", status_filter: str = "All"):
    """Get customer performance by country and region with filters."""
    where_clause = "WHERE 1=1"
    if age_filter != "All":
        where_clause += f" AND crm.age_band = '{age_filter}'"
    if status_filter != "All":
        where_clause += f" AND c360.customer_status = '{status_filter}'"
    
    query = f"""
    SELECT 
        c360.country,
        c360.region,
        c360.customer_status,
        COUNT(*) as customer_count,
        ROUND(SUM(c360.total_revenue), 2) as total_revenue,
        ROUND(AVG(c360.average_order_value), 2) as avg_basket,
        ROUND(AVG(c360.purchases_last_90_days), 1) as avg_purchases_90d
    FROM {CUSTOMER_360_TABLE} c360
    LEFT JOIN {BRONZE_CRM_CUSTOMER_TABLE} crm
        ON c360.customer_id = crm.customer_id
    {where_clause}
    GROUP BY c360.country, c360.region, c360.customer_status
    ORDER BY c360.country, c360.region, c360.customer_status
    """
    return sql_query_with_user_token(query, user_token)

def get_seasonal_trends(user_token: str, year: int = 2024):
    """Get monthly seasonal patterns for a given year."""
    query = f"""
    SELECT 
        MONTH(activity_date) as month,
        DATE_TRUNC('month', activity_date) as month_date,
        COUNT(DISTINCT customer_id) as active_customers,
        ROUND(SUM(daily_revenue), 2) as total_revenue,
        SUM(daily_transactions) as total_transactions,
        ROUND(AVG(daily_revenue), 2) as avg_daily_revenue
    FROM {CUSTOMER_ACTIVITY_DAILY_TABLE}
    WHERE YEAR(activity_date) = {year}
    GROUP BY MONTH(activity_date), DATE_TRUNC('month', activity_date)
    ORDER BY month
    """
    return sql_query_with_user_token(query, user_token)

def get_age_location_matrix(user_token: str, status_filter: str = "All"):
    """Get cross-dimensional age x country performance matrix."""
    where_clause = "WHERE crm.age_band IS NOT NULL"
    if status_filter != "All":
        where_clause += f" AND c360.customer_status = '{status_filter}'"
    
    query = f"""
    SELECT 
        crm.age_band,
        c360.country,
        COUNT(*) as customer_count,
        ROUND(AVG(c360.total_revenue), 2) as avg_revenue,
        ROUND(AVG(c360.average_order_value), 2) as avg_basket,
        ROUND(AVG(c360.marketing_engagement_rate), 2) as avg_engagement
    FROM {CUSTOMER_360_TABLE} c360
    LEFT JOIN {BRONZE_CRM_CUSTOMER_TABLE} crm
        ON c360.customer_id = crm.customer_id
    {where_clause}
    GROUP BY crm.age_band, c360.country
    ORDER BY crm.age_band, c360.country
    """
    return sql_query_with_user_token(query, user_token)

def get_active_inactive_drivers(user_token: str):
    """Compare active vs inactive customers across dimensions."""
    query = """
    WITH customer_enriched AS (
        SELECT 
            c360.*,
            crm.age_band,
            crm.preferred_channel
        FROM {CUSTOMER_360_TABLE} c360
        LEFT JOIN {BRONZE_CRM_CUSTOMER_TABLE} crm
            ON c360.customer_id = crm.customer_id
        WHERE crm.age_band IS NOT NULL
    )
    SELECT 
        customer_status,
        age_band,
        country,
        loyalty_tier,
        COUNT(*) as count,
        ROUND(AVG(total_revenue), 2) as avg_revenue,
        ROUND(AVG(average_order_value), 2) as avg_basket,
        ROUND(AVG(purchases_last_90_days), 1) as avg_purchases_90d,
        ROUND(AVG(marketing_engagement_rate), 2) as avg_engagement,
        ROUND(AVG(days_since_last_purchase), 1) as avg_days_since_purchase
    FROM customer_enriched
    GROUP BY customer_status, age_band, country, loyalty_tier
    ORDER BY customer_status, age_band, country, loyalty_tier
    """
    return sql_query_with_user_token(query, user_token)

def get_customer_top_products(user_token: str, customer_id: str, days: int = 365, limit: int = 10):
    """Get top products purchased by a customer within the last year."""
    query = f"""
    SELECT 
        product_id,
        SUM(quantity) as total_quantity,
        ROUND(SUM(revenue), 2) as total_revenue
    FROM {SILVER_PURCHASE_TABLE}
    WHERE customer_id = '{customer_id}'
    AND transaction_date >= DATE_SUB(
        (SELECT MAX(transaction_date) FROM {SILVER_PURCHASE_TABLE} WHERE customer_id = '{customer_id}'), 
        {days}
    )
    GROUP BY product_id
    ORDER BY total_quantity DESC
    LIMIT {limit}
    """
    return sql_query_with_user_token(query, user_token)

# ============================================
# VISUALIZATION FUNCTIONS
# ============================================

def create_kpi_card(label: str, value: str, change: str = None, tooltip: str = ""):
    """Create a KPI card with optional trend indicator."""
    trend_icon = ""
    if change:
        if "+" in change:
            trend_icon = "🟢"
        elif "-" in change:
            trend_icon = "🔴"
        else:
            trend_icon = "🟡"
    
    card_html = f"""
    <div style="border: 1px solid #ddd; padding: 20px; border-radius: 8px; background: #f9f9f9;">
        <div style="font-size: 14px; color: #666; margin-bottom: 8px;" title="{tooltip}">{label}</div>
        <div style="font-size: 28px; font-weight: bold; color: #333;">{value}</div>
        {f'<div style="font-size: 14px; color: #888; margin-top: 8px;">{trend_icon} {change}</div>' if change else ''}
    </div>
    """
    return card_html

def create_revenue_trend_chart(df: pd.DataFrame):
    """Create revenue trend line chart."""
    if df.empty:
        return go.Figure().update_layout(title="⚠️ Insufficient data")
    
    fig = px.line(df, x='activity_date', y='total_revenue', 
                  title='Revenue Trend',
                  labels={'activity_date': 'Date', 'total_revenue': 'Revenue'})
    fig.update_traces(line_color='#1f77b4', line_width=2)
    fig.update_layout(hovermode='x unified', height=400)
    return fig

def create_customer_activity_chart(df: pd.DataFrame):
    """Create business-focused customer activity chart with actionable insights."""
    if df.empty:
        return go.Figure().update_layout(title="⚠️ No activity data available")
    
    # Calculate business metrics
    first_30_revenue = df.iloc[:30]['daily_revenue'].sum() if len(df) >= 30 else df['daily_revenue'].sum()
    last_30_revenue = df.iloc[-30:]['daily_revenue'].sum() if len(df) >= 30 else df['daily_revenue'].sum()
    revenue_change_pct = ((last_30_revenue - first_30_revenue) / first_30_revenue * 100) if first_30_revenue > 0 else 0
    
    total_transactions = df['daily_transactions'].sum()
    avg_days_between_purchases = 90 / total_transactions if total_transactions > 0 else 0
    
    # Create subplot with 2 rows (stacked vertically)
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        subplot_titles=('Revenue Performance & Health', 'Purchase Activity Pattern'),
        vertical_spacing=0.15
    )
    
    # ROW 1: Revenue with health zones
    # Add 30-day rolling average as main metric
    fig.add_trace(go.Scatter(
        x=df['activity_date'], 
        y=df['rolling_30d_revenue'],
        mode='lines',
        name='30-Day Revenue',
        line=dict(color='#2E86AB', width=3),
        hovertemplate='<b>30-Day Revenue</b><br>€%{y:,.0f}<extra></extra>'
    ), row=1, col=1)
    
    # Add daily revenue as faded background
    fig.add_trace(go.Scatter(
        x=df['activity_date'],
        y=df['daily_revenue'],
        mode='lines',
        name='Daily Revenue',
        line=dict(color='#A8DADC', width=1),
        opacity=0.4,
        hovertemplate='<b>Daily</b><br>€%{y:,.0f}<extra></extra>'
    ), row=1, col=1)
    
    # Add trend indicator annotation
    trend_color = '#06D6A0' if revenue_change_pct > 0 else '#EF476F'
    trend_arrow = '↗' if revenue_change_pct > 0 else '↘'
    trend_text = f"{trend_arrow} {abs(revenue_change_pct):.1f}% vs 60 days ago"
    
    fig.add_annotation(
        x=df['activity_date'].iloc[-1],
        y=df['rolling_30d_revenue'].iloc[-1],
        text=f"<b>{trend_text}</b>",
        showarrow=True,
        arrowhead=2,
        arrowcolor=trend_color,
        ax=-60,
        ay=-40,
        bgcolor=trend_color,
        font=dict(color='white', size=11),
        borderpad=4,
        row=1, col=1)
    
    # ROW 2: Purchase frequency as bars
    # Show transaction days as vertical bars
    purchase_days = df[df['daily_transactions'] > 0]
    
    fig.add_trace(go.Bar(
        x=purchase_days['activity_date'],
        y=purchase_days['daily_transactions'],
        name='Transactions',
        marker=dict(color='#06D6A0'),
        hovertemplate='<b>%{y} transactions</b><br>€%{customdata:,.0f}<extra></extra>',
        customdata=purchase_days['daily_revenue']
    ), row=2, col=1)
    
    # Add business insight annotation
    insight_text = f"📊 Purchase Frequency: Every {avg_days_between_purchases:.1f} days | Total: {int(total_transactions)} transactions in 90 days"
    
    fig.add_annotation(
        text=insight_text,
        xref='paper', yref='paper',
        x=0, y=1.12,
        showarrow=False,
        font=dict(size=12, color='#555'),
        align='left',
        xanchor='left'
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        autosize=True,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified',
        plot_bgcolor='#FAFAFA',
        paper_bgcolor='white',
        margin=dict(t=80, b=40, l=60, r=40)
    )
    
    # Update axes
    fig.update_xaxes(title_text='Date', showgrid=True, gridcolor='#E8E8E8', row=2, col=1)
    fig.update_yaxes(title_text='Revenue (€)', showgrid=True, gridcolor='#E8E8E8', row=1, col=1)
    fig.update_yaxes(title_text='Transactions', showgrid=True, gridcolor='#E8E8E8', row=2, col=1)
    
    return fig

def create_demographic_chart(df: pd.DataFrame):
    """Create stacked bar chart showing revenue by age band and status."""
    logger.debug("create_demographic_chart - shape: %s, empty: %s", df.shape, df.empty)
    
    if df.empty:
        return go.Figure().update_layout(title="⚠️ No data available - DataFrame is empty")
    
    # Pivot data for stacked bars
    active_df = df[df['customer_status'] == 'active']
    inactive_df = df[df['customer_status'] == 'inactive']

    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=active_df['age_band'],
        y=active_df['total_revenue'],
        name='Active',
        marker_color='#06D6A0',
        text=active_df['customer_count'],
        texttemplate='%{text} customers',
        textposition='inside',
        hovertemplate='<b>Active</b><br>Age: %{x}<br>Revenue: €%{y:,.0f}<br>Customers: %{text}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        x=inactive_df['age_band'],
        y=inactive_df['total_revenue'],
        name='Inactive',
        marker_color='#EF476F',
        text=inactive_df['customer_count'],
        texttemplate='%{text} customers',
        textposition='inside',
        hovertemplate='<b>Inactive</b><br>Age: %{x}<br>Revenue: €%{y:,.0f}<br>Customers: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Revenue by Age Band & Customer Status',
        xaxis_title='Age Band',
        yaxis_title='Total Revenue (€)',
        barmode='stack',
        height=450,
        hovermode='x unified',
        plot_bgcolor='#FAFAFA',
        paper_bgcolor='white'
    )
    
    return fig

def create_geographic_heatmap(df: pd.DataFrame):
    """Create heatmap showing avg revenue by country and region."""
    logger.debug("create_geographic_heatmap - shape: %s, empty: %s", df.shape, df.empty)
    
    if df.empty:
        return go.Figure().update_layout(title="⚠️ No data available - DataFrame is empty")
    
    # Aggregate by country and region
    agg_df = df.groupby(['country', 'region']).agg({
        'total_revenue': 'sum',
        'customer_count': 'sum',
        'avg_basket': 'mean'
    }).reset_index()
    agg_df['avg_revenue_per_customer'] = agg_df['total_revenue'] / agg_df['customer_count']
    
    # Pivot for heatmap
    pivot = agg_df.pivot(index='region', columns='country', values='avg_revenue_per_customer')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='Viridis',
        text=pivot.values.round(0),
        texttemplate='€%{text:,.0f}',
        textfont={"size": 10},
        colorbar=dict(title='Avg Revenue<br>per Customer')
    ))
    
    fig.update_layout(
        title='Average Revenue per Customer by Location',
        xaxis_title='Country',
        yaxis_title='Region',
        height=400,
        plot_bgcolor='#FAFAFA',
        paper_bgcolor='white'
    )
    
    return fig

def create_seasonal_chart(df: pd.DataFrame):
    """Create line chart showing seasonal patterns with climate zones."""
    if df.empty:
        return go.Figure().update_layout(title="⚠️ No data available")
    
    fig = go.Figure()
    
    # Extract year from the data dynamically
    year = pd.to_datetime(df['month_date'].iloc[0]).year if 'month_date' in df.columns and not df.empty else 2025

    # Add revenue line
    fig.add_trace(go.Scatter(
        x=df['month'],
        y=df['total_revenue'],
        mode='lines+markers',
        name='Total Revenue',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=8),
        yaxis='y',
        hovertemplate='<b>Month %{x}</b><br>Revenue: €%{y:,.0f}<extra></extra>'
    ))
    
    # Add active customers line on secondary axis
    fig.add_trace(go.Scatter(
        x=df['month'],
        y=df['active_customers'],
        mode='lines+markers',
        name='Active Customers',
        line=dict(color='#06D6A0', width=2, dash='dot'),
        marker=dict(size=6),
        yaxis='y2',
        hovertemplate='<b>Month %{x}</b><br>Customers: %{y:,.0f}<extra></extra>'
    ))
    
    # Add season annotations
    season_colors = {
        'Winter (Dec-Feb)': {'months': [12, 1, 2], 'color': '#A8DADC'},
        'Spring (Mar-May)': {'months': [3, 4, 5], 'color': '#F4A261'},
        'Summer (Jun-Aug)': {'months': [6, 7, 8], 'color': '#E76F51'},
        'Fall (Sep-Nov)': {'months': [9, 10, 11], 'color': '#264653'}
    }
    
    fig.update_layout(
        title=f'Seasonal Revenue & Customer Activity Patterns ({year})',
        xaxis=dict(
            title='Month',
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ),
        yaxis=dict(title='Revenue (€)', side='left'),
        yaxis2=dict(title='Active Customers', side='right', overlaying='y'),
        height=450,
        hovermode='x unified',
        plot_bgcolor='#FAFAFA',
        paper_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig

def create_age_location_heatmap(df: pd.DataFrame):
    """Create heatmap showing age x country performance matrix."""
    logger.debug("create_age_location_heatmap - shape: %s, empty: %s", df.shape, df.empty)
    
    if df.empty:
        return go.Figure().update_layout(title="⚠️ No data available - DataFrame is empty")
    
    # Pivot for heatmap
    pivot = df.pivot(index='age_band', columns='country', values='avg_revenue')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn',
        text=pivot.values.round(0),
        texttemplate='€%{text:,.0f}',
        textfont={"size": 11},
        colorbar=dict(title='Avg Revenue<br>per Customer')
    ))
    
    fig.update_layout(
        title='Average Revenue: Age Band × Country',
        xaxis_title='Country',
        yaxis_title='Age Band',
        height=450,
        plot_bgcolor='#FAFAFA',
        paper_bgcolor='white'
    )
    
    return fig

def create_engagement_rate_chart(df: pd.DataFrame):
    """Create chart showing marketing engagement rate by age band."""
    logger.debug("create_engagement_rate_chart - shape: %s, empty: %s", df.shape, df.empty)
    
    if df.empty:
        return go.Figure().update_layout(title="⚠️ No data available - DataFrame is empty")
    
    # Calculate weighted average engagement rate by age band
    # Data has rows per (age_band, customer_status), so weight by customer_count
    df = df.copy()
    df['weighted_engagement'] = df['avg_engagement'] * df['customer_count']
    engagement_df = df.groupby('age_band').agg({
        'weighted_engagement': 'sum',
        'customer_count': 'sum'
    }).reset_index()
    engagement_df['avg_engagement_rate'] = (engagement_df['weighted_engagement'] / engagement_df['customer_count']).round(2)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=engagement_df['age_band'],
        y=engagement_df['avg_engagement_rate'],
        marker=dict(
            color=engagement_df['avg_engagement_rate'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title='Engagement')
        ),
        text=engagement_df['avg_engagement_rate'].round(2),
        texttemplate='%{text}',
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Avg Engagement: %{y:.2f}<extra></extra>'
    ))
    
    max_val = engagement_df['avg_engagement_rate'].max()
    fig.update_layout(
        title='Marketing Engagement Rate by Age Band',
        xaxis_title='Age Band',
        yaxis_title='Avg Marketing Engagement Rate (0-10)',
        height=320,
        margin=dict(t=50, b=40, l=60, r=40),
        yaxis=dict(range=[0, max_val * 1.25]),
        plot_bgcolor='#FAFAFA',
        paper_bgcolor='white'
    )
    
    return fig

def create_satisfaction_rate_chart(df: pd.DataFrame):
    """Create chart showing satisfaction score by age band."""
    logger.debug("create_satisfaction_rate_chart - shape: %s, empty: %s", df.shape, df.empty)
    
    if df.empty:
        return go.Figure().update_layout(title="⚠️ No data available - DataFrame is empty")
    
    # Calculate weighted average satisfaction score by age band
    # Data has rows per (age_band, customer_status), so weight by customer_count
    df = df.copy()
    df['weighted_satisfaction'] = df['avg_satisfaction'] * df['customer_count']
    satisfaction_df = df.groupby('age_band').agg({
        'weighted_satisfaction': 'sum',
        'customer_count': 'sum'
    }).reset_index()
    satisfaction_df['avg_satisfaction_score'] = (satisfaction_df['weighted_satisfaction'] / satisfaction_df['customer_count']).round(2)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=satisfaction_df['age_band'],
        y=satisfaction_df['avg_satisfaction_score'],
        marker=dict(
            color=satisfaction_df['avg_satisfaction_score'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title='Satisfaction')
        ),
        text=satisfaction_df['avg_satisfaction_score'].round(2),
        texttemplate='%{text}',
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Avg Satisfaction: %{y:.2f}<extra></extra>'
    ))
    
    max_val = satisfaction_df['avg_satisfaction_score'].max()
    fig.update_layout(
        title='Satisfaction Score by Age Band',
        xaxis_title='Age Band',
        yaxis_title='Avg Satisfaction Score (0-5)',
        height=320,
        margin=dict(t=50, b=40, l=60, r=40),
        yaxis=dict(range=[0, max_val * 1.25]),
        plot_bgcolor='#FAFAFA',
        paper_bgcolor='white'
    )
    
    return fig

# ============================================
# RISK PREDICTIVE VISUALIZATION FUNCTIONS
# ============================================

def create_semaphore_gauge(df: pd.DataFrame):
    """Create semaphore distribution pie chart."""
    if df.empty:
        return go.Figure().update_layout(title="⚠️ No risk data available")
    
    colors = {
        '🟢 GREEN': '#10b981',
        '🟡 YELLOW': '#fbbf24',
        '🟠 ORANGE': '#f97316',
        '🔴 RED': '#ef4444'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=df['semaphore_state'],
        values=df['customer_count'],
        marker=dict(colors=[colors.get(state, '#999') for state in df['semaphore_state']]),
        textinfo='label+percent+value',
        textposition='inside',
        insidetextorientation='radial',
        hovertemplate='<b>%{label}</b><br>Customers: %{value}<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title='Customer Risk Distribution (Semaphore States)',
        height=450,
        showlegend=True,
        margin=dict(t=40, b=20, l=20, r=20)
    )
    
    return fig

def create_churn_probability_hist(df: pd.DataFrame):
    """Create histogram of churn probabilities."""
    if df.empty:
        return go.Figure().update_layout(title="⚠️ No data available")
    
    fig = go.Figure(data=[go.Histogram(
        x=df['churn_probability_pct'],
        nbinsx=20,
        marker=dict(color='#3b82f6'),
        hovertemplate='Churn Probability: %{x:.1f}%<br>Count: %{y}<extra></extra>'
    )])
    
    fig.update_layout(
        title='Distribution of Churn Probabilities',
        xaxis_title='Churn Probability (%)',
        yaxis_title='Number of Customers',
        height=400,
        plot_bgcolor='#FAFAFA'
    )
    
    return fig

def create_time_to_churn_chart(df: pd.DataFrame):
    """Create chart showing predicted days to churn by risk tier."""
    if df.empty:
        return go.Figure().update_layout(title="⚠️ No data available")
    
    # Box plot by risk tier
    fig = go.Figure()
    
    risk_tiers = ['low', 'medium', 'high', 'critical']
    colors = {'low': '#10b981', 'medium': '#fbbf24', 'high': '#f97316', 'critical': '#ef4444'}
    
    for tier in risk_tiers:
        tier_data = df[df['risk_tier'] == tier]
        if not tier_data.empty:
            fig.add_trace(go.Box(
                y=tier_data['days_to_churn'],
                name=tier.capitalize(),
                marker=dict(color=colors[tier]),
                hovertemplate='Days to Churn: %{y}<extra></extra>'
            ))
    
    fig.update_layout(
        title='Predicted Time to Churn by Risk Tier',
        xaxis_title='Risk Tier',
        yaxis_title='Days to Churn',
        height=400,
        plot_bgcolor='#FAFAFA'
    )
    
    return fig

def create_intervention_actions_chart(df: pd.DataFrame):
    """Create bar chart of recommended actions."""
    if df.empty:
        return go.Figure().update_layout(title="⚠️ No recommendations available")
    
    action_counts = df['recommended_action'].value_counts().reset_index()
    action_counts.columns = ['action', 'count']
    
    fig = go.Figure(data=[go.Bar(
        x=action_counts['action'],
        y=action_counts['count'],
        marker=dict(color='#8b5cf6'),
        text=action_counts['count'],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Customers: %{y}<extra></extra>'
    )])
    
    fig.update_layout(
        title='Recommended Intervention Actions',
        xaxis_title='Action Type',
        yaxis_title='Number of Customers',
        height=400,
        plot_bgcolor='#FAFAFA'
    )
    
    return fig

def create_uplift_vs_risk_scatter(df: pd.DataFrame):
    """Create quadrant scatter with colored action zones."""
    if df.empty:
        return go.Figure().update_layout(title="⚠️ No data available")
    
    colors_risk = {'low': '#10b981', 'medium': '#fbbf24', 'high': '#f97316', 'critical': '#ef4444'}
    
    fig = go.Figure()
    
    # No background zones - clean white background
    
    # Add quadrant labels in black bold
    fig.add_annotation(x=75, y=0.42, text='<b>SAVE NOW</b>', showarrow=False, font=dict(size=14, color='black'))
    fig.add_annotation(x=25, y=0.42, text='<b>PREVENTIVE</b>', showarrow=False, font=dict(size=14, color='black'))
    fig.add_annotation(x=75, y=0.05, text='<b>DIFFICULT TO SAVE</b>', showarrow=False, font=dict(size=12, color='black'))
    fig.add_annotation(x=25, y=0.05, text='<b>MONITOR</b>', showarrow=False, font=dict(size=14, color='black'))
    
    # Scatter points colored by risk tier
    for tier in ['low', 'medium', 'high', 'critical']:
        subset = df[df['risk_tier'] == tier]
        if not subset.empty:
            fig.add_trace(go.Scatter(
                x=subset['churn_probability_pct'],
                y=subset['uplift_score'],
                mode='markers',
                name=tier.capitalize(),
                marker=dict(size=8, color=colors_risk[tier], opacity=0.7, line=dict(width=1, color='white')),
                text=subset['customer_id'],
                hovertemplate=f'<b>{tier.capitalize()}</b><br>Customer: %{{text}}<br>Churn: %{{x:.1f}}%<br>Uplift: %{{y:.2f}}<extra></extra>'
            ))
    
    # Bold division lines
    fig.add_hline(y=0.2, line=dict(dash='solid', color='#333', width=2.5))
    fig.add_vline(x=50, line=dict(dash='solid', color='#333', width=2.5))
    
    fig.update_layout(
        title=' ',
        xaxis_title='Churn Probability (%)',
        yaxis_title='Uplift Score',
        height=450,
        margin=dict(t=50, b=40, l=60, r=20),
        plot_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    
    return fig

def create_urgency_timeline_chart(df: pd.DataFrame):
    """Create urgency timeline chart: customers grouped by time-to-churn window, stacked by action type."""
    if df.empty:
        return go.Figure().update_layout(title="⚠️ No data available")
    
    bins = [0, 30, 60, 90, 999]
    labels = ['CRITICAL (<30 days)', 'URGENT (30-60 days)', 'WARNING (60-90 days)', 'MONITOR (90+ days)']
    df = df.copy()
    df['urgency_bucket'] = pd.cut(df['days_to_churn'], bins=bins, labels=labels, right=True)
    
    bucket_action = df.groupby(['urgency_bucket', 'recommended_action'], observed=True).size().reset_index(name='count')
    bucket_totals = df.groupby('urgency_bucket', observed=True).size().reset_index(name='total')
    
    action_colors = {
        'Call': '#ef4444',
        'Discount': '#f97316',
        'Support': '#8b5cf6',
        'Product engagement': '#3b82f6',
        'Training': '#10b981',
        'No intervention': '#9ca3af',
    }
    all_actions = ['Call', 'Discount', 'Support', 'Product engagement', 'Training', 'No intervention']
    
    fig = go.Figure()
    
    for action in all_actions:
        subset = bucket_action[bucket_action['recommended_action'] == action]
        if not subset.empty:
            fig.add_trace(go.Bar(
                y=subset['urgency_bucket'].astype(str),
                x=subset['count'],
                name=action,
                orientation='h',
                marker=dict(color=action_colors.get(action, '#999')),
                hovertemplate=f'<b>{action}</b><br>Customers: %{{x}}<extra></extra>',
                text=subset['count'],
                textposition='inside',
            ))
    
    for _, row in bucket_totals.iterrows():
        fig.add_annotation(
            x=row['total'],
            y=str(row['urgency_bucket']),
            text=f" {row['total']} customers",
            showarrow=False,
            xanchor='left',
            font=dict(size=13, color='#333', family='Arial Black'),
        )
    
    fig.update_layout(
        title=' ',
        xaxis_title='Number of Customers',
        yaxis_title='Urgency Window',
        barmode='stack',
        height=450,
        plot_bgcolor='white',
        margin=dict(t=50, b=40, l=60, r=120),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        yaxis=dict(categoryorder='array', categoryarray=labels[::-1]),
    )
    
    return fig

def create_state_transition_sankey(df: pd.DataFrame):
    """Create Sankey diagram showing state transitions."""
    if df.empty:
        return go.Figure().update_layout(title="⚠️ No transition data available")
    
    # Prepare data for Sankey
    states = ['🟢 GREEN', '🟡 YELLOW', '🟠 ORANGE', '🔴 RED']
    state_to_idx = {state: i for i, state in enumerate(states)}
    
    sources = []
    targets = []
    values = []
    
    for _, row in df.iterrows():
        if row['previous_state'] in state_to_idx and row['current_state'] in state_to_idx:
            sources.append(state_to_idx[row['previous_state']])
            targets.append(state_to_idx[row['current_state']])
            values.append(1)
    
    if not sources:
        return go.Figure().update_layout(title="⚠️ No valid transitions to display")
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=states,
            color=['#10b981', '#fbbf24', '#f97316', '#ef4444']
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])
    
    fig.update_layout(
        title='Customer State Transitions (Semaphore Journey)',
        height=400
    )
    
    return fig

def create_top_products_pie(df: pd.DataFrame):
    """Create a pie chart showing top products purchased, with percentages in legend only."""
    if df.empty:
        return go.Figure().update_layout(
            title="⚠️ No purchase data available for this customer",
            annotations=[{
                'text': "No purchase records found in the last 12 months",
                'xref': 'paper', 'yref': 'paper',
                'x': 0.5, 'y': 0.5,
                'showarrow': False, 'font': {'size': 14}
            }]
        )
    
    # Calculate percentages
    total = df['total_quantity'].sum()
    df = df.copy()
    df['percentage'] = (df['total_quantity'] / total * 100).round(1)
    
    # Create legend labels with product_id and percentage
    legend_labels = [f"Product {row['product_id']} ({row['percentage']}%)" for _, row in df.iterrows()]
    
    fig = go.Figure(data=[go.Pie(
        labels=legend_labels,
        values=df['total_quantity'],
        textinfo='none',
        hovertemplate='<b>%{label}</b><br>Quantity: %{value}<br>Percentage: %{percent}<extra></extra>',
        marker=dict(colors=px.colors.qualitative.Set3[:len(df)]),
        showlegend=True
    )])
    
    fig.update_layout(
        title='Top Products Purchased (Last 12 Months)',
        height=450,
        legend=dict(
            orientation='v',
            x=1.05,
            y=0.5,
            font=dict(size=12)
        ),
        margin=dict(t=60, b=40, l=40, r=150)
    )
    
    return fig

# ============================================
# GRADIO APP
# ============================================

with gr.Blocks(title="Customer Retail Intelligence Dashboard", theme=gr.themes.Soft()) as demo:
    # Hidden state for user token
    user_token_state = gr.State("service_principal")
    
    # Header
    gr.Markdown("# 📊 Customer Retail Intelligence Dashboard")
    gr.Markdown("*Actionable insights into customer behavior and trends*")
    
    # Main Tabs
    with gr.Tabs():
        
        # ==================== TAB 1: EXECUTIVE OVERVIEW ====================
        with gr.Tab("Executive Overview"):
            gr.Markdown("### Key Performance Indicators")
            #gr.Markdown("*Dashboard auto-loads on page open. Use filters to refine the view.*")
            
            with gr.Row():
                country_filter = gr.Dropdown(["All", "FRANCE", "ITALY", "SPAIN", "PORTUGAL"],
                                            value="All", label="Country")
                status_filter = gr.Dropdown(["All", "active", "inactive"],
                                           value="All", label="Status")
            
            with gr.Row():
                kpi_col1 = gr.HTML()
                kpi_col2 = gr.HTML()
                kpi_col3 = gr.HTML()
                kpi_col4 = gr.HTML()
            
            with gr.Row():
                kpi_col5 = gr.HTML()
                kpi_col6 = gr.HTML()
                kpi_col7 = gr.HTML()
                kpi_col8 = gr.HTML()
            
            def update_executive_kpis(user_token, country, status):
                df = get_executive_overview(user_token, country, status)
                if df.empty:
                    return ["⚠️ No data"] * 8
                
                row = df.iloc[0]
                
                kpi1 = create_kpi_card("Total Customers", f"{int(row['total_customers']):,}",
                                      tooltip="Total unique customers in database")
                kpi2 = create_kpi_card("Currently Active", f"{int(row['active_customers']):,}",
                                      tooltip="Customers with active status")
                kpi3 = create_kpi_card("All-Time Revenue", f"€{row['total_revenue']:,.0f}",
                                      tooltip="Cumulative revenue across all time")
                kpi4 = create_kpi_card("Last 12 Months Revenue", f"€{row['revenue_12m']:,.0f}",
                                      tooltip="Total revenue in the trailing 12-month period")
                kpi5 = create_kpi_card("All-Time Orders", f"{int(row['total_transactions']):,}",
                                      tooltip="Cumulative transaction count across all time")
                kpi6 = create_kpi_card("Last 12 Months Orders", f"{int(row['transactions_12m']):,}",
                                      tooltip="Total orders in the trailing 12-month period")
                kpi7 = create_kpi_card("Average Basket Size", f"€{row['avg_order_value']:.2f}",
                                      tooltip="Average value per transaction")
                kpi8 = create_kpi_card("At Risk (Last 60d Revenue Down >15%)", f"{int(row['declining_customers']):,}",
                                      tooltip="Revenue in last 60 days dropped >15% vs previous 60-day period")
                
                return [kpi1, kpi2, kpi3, kpi4, kpi5, kpi6, kpi7, kpi8]
            
            # Auto-load KPIs on page mount
            demo.load(fn=update_executive_kpis,
                     inputs=[user_token_state, country_filter, status_filter],
                     outputs=[kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6, kpi_col7, kpi_col8])
            
            # Auto-refresh on filter change
            country_filter.change(fn=update_executive_kpis,
                                 inputs=[user_token_state, country_filter, status_filter],
                                 outputs=[kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6, kpi_col7, kpi_col8])
            
            status_filter.change(fn=update_executive_kpis,
                                inputs=[user_token_state, country_filter, status_filter],
                                outputs=[kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6, kpi_col7, kpi_col8])
            
            # Top 10 Customers by Revenue
            gr.Markdown("---")
            gr.Markdown("### 🏆 Top 10 Customers by Revenue")
            gr.Markdown("*These customers generate the highest total revenue (filtered by country and status)*")
            
            top_customers_table = gr.Dataframe(
                label="Top Revenue Generators",
                interactive=False,
                wrap=True
            )
            
            def update_top_customers(user_token, country, status):
                """Load top 10 customers by revenue."""
                df = get_top_customers_by_revenue(user_token, country, status, limit=10)
                if df.empty:
                    return pd.DataFrame({"Message": ["No data available with current filters"]})
                return df
            
            # Auto-load table on page mount
            demo.load(fn=update_top_customers,
                     inputs=[user_token_state, country_filter, status_filter],
                     outputs=[top_customers_table])
            
            # Auto-refresh on filter change
            country_filter.change(fn=update_top_customers,
                                 inputs=[user_token_state, country_filter, status_filter],
                                 outputs=[top_customers_table])
            
            status_filter.change(fn=update_top_customers,
                                inputs=[user_token_state, country_filter, status_filter],
                                outputs=[top_customers_table])
        
        # ==================== TAB 2: CUSTOMER EXPLORER ====================
        with gr.Tab("Customer Explorer"):
            gr.Markdown("### Searchable Customer Database")
            gr.Markdown("""
            **Column Legend:**
            - **status**: active (recent purchase) / inactive (no recent activity)
            - **loyalty**: bronze / silver / gold / none (based on cumulative spend & engagement)
            - **AOV(€)**: Average Order Value per transaction
            - **Engagement**: Marketing engagement rate (0-10 scale)
            """)
           # gr.Markdown("*💡 Tip: Click on any row in the table below, then click 'View in Customer 360' to see that customer's full profile.*")
             #- **segment**: Customer classification (established = regular customer)
            
            # ===== QUICK FILTERS (Auto-apply) =====
            gr.Markdown("### 🔍 Quick Filters")
            with gr.Row():
                # SEGMENT FILTER - Commented out (can be re-enabled in future)
                # segment_filter = gr.Dropdown(
                #     ["All", "established"], 
                #     value="All", 
                #     label="Segment",
                #     scale=1
                # )
                status_filter_explorer = gr.Dropdown(
                    ["All", "active", "inactive"], 
                    value="All", 
                    label="Status",
                    scale=1
                )
                loyalty_filter = gr.Dropdown(
                    ["All", "bronze", "silver", "gold", "none"], 
                    value="All", 
                    label="Loyalty Tier",
                    multiselect=True,
                    scale=1
                )
            
            # ===== ADVANCED FILTERS (Manual apply) =====
            with gr.Accordion("⚙️ Advanced Filters (Numeric Ranges)", open=False):
                gr.Markdown("*Set numeric ranges below and click **Apply** to filter*")
                
                with gr.Row():
                    revenue_min = gr.Number(label="Revenue Min (€)", value=None, scale=1)
                    revenue_max = gr.Number(label="Revenue Max (€)", value=None, scale=1)
                    aov_min = gr.Number(label="AOV Min (€)", value=None, scale=1)
                    aov_max = gr.Number(label="AOV Max (€)", value=None, scale=1)
                
                with gr.Row():
                    trans_min = gr.Number(label="Transactions Min", value=None, scale=1)
                    trans_max = gr.Number(label="Transactions Max", value=None, scale=1)
                    days_since_min = gr.Number(label="Days Since Purchase Min", value=None, scale=1)
                    days_since_max = gr.Number(label="Days Since Purchase Max", value=None, scale=1)
                
                with gr.Row():
                    purch30_min = gr.Number(label="Purch. 30d Min", value=None, scale=1)
                    purch30_max = gr.Number(label="Purch. 30d Max", value=None, scale=1)
                    purch60_min = gr.Number(label="Purch. 60d Min", value=None, scale=1)
                    purch60_max = gr.Number(label="Purch. 60d Max", value=None, scale=1)
                
                with gr.Row():
                    purch90_min = gr.Number(label="Purch. 90d Min", value=None, scale=1)
                    purch90_max = gr.Number(label="Purch. 90d Max", value=None, scale=1)
                    engagement_min = gr.Number(label="Engagement Min (0-10)", value=None, scale=1)
                    engagement_max = gr.Number(label="Engagement Max (0-10)", value=None, scale=1)
                
                with gr.Row():
                    apply_filters_btn = gr.Button("✅ Apply Advanced Filters", variant="primary", scale=1)
                    clear_filters_btn = gr.Button("🔄 Clear All Filters", variant="secondary", scale=1)
            

            search_box = gr.Textbox(
                label="🔍 Search Customer ID", 
                placeholder="Enter customer ID and press Enter...",
                show_label=True
            )
            
            customer_table = gr.Dataframe(
                label="Customer List (Auto-loads 50 customers, click 'Load More' to see next 50)",
                interactive=False,
                wrap=True,
                column_widths=["10%", "10%", "8%", "10%", "10%", "8%", "8%", "8%", "6%", "6%", "6%", "8%"]
            )
            
            with gr.Row():
                load_more_btn = gr.Button("📥 Load More (Next 50)", variant="secondary", size="sm")
                reset_btn = gr.Button("🔄 Reset", variant="secondary", size="sm")
                view_in_360_btn = gr.Button("👉 View in Customer 360", variant="primary", size="sm")
            
            # Hidden state to track loaded data, page, and current filter clause
            page_state = gr.State(0)
            loaded_data_state = gr.State(None)
            filter_clause_state = gr.State("1=1")  # Stores current filter SQL
            
            def load_initial_customers(user_token, 
                                     # segment="All",  # SEGMENT - Commented out (can be re-enabled in future)
                                     status="All", loyalty="All",
                                     revenue_min=None, revenue_max=None, aov_min=None, aov_max=None,
                                     trans_min=None, trans_max=None, days_since_min=None, days_since_max=None,
                                     purch30_min=None, purch30_max=None, purch60_min=None, purch60_max=None,
                                     purch90_min=None, purch90_max=None, engagement_min=None, engagement_max=None):
                """Load first 50 customers with filters applied."""
                try:
                    # Build filter clause from all parameters
                    filter_clause = build_filter_clause(
                        # segment=segment,  # SEGMENT - Commented out (can be re-enabled in future)
                        status=status, loyalty=loyalty,
                        revenue_min=revenue_min, revenue_max=revenue_max,
                        aov_min=aov_min, aov_max=aov_max,
                        trans_min=trans_min, trans_max=trans_max,
                        days_since_min=days_since_min, days_since_max=days_since_max,
                        purch30_min=purch30_min, purch30_max=purch30_max,
                        purch60_min=purch60_min, purch60_max=purch60_max,
                        purch90_min=purch90_min, purch90_max=purch90_max,
                        engagement_min=engagement_min, engagement_max=engagement_max
                    )
                    
                    df = get_customer_list(user_token, page=0, search="", filter_clause=filter_clause)
                    logger.debug("Got %d rows", len(df))
                    
                    # Rename columns for clarity
                    df = df.rename(columns={
                        'customer_id': 'customer id',
                        'loyalty_tier': 'loyalty',
                        'revenue': 'Revenue (€)',
                        'aov': 'AOV(€)',
                        'days_since_last_purchase': 'days since last purchase',
                        'purchases_30d': 'Purch. 30d',
                        'purchases_60d': 'Purch. 60d',
                        'purchases_90d': 'Purch. 90d',
                        'engagement': 'Engagement'
                    })
                    return df, 0, df, filter_clause  # Also return filter_clause to store in state
                except Exception as e:
                    logger.error("Error in load_initial_customers: %s", e, exc_info=True)
                    # Return empty dataframe on error
                    import pandas as pd
                    empty_df = pd.DataFrame(columns=['customer id', 'segment', 'status', 'loyalty', 
                                                      'Revenue (€)', 'AOV(€)', 'transactions',
                                                      'days since last purchase', 'Purch. 30d', 'Purch. 60d', 
                                                      'Purch. 90d', 'Engagement'])
                    return empty_df, 0, empty_df, "1=1"
            
            def load_more_customers(user_token, current_page, current_data, filter_clause):
                """Append next 50 customers to existing data, maintaining current filters."""
                next_page = current_page + 1
                new_df = get_customer_list(user_token, page=next_page, search="", filter_clause=filter_clause)
                
                if new_df.empty:
                    return current_data, current_page, current_data  # No more data
                
                # Rename columns to match
                new_df = new_df.rename(columns={
                    'customer_id': 'customer id',
                    'loyalty_tier': 'loyalty',
                    'revenue': 'Revenue (€)',
                    'aov': 'AOV(€)',
                    'days_since_last_purchase': 'days since last purchase',
                    'purchases_30d': 'Purch. 30d',
                    'purchases_60d': 'Purch. 60d',
                    'purchases_90d': 'Purch. 90d',
                    'engagement': 'Engagement'
                })
                
                # Append to existing data
                combined = pd.concat([current_data, new_df], ignore_index=True)
                return combined, next_page, combined
            
            def search_customers(user_token, search_text, filter_clause):
                """Search for specific customer ID, maintaining current filters."""
                df = get_customer_list(user_token, page=0, search=search_text, filter_clause=filter_clause)
                # Rename columns
                df = df.rename(columns={
                    'customer_id': 'customer id',
                    'loyalty_tier': 'loyalty',
                    'revenue': 'Revenue (€)',
                    'aov': 'AOV(€)',
                    'days_since_last_purchase': 'days since last purchase',
                    'purchases_30d': 'Purch. 30d',
                    'purchases_60d': 'Purch. 60d',
                    'purchases_90d': 'Purch. 90d',
                    'engagement': 'Engagement'
                })
                return df, 0, df
            
            # Auto-load first 50 customers on tab open
            demo.load(fn=load_initial_customers,
                     inputs=[user_token_state],
                     outputs=[customer_table, page_state, loaded_data_state, filter_clause_state])
            
            # Load more button - appends next 50
            load_more_btn.click(fn=load_more_customers,
                               inputs=[user_token_state, page_state, loaded_data_state, filter_clause_state],
                               outputs=[customer_table, page_state, loaded_data_state])
            
            # Search on Enter key press
            search_box.submit(fn=search_customers,
                            inputs=[user_token_state, search_box, filter_clause_state],
                            outputs=[customer_table, page_state, loaded_data_state])
            
            # Reset button - reload first 50 with no filters
            reset_btn.click(fn=load_initial_customers,
                          inputs=[user_token_state],
                          outputs=[customer_table, page_state, loaded_data_state, filter_clause_state])
            
            # Helper function for quick filter changes (only categorical filters, no advanced)
            def apply_quick_filters(user_token, status, loyalty):
                """Apply only quick filters (categorical), reset numeric filters."""
                # segment parameter removed - can be re-added if segment filter is re-enabled
                return load_initial_customers(user_token, status, loyalty)
            
            # ===== QUICK FILTERS - Auto-apply on change =====
            # SEGMENT FILTER EVENT - Commented out (segment_filter component is commented out above)
            # To re-enable: uncomment segment_filter component definition, then uncomment this event handler
            # segment_filter.change(fn=apply_quick_filters,
            #                      inputs=[user_token_state, segment_filter, status_filter_explorer, loyalty_filter],
            #                      outputs=[customer_table, page_state, loaded_data_state, filter_clause_state])
            
            status_filter_explorer.change(fn=apply_quick_filters,
                                         inputs=[user_token_state, status_filter_explorer, loyalty_filter],
                                         outputs=[customer_table, page_state, loaded_data_state, filter_clause_state])
            
            loyalty_filter.change(fn=apply_quick_filters,
                                 inputs=[user_token_state, status_filter_explorer, loyalty_filter],
                                 outputs=[customer_table, page_state, loaded_data_state, filter_clause_state])
            
            # ===== ADVANCED FILTERS - Apply button required =====
            apply_filters_btn.click(fn=load_initial_customers,
                                   inputs=[user_token_state, status_filter_explorer, loyalty_filter,
                                          revenue_min, revenue_max, aov_min, aov_max,
                                          trans_min, trans_max, days_since_min, days_since_max,
                                          purch30_min, purch30_max, purch60_min, purch60_max,
                                          purch90_min, purch90_max, engagement_min, engagement_max],
                                   outputs=[customer_table, page_state, loaded_data_state, filter_clause_state])
            
            # Clear Filters button - reset all filters to default and reload
            def clear_all_filters(user_token):
                """Reset all filters and reload data."""
                try:
                    df = get_customer_list(user_token, page=0)
                    df = df.rename(columns={
                        'customer_id': 'customer id',
                        'loyalty_tier': 'loyalty',
                        'revenue': 'Revenue (€)',
                        'aov': 'AOV(€)',
                        'days_since_last_purchase': 'days since last purchase',
                        'purchases_30d': 'Purch. 30d',
                        'purchases_60d': 'Purch. 60d',
                        'purchases_90d': 'Purch. 90d',
                        'engagement': 'Engagement'
                    })
                    # segment value removed from return tuple (was between filter_clause and status)
                    # loyalty changed to ["All"] to match multiselect format
                    return (df, 0, df, "1=1", "All", ["All"], 
                           None, None, None, None, None, None, None, None,
                           None, None, None, None, None, None, None, None)
                except Exception as e:
                    error_df = pd.DataFrame([[f"Error: {str(e)}"]], columns=["Error"])
                    return (error_df, 0, error_df, "1=1", "All", ["All"],
                           None, None, None, None, None, None, None, None,
                           None, None, None, None, None, None, None, None)
            
            # segment_filter removed from outputs (component is commented out above)
            clear_filters_btn.click(fn=clear_all_filters,
                                   inputs=[user_token_state],
                                   outputs=[customer_table, page_state, loaded_data_state, filter_clause_state,
                                           status_filter_explorer, loyalty_filter,
                                           revenue_min, revenue_max, aov_min, aov_max,
                                           trans_min, trans_max, days_since_min, days_since_max,
                                           purch30_min, purch30_max, purch60_min, purch60_max,
                                           purch90_min, purch90_max, engagement_min, engagement_max])
        
            # # ==================== TAB 3: CUSTOMER 360 ====================
        with gr.Tab("Customer 360"):
            gr.Markdown("### Single Customer Profile")
            gr.Markdown("*Enter a customer ID in the box below and press Enter to load their complete profile*")
            
            customer_id_input = gr.Textbox(
                label="🔍 Customer ID", 
                placeholder="Type customer ID and press Enter to search...",
                show_label=True
            )
            
            # Dynamic Metric Values - 4 Column Layout (updates with customer data)
            metrics_table_html = gr.HTML("""
            <div style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 8px;">
                <h4 style="margin-top: 0; color: #333; font-size: 16px;">📊 Customer Metrics</h4>
                <p style="color: #666; font-size: 13px; margin-top: 5px;">Load a customer profile to see their metrics here</p>
            </div>
            """)
            
            profile_output = gr.Markdown()
            activity_chart = gr.Plot()
            # top_products_chart = gr.Plot()  # Commented out - top products pie chart
            
            def load_customer_profile(user_token, customer_id):
                if not customer_id:
                    empty_metrics_html = """
                    <div style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 8px;">
                        <h4 style="margin-top: 0; color: #333; font-size: 16px;">📊 Customer Metrics</h4>
                        <p style="color: #666; font-size: 13px; margin-top: 5px;">Load a customer profile to see their metrics here</p>
                    </div>
                    """
                    return empty_metrics_html, "⚠️ Please enter a customer ID", None
                
                # Get customer profile
                df = get_customer_360(user_token, customer_id)
                if df.empty:
                    empty_metrics_html = """
                    <div style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 8px;">
                        <h4 style="margin-top: 0; color: #333; font-size: 16px;">📊 Customer Metrics</h4>
                        <p style="color: #666; font-size: 13px; margin-top: 5px;">Load a customer profile to see their metrics here</p>
                    </div>
                    """
                    return empty_metrics_html, f"⚠️ Customer {customer_id} not found", None
                
                customer = df.iloc[0]
                
                # Helper function to safely format nullable fields
                def fmt_nullable(value, format_str=".2f", default="N/A"):
                    """Safely format a value that might be null/None."""
                    if pd.isna(value) or value is None:
                        return default
                    if format_str:
                        return f"{value:{format_str}}"
                    return str(value)
                
                # Generate dynamic metrics table HTML with actual customer values
                metrics_html = f"""
                <div style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 8px;">
                    <h4 style="margin-top: 0; color: #333; font-size: 16px;">📊 Customer Metrics</h4>
                    
                    <!-- Customer Info Header -->
                    <div style="background-color: #ffffff; padding: 12px; border-radius: 6px; margin-bottom: 15px; border-left: 4px solid #2E86AB;">
                        <div style="display: flex; gap: 25px; flex-wrap: wrap; align-items: center;">
                            <div>
                                <span style="font-size: 12px; color: #666; text-transform: uppercase; font-weight: 600;">Customer ID</span><br>
                                <span style="font-size: 18px; color: #2E86AB; font-weight: bold;">{customer['customer_id']}</span>
                            </div>
                            <div>
                                <span style="font-size: 12px; color: #666; text-transform: uppercase; font-weight: 600;">Status</span><br>
                                <span style="font-size: 16px; color: {'#10b981' if customer['customer_status'] == 'active' else '#EF476F'}; font-weight: 600;">{customer['customer_status'].upper()}</span>
                            </div>
                            

                            <div>
                                <span style="font-size: 12px; color: #666; text-transform: uppercase; font-weight: 600;">Loyalty Tier</span><br>
                                <span style="font-size: 16px; color: #F4A261; font-weight: 600;">{customer['loyalty_tier'].upper() if customer['loyalty_tier'] != 'none' else 'NONE'}</span>
                            </div>
                        </div>
                    </div>
                    
                    <table style="width: 100%; border-collapse: collapse; border: none;">
                        <tr>
                            <!-- Column 1: Revenue Metrics -->
                            <td style="width: 25%; vertical-align: top; padding: 10px; border: none;">
                                <div style="font-weight: bold; color: #2E86AB; margin-bottom: 12px; font-size: 15px;">💰 Revenue Metrics</div>
                                <div style="font-size: 14px; line-height: 2.2; color: #333;">
                                    <strong>Total Revenue:</strong> <span style="font-size: 14px; color: #2E86AB;">€{customer['total_revenue']:,.2f}</span><br>
                                    <strong>Revenue (12M):</strong> <span style="font-size: 14px; color: #2E86AB;">€{customer['revenue_last_12_months']:,.2f}</span><br>
                                    <strong>Average Order Value:</strong> <span style="font-size: 14px; color: #2E86AB;">€{customer['average_order_value']:.2f}</span><br>
                                    <strong>Total Transactions:</strong> <span style="font-size: 14px; color: #2E86AB;">{int(customer['number_of_transactions'])}</span>
                                </div>
                            </td>
                            
                            <!-- Column 2: Recency & Activity -->
                            <td style="width: 25%; vertical-align: top; padding: 10px; border: none;">
                                <div style="font-weight: bold; color: #06D6A0; margin-bottom: 12px; font-size: 15px;">🕒 Recency & Activity</div>
                                <div style="font-size: 14px; line-height: 2.2; color: #333;">
                                    <strong>Last Purchase Date:</strong> <span style="font-size: 14px; color: #06D6A0;">{customer['last_purchase_date']}</span><br>
                                    <strong>Days Since Purchase:</strong> <span style="font-size: 14px; color: #06D6A0;">{int(customer['days_since_last_purchase'])}</span><br>
                                    <strong>Purchases (30d):</strong> <span style="font-size: 14px; color: #06D6A0;">{int(customer['purchases_last_30_days'])}</span><br>
                                    <strong>Purchases (60d):</strong> <span style="font-size: 14px; color: #06D6A0;">{int(customer['purchases_last_60_days'])}</span><br>
                                    <strong>Purchases (90d):</strong> <span style="font-size: 14px; color: #06D6A0;">{int(customer['purchases_last_90_days'])}</span>
                                </div>
                            </td>
                            
                            <!-- Column 3: Engagement & Satisfaction -->
                            <td style="width: 25%; vertical-align: top; padding: 10px; border: none;">
                                <div style="font-weight: bold; color: #F4A261; margin-bottom: 12px; font-size: 15px;">🎯 Engagement & Satisfaction</div>
                                <div style="font-size: 14px; line-height: 2.2; color: #333;">
                                    <strong>Engagement Rate:</strong> <span style="font-size: 14px; color: #F4A261;">{fmt_nullable(customer['marketing_engagement_rate'], '.2f')}</span><br>
                                    <strong>Satisfaction Score:</strong> <span style="font-size: 14px; color: #F4A261;">{fmt_nullable(customer['average_satisfaction_score'], '.2f')}</span><br>
                                    <strong>Unresolved Interactions:</strong> <span style="font-size: 14px; color: #F4A261;">{int(customer['unresolved_interactions'])}</span>
                                </div>
                            </td>
                            
                            <!-- Column 4: Change Indicators -->
                            <td style="width: 25%; vertical-align: top; padding: 10px; border: none;">
                                <div style="font-weight: bold; color: #EF476F; margin-bottom: 12px; font-size: 15px;">📉 Change Indicators</div>
                                <div style="font-size: 14px; line-height: 2.2; color: #333;">
                                    <strong>Revenue Change %:</strong> <span style="font-size: 14px; color: {'#10b981' if pd.notna(customer['revenue_change_pct']) and customer['revenue_change_pct'] >= 0 else '#EF476F' if pd.notna(customer['revenue_change_pct']) else '#999'};">{fmt_nullable(customer['revenue_change_pct'], '+.1f') + '%' if pd.notna(customer['revenue_change_pct']) else 'N/A'}</span><br>
                                    <strong>Frequency Change %:</strong> <span style="font-size: 14px; color: {'#10b981' if pd.notna(customer['purchase_frequency_change_pct']) and customer['purchase_frequency_change_pct'] >= 0 else '#EF476F' if pd.notna(customer['purchase_frequency_change_pct']) else '#999'};">{fmt_nullable(customer['purchase_frequency_change_pct'], '+.1f') + '%' if pd.notna(customer['purchase_frequency_change_pct']) else 'N/A'}</span>
                                </div>
                            </td>
                        </tr>
                    </table>
                    
                    <!-- Metric Explanations Section -->
                    <div style="margin-top: 20px; padding-top: 15px; border-top: 2px solid #e0e0e0;">
                        <h5 style="color: #555; font-size: 14px; margin-bottom: 10px;">📘 Metric Explanations</h5>
                        <table style="width: 100%; border-collapse: collapse; border: none;">
                            <tr>
                                <!-- Column 1 Explanation -->
                                <td style="width: 25%; vertical-align: top; padding: 10px; border: none;">
                                    <div style="font-weight: 600; color: #2E86AB; margin-bottom: 6px; font-size: 13px;">💰 Revenue Metrics</div>
                                    <div style="font-size: 12px; line-height: 1.6; color: #666;">
                                        <!-- Measures customer monetary value:<br> -->
                                        • <strong>Total Revenue</strong>: Lifetime spend<br>
                                        • <strong>Revenue (12M)</strong>: Last 12 months<br>
                                        • <strong>Avg Order Value</strong>: Basket size<br>
                                        • <strong>Total Transactions</strong>: Purchase count
                                    </div>
                                </td>
                                
                                <!-- Column 2 Explanation -->
                                <td style="width: 25%; vertical-align: top; padding: 10px; border: none;">
                                    <div style="font-weight: 600; color: #06D6A0; margin-bottom: 6px; font-size: 13px;">🕒 Recency & Activity</div>
                                    <div style="font-size: 12px; line-height: 1.6; color: #666;">
                                        <!-- Tracks purchase timing patterns:<br> -->
                                        • <strong>Last Purchase Date</strong>: Most recent order<br>
                                        • <strong>Days Since Purchase</strong>: Recency indicator<br>
                                        • <strong>Purchases (30d/60d/90d)</strong>: Recent activity levels
                                    </div>
                                </td>
                                
                                <!-- Column 3 Explanation -->
                                <td style="width: 25%; vertical-align: top; padding: 10px; border: none;">
                                    <div style="font-weight: 600; color: #F4A261; margin-bottom: 6px; font-size: 13px;">🎯 Engagement & Satisfaction</div>
                                    <div style="font-size: 12px; line-height: 1.6; color: #666;">
                                        <!-- Measures customer experience:<br> -->
                                        • <strong>Engagement Rate</strong>: customer interaction with marketing communications (emails, campaigns, promotions)<br>
                                        • <strong>Satisfaction Score</strong>: Average rating from customer feedback surveys, support interactions, and product reviews<br>
                                        • <strong>Unresolved Interactions</strong>: Count of open support tickets, pending issues, or customer complaints that haven't been resolved yet.
                                    </div>
                                </td>
                                
                                <!-- Column 4 Explanation -->
                                <td style="width: 25%; vertical-align: top; padding: 10px; border: none;">
                                    <div style="font-weight: 600; color: #EF476F; margin-bottom: 6px; font-size: 13px;">📉 Change Indicators</div>
                                    <div style="font-size: 12px; line-height: 1.6; color: #666;">
                                        <!-- Tracks behavioral changes:<br>  -->
                                        • <strong>Revenue Change %</strong>: Spending trend (last 60d vs previous 60d). Negative values (e.g., -25%) indicate declining spend and potential churn risk<br>
                                        • <strong>Frequency Change %</strong>: Purchase pattern shift, measure if the customer is buying more or less often. A drop suggests reduced engagement or interest
                                    </div>
                                </td>
                            </tr>
                        </table>
                    </div>
                </div>
                """
                
                
                # Get activity trend
                activity_df = get_customer_activity_trend(user_token, customer_id, days=90)
                chart = create_customer_activity_chart(activity_df)
                
                # Get top products purchased in last 12 months (COMMENTED OUT)
                # top_products_df = get_customer_top_products(user_token, customer_id, days=365)
                # products_pie = create_top_products_pie(top_products_df)
                
                # Simple profile summary text
                profile_md = f"""**Profile loaded successfully for Customer {customer_id}**
                
📍 Location: {customer['country']} - {customer['region']} | 📅 Customer Since: {customer['customer_since']}  

                """
                
                return metrics_html, profile_md, chart
            
            # Trigger search on Enter key press in text box
            customer_id_input.submit(fn=load_customer_profile,
                                     inputs=[user_token_state, customer_id_input],
                                     outputs=[metrics_table_html, profile_output, activity_chart])
            
            # Row click handler: silently loads the customer's profile data (no tab switch)
            def on_customer_row_select(evt: gr.SelectData, user_token, loaded_data):
                """Handle row click in customer table - load profile data silently."""
                try:
                    row_idx = evt.index[0]
                    if loaded_data is not None and hasattr(loaded_data, 'iloc') and row_idx < len(loaded_data):
                        customer_id = str(loaded_data.iloc[row_idx]['customer id'])
                        result = load_customer_profile(user_token, customer_id)
                        return result[0], result[1], result[2], customer_id
                    return "", "⚠️ Please select a valid customer", None, ""
                except Exception as e:
                    logger.warning("Error in on_customer_row_select: %s", e)
                    return "", f"⚠️ Error loading profile: {str(e)}", None, ""
            
            customer_table.select(
                fn=on_customer_row_select,
                inputs=[user_token_state, loaded_data_state],
                outputs=[metrics_table_html, profile_output, activity_chart, customer_id_input]
            )
            
            # Button to switch to Customer 360 tab (separate from row select to avoid scroll-triggered redirects)
            view_in_360_btn.click(
                fn=None,
                js="() => { const btns = document.querySelectorAll('button'); for (let b of btns) { if (b.textContent.includes('Customer 360')) { b.click(); break; } } setTimeout(() => window.scrollTo(0, 0), 100); }"
            )
        
        # ==================== TAB 4: BEHAVIOUR & TRENDS ====================
        with gr.Tab("Behaviour & Trends"):
            gr.Markdown("### Multi-Dimensional Customer Behaviour Analysis")
            gr.Markdown("*Explore how location, age, seasonality, and other factors drive customer activity and revenue*")
            
            # Filter
            country_filter_trends = gr.Dropdown(
                ["All", "FRANCE", "ITALY", "SPAIN", "PORTUGAL"],
                value="All",
                label="🌍 Country"
            )
            
            # Row 1: Demographic & Geographic Analysis
            gr.Markdown("#### 👥 Demographic & Geographic Performance")
            with gr.Row():
                demographic_chart = gr.Plot(value=go.Figure().update_layout(title="Loading..."))
                active_rate_chart = gr.Plot(value=go.Figure().update_layout(title="Loading..."))
            
            # Row 2: Cross-dimensional Analysis
            gr.Markdown("#### 🔀 Cross-Dimensional Insights")
            with gr.Row():
                age_location_heatmap = gr.Plot(value=go.Figure().update_layout(title="Loading..."))
                geographic_heatmap = gr.Plot(value=go.Figure().update_layout(title="Loading..."))
            
            # Row 3: Satisfaction & Seasonal Analysis
            gr.Markdown("#### 😊 Satisfaction Score & 🌡️ Seasonal Patterns")
            with gr.Row():
                satisfaction_chart = gr.Plot(value=go.Figure().update_layout(title="Loading..."))
                seasonal_chart = gr.Plot(value=go.Figure().update_layout(title="Loading..."))

            
            def load_behaviour_analysis(user_token, country):
                """Load all behaviour analysis charts."""
                # Create empty figure helper
                def make_empty_fig(msg):
                    return go.Figure().update_layout(
                        title=msg,
                        annotations=[{
                            'text': msg,
                            'xref': 'paper',
                            'yref': 'paper',
                            'x': 0.5,
                            'y': 0.5,
                            'showarrow': False,
                            'font': {'size': 14}
                        }]
                    )

                try:
                    logger.debug("load_behaviour_analysis called, country=%s", country)

                    # Run 4 independent SQL queries concurrently (each thread
                    # gets its own connection from the thread-local pool).
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        future_demo = pool.submit(get_demographic_performance, user_token, country, "All")
                        future_age_loc = pool.submit(get_age_location_matrix, user_token, "All")
                        future_geo = pool.submit(get_geographic_performance, user_token, "All", "All")
                        future_seasonal = pool.submit(get_seasonal_trends, user_token, 2025)

                        demo_df = future_demo.result()
                        age_loc_df = future_age_loc.result()
                        geo_df = future_geo.result()
                        seasonal_df = future_seasonal.result()

                    # Build charts from fetched data (CPU-bound, fast)
                    try:
                        demo_chart = create_demographic_chart(demo_df)
                        if demo_chart is None:
                            demo_chart = make_empty_fig("⚠️ Demographic chart returned None")
                    except Exception as e:
                        logger.warning("Error in demographic chart: %s", e)
                        demo_chart = make_empty_fig(f"⚠️ Error: {str(e)}")

                    try:
                        engagement_chart = create_engagement_rate_chart(demo_df)
                        if engagement_chart is None:
                            engagement_chart = make_empty_fig("⚠️ Engagement rate chart returned None")
                    except Exception as e:
                        logger.warning("Error in engagement rate chart: %s", e)
                        engagement_chart = make_empty_fig(f"⚠️ Error: {str(e)}")

                    try:
                        satisfaction_chart_fig = create_satisfaction_rate_chart(demo_df)
                        if satisfaction_chart_fig is None:
                            satisfaction_chart_fig = make_empty_fig("⚠️ Satisfaction chart returned None")
                    except Exception as e:
                        logger.warning("Error in satisfaction chart: %s", e)
                        satisfaction_chart_fig = make_empty_fig(f"⚠️ Error: {str(e)}")

                    try:
                        age_loc_chart = create_age_location_heatmap(age_loc_df)
                        if age_loc_chart is None:
                            age_loc_chart = make_empty_fig("⚠️ Age/location chart returned None")
                    except Exception as e:
                        logger.warning("Error in age/location chart: %s", e)
                        age_loc_chart = make_empty_fig(f"⚠️ Error: {str(e)}")

                    try:
                        geo_chart = create_geographic_heatmap(geo_df)
                        if geo_chart is None:
                            geo_chart = make_empty_fig("⚠️ Geographic chart returned None")
                    except Exception as e:
                        logger.warning("Error in geographic chart: %s", e)
                        geo_chart = make_empty_fig(f"⚠️ Error: {str(e)}")

                    try:
                        seasonal_chart_fig = create_seasonal_chart(seasonal_df)
                        if seasonal_chart_fig is None:
                            seasonal_chart_fig = make_empty_fig("⚠️ Seasonal chart returned None")
                    except Exception as e:
                        logger.warning("Error in seasonal chart: %s", e)
                        seasonal_chart_fig = make_empty_fig(f"⚠️ Error: {str(e)}")

                    return demo_chart, engagement_chart, satisfaction_chart_fig, age_loc_chart, geo_chart, seasonal_chart_fig

                except Exception as e:
                    logger.error("CRITICAL ERROR in load_behaviour_analysis: %s", e, exc_info=True)
                    error_fig = make_empty_fig(f"⚠️ Critical Error: {str(e)}")
                    return error_fig, error_fig, error_fig, error_fig, error_fig, error_fig
            
            # Auto-load on page mount
            demo.load(fn=load_behaviour_analysis,
                     inputs=[user_token_state, country_filter_trends],
                     outputs=[demographic_chart, active_rate_chart, satisfaction_chart, age_location_heatmap, geographic_heatmap, seasonal_chart])
            
            # Auto-apply on country change
            country_filter_trends.change(fn=load_behaviour_analysis,
                               inputs=[user_token_state, country_filter_trends],
                               outputs=[demographic_chart, active_rate_chart, satisfaction_chart, age_location_heatmap, geographic_heatmap, seasonal_chart])
        
        # ==================== TAB 5: RISK SIGNALS (COMMENTED OUT) ====================
        # with gr.Tab("Risk Signals"):
        #     gr.Markdown("### At-Risk Customers")
        #     gr.Markdown("*Rule-based risk indicators: declining revenue/frequency, extended inactivity, low engagement*")
        #     
        #     load_risk_btn = gr.Button("⚠️ Load At-Risk Customers", variant="primary")
        #     
        #     risk_table = gr.Dataframe(
        #         headers=["customer_id", "status", "revenue", "days_since_last_purchase",
        #                 "freq_change_pct", "revenue_change_pct", "engagement", 
        #                 "unresolved_interactions", "risk_level"],
        #         label="At-Risk Customers",
        #         interactive=False
        #     )
        #     
        #     def load_risk_customers(user_token):
        #         df = get_at_risk_customers(user_token)
        #         return df
        #     
        #     load_risk_btn.click(fn=load_risk_customers,
        #                        inputs=user_token_state,
        #                        outputs=risk_table)
        
        # ==================== TAB 6: RISK PREDICTIVE ====================
        with gr.Tab("Risk Predictive"):
            gr.Markdown("""### 🎯 AI-Powered Churn Prediction & Intervention Framework
            **WHO** is at risk? | **WHEN** will they churn? | **CAN** we save them? | **WHAT** should we do?""")
            
            gr.Markdown("""**System Components powered by XGBoost:**
            * **Churn Risk Model**: Predicts probability each customer will churn
            * **Survival Model**: Estimates time-to-churn in days
            * **Uplift Model**: Identifies who is likely to respond to interventions
            """)
            #* **Semaphore States**: 🟢 GREEN (low risk) | 🟡 YELLOW (medium) | 🟠 ORANGE (high) | 🔴 RED (critical)
            
            # Filters
            with gr.Row():
                risk_tier_filter = gr.Dropdown(
                    ["All", "low", "medium", "high", "critical"],
                    value="All",
                    label="Risk Tier"
                )
                action_filter = gr.Dropdown(
                    ["All", "Call", "Discount", "Support", "Product engagement", "Training", "No intervention"],
                    value="All",
                    label="Recommended Action"
                )
            
            load_predictions_btn = gr.Button("🎯 Load Risk Predictions", variant="primary")
            
            # Two charts side by side: Urgency Timeline + Risk vs Intervention
            with gr.Row():
                urgency_chart = gr.Plot(label="Urgency Timeline by Action")
                uplift_scatter = gr.Plot(label="Risk vs Intervention Response")
            
            # Merged Table
            gr.Markdown("#### 📋 Risk Predictions & Intervention Recommendations")
            merged_risk_table = gr.Dataframe(
                label="Merged Risk Predictions (Top 100 by Priority)",
                interactive=False,
                wrap=True
            )
            
            gr.Markdown("""            
            **🔄 Model Updates:**
            Models are retrained weekly on latest customer behavior. Predictions refresh daily via batch scoring.
            """)
            
            def load_all_predictions(user_token, risk_tier, action_filter):
                """Load all risk predictive charts and tables."""
                try:
                    # Run 3 independent queries concurrently for ~3x speedup
                    with ThreadPoolExecutor(max_workers=3) as pool:
                        future_pred = pool.submit(get_risk_predictions, user_token, risk_tier)
                        future_rec = pool.submit(get_intervention_recommendations, user_token, action_filter, 1000)
                        future_merged = pool.submit(get_merged_risk_predictions, user_token, risk_tier, action_filter, 100)

                        predictions_df = future_pred.result()
                        recommendations_df = future_rec.result()
                        merged_df = future_merged.result()

                    # Merge predictions with recommendations for urgency chart
                    # Cast customer_id to string on both sides: risk_predictions has INT, intervention_recommendations has STRING
                    if not predictions_df.empty and not recommendations_df.empty:
                        pred_subset = predictions_df.copy()
                        pred_subset['customer_id'] = pred_subset['customer_id'].astype(str)
                        rec_subset = recommendations_df[['customer_id', 'recommended_action']].copy()
                        rec_subset['customer_id'] = rec_subset['customer_id'].astype(str)
                        urgency_data = pred_subset.merge(
                            rec_subset,
                            on='customer_id',
                            how='inner'
                        )
                    else:
                        urgency_data = pd.DataFrame()

                    # Create charts
                    urgency_fig = create_urgency_timeline_chart(urgency_data)
                    uplift_fig = create_uplift_vs_risk_scatter(predictions_df)

                    return (urgency_fig, uplift_fig, merged_df)

                except Exception as e:
                    logger.error("ERROR in load_all_predictions: %s", e, exc_info=True)

                    error_fig = go.Figure().update_layout(
                        title=f"⚠️ Error loading predictions: {str(e)}",
                        annotations=[{
                            'text': "Check that the training notebook has been run and tables exist",
                            'xref': 'paper',
                            'yref': 'paper',
                            'showarrow': False
                        }]
                    )
                    empty_df = pd.DataFrame()
                    return (error_fig, error_fig, empty_df)
            
            load_predictions_btn.click(
                fn=load_all_predictions,
                inputs=[user_token_state, risk_tier_filter, action_filter],
                outputs=[urgency_chart, uplift_scatter, merged_risk_table]
            )
        
        # ==================== TAB: ASK GENIE ====================
        with gr.Tab("Ask Genie") as ask_genie_tab:
            gr.Markdown("### Ask Genie")
            
            # ===== SHORTCUT QUERIES SECTION =====
            with gr.Accordion("⚡ Quick Insights", open=True):
                #gr.Markdown("*Pre-optimized queries for common questions — results in < 1 second*")
                
                # Customer Metrics
                with gr.Group():
                    gr.Markdown("**👥 Customer Metrics**")
                    with gr.Row():
                        q1_btn = gr.Button("How many customers?", size="sm", scale=1)
                        q2_btn = gr.Button("Active vs inactive?", size="sm", scale=1)
                        q3_btn = gr.Button("Avg satisfaction score?", size="sm", scale=1)
                
                # Revenue Metrics
                with gr.Group():
                    gr.Markdown("**💰 Revenue Metrics**")
                    with gr.Row():
                        q4_btn = gr.Button("Total revenue?", size="sm", scale=1)
                        q5_btn = gr.Button("Avg revenue per customer?", size="sm", scale=1)
                        q6_btn = gr.Button("Revenue by country?", size="sm", scale=1)
                    with gr.Row():
                        q7_btn = gr.Button("Avg order value?", size="sm", scale=1)
                        q8_btn = gr.Button("Top 10 customers?", size="sm", scale=1)
                
                # Risk & Interventions
                with gr.Group():
                    gr.Markdown("**⚠️ Risk & Interventions**")
                    with gr.Row():
                        q9_btn = gr.Button("How many at high risk?", size="sm", scale=1)
                        q10_btn = gr.Button("Which customers at risk?", size="sm", scale=1)
                    with gr.Row():
                        q11_btn = gr.Button("Critical interventions?", size="sm", scale=1)
            
            # ===== GENIE CUSTOM QUESTIONS SECTION =====
            gr.Markdown("---")
            gr.Markdown("### 🤖 Ask Genie (Custom Questions)")
            gr.Markdown("""
            For complex or custom questions not covered by shortcuts above.

            **Examples:**
            * What are the trends in customer satisfaction by region over time?
            * Compare high-value customers between France and Italy
            * Show me customers who decreased spending but increased purchase frequency
            """)
            
            genie_output = gr.Markdown("")
            with gr.Row():
                genie_input = gr.Textbox(
                    placeholder="Ask a custom question...",
                    scale=8,
                    show_label=False
                )
                genie_btn = gr.Button("Ask Genie", variant="primary", scale=2)

            # "Don't trust this?" flag button + collapsible report popup
            dont_trust_btn = gr.Button("🚩 Don't trust this answer?", variant="stop", size="sm")

            with gr.Column(visible=False) as trust_popup:
                gr.Markdown("### 📋 Query Sent to Engineers for Analysis\n*Review the Genie response details below:*")
                trust_report = gr.Markdown()
                trust_close_btn = gr.Button("✖ Close", size="sm")

            # ---- Genie response cache (TTL 5 min for repeat questions) ----
            _genie_cache = {}
            _GENIE_CACHE_TTL = 300

            def _prewarm_genie():
                """Lightweight page-load check to ensure warehouse is still warm.
                
                The heavy prewarm (multi-table queries) already ran at module init.
                This just does a quick ping to keep the warehouse active if the user
                takes a while before asking their first question.
                """
                try:
                    # Quick warehouse ping (reuses existing connection pool)
                    sql_query_with_service_principal(f"SELECT 1 FROM {CUSTOMER_360_TABLE} LIMIT 1")
                except Exception:
                    pass  # Best-effort keep-alive; ignore errors
                return ""

            # Warm the Genie space when the app loads
            demo.load(fn=_prewarm_genie, outputs=[genie_output])

            def ask_genie(question):
                """Send a question to Genie and return the answer."""
                if not question.strip():
                    yield ""
                    return

                import time

                # ---- Cache check (instant for repeat questions) ----
                cached = _genie_cache.get(question)
                if cached and time.time() - cached['time'] < _GENIE_CACHE_TTL:
                    yield cached['answer']
                    return

                import requests as req

                SPACE_ID = "01f1a7bebfb61f4eb98e17b798e94231"

                yield "*Genie is thinking...*"

                try:
                    auth_headers = cfg.authenticate()
                    host = cfg.host
                    if not host.startswith("http"):
                        host = f"https://{host}"
                    base = f"{host}/api/2.0/genie/spaces/{SPACE_ID}"

                    # Start conversation
                    resp = req.post(f"{base}/start-conversation", json={"content": question}, headers=auth_headers, timeout=30)
                    if resp.status_code != 200:
                        yield f"Genie error ({resp.status_code}): {resp.text}"
                        return

                    data = resp.json()
                    conv_id = data["conversation"]["id"]
                    msg_id = data["message"]["id"]
                    status = data["message"]["status"]

                    answer_text = None
                    sql_query = None

                    # Check if start-conversation already returned a completed
                    # response (happens for cached/simple queries)
                    if status == "COMPLETED":
                        for att in data["message"].get("attachments", []):
                            if "text" in att and att["text"].get("purpose") == "TEXT_ATTACHMENT_PURPOSE_ANSWER":
                                answer_text = att["text"].get("content", "")
                            if "query" in att:
                                sql_query = att["query"].get("query", "")

                    # Poll for response. Poll FIRST (before sleeping) to catch
                    # fast responses, then use adaptive interval:
                    #   1s for first 20s, 2s for 20-40s, 5s after. Total 90s.
                    elapsed = 0
                    max_total = 90
                    last_ui_update = 0

                    while status not in ("COMPLETED", "FAILED") and elapsed < max_total:
                        # Poll immediately (no sleep before first poll)
                        msg_resp = req.get(f"{base}/conversations/{conv_id}/messages", headers=auth_headers, timeout=15)
                        if msg_resp.status_code == 200:
                            for msg in msg_resp.json().get("messages", []):
                                if msg.get("message_id") == msg_id:
                                    status = msg.get("status", status)
                                    if status == "COMPLETED":
                                        for att in msg.get("attachments", []):
                                            if "text" in att and att["text"].get("purpose") == "TEXT_ATTACHMENT_PURPOSE_ANSWER":
                                                answer_text = att["text"].get("content", "")
                                            if "query" in att:
                                                sql_query = att["query"].get("query", "")
                                    break
                        if status in ("COMPLETED", "FAILED"):
                            break

                        # Adaptive sleep
                        if elapsed < 20:
                            delay = 1
                        elif elapsed < 40:
                            delay = 2
                        else:
                            delay = 5
                        time.sleep(delay)
                        elapsed += delay
                        if elapsed - last_ui_update >= 5:
                            last_ui_update = elapsed
                            yield f"*Genie is thinking... ({elapsed}s)*"

                    if status == "FAILED":
                        yield "Genie could not process this question. Try rephrasing."
                        return
                    if status != "COMPLETED":
                        yield "Genie response timed out. Try again."
                        return

                    if answer_text:
                        result = (
                            '<div style="background: #e8f4fd; border-left: 4px solid #2196F3; '
                            'padding: 12px 16px; border-radius: 8px; margin: 10px 0 15px 0; '
                            'font-weight: bold; color: #2196F3; font-size: 16px;">'
                            '🤖 Genie Answer'
                            '</div>\n\n'
                            f'{answer_text}'
                        )
                        # Cache the response for repeat questions (with metadata for flagging)
                        _genie_cache[question] = {
                            'answer': result,
                            'time': time.time(),
                            'meta': {
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'question': question,
                                'answer': answer_text,
                                'sql_query': sql_query or 'No query generated'
                            }
                        }
                        yield result
                    else:
                        yield "No response from Genie."

                except Exception as e:
                    yield f"Error: {str(e)}"

            genie_btn.click(
                fn=ask_genie,
                inputs=[genie_input],
                outputs=[genie_output]
            )

            genie_input.submit(
                fn=ask_genie,
                inputs=[genie_input],
                outputs=[genie_output]
            )

            def show_genie_report(question):
                """Show Genie response details in a popup for engineer review."""
                if not question or not question.strip():
                    return gr.update(visible=True), "⚠️ No question asked yet. Ask Genie a question first."
                cached = _genie_cache.get(question)
                if not cached or 'meta' not in cached:
                    return gr.update(visible=True), "⚠️ No cached response found for this question."
                meta = cached['meta']
                report = (
                    f"**Timestamp:** {meta['timestamp']}\n\n"
                    f"**Question:**\n> {meta['question']}\n\n"
                    f"**Answer:**\n> {meta['answer']}\n\n"
                    f"**Query Used to Produce the Answer:**\n"
                    f"```sql\n{meta['sql_query']}\n```"
                )
                return gr.update(visible=True), report

            dont_trust_btn.click(
                fn=show_genie_report,
                inputs=[genie_input],
                outputs=[trust_popup, trust_report]
            )

            def close_genie_report():
                """Hide the report popup."""
                return gr.update(visible=False)

            trust_close_btn.click(
                fn=close_genie_report,
                inputs=None,
                outputs=[trust_popup]
            )

            # Clear Genie output and input when user re-enters the tab (fresh state each visit)
            ask_genie_tab.select(
                fn=lambda: ("", gr.update(visible=False), ""),
                outputs=[genie_output, trust_popup, genie_input]
            )

            # Wire up shortcut buttons
            q1_btn.click(fn=run_shortcut_query, 
                         inputs=[user_token_state, gr.State("How many customers are there?")],
                         outputs=[genie_output])
            
            q2_btn.click(fn=run_shortcut_query,
                         inputs=[user_token_state, gr.State("How many active vs inactive customers?")],
                         outputs=[genie_output])
            
            q3_btn.click(fn=run_shortcut_query,
                         inputs=[user_token_state, gr.State("What is the average satisfaction score?")],
                         outputs=[genie_output])
            
            q4_btn.click(fn=run_shortcut_query,
                         inputs=[user_token_state, gr.State("What is the total revenue?")],
                         outputs=[genie_output])
            
            q5_btn.click(fn=run_shortcut_query,
                         inputs=[user_token_state, gr.State("What is the average revenue per customer?")],
                         outputs=[genie_output])
            
            q6_btn.click(fn=run_shortcut_query,
                         inputs=[user_token_state, gr.State("Revenue by country?")],
                         outputs=[genie_output])
            
            q7_btn.click(fn=run_shortcut_query,
                         inputs=[user_token_state, gr.State("What is the average order value?")],
                         outputs=[genie_output])
            
            q8_btn.click(fn=run_shortcut_query,
                         inputs=[user_token_state, gr.State("Top 10 customers by revenue?")],
                         outputs=[genie_output])
            
            q9_btn.click(fn=run_shortcut_query,
                         inputs=[user_token_state, gr.State("How many customers are at high churn risk?")],
                         outputs=[genie_output])
            
            q10_btn.click(fn=run_shortcut_query,
                          inputs=[user_token_state, gr.State("Which customers are at high churn risk?")],
                          outputs=[genie_output])
            
            q11_btn.click(fn=run_shortcut_query,
                          inputs=[user_token_state, gr.State("What interventions are recommended for critical customers?")],
                          outputs=[genie_output])
    
    gr.Markdown("---")
    #gr.Markdown("*Data Source: workspace.db_analytics | Powered by Databricks + Gradio*")

demo.launch()