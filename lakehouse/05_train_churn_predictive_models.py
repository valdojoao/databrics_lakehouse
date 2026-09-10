# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Setup: Install Required Libraries
# MAGIC %pip install mlflow xgboost lifelines scikit-learn imbalanced-learn shap --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Imports and Configuration
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
from xgboost import XGBClassifier
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings('ignore')

# Unity Catalog configuration
CATALOG = "workspace"
SCHEMA = "retail"
VOLUME = "databrics_customer_analytics"

# MLflow experiment
mlflow.set_experiment("/Users/valdojoao90@gmail.com/retail_analytics/churn_predictive_models")

print("✅ Libraries loaded")
print(f"📊 MLflow experiment: /Users/valdojoao90@gmail.com/retail_analytics/churn_predictive_models")
print(f"📁 Data source: {CATALOG}.{SCHEMA}")



# DBTITLE 1,Feature Engineering: Calculate Derived Features
# Load data from SQL temp view
df = spark.sql("SELECT * FROM customer_churn_labels").toPandas()

print(f"📊 Dataset shape: {df.shape}")
print(f"📊 Churn rate: {df['is_churned'].mean():.2%}")
print(f"\n📋 Feature columns: {len(df.columns)}")

# Add derived features
df['revenue_per_transaction'] = df['total_revenue'] / (df['number_of_transactions'] + 1)
df['support_interaction_rate'] = df['service_interactions'] / (df['number_of_transactions'] + 1)
df['unresolved_ratio'] = df['unresolved_interactions'] / (df['service_interactions'] + 1)
df['purchase_velocity_30d'] = df['purchases_last_30_days'] / 30
df['purchase_velocity_60d'] = df['purchases_last_60_days'] / 60
df['purchase_acceleration'] = df['purchase_velocity_30d'] / (df['purchase_velocity_60d'] + 0.01)
df['recency_risk'] = np.where(df['days_since_last_purchase'] > 60, 1, 0)
df['low_engagement_risk'] = np.where(df['marketing_engagement_rate'] < 0.3, 1, 0)
df['negative_trend_risk'] = np.where(
    (df['revenue_change_pct'] < -10) | (df['purchase_frequency_change_pct'] < -10), 1, 0
)

print(f"✅ Feature engineering complete: {df.shape[1]} total features")

# COMMAND ----------

# DBTITLE 1,Prepare Training Data
# Define feature columns
categorical_features = ['customer_segment', 'loyalty_tier', 'country', 'region', 'age_band', 'preferred_channel']
numeric_features = [
    'total_revenue', 'number_of_transactions', 'average_order_value',
    'days_since_last_purchase', 'purchases_last_30_days', 'purchases_last_60_days', 'purchases_last_90_days',
    'marketing_engagement_rate', 'average_satisfaction_score',
    'service_interactions', 'unresolved_interactions',
    'revenue_change_pct', 'purchase_frequency_change_pct', 'avg_basket_change_pct',
    'revenue_per_transaction', 'support_interaction_rate', 'unresolved_ratio',
    'purchase_velocity_30d', 'purchase_velocity_60d', 'purchase_acceleration',
    'recency_risk', 'low_engagement_risk', 'negative_trend_risk'
]

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder

le_dict = {}
for col in categorical_features:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('unknown'))
    le_dict[col] = le

encoded_categorical = [f'{col}_encoded' for col in categorical_features]
all_features = numeric_features + encoded_categorical

# Prepare X and y
X = df[all_features].fillna(0)
y = df['is_churned']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"✅ Test set: {X_test.shape[0]} samples")
print(f"   Train churn rate: {y_train.mean():.2%}")
print(f"   Test churn rate: {y_test.mean():.2%}")

# COMMAND ----------

# DBTITLE 1,Train Churn Risk Model (XGBoost)
# Start MLflow run
with mlflow.start_run(run_name="churn_risk_xgboost") as run:
    
    # Train XGBoost model
    print("🚀 Training XGBoost churn risk model...")
    
    model_churn = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # Handle imbalance
        random_state=42,
        eval_metric='auc'
    )
    
    model_churn.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Predictions
    y_pred_proba = model_churn.predict_proba(X_test)[:, 1]
    y_pred = model_churn.predict(X_test)
    
    # Metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n✅ Churn Risk Model Trained")
    print(f"   AUC: {auc_score:.4f}")
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Active', 'Churned']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': model_churn.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔝 Top 10 Features:")
    print(feature_importance.head(10))
    
    # Log to MLflow
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("n_features", len(all_features))
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_metric("auc", auc_score)
    mlflow.log_metric("churn_rate", y.mean())
    
    # Log model with signature
    from mlflow.models import infer_signature
    signature = infer_signature(X_train, model_churn.predict_proba(X_train))
    
    mlflow.xgboost.log_model(
        model_churn, 
        "churn_risk_model",
        signature=signature,
        registered_model_name="churn_risk_model"
    )
    
    churn_model_uri = f"runs:/{run.info.run_id}/churn_risk_model"
    print(f"\n✅ Model logged to MLflow: {churn_model_uri}")

# COMMAND ----------

# DBTITLE 1,Train Time-to-Churn Model (XGBoost Regression)
# Train time-to-churn model using XGBoost regression (more robust than Cox PH)
from xgboost import XGBRegressor
import numpy as np

print("🚀 Training XGBoost time-to-churn regression model...")

# Prepare data - predict days_to_event for active customers
time_to_churn_data = df[df['event_observed'] == 0].copy()  # Only active customers
X_time = time_to_churn_data[all_features].fillna(0)
y_time = time_to_churn_data['days_to_event'].fillna(df['days_to_event'].median())

print(f"   Training on {len(X_time)} active customers")

with mlflow.start_run(run_name="time_to_churn_xgboost") as run:
    
    # Train XGBoost regression model
    model_time = XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model_time.fit(X_time, y_time)
    
    # Make predictions and calculate metrics
    y_pred_time = model_time.predict(X_time)
    
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_time, y_pred_time)
    r2 = r2_score(y_time, y_pred_time)
    
    print(f"\n✅ Time-to-Churn Model Trained")
    print(f"   MAE: {mae:.2f} days")
    print(f"   R²: {r2:.4f}")
    
    # Feature importance
    time_feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': model_time.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔝 Top 10 Features for Time Prediction:")
    print(time_feature_importance.head(10))
    
    # Log to MLflow
    mlflow.log_param("model_type", "XGBoost_Regression")
    mlflow.log_param("n_features", len(all_features))
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    
    # Log model with signature
    from mlflow.models import infer_signature
    signature = infer_signature(X_time, model_time.predict(X_time))
    
    mlflow.xgboost.log_model(
        model_time,
        "time_to_churn_model",
        signature=signature,
        registered_model_name="time_to_churn_model"
    )
    
    time_model_uri = f"runs:/{run.info.run_id}/time_to_churn_model"
    print(f"\n✅ Time-to-churn model logged to MLflow: {time_model_uri}")

# COMMAND ----------

# DBTITLE 1,Simple Uplift Model (Treatment Effect Estimation)
# Simplified uplift: estimate who is more likely to respond to intervention
# Using two models: one for "treated" (simulated), one for "control"
# In production, this needs A/B test data with actual interventions

print("🚀 Training simplified uplift model...")
print("⚠️  Note: True uplift requires A/B test data with interventions")
print("    This is a proxy using engagement as pseudo-treatment indicator\n")

with mlflow.start_run(run_name="uplift_intervention_response") as run:
    
    # Proxy for treatment: customers with high engagement are "treated"
    # In reality, you'd use actual intervention data
    df['pseudo_treatment'] = (df['marketing_engagement_rate'] > df['marketing_engagement_rate'].median()).astype(int)
    
    # Split into treatment and control groups
    treated = df[df['pseudo_treatment'] == 1]
    control = df[df['pseudo_treatment'] == 0]
    
    X_treated = treated[all_features].fillna(0)
    y_treated = treated['is_churned']
    
    X_control = control[all_features].fillna(0)
    y_control = control['is_churned']
    
    # Train separate models
    model_treated = XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
    model_control = XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
    
    model_treated.fit(X_treated, y_treated)
    model_control.fit(X_control, y_control)
    
    # Uplift = P(churn | control) - P(churn | treated)
    # Higher uplift = customer who benefits more from intervention
    
    X_test_full = df[all_features].fillna(0)
    uplift_score = model_control.predict_proba(X_test_full)[:, 1] - model_treated.predict_proba(X_test_full)[:, 1]
    
    df['uplift_score'] = uplift_score
    
    print(f"✅ Uplift Model Trained")
    print(f"   Avg uplift score: {uplift_score.mean():.4f}")
    print(f"   High uplift (>0.2): {(uplift_score > 0.2).sum()} customers")
    print(f"   Low uplift (<0): {(uplift_score < 0).sum()} customers")
    
    # Log models
    mlflow.xgboost.log_model(model_treated, "uplift_treated_model")
    mlflow.xgboost.log_model(model_control, "uplift_control_model")
    mlflow.log_param("uplift_method", "two_model_approach")
    mlflow.log_metric("avg_uplift", uplift_score.mean())
    
    print(f"\n✅ Uplift models logged to MLflow")

# COMMAND ----------

# DBTITLE 1,Generate Predictions and Semaphore States
# Generate predictions for all customers
print("🎯 Generating predictions for all customers...\n")

X_all = df[all_features].fillna(0)

# Churn probability
churn_probability = model_churn.predict_proba(X_all)[:, 1]

# Time-to-churn prediction using XGBoost regression model
time_pred_data = X_all.copy()
time_predictions = model_time.predict(time_pred_data)

# Map churn probability to semaphore states
def get_semaphore_state(prob):
    if prob < 0.25:
        return '🟢 GREEN'
    elif prob < 0.50:
        return '🟡 YELLOW'
    elif prob < 0.75:
        return '🟠 ORANGE'
    else:
        return '🔴 RED'

semaphore_state = [get_semaphore_state(p) for p in churn_probability]

# Create predictions dataframe
predictions_df = pd.DataFrame({
    'customer_id': df['customer_id'],
    'churn_probability': churn_probability,
    'predicted_days_to_churn': time_predictions,
    'uplift_score': df['uplift_score'],
    'semaphore_state': semaphore_state,
    'risk_tier': pd.cut(churn_probability, bins=[0, 0.25, 0.5, 0.75, 1.0], labels=['low', 'medium', 'high', 'critical']),
    'is_high_value': (df['total_revenue'] > df['total_revenue'].quantile(0.75)).astype(int),
    'intervention_priority': (churn_probability * df['uplift_score'] * (df['total_revenue'] / df['total_revenue'].max())),
    'prediction_date': datetime.now(),
    'model_version': '1.0'
})

# Rank customers by intervention priority
predictions_df['priority_rank'] = predictions_df['intervention_priority'].rank(ascending=False, method='dense').astype(int)

print(f"✅ Predictions generated for {len(predictions_df)} customers\n")
print("📊 Semaphore Distribution:")
print(predictions_df['semaphore_state'].value_counts())
print(f"\n📊 Risk Tier Distribution:")
print(predictions_df['risk_tier'].value_counts())
print(f"\n🎯 Top 10 Priority Customers:")
print(predictions_df.nlargest(10, 'intervention_priority')[[
    'customer_id', 'churn_probability', 'uplift_score', 'semaphore_state', 'priority_rank'
]])

# COMMAND ----------

# DBTITLE 1,Create Unity Catalog Tables for Predictions
# Convert to Spark DataFrame and save to UC
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType

print("💾 Saving predictions to Unity Catalog...\n")

# Create Spark DataFrame
predictions_spark = spark.createDataFrame(predictions_df)

# Save to UC table
predictions_spark.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.risk_predictions")

print(f"✅ Table created: {CATALOG}.{SCHEMA}.risk_predictions")
print(f"   Rows: {predictions_spark.count()}")
print(f"   Columns: {len(predictions_spark.columns)}")


# DBTITLE 1,Summary and Next Steps
print("="*70)
print("🎉 CHURN PREDICTIVE SYSTEM - TRAINING COMPLETE")
print("="*70)
print("\n📦 Models Logged to MLflow:")
print("   1. churn_risk_model (XGBoost) - WHO is at risk?")
print("   2. survival_time_to_churn (Cox PH) - WHEN will they churn?")
print("   3. uplift_intervention_response - CAN we save them?")
print("\n💾 Unity Catalog Tables Created:")
print(f"   1. {CATALOG}.{SCHEMA}.risk_predictions")
print(f"   2. {CATALOG}.{SCHEMA}.customer_state_history")
print(f"   3. {CATALOG}.{SCHEMA}.intervention_recommendations")
print(f"   4. {CATALOG}.{SCHEMA}.intervention_outcomes")
print("\n🎯 Next Steps:")
print("   1. Register models in MLflow Model Registry")
print("   2. Add 'Risk Predictive' tab to app.py")
print("   3. Schedule daily batch inference to update predictions")
print("   4. Monitor intervention outcomes and retrain models")
print("\n📊 Quick Stats:")
print(f"   Total customers scored: {len(predictions_df):,}")
print(f"   High risk (🔴 RED): {(predictions_df['semaphore_state'] == '🔴 RED').sum():,}")
print(f"   Interventions recommended: {spark.table(f'{CATALOG}.{SCHEMA}.intervention_recommendations').count():,}")
print("="*70)

# COMMAND ----------

