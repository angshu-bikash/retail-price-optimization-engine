import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
import os

# CONFIGURATION
warnings.filterwarnings('ignore')
INPUT_FILE = 'data/retail_price.csv'
OUTPUT_HISTORICAL = 'data/historical_enriched.csv' # Feeds Power BI
OUTPUT_MODELLING = 'data/modelling_data.csv'       # Feeds the next Python script

# Thresholds (from your Analysis)
CAP_PRICE = 349.90  # 99th Percentile
CAP_QTY = 82        # 99th Percentile
MIN_MONTHS = 5      # Minimum history required to keep a product

def load_and_cap(filepath):
    print(f"Loading and Capping {filepath}...")
    df = pd.read_csv(filepath)
    
    # Date Setup
    if 'month_year' in df.columns:
        df['date'] = pd.to_datetime(df['month_year'], format='%d-%m-%Y')
    else:
        df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['product_id', 'date'])

    # Revenue Calculation
    df['revenue'] = df['unit_price'] * df['qty']

    # Capping Outliers (Safety Rail)
    df['unit_price'] = df['unit_price'].clip(upper=CAP_PRICE)
    df['qty'] = df['qty'].clip(upper=CAP_QTY)
    
    return df

def create_base_features(input_df):
    print("Engineering Features")
    df_eng = input_df.copy()
    
    # Time Features
    df_eng["month"] = df_eng["date"].dt.month
    df_eng["month_sin"] = np.sin(2 * np.pi * df_eng["month"] / 12)
    df_eng["month_cos"] = np.cos(2 * np.pi * df_eng["month"] / 12)
    
    # Price Ratio
    comp_cols = [c for c in df_eng.columns if "comp_" in c]
    if comp_cols:
        df_eng["avg_comp_price"] = df_eng[comp_cols].replace(0, np.nan).mean(axis=1)
        df_eng["avg_comp_price"] = df_eng["avg_comp_price"].fillna(df_eng["unit_price"])
        df_eng["price_ratio"] = df_eng["unit_price"] / df_eng["avg_comp_price"]
    else:
        df_eng["price_ratio"] = 1.0

    # Lag Price
    df_eng["lag_price"] = df_eng.groupby("product_id")["unit_price"].shift(1)
    df_eng["lag_price"] = df_eng["lag_price"].fillna(df_eng["unit_price"])

    return df_eng

def filter_min_history(df):
    print(f"Filtering products with < {MIN_MONTHS} months history...")
    counts = df.groupby('product_id')['date'].nunique()
    valid_ids = counts[counts >= MIN_MONTHS].index
    return df[df['product_id'].isin(valid_ids)].copy()


def assign_clusters(df):
    print("Calculating Elasticity & Clusters (Standard Rule)...")
    results = []
    
    for pid in df['product_id'].unique():
        d_prod = df[df['product_id'] == pid]
        
        # Log-Log Regression
        y = np.log1p(d_prod['qty'])
        X = np.log1p(d_prod[['unit_price']])
        model = LinearRegression().fit(X, y)
        elasticity = model.coef_[0]
        
        # Safety: Cap positive elasticity at 0 (treat as perfectly inelastic)
        if elasticity > 0:
            elasticity = 0
            
        results.append({'product_id': pid, 'elasticity': elasticity})
        
    df_elast = pd.DataFrame(results)
    
    # --- THE STATIC RULE ---
    # -1.0 is the Unit Elasticity point.
    # Lower than -1.0 (e.g. -1.5) = Elastic = Volume Driver
    # Higher than -1.0 (e.g. -0.8) = Inelastic = Margin Driver
    
    threshold_val = -1.0
    print(f"Splitting Clusters at Static Threshold: {threshold_val}")
    
    df_elast['cluster'] = np.where(
        df_elast['elasticity'] < threshold_val, 
        'Cluster_Volume',   # High Sensitivity (Don't raise price)
        'Cluster_Margin'    # Low Sensitivity (Can raise price)
    )
    
    # Check the counts
    print("   Final Cluster Counts:")
    print(df_elast['cluster'].value_counts())
    
    return df.merge(df_elast, on='product_id', how='left')


def add_dashboard_labels(df):
    print("5. Adding Power BI Labels...")
    # Label: Price Position
    conditions = [
        (df['price_ratio'] < 0.96),
        (df['price_ratio'] > 1.04)
    ]
    choices = ['Cheaper', 'Expensive']
    df['price_position'] = np.select(conditions, choices, default='Equal')
    return df

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # The Pipeline
    df = load_and_cap(INPUT_FILE)
    df = create_base_features(df)
    df = filter_min_history(df)
    df = assign_clusters(df)
    df = add_dashboard_labels(df)
    
    # EXPORT 1: Power BI (Human Readable)
    cols_bi = [
        'product_id', 'date', 'product_category_name', 
        'unit_price', 'qty', 'revenue', 
        'comp_1', 'comp_2', 'comp_3', 
        'price_ratio', 'price_position', 
        'elasticity', 'cluster'
    ]
    # Filter columns that exist
    final_bi_cols = [c for c in cols_bi if c in df.columns]
    df[final_bi_cols].to_csv(OUTPUT_HISTORICAL, index=False)
    print(f" SAVED: {OUTPUT_HISTORICAL}")
    
    # EXPORT 2: Modelling (Machine Readable)
    cols_model = [
        'product_id', 'date', 'cluster', 
        'unit_price', 'lag_price', 'price_ratio', 
        'month_sin', 'month_cos', 
        'qty', 'revenue',
        'product_score', 'freight_price', 'product_weight_g'
    ]
    final_model_cols = [c for c in cols_model if c in df.columns]
    df[final_model_cols].to_csv(OUTPUT_MODELLING, index=False)
    print(f" SAVED: {OUTPUT_MODELLING}")