import pandas as pd
import numpy as np
import joblib
import os
import warnings

# --- CONFIGURATION ---
warnings.filterwarnings('ignore')
INPUT_FILE = 'data/modelling_data.csv'       # The clean history
MODEL_DIR = 'models/'                        # Where the best models live
OUTPUT_FILE = 'data/future_simulations.csv'  # The Final Power BI Feed

#Grid (2.5% steps from -20% to +20%)
PRICE_MULTIPLIERS = [
    0.80, 0.825, 0.85, 0.875, 
    0.90, 0.925, 0.95, 0.975, 
    1.00, 
    1.025, 1.05, 1.075, 1.10, 
    1.125, 1.15, 1.175, 1.20
]

def generate_scenarios(df, model_path, cluster_name):
    print(f" Generating 17 scenarios for {cluster_name}...")
    
    # Load the models
    artifact = joblib.load(model_path)
    model = artifact['model']
    feature_names = artifact['features']
    
    # Get the most recent state for every product (The "Now")
    latest_data = df.sort_values('date').groupby('product_id').tail(1).copy()
    
    all_simulations = []
    
    # Loop through every product
    for _, row in latest_data.iterrows():
        base_price = row['unit_price']
        
        # Loop through the +-2.5%
        for mult in PRICE_MULTIPLIERS:
            # 1. Create the Scenario
            sim_row = row.copy()
            
            # 2. Adjust Price
            sim_price = base_price * mult
            
            # 3. Adjust Price Ratio (If we move, our position vs competitors moves)
            sim_row['unit_price'] = sim_price
            sim_row['price_ratio'] = sim_row['price_ratio'] * mult
            
            # 4. Predict Qty (Ask the AI)
            input_data = sim_row[feature_names].to_frame().T.astype(float)
            pred_qty = model.predict(input_data)[0]
            
            # Clip negative predictions (Safety)
            pred_qty = max(0, pred_qty)
            
            # 5. Calculate Revenue
            pred_rev = sim_price * pred_qty
            
            # 6. Store
            all_simulations.append({
                'product_id': row['product_id'],
                'cluster': cluster_name,
                'scenario_multiplier': mult,
                'current_price': base_price,
                'scenario_price': sim_price,
                'pred_qty': pred_qty,
                'pred_revenue': pred_rev
            })
            
    return pd.DataFrame(all_simulations)

if __name__ == "__main__":
    print(" STARTING SCENARIO ENGINE...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found.")
    else:
        df = pd.read_csv(INPUT_FILE)
        

        # Force these columns to be numbers (Floats), not Objects (Strings)
        numeric_cols = [
            'unit_price', 'lag_price', 'price_ratio', 
            'month_sin', 'month_cos', 
            'product_score', 'freight_price', 'product_weight_g'
        ]
        
        #Fixing data types
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')


        final_results = []
        
        # Check for models
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
        if not model_files:
            print("ERROR: No models found. Run Script 2 first.")
            exit()
            
        # Run Simulation for each Cluster
        for m_file in model_files:
            cluster_name = m_file.replace('model_', '').replace('.pkl', '')
            model_path = os.path.join(MODEL_DIR, m_file)
            
            # Filter data for this cluster
            cluster_df = df[df['cluster'] == cluster_name]
            
            if not cluster_df.empty:
                sim_df = generate_scenarios(cluster_df, model_path, cluster_name)
                final_results.append(sim_df)
                
        # Consolidate
        if final_results:
            full_df = pd.concat(final_results, ignore_index=True)
            
            # Identify the "Optimal" Price for each product
            # (The row with Max Revenue)
            idx_max = full_df.groupby('product_id')['pred_revenue'].idxmax()
            full_df['is_optimal'] = 0
            full_df.loc[idx_max, 'is_optimal'] = 1
            
            # Save
            full_df.to_csv(OUTPUT_FILE, index=False)
            print(f"\n SUCCESS: Simulations saved to {OUTPUT_FILE}")
            print("   Ready for Power BI Page 3.")
        else:
            print(" No simulations generated.")