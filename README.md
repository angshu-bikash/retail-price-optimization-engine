# Retail Price Optimization Engine 🏷️

### 📌 Project Overview
A machine learning-based pricing strategy tool designed to optimize revenue for a retail portfolio. This project moves away from "Competitor-Matching" strategies to a "Data-Driven" approach using Price Elasticity and Demand Forecasting.

**Impact:** Identified a **more than $40k** monthly revenue opportunity by optimizing prices for 52 products.

### 🛠️ Tech Stack
* **Python:** Pandas, Scikit-Learn, log-log regression (elasticity), XGBoost (for Demand Forecasting)
* **Power BI:** DAX, Data Modeling, Scenario Simulation
* **Statistics:** Price Elasticity of Demand (PED), Clustering

### 📊 Key Features
1.  **Demand Prediction Model:** An XGBoost Regressor trained on 2 years of transaction data to predict sales volume at different price points.
2.  **Scenario Simulator:** A custom engine that generates 17 price scenarios (-20% to +20%) for every product to find the revenue-maximizing price.
3.  **Interactive Dashboard:** A Power BI tool allowing stakeholders to visualize the trade-off between "Volume" and "Margin."

### 📂 File Structure
* `scripts/`: Contains the Python logic for data enrichment and ML modeling.
* `dashboard/`: The `.pbix` file with the Executive Summary and Simulator.
* `data/`: Anonymized sample data used for the analysis.

### 🚀 How to Run
1.  Install dependencies: `pip install pandas xgboost scikit-learn`
2.  Run the engine: `python scripts/03_scenario_engine.py`
3.  Open `dashboard/Pricing_Simulator.pbix` in Power BI Desktop.