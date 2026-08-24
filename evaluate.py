"""
Project Evaluation & Demo Script
Run this to demonstrate all 4 weeks of work with actual outputs.
"""
import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("LOGISTICS DATA SCIENCE PROJECT - EVALUATION DEMO")
print("YuvaIntern | Skill India Development | Intern ID: 383600")
print("="*70)

# ============================================================
# WEEK 1: DATA COLLECTION & SIMULATION
# ============================================================
print("\n" + "="*70)
print("WEEK 1: STRATEGIC PLANNING & DATA COLLECTION")
print("="*70)

from src.data_collection.generate_dataset import generate_logistics_dataset

print("\n[1/4] Generating logistics dataset...")
df = generate_logistics_dataset(n_records=5000)  # Smaller for quick demo
print(f"✓ Dataset generated: {df.shape[0]} records, {df.shape[1]} features")
print(f"✓ Cities covered: {df['warehouse_city'].nunique()}")
print(f"✓ Date range: {df['order_timestamp'].min()} to {df['order_timestamp'].max()}")

# Show sample
print("\nSample Data:")
print(df[['order_id', 'warehouse_city', 'distance_km', 'delivery_time_min', 
          'on_time_flag', 'transport_cost_inr']].head().to_string(index=False))

# KPIs
print("\n--- Key Performance Indicators ---")
print(f"On-Time Delivery Rate: {df['on_time_flag'].mean()*100:.2f}%")
print(f"Average Cost Per Delivery: ₹{df['transport_cost_inr'].mean():.2f}")
print(f"Average Delivery Time: {df['delivery_time_min'].mean():.2f} min")

# Save for next weeks
df.to_csv('data/raw/demo_dataset.csv', index=False)

# ============================================================
# WEEK 2: DATA PREPROCESSING
# ============================================================
print("\n" + "="*70)
print("WEEK 2: DATA CLEANING & PREPROCESSING")
print("="*70)

from src.data_preprocessing.preprocess import LogisticsDataPreprocessor

print("\n[2/4] Running preprocessing pipeline...")
preprocessor = LogisticsDataPreprocessor()
preprocessor.load_data('data/raw/demo_dataset.csv')
preprocessor.clean_missing_values()
preprocessor.detect_outliers()
preprocessor.remove_duplicates()
preprocessor.encode_categorical()
preprocessor.scale_features()
preprocessor.engineer_features()
preprocessor.split_data()
preprocessor.save_processed_data('data/processed/')

print(f"✓ Missing values handled")
print(f"✓ Outliers detected and treated")
print(f"✓ Duplicates removed")
print(f"✓ Categorical variables encoded")
print(f"✓ Features scaled and engineered")
print(f"✓ Data split: Train={len(preprocessor.train)}, Val={len(preprocessor.val)}, Test={len(preprocessor.test)}")

# ============================================================
# WEEK 3: EDA & VISUALIZATION
# ============================================================
print("\n" + "="*70)
print("WEEK 3: EXPLORATORY DATA ANALYSIS & VISUALIZATION")
print("="*70)

from src.eda_visualization.eda import LogisticsVisualizer

print("\n[3/4] Generating visualizations...")
viz = LogisticsVisualizer('data/processed/train.csv')
viz.output_dir = 'visualizations/week3/'
os.makedirs(viz.output_dir, exist_ok=True)

viz.plot_delivery_time_distribution()
viz.plot_correlation_heatmap()
viz.plot_traffic_boxplot()
viz.plot_distance_scatter()
viz.plot_monthly_trends()
viz.plot_cost_violin()

print("✓ 6 visualizations generated and saved to visualizations/week3/")

# Key insights
print("\n--- Key EDA Insights ---")
train_df = pd.read_csv('data/processed/train.csv')
print(f"Distance-Delivery Time Correlation: {train_df[['distance_km', 'delivery_time_min']].corr().iloc[0,1]:.3f}")
print(f"Peak Hour Impact: {train_df[train_df['is_peak_hour']==1]['delivery_time_min'].mean():.1f} min vs {train_df[train_df['is_peak_hour']==0]['delivery_time_min'].mean():.1f} min (off-peak)")

# ============================================================
# WEEK 4: PREDICTIVE MODELING & OPTIMIZATION
# ============================================================
print("\n" + "="*70)
print("WEEK 4: PREDICTIVE MODELING & OPTIMIZATION")
print("="*70)

from src.predictive_modeling.train_models import DeliveryTimePredictor

print("\n[4/4] Training predictive models...")
predictor = DeliveryTimePredictor()
predictor.load_data(
    'data/processed/train.csv',
    'data/processed/validation.csv',
    'data/processed/test.csv'
)

# Train all models
predictor.train_linear_regression()
predictor.train_random_forest()
predictor.train_xgboost()
predictor.train_gradient_boosting()

# Compare
print("\n--- Model Performance Comparison ---")
comparison = predictor.compare_models()

# Save best model
predictor.save_best_model('models/')

# Route Optimization Demo
print("\n--- Route Optimization Demo ---")
from src.optimization.route_optimizer import RouteOptimizer

np.random.seed(42)
warehouse = pd.DataFrame({'lat': [19.0760], 'lon': [72.8777]})
deliveries = pd.DataFrame({
    'lat': 19.0760 + np.random.normal(0, 0.08, 15),
    'lon': 72.8777 + np.random.normal(0, 0.08, 15)
})
locations = pd.concat([warehouse, deliveries], ignore_index=True)
demands = [0] + list(np.random.uniform(1, 12, 15))

optimizer = RouteOptimizer()
routes, distance, vehicles = optimizer.genetic_algorithm_vrp(
    locations, demands, vehicle_capacity=40,
    population_size=50, generations=100
)

print(f"✓ Optimized routes: {vehicles} vehicles, {distance:.2f} km total")
print(f"✓ Average distance per vehicle: {distance/vehicles:.2f} km")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*70)
print("PROJECT EVALUATION SUMMARY")
print("="*70)

print("""
✅ WEEK 1: Data Collection
   - Generated 5,000 realistic logistics records
   - 20 cities, 4 vehicle types, realistic distributions

✅ WEEK 2: Data Preprocessing  
   - Missing value imputation (KNN, mode, rule-based)
   - Outlier detection (IQR + domain rules)
   - Feature engineering (temporal, spatial, operational)
   - Time-based train/val/test split

✅ WEEK 3: EDA & Visualization
   - 6 professional visualizations generated
   - Correlation analysis, distribution studies
   - Traffic and seasonal pattern identification

✅ WEEK 4: Predictive Modeling & Optimization
   - 4 ML models trained and compared
   - Best Model: {best_model} (Test R² = {best_r2:.4f})
   - Genetic Algorithm VRP: {vehicles} vehicles, {dist:.2f} km

📁 Output Files:
   - data/raw/demo_dataset.csv
   - data/processed/train.csv, validation.csv, test.csv
   - visualizations/week3/*.png (6 charts)
   - models/best_model.pkl

🔗 GitHub: https://github.com/rvdhanush02-lab/logistics-data-science-project
""".format(
    best_model=predictor.best_model_name,
    best_r2=predictor.results[predictor.best_model_name]['test_r2'],
    vehicles=vehicles,
    dist=distance
))

print("="*70)
print("EVALUATION COMPLETE!")
print("="*70)
