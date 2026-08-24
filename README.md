# Logistics Data Science Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-green)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

A comprehensive data science project analyzing logistics operations for a national e-commerce distribution network across India. This project covers the complete data science lifecycle — from strategic planning and data preprocessing to exploratory analysis, predictive modeling, and operational optimization.

**Internship:** YuvaIntern Data Science Internship (Skill India Development)  
**Intern ID:** 383600  
**Duration:** 4 Weeks (July 30 – August 27, 2026)

## Project Structure

```
logistics-data-science-project/
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Cleaned & preprocessed data
├── src/
│   ├── data_collection/        # Week 1: Data simulation & collection
│   ├── data_preprocessing/     # Week 2: Cleaning & preprocessing
│   ├── eda_visualization/      # Week 3: Analysis & visualizations
│   ├── predictive_modeling/    # Week 4: ML models & predictions
│   └── optimization/           # Week 4: Route & cost optimization
├── notebooks/                  # Jupyter notebooks for each week
├── reports/                    # DOC reports for each week
├── visualizations/             # Generated charts & plots
├── models/                     # Trained model artifacts
├── tests/                      # Unit tests
├── docs/                       # Documentation
└── README.md                   # This file
```

## Dataset

- **Size:** 100,000 delivery records
- **Period:** January – December 2025
- **Coverage:** 20 major Indian cities
- **Features:** 24 variables including delivery times, costs, traffic, weather, vehicle types, and geographic data

## Weekly Progress

### Week 1: Strategic Planning & Data Exploration
- Defined logistics scenario (500+ vehicles, 12 warehouses, 50K daily orders)
- Identified 3 KPIs: On-Time Delivery Rate, Cost Per Delivery, Inventory Turnover
- Established data science methodology roadmap
- **Report:** [Week 1 - Strategic Planning](reports/week1_strategic_planning/)

### Week 2: Data Collection, Cleaning & Preprocessing
- Simulated realistic logistics dataset with 8.5% missing values
- Implemented 5-stage cleaning pipeline (missing values, outliers, duplicates, format standardization, logical validation)
- Feature engineering: temporal, spatial, operational, and interaction features
- **Report:** [Week 2 - Data Preprocessing](reports/week2_data_preprocessing/)

### Week 3: Advanced Data Analysis & Visualization
- Comprehensive EDA with descriptive statistics and correlation analysis
- Created 8 professional visualizations using matplotlib and seaborn
- Identified seasonal bottlenecks, cost drivers, and performance gaps
- **Report:** [Week 3 - EDA & Visualization](reports/week3_eda_visualization/)

### Week 4: Predictive Modeling & Optimization
- Built 4 predictive models: Linear Regression, Random Forest, XGBoost, and Neural Network
- Forecasted delivery times with RMSE of 8.2 minutes (R² = 0.91)
- Implemented route optimization using Genetic Algorithm VRP solver
- Proposed 5 optimization strategies with projected savings of ₹68-75 lakhs annually
- **Report:** [Week 4 - Predictive Modeling](reports/week4_predictive_modeling/)

## Key Results

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| On-Time Delivery Rate | 91.7% | 95.5% | +3.8 pp |
| Avg Delivery Time | 52.3 min | 44.1 min | -15.7% |
| Cost Per Delivery | ₹78.4 | ₹64.2 | -18.1% |
| Fuel Efficiency | 3.2 km/L | 3.8 km/L | +18.8% |
| Annual Savings | — | ₹68-75L | — |

## Technologies Used

- **Python 3.10+**
- **Pandas** – Data manipulation
- **NumPy** – Numerical computing
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn** – Machine learning
- **XGBoost** – Gradient boosting
- **TensorFlow/Keras** – Neural networks
- **SciPy** – Statistical analysis
- **NetworkX** – Graph optimization

## Installation

```bash
# Clone the repository
git clone https://github.com/rvdhanush02-lab/logistics-data-science-project.git
cd logistics-data-science-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Evaluation

For evaluators and reviewers, run the evaluation script to see all results:

```bash
python evaluate.py
```

This executes all 4 weeks of work and displays:
- Dataset generation and KPIs
- Preprocessing pipeline results
- Generated visualizations
- Model performance comparison
- Route optimization output

## Usage

```bash
# Run complete pipeline
python src/main.py

# Run individual modules
python src/data_preprocessing/clean_data.py
python src/predictive_modeling/train_models.py
python src/optimization/route_optimizer.py

# Launch Jupyter notebooks
jupyter notebook notebooks/
```

## Visualizations

All generated visualizations are saved in the `visualizations/` directory:
- Delivery time distributions
- Correlation heatmaps
- Traffic condition box plots
- Monthly performance trends
- Cost structure violin plots
- Model performance comparisons
- Route optimization maps

## Models

Trained models are saved in the `models/` directory:
- `linear_regression.pkl` – Baseline linear model
- `random_forest.pkl` – Ensemble tree model (best performer)
- `xgboost_model.pkl` – Gradient boosting model
- `neural_network.h5` – Deep learning model
- `scalers.pkl` – Feature scaling artifacts

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YuvaIntern & Skill India Development for the internship opportunity
- Kaggle for publicly available logistics benchmark datasets
- Open Government Data (India) for traffic and infrastructure data

## Contact

For questions or collaboration, please reach out via the internship portal (ID: 383600).

---

**Note:** This is an educational project completed as part of the YuvaIntern Data Science Internship program. All data is synthetic and generated for analytical purposes.
