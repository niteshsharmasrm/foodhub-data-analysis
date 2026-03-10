# Configuration file for FoodHub Analysis
# =====================================

# Data Configuration
DATA_CONFIG = {
    'default_data_path': 'data/foodhub_order.csv',
    'output_directory': 'results',
    'processed_data_dir': 'data/processed',
    'models_dir': 'models',
    'visualizations_dir': 'assets/images'
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    'customer_segments': 4,
    'forecast_periods': 5,
    'test_size': 0.2,
    'random_state': 42,
    'cross_validation_folds': 5
}

# Visualization Configuration
VIZ_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'],
    'save_formats': ['png', 'html']
}

# Model Configuration
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'random_state': 42,
        'max_depth': 10
    },
    'kmeans': {
        'n_clusters': 4,
        'random_state': 42,
        'max_iter': 300
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'random_state': 42
    }
}

# Business Rules
BUSINESS_RULES = {
    'high_value_customer_threshold': 100,  # Total spent
    'frequent_customer_threshold': 5,      # Number of orders
    'excellent_rating_threshold': 4,       # Rating >= 4
    'fast_delivery_threshold': 25,         # Minutes
    'target_profit_margin': 0.3           # 30%
}