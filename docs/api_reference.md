# FoodHub Data Analysis - API Reference

## Core Classes and Functions

### FoodHubAnalyzer

Main analysis engine for FoodHub data processing and insights generation.

#### Constructor
```python
FoodHubAnalyzer(data_path: str = None)
```

**Parameters:**
- `data_path` (str, optional): Path to the CSV data file

#### Methods

##### `load_data(data_path: str = None) -> pd.DataFrame`
Load the FoodHub dataset from CSV file.

**Parameters:**
- `data_path` (str, optional): Path to the CSV file

**Returns:**
- `pd.DataFrame`: Loaded dataset

**Example:**
```python
analyzer = FoodHubAnalyzer()
data = analyzer.load_data('data/foodhub_orders.csv')
```

##### `get_data_overview() -> Dict`
Get comprehensive overview of the dataset including shape, columns, data types, and missing values.

**Returns:**
- `Dict`: Dataset overview statistics

##### `analyze_restaurants() -> Dict`
Analyze restaurant performance and popularity metrics.

**Returns:**
- `Dict`: Restaurant analysis results including top performers and revenue metrics

##### `analyze_cuisines() -> Dict`
Analyze cuisine popularity and performance trends.

**Returns:**
- `Dict`: Cuisine analysis results

##### `run_complete_analysis(save_visualizations: bool = True) -> Dict`
Execute the complete analysis pipeline.

**Parameters:**
- `save_visualizations` (bool): Whether to save visualization plots

**Returns:**
- `Dict`: Complete analysis results

---

### DataPreprocessor

Comprehensive data preprocessing pipeline for data cleaning and feature engineering.

#### Constructor
```python
DataPreprocessor()
```

#### Methods

##### `validate_data(df: pd.DataFrame) -> Dict`
Validate the input dataset for common data quality issues.

**Parameters:**
- `df` (pd.DataFrame): Input dataset

**Returns:**
- `Dict`: Validation results including issues found

##### `clean_data(df: pd.DataFrame) -> pd.DataFrame`
Clean the dataset by handling missing values and duplicates.

**Parameters:**
- `df` (pd.DataFrame): Input dataset

**Returns:**
- `pd.DataFrame`: Cleaned dataset

##### `engineer_features(df: pd.DataFrame) -> pd.DataFrame`
Create new features from existing data.

**Parameters:**
- `df` (pd.DataFrame): Input dataset

**Returns:**
- `pd.DataFrame`: Dataset with engineered features

##### `preprocess_pipeline(df: pd.DataFrame, **kwargs) -> pd.DataFrame`
Run the complete preprocessing pipeline.

**Parameters:**
- `df` (pd.DataFrame): Input dataset
- `clean_data` (bool): Whether to clean the data
- `engineer_features` (bool): Whether to engineer new features
- `encode_categorical` (bool): Whether to encode categorical variables
- `scale_numerical` (bool): Whether to scale numerical features

**Returns:**
- `pd.DataFrame`: Preprocessed dataset

---

### FoodHubVisualizer

Advanced visualization engine for creating interactive dashboards and plots.

#### Constructor
```python
FoodHubVisualizer(style: str = 'seaborn-v0_8')
```

**Parameters:**
- `style` (str): Matplotlib style to use

#### Methods

##### `create_restaurant_analysis(df: pd.DataFrame, save_path: str = None) -> go.Figure`
Create comprehensive restaurant performance analysis dashboard.

**Parameters:**
- `df` (pd.DataFrame): Dataset
- `save_path` (str, optional): Path to save the plot

**Returns:**
- `go.Figure`: Plotly figure object

##### `create_comprehensive_dashboard(df: pd.DataFrame, save_path: str = None) -> go.Figure`
Create executive summary dashboard with key metrics.

**Parameters:**
- `df` (pd.DataFrame): Dataset
- `save_path` (str, optional): Path to save the plot

**Returns:**
- `go.Figure`: Plotly figure object

##### `save_all_visualizations(df: pd.DataFrame, save_path: str = "assets/images/") -> None`
Generate and save all visualizations.

**Parameters:**
- `df` (pd.DataFrame): Dataset
- `save_path` (str): Directory to save visualizations

---

### Machine Learning Models

#### CustomerSegmentation

Advanced customer segmentation using K-Means clustering.

##### Constructor
```python
CustomerSegmentation(n_clusters: int = 4)
```

##### Methods

##### `fit_predict(df: pd.DataFrame) -> np.ndarray`
Fit the segmentation model and predict customer segments.

**Parameters:**
- `df` (pd.DataFrame): Input dataset

**Returns:**
- `np.ndarray`: Customer segments

#### DemandForecaster

Demand forecasting model for restaurant orders.

##### Constructor
```python
DemandForecaster(model_type: str = 'random_forest')
```

**Parameters:**
- `model_type` (str): Type of model ('random_forest', 'gradient_boosting', 'linear')

##### Methods

##### `train(df: pd.DataFrame) -> Dict[str, float]`
Train the demand forecasting model.

**Parameters:**
- `df` (pd.DataFrame): Training dataset

**Returns:**
- `Dict[str, float]`: Training metrics

##### `predict_demand(df: pd.DataFrame, periods: int = 5) -> np.ndarray`
Predict future demand.

**Parameters:**
- `df` (pd.DataFrame): Historical data
- `periods` (int): Number of periods to forecast

**Returns:**
- `np.ndarray`: Demand predictions

---

## Usage Examples

### Basic Analysis
```python
from src.foodhub_analysis import FoodHubAnalyzer

# Initialize analyzer
analyzer = FoodHubAnalyzer()

# Load data
data = analyzer.load_data('data/foodhub_orders.csv')

# Run complete analysis
results = analyzer.run_complete_analysis()

# Get insights
insights = analyzer.generate_insights()
print(f"Total Revenue: ${insights['revenue_metrics']['total_revenue']:,.2f}")
```

### Data Preprocessing
```python
from src.data_preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Run preprocessing pipeline
processed_data = preprocessor.preprocess_pipeline(
    data, 
    clean_data=True, 
    engineer_features=True
)

# Get preprocessing summary
summary = preprocessor.get_preprocessing_summary()
```

### Machine Learning
```python
from src.ml_models import MLPipeline

# Initialize ML pipeline
ml_pipeline = MLPipeline()

# Run complete ML analysis
ml_results = ml_pipeline.run_complete_analysis(data)

# Save trained models
ml_pipeline.save_models('models/')
```

### Visualization
```python
from src.visualization import FoodHubVisualizer

# Initialize visualizer
visualizer = FoodHubVisualizer()

# Create executive dashboard
dashboard = visualizer.create_comprehensive_dashboard(data)
dashboard.show()

# Save all visualizations
visualizer.save_all_visualizations(data, 'assets/images/')
```

---

## Configuration

The project uses a configuration file (`config.py`) for customizing analysis parameters:

```python
from config import ANALYSIS_CONFIG, MODEL_CONFIG

# Access configuration
n_clusters = ANALYSIS_CONFIG['customer_segments']
random_state = ANALYSIS_CONFIG['random_state']
```

---

## Error Handling

All methods include proper error handling and will raise appropriate exceptions:

- `ValueError`: For invalid input parameters
- `FileNotFoundError`: For missing data files
- `KeyError`: For missing required columns

Always wrap method calls in try-except blocks for production use:

```python
try:
    results = analyzer.run_complete_analysis()
except FileNotFoundError:
    print("Data file not found")
except ValueError as e:
    print(f"Invalid input: {e}")