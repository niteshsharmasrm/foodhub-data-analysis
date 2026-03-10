# FoodHub Data Analysis - User Guide

## 🚀 Quick Start

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/nitesh1396975/foodhub-data-analysis.git
   cd foodhub-data-analysis
   ```

2. **Set Up Environment**
   ```bash
   python -m venv foodhub_env
   source foodhub_env/bin/activate  # On Windows: foodhub_env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   ```bash
   python test_analysis.py
   ```

## 📊 Running Analysis

### Option 1: Complete Pipeline
```bash
python main.py --data_path data/foodhub_order.csv --output_dir results
```

### Option 2: Interactive Analysis
```python
from src.foodhub_analysis import FoodHubAnalyzer

# Initialize and run analysis
analyzer = FoodHubAnalyzer('data/foodhub_order.csv')
analyzer.load_data()
results = analyzer.run_complete_analysis()

# View insights
insights = analyzer.generate_insights()
print(f"Total Revenue: ${insights['revenue_metrics']['total_revenue']:,.2f}")
```

### Option 3: Custom Analysis
```python
from src.data_preprocessing import DataPreprocessor
from src.visualization import FoodHubVisualizer
from src.ml_models import MLPipeline

# Load and preprocess data
preprocessor = DataPreprocessor()
data = pd.read_csv('data/foodhub_order.csv')
processed_data = preprocessor.preprocess_pipeline(data)

# Create visualizations
visualizer = FoodHubVisualizer()
visualizer.save_all_visualizations(processed_data)

# Run ML analysis
ml_pipeline = MLPipeline()
ml_results = ml_pipeline.run_complete_analysis(processed_data)
```

## 📈 Understanding Results

### Key Metrics
- **Total Orders**: Number of orders analyzed
- **Revenue Metrics**: Total and average order values
- **Customer Insights**: Segmentation and behavior patterns
- **Operational Metrics**: Delivery and preparation times

### Visualizations
The analysis generates several types of visualizations:
- Restaurant performance dashboards
- Cuisine popularity charts
- Delivery performance metrics
- Customer behavior analysis
- Executive summary dashboard

### Machine Learning Results
- **Customer Segmentation**: Groups customers based on behavior
- **Demand Forecasting**: Predicts future order volumes
- **Delivery Optimization**: Identifies efficiency improvements
- **Revenue Analysis**: Pricing and profitability insights

## 🛠️ Customization

### Configuration
Edit `config.py` to customize:
- Analysis parameters
- Visualization settings
- Model configurations
- Business rules

### Adding New Features
1. Create new analysis methods in appropriate modules
2. Add tests in `test_analysis.py`
3. Update documentation
4. Submit pull request

## 📋 Troubleshooting

### Common Issues

**Data Loading Errors**
- Ensure CSV file exists and is readable
- Check column names match expected format
- Verify data types are correct

**Memory Issues**
- Use data sampling for large datasets
- Increase system memory allocation
- Process data in chunks

**Visualization Errors**
- Install required visualization libraries
- Check output directory permissions
- Verify data contains required columns

### Getting Help
- Check the [API Reference](docs/api_reference.md)
- Review [Contributing Guidelines](CONTRIBUTING.md)
- Create an issue on GitHub
- Contact: nitesh1396975@gmail.com

## 🎯 Best Practices

### Data Preparation
- Clean data before analysis
- Handle missing values appropriately
- Validate data quality

### Analysis Workflow
1. Start with exploratory data analysis
2. Apply preprocessing as needed
3. Run comprehensive analysis
4. Generate visualizations
5. Apply machine learning models
6. Interpret and document results

### Performance Optimization
- Use appropriate data types
- Leverage vectorized operations
- Cache intermediate results
- Monitor memory usage

## 📊 Sample Outputs

### Console Output
```
🚀 FoodHub Complete Analysis Pipeline
============================================================
📊 Data Source: data/foodhub_order.csv
📁 Output Directory: results
⏰ Analysis Started: 2024-03-10 12:00:00
============================================================

📥 Step 1: Loading and Validating Data
----------------------------------------
✅ Data loaded: 1,898 rows, 9 columns

🔧 Step 2: Data Preprocessing
----------------------------------------
✅ Data preprocessing completed: 8 steps

📊 Step 3: Exploratory Data Analysis
----------------------------------------
🍳 Food Preparation Time Analysis:
   Minimum: 20 minutes
   Average: 27.37 minutes
   Maximum: 35 minutes

🏪 Top 5 Restaurants by Order Count:
   1. Shake Shack: 219 orders
   2. The Meatball Shop: 132 orders
   3. Blue Ribbon Sushi: 119 orders
   4. Parm: 115 orders
   5. Joe's Pizza: 108 orders

✅ Analysis Complete!
```

### Generated Files
- `results/reports/analysis_report.html` - Executive summary
- `results/images/` - All visualization files
- `results/models/` - Trained ML models
- `results/processed_data/` - Cleaned datasets

## 🔄 Automation

### Scheduled Analysis
Set up automated analysis using cron jobs:
```bash
# Run daily analysis at 2 AM
0 2 * * * /path/to/python /path/to/main.py --data_path /path/to/data.csv
```

### CI/CD Integration
Include in your CI/CD pipeline:
```yaml
- name: Run FoodHub Analysis
  run: |
    python main.py --data_path data/foodhub_order.csv
    python test_analysis.py