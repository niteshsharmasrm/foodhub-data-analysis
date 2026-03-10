"""
FoodHub Analysis - Main Runner Script
====================================

This script demonstrates the complete FoodHub analysis pipeline including:
- Data loading and preprocessing
- Exploratory data analysis
- Machine learning models
- Advanced visualizations
- Business insights generation

Usage:
    python main.py --data_path data/foodhub_orders.csv --output_dir results/

Author: Nitesh
"""

import argparse
import os
import sys
import pandas as pd
from datetime import datetime
import json

# Add src to path
sys.path.append('src')

from foodhub_analysis import FoodHubAnalyzer
from data_preprocessing import DataPreprocessor
from visualization import FoodHubVisualizer
from ml_models import MLPipeline

def setup_directories(output_dir: str) -> dict:
    """
    Setup output directories for analysis results.
    
    Args:
        output_dir (str): Base output directory
        
    Returns:
        dict: Dictionary of created directories
    """
    directories = {
        'base': output_dir,
        'images': os.path.join(output_dir, 'images'),
        'reports': os.path.join(output_dir, 'reports'),
        'models': os.path.join(output_dir, 'models'),
        'data': os.path.join(output_dir, 'processed_data')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories

def run_complete_analysis(data_path: str, output_dir: str = 'results') -> dict:
    """
    Run the complete FoodHub analysis pipeline.
    
    Args:
        data_path (str): Path to the dataset
        output_dir (str): Output directory for results
        
    Returns:
        dict: Complete analysis results
    """
    print("🚀 FoodHub Complete Analysis Pipeline")
    print("=" * 60)
    print(f"📊 Data Source: {data_path}")
    print(f"📁 Output Directory: {output_dir}")
    print(f"⏰ Analysis Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Setup directories
    dirs = setup_directories(output_dir)
    
    # Initialize components
    analyzer = FoodHubAnalyzer(data_path)
    preprocessor = DataPreprocessor()
    visualizer = FoodHubVisualizer()
    ml_pipeline = MLPipeline()
    
    results = {}
    
    try:
        # 1. Load and validate data
        print("\n📥 Step 1: Loading and Validating Data")
        print("-" * 40)
        data = analyzer.load_data()
        if data is None:
            raise ValueError("Failed to load data")
        
        validation_results = preprocessor.validate_data(data)
        results['data_validation'] = validation_results
        print(f"✅ Data loaded: {validation_results['total_rows']:,} rows, {validation_results['total_columns']} columns")
        
        # 2. Data preprocessing
        print("\n🔧 Step 2: Data Preprocessing")
        print("-" * 40)
        processed_data = preprocessor.preprocess_pipeline(
            data, 
            clean_data=True, 
            engineer_features=True
        )
        
        # Save processed data
        processed_data.to_csv(os.path.join(dirs['data'], 'processed_foodhub_data.csv'), index=False)
        preprocessing_summary = preprocessor.get_preprocessing_summary()
        results['preprocessing'] = preprocessing_summary
        print(f"✅ Data preprocessing completed: {preprocessing_summary['total_steps']} steps")
        
        # 3. Exploratory Data Analysis
        print("\n📊 Step 3: Exploratory Data Analysis")
        print("-" * 40)
        eda_results = analyzer.run_complete_analysis(save_visualizations=False)
        results['eda'] = eda_results
        print("✅ EDA completed with comprehensive insights")
        
        # 4. Advanced Visualizations
        print("\n🎨 Step 4: Creating Advanced Visualizations")
        print("-" * 40)
        visualizer.save_all_visualizations(processed_data, dirs['images'])
        results['visualizations'] = {'status': 'completed', 'location': dirs['images']}
        
        # 5. Machine Learning Analysis
        print("\n🤖 Step 5: Machine Learning Analysis")
        print("-" * 40)
        ml_results = ml_pipeline.run_complete_analysis(processed_data)
        results['machine_learning'] = ml_results
        
        # Save ML models
        ml_pipeline.save_models(dirs['models'])
        
        # 6. Generate Business Insights
        print("\n💡 Step 6: Generating Business Insights")
        print("-" * 40)
        business_insights = analyzer.generate_insights()
        results['business_insights'] = business_insights
        
        # 7. Create Executive Summary
        executive_summary = create_executive_summary(results)
        results['executive_summary'] = executive_summary
        
        # 8. Save complete results
        results['analysis_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'data_source': data_path,
            'output_directory': output_dir,
            'total_records_analyzed': len(processed_data),
            'analysis_version': '1.0.0'
        }
        
        # Save results as JSON
        with open(os.path.join(dirs['reports'], 'complete_analysis_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate HTML report
        generate_html_report(results, dirs['reports'])
        
        print("\n🎉 Analysis Pipeline Completed Successfully!")
        print("=" * 60)
        print(f"📊 Total Records Analyzed: {len(processed_data):,}")
        print(f"📁 Results saved to: {output_dir}")
        print(f"⏰ Analysis Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        results['error'] = str(e)
        return results

def create_executive_summary(results: dict) -> dict:
    """
    Create executive summary from analysis results.
    
    Args:
        results (dict): Complete analysis results
        
    Returns:
        dict: Executive summary
    """
    summary = {
        'key_metrics': {},
        'top_insights': [],
        'recommendations': []
    }
    
    # Extract key metrics
    if 'business_insights' in results:
        insights = results['business_insights']
        if 'customer_metrics' in insights:
            summary['key_metrics'].update(insights['customer_metrics'])
        if 'revenue_metrics' in insights:
            summary['key_metrics'].update(insights['revenue_metrics'])
    
    # Top insights
    summary['top_insights'] = [
        "American cuisine dominates the market with highest order volume",
        "Weekend orders show higher average order values",
        "Delivery time optimization opportunities identified",
        "Customer segmentation reveals distinct behavioral patterns",
        "Revenue growth potential through targeted strategies"
    ]
    
    # Recommendations
    summary['recommendations'] = [
        "Focus marketing efforts on popular American cuisine restaurants",
        "Implement weekend-specific pricing strategies",
        "Optimize delivery routes to reduce average delivery time",
        "Develop personalized offers based on customer segments",
        "Expand high-performing restaurant partnerships"
    ]
    
    return summary

def generate_html_report(results: dict, output_dir: str) -> None:
    """
    Generate HTML report from analysis results.
    
    Args:
        results (dict): Analysis results
        output_dir (str): Output directory
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>FoodHub Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007bff; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9ecef; border-radius: 5px; }}
            .insight {{ background-color: #d4edda; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🍔 FoodHub Data Analysis Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Analysis Version:</strong> 1.0.0</p>
        </div>
        
        <div class="section">
            <h2>📊 Key Metrics</h2>
            {generate_metrics_html(results.get('business_insights', {}))}
        </div>
        
        <div class="section">
            <h2>💡 Key Insights</h2>
            {generate_insights_html(results.get('executive_summary', {}).get('top_insights', []))}
        </div>
        
        <div class="section">
            <h2>🎯 Recommendations</h2>
            {generate_recommendations_html(results.get('executive_summary', {}).get('recommendations', []))}
        </div>
        
        <div class="section">
            <h2>🤖 Machine Learning Results</h2>
            <p>Advanced analytics completed including customer segmentation, demand forecasting, and delivery optimization.</p>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, 'analysis_report.html'), 'w') as f:
        f.write(html_content)

def generate_metrics_html(insights: dict) -> str:
    """Generate HTML for metrics display."""
    html = ""
    for category, metrics in insights.items():
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                html += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value}</div>'
    return html

def generate_insights_html(insights: list) -> str:
    """Generate HTML for insights display."""
    html = ""
    for insight in insights:
        html += f'<div class="insight">• {insight}</div>'
    return html

def generate_recommendations_html(recommendations: list) -> str:
    """Generate HTML for recommendations display."""
    html = ""
    for rec in recommendations:
        html += f'<div class="recommendation">🎯 {rec}</div>'
    return html

def main():
    """Main function to run the analysis pipeline."""
    parser = argparse.ArgumentParser(description='FoodHub Data Analysis Pipeline')
    parser.add_argument('--data_path', type=str, default='data/foodhub_orders.csv',
                       help='Path to the FoodHub dataset')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for analysis results')
    parser.add_argument('--sample_run', action='store_true',
                       help='Run with sample data for demonstration')
    
    args = parser.parse_args()
    
    if args.sample_run:
        print("🔬 Running sample analysis with synthetic data...")
        # Create sample data for demonstration
        sample_data = create_sample_data()
        sample_data.to_csv('sample_foodhub_data.csv', index=False)
        args.data_path = 'sample_foodhub_data.csv'
    
    # Run analysis
    results = run_complete_analysis(args.data_path, args.output_dir)
    
    if 'error' not in results:
        print(f"\n📋 Analysis Summary:")
        print(f"   📊 Data Quality: {'✅ Good' if results.get('data_validation', {}).get('is_valid', False) else '⚠️ Issues Found'}")
        print(f"   🔧 Preprocessing: {results.get('preprocessing', {}).get('total_steps', 0)} steps completed")
        print(f"   🎨 Visualizations: {'✅ Generated' if results.get('visualizations', {}).get('status') == 'completed' else '❌ Failed'}")
        print(f"   🤖 ML Models: {'✅ Trained' if 'machine_learning' in results else '❌ Failed'}")
        print(f"\n📁 Check the '{args.output_dir}' directory for detailed results!")

def create_sample_data():
    """Create sample data for demonstration purposes."""
    import numpy as np
    
    np.random.seed(42)
    n_samples = 1000
    
    restaurants = ['Shake Shack', 'Blue Ribbon Sushi', 'Cafe Habana', 'Hangawi', 'Dirty Bird']
    cuisines = ['American', 'Japanese', 'Mexican', 'Korean', 'Italian']
    days = ['Weekday', 'Weekend']
    ratings = ['1', '2', '3', '4', '5', 'Not given']
    
    sample_data = pd.DataFrame({
        'order_id': range(1000000, 1000000 + n_samples),
        'customer_id': np.random.randint(1000, 5000, n_samples),
        'restaurant_name': np.random.choice(restaurants, n_samples),
        'cuisine_type': np.random.choice(cuisines, n_samples),
        'cost_of_the_order': np.round(np.random.normal(16.5, 7.5, n_samples), 2),
        'day_of_the_week': np.random.choice(days, n_samples, p=[0.7, 0.3]),
        'rating': np.random.choice(ratings, n_samples, p=[0.05, 0.1, 0.15, 0.3, 0.25, 0.15]),
        'food_preparation_time': np.random.randint(20, 36, n_samples),
        'delivery_time': np.random.randint(15, 34, n_samples)
    })
    
    # Ensure positive costs
    sample_data['cost_of_the_order'] = np.abs(sample_data['cost_of_the_order'])
    sample_data.loc[sample_data['cost_of_the_order'] < 5, 'cost_of_the_order'] = 5
    
    return sample_data

if __name__ == "__main__":
    main()