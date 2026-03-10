"""
Test Script for FoodHub Data Analysis
====================================

This script tests the main functionality of the FoodHub analysis pipeline.
Run this to verify that all components are working correctly.

Usage:
    python test_analysis.py
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append('src')

def create_test_data():
    """Create sample test data for validation."""
    np.random.seed(42)
    n_samples = 100
    
    restaurants = ['Test Restaurant A', 'Test Restaurant B', 'Test Restaurant C']
    cuisines = ['American', 'Italian', 'Asian']
    days = ['Weekday', 'Weekend']
    ratings = ['1', '2', '3', '4', '5', 'Not given']
    
    test_data = pd.DataFrame({
        'order_id': range(1000, 1000 + n_samples),
        'customer_id': np.random.randint(100, 200, n_samples),
        'restaurant_name': np.random.choice(restaurants, n_samples),
        'cuisine_type': np.random.choice(cuisines, n_samples),
        'cost_of_the_order': np.round(np.random.normal(15, 5, n_samples), 2),
        'day_of_the_week': np.random.choice(days, n_samples),
        'rating': np.random.choice(ratings, n_samples),
        'food_preparation_time': np.random.randint(20, 35, n_samples),
        'delivery_time': np.random.randint(15, 30, n_samples)
    })
    
    # Ensure positive costs
    test_data['cost_of_the_order'] = np.abs(test_data['cost_of_the_order'])
    test_data.loc[test_data['cost_of_the_order'] < 5, 'cost_of_the_order'] = 5
    
    return test_data

def test_data_loading():
    """Test data loading functionality."""
    print("🧪 Testing Data Loading...")
    
    try:
        from foodhub_analysis import FoodHubAnalyzer
        
        # Create test data
        test_data = create_test_data()
        test_data.to_csv('test_data.csv', index=False)
        
        # Test analyzer
        analyzer = FoodHubAnalyzer('test_data.csv')
        data = analyzer.load_data()
        
        assert data is not None, "Data loading failed"
        assert len(data) == 100, "Incorrect data size"
        
        print("✅ Data loading test passed")
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {str(e)}")
        return False

def test_data_preprocessing():
    """Test data preprocessing functionality."""
    print("🧪 Testing Data Preprocessing...")
    
    try:
        from data_preprocessing import DataPreprocessor
        
        test_data = create_test_data()
        preprocessor = DataPreprocessor()
        
        # Test validation
        validation_results = preprocessor.validate_data(test_data)
        assert 'total_rows' in validation_results, "Validation failed"
        
        # Test preprocessing pipeline
        processed_data = preprocessor.preprocess_pipeline(test_data)
        assert len(processed_data) > 0, "Preprocessing failed"
        
        print("✅ Data preprocessing test passed")
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing test failed: {str(e)}")
        return False

def test_analysis():
    """Test main analysis functionality."""
    print("🧪 Testing Analysis Engine...")
    
    try:
        from foodhub_analysis import FoodHubAnalyzer
        
        test_data = create_test_data()
        test_data.to_csv('test_data.csv', index=False)
        
        analyzer = FoodHubAnalyzer('test_data.csv')
        analyzer.load_data()
        
        # Test individual analysis methods
        overview = analyzer.get_data_overview()
        assert 'shape' in overview, "Overview analysis failed"
        
        restaurant_analysis = analyzer.analyze_restaurants()
        assert 'top_restaurants' in restaurant_analysis, "Restaurant analysis failed"
        
        cuisine_analysis = analyzer.analyze_cuisines()
        assert 'cuisine_popularity' in cuisine_analysis, "Cuisine analysis failed"
        
        print("✅ Analysis engine test passed")
        return True
        
    except Exception as e:
        print(f"❌ Analysis engine test failed: {str(e)}")
        return False

def test_machine_learning():
    """Test machine learning functionality."""
    print("🧪 Testing Machine Learning Models...")
    
    try:
        from ml_models import CustomerSegmentation, DemandForecaster
        
        test_data = create_test_data()
        
        # Test customer segmentation
        segmenter = CustomerSegmentation(n_clusters=3)
        segments = segmenter.fit_predict(test_data)
        assert len(segments) > 0, "Customer segmentation failed"
        
        # Test demand forecasting
        forecaster = DemandForecaster()
        metrics = forecaster.train(test_data)
        assert 'rmse' in metrics, "Demand forecasting failed"
        
        print("✅ Machine learning test passed")
        return True
        
    except Exception as e:
        print(f"❌ Machine learning test failed: {str(e)}")
        return False

def test_visualization():
    """Test visualization functionality."""
    print("🧪 Testing Visualization Engine...")
    
    try:
        from visualization import FoodHubVisualizer
        
        test_data = create_test_data()
        visualizer = FoodHubVisualizer()
        
        # Test dashboard creation
        dashboard = visualizer.create_comprehensive_dashboard(test_data)
        assert dashboard is not None, "Dashboard creation failed"
        
        print("✅ Visualization test passed")
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {str(e)}")
        return False

def test_complete_pipeline():
    """Test the complete analysis pipeline."""
    print("🧪 Testing Complete Pipeline...")
    
    try:
        # Create test data
        test_data = create_test_data()
        test_data.to_csv('test_data.csv', index=False)
        
        # Import main runner
        from main import run_complete_analysis
        
        # Run analysis with test data
        results = run_complete_analysis('test_data.csv', 'test_results')
        
        assert 'data_validation' in results, "Pipeline validation failed"
        assert 'eda' in results, "EDA pipeline failed"
        
        print("✅ Complete pipeline test passed")
        return True
        
    except Exception as e:
        print(f"❌ Complete pipeline test failed: {str(e)}")
        return False

def cleanup_test_files():
    """Clean up test files."""
    test_files = ['test_data.csv']
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
    
    # Clean up test results directory
    import shutil
    if os.path.exists('test_results'):
        shutil.rmtree('test_results')

def main():
    """Run all tests."""
    print("🚀 FoodHub Data Analysis - Test Suite")
    print("=" * 50)
    print(f"⏰ Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Data Preprocessing", test_data_preprocessing),
        ("Analysis Engine", test_analysis),
        ("Machine Learning", test_machine_learning),
        ("Visualization", test_visualization),
        ("Complete Pipeline", test_complete_pipeline)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test...")
        if test_func():
            passed += 1
        print("-" * 30)
    
    # Cleanup
    cleanup_test_files()
    
    print(f"\n🎯 Test Results Summary:")
    print(f"   ✅ Passed: {passed}/{total}")
    print(f"   ❌ Failed: {total - passed}/{total}")
    print(f"   📊 Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is ready for use.")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)