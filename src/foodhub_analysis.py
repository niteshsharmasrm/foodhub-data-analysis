"""
FoodHub Data Analysis - Comprehensive Food Delivery Analytics
============================================================

This module provides comprehensive analysis of food delivery data including:
- Customer behavior analysis
- Restaurant performance metrics
- Cuisine popularity trends
- Delivery time optimization
- Revenue analysis and insights

Author: Nitesh
Project: PG Course AIML - Food Hub Analysis
Technologies: Python, Pandas, NumPy, Matplotlib, Seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class FoodHubAnalyzer:
    """
    A comprehensive analyzer for FoodHub delivery data.
    
    This class provides methods for analyzing various aspects of food delivery
    business including customer patterns, restaurant performance, and operational metrics.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the FoodHub analyzer.
        
        Args:
            data_path (str): Path to the CSV data file
        """
        self.data = None
        self.data_path = data_path
        self.analysis_results = {}
        
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load the FoodHub dataset.
        
        Args:
            data_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        if data_path:
            self.data_path = data_path
            
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"✅ Data loaded successfully! Shape: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            print(f"❌ Error: File not found at {self.data_path}")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None
    
    def get_data_overview(self) -> Dict:
        """
        Get comprehensive overview of the dataset.
        
        Returns:
            Dict: Dataset overview statistics
        """
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return {}
            
        overview = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'data_types': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage': f"{self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        
        print("📊 Dataset Overview:")
        print(f"   Rows: {overview['shape'][0]:,}")
        print(f"   Columns: {overview['shape'][1]}")
        print(f"   Memory Usage: {overview['memory_usage']}")
        
        return overview
    
    def analyze_missing_values(self) -> Dict:
        """
        Analyze missing values and data quality issues.
        
        Returns:
            Dict: Missing value analysis results
        """
        if self.data is None:
            return {}
            
        missing_analysis = {}
        
        # Check for missing values
        missing_counts = self.data.isnull().sum()
        missing_analysis['missing_counts'] = missing_counts.to_dict()
        
        # Check for 'Not given' ratings
        if 'rating' in self.data.columns:
            not_given_count = (self.data['rating'] == 'Not given').sum()
            missing_analysis['not_given_ratings'] = not_given_count
            print(f"📈 Orders with 'Not given' rating: {not_given_count:,}")
        
        return missing_analysis
    
    def get_statistical_summary(self) -> Dict:
        """
        Get statistical summary of numerical columns.
        
        Returns:
            Dict: Statistical summary
        """
        if self.data is None:
            return {}
            
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        summary = self.data[numerical_cols].describe().to_dict()
        
        # Food preparation time insights
        if 'food_preparation_time' in self.data.columns:
            prep_stats = {
                'min_prep_time': self.data['food_preparation_time'].min(),
                'avg_prep_time': self.data['food_preparation_time'].mean(),
                'max_prep_time': self.data['food_preparation_time'].max()
            }
            print(f"🍳 Food Preparation Time Analysis:")
            print(f"   Minimum: {prep_stats['min_prep_time']} minutes")
            print(f"   Average: {prep_stats['avg_prep_time']:.2f} minutes")
            print(f"   Maximum: {prep_stats['max_prep_time']} minutes")
            
        return summary
    
    def analyze_restaurants(self) -> Dict:
        """
        Analyze restaurant performance and popularity.
        
        Returns:
            Dict: Restaurant analysis results
        """
        if self.data is None:
            return {}
            
        restaurant_analysis = {}
        
        # Top restaurants by order count
        restaurant_counts = self.data['restaurant_name'].value_counts()
        restaurant_analysis['top_restaurants'] = restaurant_counts.head(10).to_dict()
        
        # Restaurant revenue analysis
        if 'cost_of_the_order' in self.data.columns:
            restaurant_revenue = self.data.groupby('restaurant_name')['cost_of_the_order'].agg([
                'sum', 'mean', 'count'
            ]).round(2)
            restaurant_analysis['restaurant_metrics'] = restaurant_revenue.to_dict()
        
        print(f"🏪 Top 5 Restaurants by Order Count:")
        for i, (restaurant, count) in enumerate(restaurant_counts.head(5).items(), 1):
            print(f"   {i}. {restaurant}: {count:,} orders")
            
        return restaurant_analysis
    
    def analyze_cuisines(self) -> Dict:
        """
        Analyze cuisine popularity and trends.
        
        Returns:
            Dict: Cuisine analysis results
        """
        if self.data is None:
            return {}
            
        cuisine_analysis = {}
        
        # Cuisine popularity
        cuisine_counts = self.data['cuisine_type'].value_counts()
        cuisine_analysis['cuisine_popularity'] = cuisine_counts.to_dict()
        
        # Cuisine revenue analysis
        if 'cost_of_the_order' in self.data.columns:
            cuisine_revenue = self.data.groupby('cuisine_type')['cost_of_the_order'].agg([
                'sum', 'mean', 'count'
            ]).round(2)
            cuisine_analysis['cuisine_metrics'] = cuisine_revenue.to_dict()
        
        print(f"🍽️ Top 5 Cuisines by Popularity:")
        for i, (cuisine, count) in enumerate(cuisine_counts.head(5).items(), 1):
            print(f"   {i}. {cuisine}: {count:,} orders")
            
        return cuisine_analysis
    
    def analyze_order_patterns(self) -> Dict:
        """
        Analyze order patterns including weekday vs weekend trends.
        
        Returns:
            Dict: Order pattern analysis
        """
        if self.data is None:
            return {}
            
        pattern_analysis = {}
        
        # Weekday vs Weekend analysis
        if 'day_of_the_week' in self.data.columns:
            day_counts = self.data['day_of_the_week'].value_counts()
            pattern_analysis['day_distribution'] = day_counts.to_dict()
            
            # Average order value by day type
            if 'cost_of_the_order' in self.data.columns:
                avg_order_by_day = self.data.groupby('day_of_the_week')['cost_of_the_order'].mean()
                pattern_analysis['avg_order_value_by_day'] = avg_order_by_day.to_dict()
        
        # Rating distribution
        if 'rating' in self.data.columns:
            rating_dist = self.data['rating'].value_counts()
            pattern_analysis['rating_distribution'] = rating_dist.to_dict()
        
        return pattern_analysis
    
    def analyze_delivery_performance(self) -> Dict:
        """
        Analyze delivery and preparation time performance.
        
        Returns:
            Dict: Delivery performance metrics
        """
        if self.data is None:
            return {}
            
        delivery_analysis = {}
        
        # Preparation time analysis
        if 'food_preparation_time' in self.data.columns:
            prep_stats = {
                'mean': self.data['food_preparation_time'].mean(),
                'median': self.data['food_preparation_time'].median(),
                'std': self.data['food_preparation_time'].std(),
                'min': self.data['food_preparation_time'].min(),
                'max': self.data['food_preparation_time'].max()
            }
            delivery_analysis['preparation_time_stats'] = prep_stats
        
        # Delivery time analysis
        if 'delivery_time' in self.data.columns:
            delivery_stats = {
                'mean': self.data['delivery_time'].mean(),
                'median': self.data['delivery_time'].median(),
                'std': self.data['delivery_time'].std(),
                'min': self.data['delivery_time'].min(),
                'max': self.data['delivery_time'].max()
            }
            delivery_analysis['delivery_time_stats'] = delivery_stats
        
        return delivery_analysis
    
    def create_visualizations(self, save_path: str = "assets/images/") -> None:
        """
        Create comprehensive visualizations for the analysis.
        
        Args:
            save_path (str): Directory to save visualization images
        """
        if self.data is None:
            print("❌ No data loaded for visualization.")
            return
            
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Set style for better visualizations
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Restaurant popularity
        plt.figure(figsize=(12, 8))
        top_restaurants = self.data['restaurant_name'].value_counts().head(10)
        sns.barplot(y=top_restaurants.index, x=top_restaurants.values, orient='h')
        plt.title('Top 10 Restaurants by Order Count', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Orders')
        plt.ylabel('Restaurant Name')
        plt.tight_layout()
        plt.savefig(f"{save_path}top_restaurants.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Cuisine popularity
        plt.figure(figsize=(10, 8))
        cuisine_counts = self.data['cuisine_type'].value_counts()
        sns.barplot(y=cuisine_counts.index, x=cuisine_counts.values, orient='h')
        plt.title('Cuisine Popularity Analysis', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Orders')
        plt.ylabel('Cuisine Type')
        plt.tight_layout()
        plt.savefig(f"{save_path}cuisine_popularity.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Order cost distribution
        if 'cost_of_the_order' in self.data.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram
            sns.histplot(data=self.data, x='cost_of_the_order', kde=True, ax=ax1)
            ax1.set_title('Order Cost Distribution', fontweight='bold')
            ax1.set_xlabel('Order Cost ($)')
            
            # Box plot
            sns.boxplot(data=self.data, x='cost_of_the_order', ax=ax2)
            ax2.set_title('Order Cost Box Plot', fontweight='bold')
            ax2.set_xlabel('Order Cost ($)')
            
            plt.tight_layout()
            plt.savefig(f"{save_path}order_cost_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Delivery performance
        if 'food_preparation_time' in self.data.columns and 'delivery_time' in self.data.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Preparation time
            sns.histplot(data=self.data, x='food_preparation_time', kde=True, ax=ax1)
            ax1.set_title('Food Preparation Time Distribution', fontweight='bold')
            ax1.set_xlabel('Preparation Time (minutes)')
            
            # Delivery time
            sns.histplot(data=self.data, x='delivery_time', kde=True, ax=ax2)
            ax2.set_title('Delivery Time Distribution', fontweight='bold')
            ax2.set_xlabel('Delivery Time (minutes)')
            
            plt.tight_layout()
            plt.savefig(f"{save_path}delivery_performance.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Weekday vs Weekend analysis
        if 'day_of_the_week' in self.data.columns:
            plt.figure(figsize=(10, 6))
            day_counts = self.data['day_of_the_week'].value_counts()
            colors = ['#FF6B6B', '#4ECDC4']
            plt.pie(day_counts.values, labels=day_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
            plt.title('Weekday vs Weekend Order Distribution', fontsize=16, fontweight='bold')
            plt.axis('equal')
            plt.savefig(f"{save_path}weekday_weekend_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 Visualizations saved to {save_path}")
    
    def generate_insights(self) -> Dict:
        """
        Generate business insights from the analysis.
        
        Returns:
            Dict: Key business insights
        """
        if self.data is None:
            return {}
            
        insights = {}
        
        # Customer insights
        total_customers = self.data['customer_id'].nunique()
        total_orders = len(self.data)
        avg_orders_per_customer = total_orders / total_customers
        
        insights['customer_metrics'] = {
            'total_customers': total_customers,
            'total_orders': total_orders,
            'avg_orders_per_customer': round(avg_orders_per_customer, 2)
        }
        
        # Revenue insights
        if 'cost_of_the_order' in self.data.columns:
            total_revenue = self.data['cost_of_the_order'].sum()
            avg_order_value = self.data['cost_of_the_order'].mean()
            
            insights['revenue_metrics'] = {
                'total_revenue': round(total_revenue, 2),
                'avg_order_value': round(avg_order_value, 2)
            }
        
        # Operational insights
        if 'rating' in self.data.columns:
            rated_orders = len(self.data[self.data['rating'] != 'Not given'])
            rating_completion_rate = (rated_orders / total_orders) * 100
            
            insights['operational_metrics'] = {
                'rating_completion_rate': round(rating_completion_rate, 2)
            }
        
        return insights
    
    def run_complete_analysis(self, save_visualizations: bool = True) -> Dict:
        """
        Run complete analysis pipeline.
        
        Args:
            save_visualizations (bool): Whether to save visualization plots
            
        Returns:
            Dict: Complete analysis results
        """
        print("🚀 Starting FoodHub Complete Analysis...")
        print("=" * 50)
        
        # Data overview
        overview = self.get_data_overview()
        
        # Missing values analysis
        missing_analysis = self.analyze_missing_values()
        
        # Statistical summary
        stats_summary = self.get_statistical_summary()
        
        # Restaurant analysis
        restaurant_analysis = self.analyze_restaurants()
        
        # Cuisine analysis
        cuisine_analysis = self.analyze_cuisines()
        
        # Order patterns
        pattern_analysis = self.analyze_order_patterns()
        
        # Delivery performance
        delivery_analysis = self.analyze_delivery_performance()
        
        # Generate insights
        insights = self.generate_insights()
        
        # Create visualizations
        if save_visualizations:
            self.create_visualizations()
        
        # Compile results
        complete_results = {
            'overview': overview,
            'missing_analysis': missing_analysis,
            'statistical_summary': stats_summary,
            'restaurant_analysis': restaurant_analysis,
            'cuisine_analysis': cuisine_analysis,
            'pattern_analysis': pattern_analysis,
            'delivery_analysis': delivery_analysis,
            'insights': insights,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        self.analysis_results = complete_results
        
        print("\n✅ Analysis Complete!")
        print("=" * 50)
        
        return complete_results


def main():
    """
    Main function to demonstrate the FoodHub analysis.
    """
    # Initialize analyzer
    analyzer = FoodHubAnalyzer()
    
    # Example usage (uncomment when you have data)
    # analyzer.load_data('data/foodhub_order.csv')
    # results = analyzer.run_complete_analysis()
    
    print("FoodHub Data Analysis Tool Ready!")
    print("Usage:")
    print("1. analyzer = FoodHubAnalyzer()")
    print("2. analyzer.load_data('path/to/your/data.csv')")
    print("3. results = analyzer.run_complete_analysis()")


if __name__ == "__main__":
    main()