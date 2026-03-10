"""
Advanced Visualization Module for FoodHub Analysis
==================================================

This module provides comprehensive visualization capabilities including:
- Interactive dashboards
- Statistical plots
- Business intelligence charts
- Performance metrics visualization

Author: Nitesh
Technologies: Matplotlib, Seaborn, Plotly, Bokeh
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
import os

warnings.filterwarnings('ignore')

class FoodHubVisualizer:
    """
    Advanced visualization engine for FoodHub analytics.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize the visualizer with styling preferences.
        
        Args:
            style (str): Matplotlib style to use
        """
        plt.style.use(style)
        sns.set_palette("husl")
        self.color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
    def create_restaurant_analysis(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """
        Create comprehensive restaurant performance analysis.
        
        Args:
            df (pd.DataFrame): Dataset
            save_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Top restaurants by order count
        restaurant_counts = df['restaurant_name'].value_counts().head(15)
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Top Restaurants by Orders', 'Revenue by Restaurant', 
                          'Average Order Value', 'Order Distribution'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "pie"}]]
        )
        
        # Top restaurants bar chart
        fig.add_trace(
            go.Bar(x=restaurant_counts.values, y=restaurant_counts.index,
                   orientation='h', name='Order Count',
                   marker_color='#FF6B6B'),
            row=1, col=1
        )
        
        # Revenue analysis
        if 'cost_of_the_order' in df.columns:
            restaurant_revenue = df.groupby('restaurant_name')['cost_of_the_order'].sum().sort_values(ascending=False).head(10)
            fig.add_trace(
                go.Bar(x=restaurant_revenue.values, y=restaurant_revenue.index,
                       orientation='h', name='Revenue',
                       marker_color='#4ECDC4'),
                row=1, col=2
            )
            
            # Average order value
            avg_order_value = df.groupby('restaurant_name')['cost_of_the_order'].mean().sort_values(ascending=False).head(10)
            fig.add_trace(
                go.Scatter(x=avg_order_value.values, y=avg_order_value.index,
                          mode='markers', name='Avg Order Value',
                          marker=dict(size=10, color='#45B7D1')),
                row=2, col=1
            )
        
        # Order distribution pie chart
        top_5_restaurants = restaurant_counts.head(5)
        fig.add_trace(
            go.Pie(labels=top_5_restaurants.index, values=top_5_restaurants.values,
                   name="Order Distribution"),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Restaurant Performance Dashboard",
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(f"{save_path}/restaurant_dashboard.html")
            fig.write_image(f"{save_path}/restaurant_dashboard.png")
        
        return fig
    
    def create_cuisine_analysis(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """
        Create cuisine popularity and performance analysis.
        
        Args:
            df (pd.DataFrame): Dataset
            save_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        cuisine_counts = df['cuisine_type'].value_counts()
        
        # Create interactive bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=cuisine_counts.index,
            y=cuisine_counts.values,
            marker_color=self.color_palette[:len(cuisine_counts)],
            text=cuisine_counts.values,
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Cuisine Popularity Analysis",
            title_x=0.5,
            xaxis_title="Cuisine Type",
            yaxis_title="Number of Orders",
            template="plotly_white",
            height=600
        )
        
        if save_path:
            fig.write_html(f"{save_path}/cuisine_analysis.html")
            fig.write_image(f"{save_path}/cuisine_analysis.png")
        
        return fig
    
    def create_delivery_performance_dashboard(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """
        Create delivery and preparation time performance dashboard.
        
        Args:
            df (pd.DataFrame): Dataset
            save_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Preparation Time Distribution', 'Delivery Time Distribution',
                          'Time Performance by Day', 'Efficiency Metrics'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "box"}, {"type": "indicator"}]]
        )
        
        # Preparation time histogram
        if 'food_preparation_time' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['food_preparation_time'], name='Prep Time',
                           marker_color='#FF6B6B', opacity=0.7),
                row=1, col=1
            )
        
        # Delivery time histogram
        if 'delivery_time' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['delivery_time'], name='Delivery Time',
                           marker_color='#4ECDC4', opacity=0.7),
                row=1, col=2
            )
        
        # Box plot by day of week
        if 'day_of_the_week' in df.columns and 'food_preparation_time' in df.columns:
            for day in df['day_of_the_week'].unique():
                day_data = df[df['day_of_the_week'] == day]['food_preparation_time']
                fig.add_trace(
                    go.Box(y=day_data, name=day, boxpoints='outliers'),
                    row=2, col=1
                )
        
        # Efficiency indicator
        if 'food_preparation_time' in df.columns:
            avg_prep_time = df['food_preparation_time'].mean()
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=avg_prep_time,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Avg Prep Time (min)"},
                    delta={'reference': 30},
                    gauge={'axis': {'range': [None, 50]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 25], 'color': "lightgray"},
                                   {'range': [25, 35], 'color': "gray"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': 35}}
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Delivery Performance Dashboard",
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(f"{save_path}/delivery_dashboard.html")
            fig.write_image(f"{save_path}/delivery_dashboard.png")
        
        return fig
    
    def create_revenue_analysis(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """
        Create comprehensive revenue analysis dashboard.
        
        Args:
            df (pd.DataFrame): Dataset
            save_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        if 'cost_of_the_order' not in df.columns:
            print("Cost column not found for revenue analysis")
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Order Value Distribution', 'Revenue by Day Type',
                          'Cumulative Revenue', 'Revenue Metrics'),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "indicator"}]]
        )
        
        # Order value distribution
        fig.add_trace(
            go.Histogram(x=df['cost_of_the_order'], nbinsx=30,
                        marker_color='#96CEB4', opacity=0.7),
            row=1, col=1
        )
        
        # Revenue by day type
        if 'day_of_the_week' in df.columns:
            day_revenue = df.groupby('day_of_the_week')['cost_of_the_order'].sum()
            fig.add_trace(
                go.Bar(x=day_revenue.index, y=day_revenue.values,
                       marker_color=['#FF6B6B', '#4ECDC4']),
                row=1, col=2
            )
        
        # Cumulative revenue (if order_id can be used as proxy for time)
        df_sorted = df.sort_values('order_id')
        cumulative_revenue = df_sorted['cost_of_the_order'].cumsum()
        fig.add_trace(
            go.Scatter(x=df_sorted.index, y=cumulative_revenue,
                      mode='lines', name='Cumulative Revenue',
                      line=dict(color='#45B7D1', width=3)),
            row=2, col=1
        )
        
        # Revenue metrics indicator
        total_revenue = df['cost_of_the_order'].sum()
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=total_revenue,
                title={'text': "Total Revenue ($)"},
                number={'prefix': "$", 'font': {'size': 40}}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Revenue Analysis Dashboard",
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(f"{save_path}/revenue_dashboard.html")
            fig.write_image(f"{save_path}/revenue_dashboard.png")
        
        return fig
    
    def create_customer_analysis(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """
        Create customer behavior analysis dashboard.
        
        Args:
            df (pd.DataFrame): Dataset
            save_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Customer Order Frequency', 'Rating Distribution',
                          'Customer Segments', 'Satisfaction Metrics'),
            specs=[[{"type": "histogram"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Customer order frequency
        if 'customer_id' in df.columns:
            customer_orders = df['customer_id'].value_counts()
            fig.add_trace(
                go.Histogram(x=customer_orders.values, nbinsx=20,
                           marker_color='#FFEAA7', opacity=0.7),
                row=1, col=1
            )
        
        # Rating distribution
        if 'rating' in df.columns:
            rating_counts = df['rating'].value_counts()
            fig.add_trace(
                go.Pie(labels=rating_counts.index, values=rating_counts.values,
                       hole=0.3),
                row=1, col=2
            )
        
        # Customer segments (order frequency vs order value)
        if 'customer_id' in df.columns and 'cost_of_the_order' in df.columns:
            customer_metrics = df.groupby('customer_id').agg({
                'cost_of_the_order': ['mean', 'count']
            }).round(2)
            customer_metrics.columns = ['avg_order_value', 'order_frequency']
            
            fig.add_trace(
                go.Scatter(x=customer_metrics['order_frequency'],
                          y=customer_metrics['avg_order_value'],
                          mode='markers',
                          marker=dict(size=8, color='#DDA0DD', opacity=0.6)),
                row=2, col=1
            )
        
        # Satisfaction metrics
        if 'rating' in df.columns:
            # Calculate satisfaction rate (ratings 4-5 vs total rated)
            rated_orders = df[df['rating'] != 'Not given']
            if len(rated_orders) > 0:
                satisfaction_rate = len(rated_orders[rated_orders['rating'].isin(['4', '5'])]) / len(rated_orders) * 100
                
                fig.add_trace(
                    go.Bar(x=['Satisfaction Rate'], y=[satisfaction_rate],
                           marker_color='#96CEB4'),
                    row=2, col=2
                )
        
        fig.update_layout(
            title_text="Customer Analysis Dashboard",
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(f"{save_path}/customer_dashboard.html")
            fig.write_image(f"{save_path}/customer_dashboard.png")
        
        return fig
    
    def create_comprehensive_dashboard(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """
        Create a comprehensive executive dashboard.
        
        Args:
            df (pd.DataFrame): Dataset
            save_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Total Orders', 'Total Revenue', 'Avg Order Value',
                          'Top Cuisine', 'Top Restaurant', 'Customer Satisfaction',
                          'Avg Prep Time', 'Avg Delivery Time', 'Order Trends'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}, {"type": "scatter"}]]
        )
        
        # KPI Indicators
        total_orders = len(df)
        fig.add_trace(go.Indicator(mode="number", value=total_orders,
                                  title={'text': "Total Orders"}), row=1, col=1)
        
        if 'cost_of_the_order' in df.columns:
            total_revenue = df['cost_of_the_order'].sum()
            avg_order_value = df['cost_of_the_order'].mean()
            
            fig.add_trace(go.Indicator(mode="number", value=total_revenue,
                                      number={'prefix': "$"},
                                      title={'text': "Total Revenue"}), row=1, col=2)
            
            fig.add_trace(go.Indicator(mode="number", value=avg_order_value,
                                      number={'prefix': "$"},
                                      title={'text': "Avg Order Value"}), row=1, col=3)
        
        # Top performers
        if 'cuisine_type' in df.columns:
            top_cuisine = df['cuisine_type'].value_counts()
            fig.add_trace(go.Bar(x=top_cuisine.head(5).index, y=top_cuisine.head(5).values,
                               marker_color='#FF6B6B'), row=2, col=1)
        
        if 'restaurant_name' in df.columns:
            top_restaurant = df['restaurant_name'].value_counts()
            fig.add_trace(go.Bar(x=top_restaurant.head(5).index, y=top_restaurant.head(5).values,
                               marker_color='#4ECDC4'), row=2, col=2)
        
        # Performance metrics
        if 'rating' in df.columns:
            rated_orders = df[df['rating'] != 'Not given']
            if len(rated_orders) > 0:
                satisfaction = len(rated_orders[rated_orders['rating'].isin(['4', '5'])]) / len(rated_orders) * 100
                fig.add_trace(go.Indicator(mode="gauge+number", value=satisfaction,
                                          title={'text': "Satisfaction %"},
                                          gauge={'axis': {'range': [0, 100]}}), row=2, col=3)
        
        if 'food_preparation_time' in df.columns:
            avg_prep = df['food_preparation_time'].mean()
            fig.add_trace(go.Indicator(mode="number", value=avg_prep,
                                      number={'suffix': " min"},
                                      title={'text': "Avg Prep Time"}), row=3, col=1)
        
        if 'delivery_time' in df.columns:
            avg_delivery = df['delivery_time'].mean()
            fig.add_trace(go.Indicator(mode="number", value=avg_delivery,
                                      number={'suffix': " min"},
                                      title={'text': "Avg Delivery Time"}), row=3, col=2)
        
        # Order trends (using order_id as proxy)
        df_sorted = df.sort_values('order_id')
        order_trend = df_sorted.groupby(df_sorted.index // 100).size()  # Group by batches
        fig.add_trace(go.Scatter(x=order_trend.index, y=order_trend.values,
                               mode='lines+markers', line=dict(color='#45B7D1')), row=3, col=3)
        
        fig.update_layout(
            title_text="FoodHub Executive Dashboard",
            title_x=0.5,
            height=1000,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(f"{save_path}/executive_dashboard.html")
            fig.write_image(f"{save_path}/executive_dashboard.png")
        
        return fig
    
    def save_all_visualizations(self, df: pd.DataFrame, save_path: str = "assets/images/") -> None:
        """
        Generate and save all visualizations.
        
        Args:
            df (pd.DataFrame): Dataset
            save_path (str): Directory to save visualizations
        """
        os.makedirs(save_path, exist_ok=True)
        
        print("🎨 Generating comprehensive visualizations...")
        
        # Create all dashboards
        self.create_restaurant_analysis(df, save_path)
        self.create_cuisine_analysis(df, save_path)
        self.create_delivery_performance_dashboard(df, save_path)
        self.create_revenue_analysis(df, save_path)
        self.create_customer_analysis(df, save_path)
        self.create_comprehensive_dashboard(df, save_path)
        
        print(f"✅ All visualizations saved to {save_path}")


def main():
    """
    Demonstrate the visualization capabilities.
    """
    print("FoodHub Advanced Visualization Module")
    print("====================================")
    print("Available visualization types:")
    print("1. Restaurant Performance Analysis")
    print("2. Cuisine Popularity Dashboard")
    print("3. Delivery Performance Metrics")
    print("4. Revenue Analysis Dashboard")
    print("5. Customer Behavior Analysis")
    print("6. Executive Summary Dashboard")


if __name__ == "__main__":
    main()