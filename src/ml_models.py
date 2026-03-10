"""
Machine Learning Models for FoodHub Analysis
============================================

This module provides advanced machine learning capabilities including:
- Customer segmentation
- Demand forecasting
- Delivery time prediction
- Revenue optimization models

Author: Nitesh
Technologies: Scikit-learn, Pandas, NumPy
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, classification_report, silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional, Any
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

class CustomerSegmentation:
    """
    Advanced customer segmentation using machine learning.
    """
    
    def __init__(self, n_clusters: int = 4):
        """
        Initialize customer segmentation model.
        
        Args:
            n_clusters (int): Number of customer segments
        """
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        self.features = None
        self.segments = None
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for customer segmentation.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Feature matrix
        """
        if 'customer_id' not in df.columns:
            raise ValueError("customer_id column is required for segmentation")
        
        # Aggregate customer metrics
        customer_features = df.groupby('customer_id').agg({
            'cost_of_the_order': ['sum', 'mean', 'count'],
            'food_preparation_time': 'mean',
            'delivery_time': 'mean'
        }).round(2)
        
        # Flatten column names
        customer_features.columns = [
            'total_spent', 'avg_order_value', 'order_frequency',
            'avg_prep_time', 'avg_delivery_time'
        ]
        
        # Add rating information if available
        if 'rating' in df.columns:
            rating_info = df[df['rating'] != 'Not given'].groupby('customer_id')['rating'].apply(
                lambda x: x.astype(int).mean() if len(x) > 0 else 0
            )
            customer_features['avg_rating'] = rating_info.fillna(0)
        
        # Calculate customer lifetime value (CLV)
        customer_features['clv'] = customer_features['total_spent']
        
        # Calculate recency (using order_id as proxy)
        if 'order_id' in df.columns:
            last_order = df.groupby('customer_id')['order_id'].max()
            customer_features['recency'] = last_order
        
        self.features = customer_features.fillna(0)
        return self.features
    
    def fit_predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit the segmentation model and predict segments.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            np.ndarray: Customer segments
        """
        features = self.prepare_features(df)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit K-means
        self.segments = self.kmeans.fit_predict(features_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(features_scaled, self.segments)
        print(f"📊 Customer Segmentation Results:")
        print(f"   Silhouette Score: {silhouette_avg:.3f}")
        print(f"   Number of Segments: {self.n_clusters}")
        
        return self.segments
    
    def get_segment_profiles(self) -> pd.DataFrame:
        """
        Get detailed profiles of each customer segment.
        
        Returns:
            pd.DataFrame: Segment profiles
        """
        if self.features is None or self.segments is None:
            raise ValueError("Model must be fitted first")
        
        # Add segments to features
        features_with_segments = self.features.copy()
        features_with_segments['segment'] = self.segments
        
        # Calculate segment profiles
        segment_profiles = features_with_segments.groupby('segment').agg({
            'total_spent': ['mean', 'std'],
            'avg_order_value': ['mean', 'std'],
            'order_frequency': ['mean', 'std'],
            'avg_prep_time': 'mean',
            'avg_delivery_time': 'mean',
            'clv': 'mean'
        }).round(2)
        
        # Add segment sizes
        segment_sizes = features_with_segments['segment'].value_counts().sort_index()
        
        return segment_profiles, segment_sizes
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        joblib.dump({
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'n_clusters': self.n_clusters
        }, filepath)


class DemandForecaster:
    """
    Demand forecasting model for restaurant orders.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize demand forecasting model.
        
        Args:
            model_type (str): Type of model ('random_forest', 'gradient_boosting', 'linear')
        """
        self.model_type = model_type
        self.model = self._get_model()
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def _get_model(self):
        """Get the specified model."""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'linear':
            return LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for demand forecasting.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        # Create time-based features (using order_id as proxy for time)
        df_sorted = df.sort_values('order_id').copy()
        df_sorted['order_sequence'] = range(len(df_sorted))
        
        # Aggregate by time windows (every 50 orders)
        window_size = 50
        df_sorted['time_window'] = df_sorted['order_sequence'] // window_size
        
        # Aggregate features by time window
        features = df_sorted.groupby('time_window').agg({
            'order_id': 'count',  # Number of orders (target)
            'cost_of_the_order': ['mean', 'sum'],
            'food_preparation_time': 'mean',
            'delivery_time': 'mean'
        }).round(2)
        
        # Flatten column names
        features.columns = [
            'order_count', 'avg_order_value', 'total_revenue',
            'avg_prep_time', 'avg_delivery_time'
        ]
        
        # Add day of week information if available
        if 'day_of_the_week' in df.columns:
            day_counts = df_sorted.groupby('time_window')['day_of_the_week'].apply(
                lambda x: (x == 'Weekend').sum() / len(x)
            )
            features['weekend_ratio'] = day_counts
        
        # Add cuisine diversity
        cuisine_diversity = df_sorted.groupby('time_window')['cuisine_type'].nunique()
        features['cuisine_diversity'] = cuisine_diversity
        
        # Create lagged features
        features['prev_order_count'] = features['order_count'].shift(1)
        features['prev_avg_value'] = features['avg_order_value'].shift(1)
        
        # Remove first row due to lagged features
        features = features.dropna()
        
        # Separate features and target
        target = features['order_count']
        feature_cols = [col for col in features.columns if col != 'order_count']
        X = features[feature_cols]
        
        return X, target
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the demand forecasting model.
        
        Args:
            df (pd.DataFrame): Training dataset
            
        Returns:
            Dict[str, float]: Training metrics
        """
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2_score': r2
        }
        
        print(f"🔮 Demand Forecasting Results:")
        print(f"   RMSE: {metrics['rmse']:.3f}")
        print(f"   R² Score: {metrics['r2_score']:.3f}")
        
        return metrics
    
    def predict_demand(self, df: pd.DataFrame, periods: int = 5) -> np.ndarray:
        """
        Predict future demand.
        
        Args:
            df (pd.DataFrame): Historical data
            periods (int): Number of periods to forecast
            
        Returns:
            np.ndarray: Demand predictions
        """
        X, _ = self.prepare_features(df)
        
        # Use last available data point for prediction
        last_features = X.iloc[-1:].values
        last_features_scaled = self.scaler.transform(last_features)
        
        predictions = []
        for _ in range(periods):
            pred = self.model.predict(last_features_scaled)[0]
            predictions.append(pred)
            
            # Update features for next prediction (simple approach)
            # In practice, you'd want more sophisticated feature updating
            
        return np.array(predictions)


class DeliveryTimePredictor:
    """
    Predict delivery times based on various factors.
    """
    
    def __init__(self):
        """Initialize delivery time prediction model."""
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for delivery time prediction.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        if 'delivery_time' not in df.columns:
            raise ValueError("delivery_time column is required")
        
        features = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['restaurant_name', 'cuisine_type', 'day_of_the_week']
        for col in categorical_cols:
            if col in features.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                features[f'{col}_encoded'] = self.label_encoders[col].fit_transform(features[col].astype(str))
        
        # Select relevant features
        feature_cols = ['cost_of_the_order', 'food_preparation_time']
        feature_cols.extend([f'{col}_encoded' for col in categorical_cols if col in df.columns])
        
        X = features[feature_cols].fillna(0)
        y = features['delivery_time']
        
        return X, y
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the delivery time prediction model.
        
        Args:
            df (pd.DataFrame): Training dataset
            
        Returns:
            Dict[str, float]: Training metrics
        """
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2_score': r2
        }
        
        print(f"🚚 Delivery Time Prediction Results:")
        print(f"   RMSE: {metrics['rmse']:.3f} minutes")
        print(f"   R² Score: {metrics['r2_score']:.3f}")
        
        return metrics


class RevenueOptimizer:
    """
    Revenue optimization and pricing analysis.
    """
    
    def __init__(self):
        """Initialize revenue optimization model."""
        self.price_elasticity_model = LinearRegression()
        self.revenue_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def analyze_price_elasticity(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze price elasticity of demand.
        
        Args:
            df (pd.DataFrame): Dataset with order and pricing data
            
        Returns:
            Dict[str, float]: Price elasticity metrics
        """
        if 'cost_of_the_order' not in df.columns:
            raise ValueError("cost_of_the_order column is required")
        
        # Group by price ranges
        df['price_range'] = pd.cut(df['cost_of_the_order'], bins=10)
        price_demand = df.groupby('price_range').agg({
            'cost_of_the_order': 'mean',
            'order_id': 'count'
        }).dropna()
        
        # Calculate elasticity
        X = price_demand['cost_of_the_order'].values.reshape(-1, 1)
        y = price_demand['order_id'].values
        
        self.price_elasticity_model.fit(X, y)
        elasticity = self.price_elasticity_model.coef_[0]
        
        return {
            'price_elasticity': elasticity,
            'r2_score': self.price_elasticity_model.score(X, y)
        }
    
    def optimize_pricing(self, df: pd.DataFrame, target_margin: float = 0.3) -> Dict[str, Any]:
        """
        Suggest optimal pricing strategies.
        
        Args:
            df (pd.DataFrame): Historical data
            target_margin (float): Target profit margin
            
        Returns:
            Dict[str, Any]: Pricing recommendations
        """
        # Analyze current performance by cuisine and restaurant
        performance_analysis = df.groupby(['cuisine_type', 'restaurant_name']).agg({
            'cost_of_the_order': ['mean', 'count'],
            'rating': lambda x: (x != 'Not given').sum() / len(x)  # Rating completion rate
        }).round(2)
        
        # Simple optimization suggestions
        recommendations = {
            'high_demand_low_price': "Consider price increases for high-demand, low-price items",
            'low_demand_high_price': "Consider price reductions or promotions for low-demand items",
            'target_margin': target_margin
        }
        
        return recommendations


class MLPipeline:
    """
    Complete machine learning pipeline for FoodHub analysis.
    """
    
    def __init__(self):
        """Initialize the ML pipeline."""
        self.customer_segmentation = CustomerSegmentation()
        self.demand_forecaster = DemandForecaster()
        self.delivery_predictor = DeliveryTimePredictor()
        self.revenue_optimizer = RevenueOptimizer()
        self.results = {}
        
    def run_complete_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run complete machine learning analysis.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            Dict[str, Any]: Complete ML analysis results
        """
        print("🤖 Starting Machine Learning Analysis Pipeline...")
        print("=" * 50)
        
        results = {}
        
        # Customer Segmentation
        try:
            if 'customer_id' in df.columns:
                segments = self.customer_segmentation.fit_predict(df)
                segment_profiles, segment_sizes = self.customer_segmentation.get_segment_profiles()
                results['customer_segmentation'] = {
                    'segments': segments,
                    'profiles': segment_profiles,
                    'sizes': segment_sizes
                }
                print("✅ Customer segmentation completed")
        except Exception as e:
            print(f"⚠️ Customer segmentation failed: {str(e)}")
        
        # Demand Forecasting
        try:
            demand_metrics = self.demand_forecaster.train(df)
            demand_predictions = self.demand_forecaster.predict_demand(df)
            results['demand_forecasting'] = {
                'metrics': demand_metrics,
                'predictions': demand_predictions,
                'feature_importance': self.demand_forecaster.feature_importance
            }
            print("✅ Demand forecasting completed")
        except Exception as e:
            print(f"⚠️ Demand forecasting failed: {str(e)}")
        
        # Delivery Time Prediction
        try:
            if 'delivery_time' in df.columns:
                delivery_metrics = self.delivery_predictor.train(df)
                results['delivery_prediction'] = {
                    'metrics': delivery_metrics
                }
                print("✅ Delivery time prediction completed")
        except Exception as e:
            print(f"⚠️ Delivery time prediction failed: {str(e)}")
        
        # Revenue Optimization
        try:
            price_elasticity = self.revenue_optimizer.analyze_price_elasticity(df)
            pricing_recommendations = self.revenue_optimizer.optimize_pricing(df)
            results['revenue_optimization'] = {
                'price_elasticity': price_elasticity,
                'recommendations': pricing_recommendations
            }
            print("✅ Revenue optimization completed")
        except Exception as e:
            print(f"⚠️ Revenue optimization failed: {str(e)}")
        
        self.results = results
        print("\n🎯 Machine Learning Analysis Complete!")
        print("=" * 50)
        
        return results
    
    def save_models(self, directory: str = "models/") -> None:
        """
        Save all trained models.
        
        Args:
            directory (str): Directory to save models
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save customer segmentation model
        self.customer_segmentation.save_model(f"{directory}/customer_segmentation.pkl")
        
        # Save other models
        joblib.dump(self.demand_forecaster, f"{directory}/demand_forecaster.pkl")
        joblib.dump(self.delivery_predictor, f"{directory}/delivery_predictor.pkl")
        joblib.dump(self.revenue_optimizer, f"{directory}/revenue_optimizer.pkl")
        
        print(f"💾 All models saved to {directory}")


def main():
    """
    Demonstrate the machine learning capabilities.
    """
    print("FoodHub Machine Learning Module")
    print("==============================")
    print("Available ML models:")
    print("1. Customer Segmentation (K-Means)")
    print("2. Demand Forecasting (Random Forest)")
    print("3. Delivery Time Prediction")
    print("4. Revenue Optimization")
    print("5. Complete ML Pipeline")


if __name__ == "__main__":
    main()