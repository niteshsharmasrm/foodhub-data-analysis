"""
Data Preprocessing Module for FoodHub Analysis
==============================================

This module provides comprehensive data preprocessing capabilities including:
- Data cleaning and validation
- Feature engineering
- Data transformation
- Quality assessment

Author: Nitesh
Technologies: Pandas, NumPy, Scikit-learn
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for FoodHub data.
    """
    
    def __init__(self):
        """Initialize the preprocessor with default configurations."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
        self.preprocessing_log = []
        
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate the input dataset for common issues.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            Dict: Validation results
        """
        validation_results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Check for potential data quality issues
        issues = []
        
        # Missing values check
        missing_cols = [col for col, count in validation_results['missing_values'].items() if count > 0]
        if missing_cols:
            issues.append(f"Missing values found in columns: {missing_cols}")
            
        # Duplicate rows check
        if validation_results['duplicate_rows'] > 0:
            issues.append(f"Found {validation_results['duplicate_rows']} duplicate rows")
            
        validation_results['issues'] = issues
        validation_results['is_valid'] = len(issues) == 0
        
        self.preprocessing_log.append(f"Data validation completed: {len(issues)} issues found")
        
        return validation_results
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and data quality issues.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        df_cleaned = df.copy()
        
        # Remove duplicate rows
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        removed_duplicates = initial_rows - len(df_cleaned)
        
        if removed_duplicates > 0:
            self.preprocessing_log.append(f"Removed {removed_duplicates} duplicate rows")
        
        # Handle missing values in numerical columns
        numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_cleaned[col].isnull().sum() > 0:
                # Use median for numerical columns
                median_value = df_cleaned[col].median()
                df_cleaned[col].fillna(median_value, inplace=True)
                self.preprocessing_log.append(f"Filled missing values in {col} with median: {median_value}")
        
        # Handle missing values in categorical columns
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_cleaned[col].isnull().sum() > 0:
                # Use mode for categorical columns
                mode_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown'
                df_cleaned[col].fillna(mode_value, inplace=True)
                self.preprocessing_log.append(f"Filled missing values in {col} with mode: {mode_value}")
        
        return df_cleaned
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing data.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with engineered features
        """
        df_featured = df.copy()
        
        # Total time (preparation + delivery)
        if 'food_preparation_time' in df.columns and 'delivery_time' in df.columns:
            df_featured['total_time'] = df_featured['food_preparation_time'] + df_featured['delivery_time']
            self.preprocessing_log.append("Created feature: total_time")
        
        # Order value categories
        if 'cost_of_the_order' in df.columns:
            df_featured['order_value_category'] = pd.cut(
                df_featured['cost_of_the_order'],
                bins=[0, 10, 20, 30, float('inf')],
                labels=['Low', 'Medium', 'High', 'Premium']
            )
            self.preprocessing_log.append("Created feature: order_value_category")
        
        # Rating categories
        if 'rating' in df.columns:
            def categorize_rating(rating):
                if rating == 'Not given':
                    return 'Not Rated'
                elif rating in ['1', '2']:
                    return 'Poor'
                elif rating == '3':
                    return 'Average'
                elif rating in ['4', '5']:
                    return 'Good'
                else:
                    return 'Unknown'
            
            df_featured['rating_category'] = df_featured['rating'].apply(categorize_rating)
            self.preprocessing_log.append("Created feature: rating_category")
        
        # Customer order frequency (if customer_id is available)
        if 'customer_id' in df.columns:
            customer_counts = df_featured['customer_id'].value_counts()
            df_featured['customer_order_frequency'] = df_featured['customer_id'].map(customer_counts)
            self.preprocessing_log.append("Created feature: customer_order_frequency")
        
        # Restaurant popularity score
        if 'restaurant_name' in df.columns:
            restaurant_counts = df_featured['restaurant_name'].value_counts()
            df_featured['restaurant_popularity'] = df_featured['restaurant_name'].map(restaurant_counts)
            self.preprocessing_log.append("Created feature: restaurant_popularity")
        
        return df_featured
    
    def encode_categorical_variables(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Encode categorical variables for machine learning.
        
        Args:
            df (pd.DataFrame): Input dataset
            columns (List[str]): Columns to encode (if None, auto-detect)
            
        Returns:
            pd.DataFrame: Dataset with encoded variables
        """
        df_encoded = df.copy()
        
        if columns is None:
            columns = df_encoded.select_dtypes(include=['object']).columns
        
        for col in columns:
            if col in df_encoded.columns:
                # Use label encoding for now (can be extended to one-hot encoding)
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                
                df_encoded[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                self.preprocessing_log.append(f"Encoded categorical variable: {col}")
        
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Scale numerical features for machine learning.
        
        Args:
            df (pd.DataFrame): Input dataset
            columns (List[str]): Columns to scale (if None, auto-detect)
            
        Returns:
            pd.DataFrame: Dataset with scaled features
        """
        df_scaled = df.copy()
        
        if columns is None:
            columns = df_scaled.select_dtypes(include=[np.number]).columns
        
        # Exclude ID columns from scaling
        columns = [col for col in columns if 'id' not in col.lower()]
        
        if columns:
            scaled_values = self.scaler.fit_transform(df_scaled[columns])
            
            for i, col in enumerate(columns):
                df_scaled[f'{col}_scaled'] = scaled_values[:, i]
                self.preprocessing_log.append(f"Scaled numerical feature: {col}")
        
        return df_scaled
    
    def create_time_features(self, df: pd.DataFrame, datetime_col: str = None) -> pd.DataFrame:
        """
        Create time-based features if datetime column exists.
        
        Args:
            df (pd.DataFrame): Input dataset
            datetime_col (str): Name of datetime column
            
        Returns:
            pd.DataFrame: Dataset with time features
        """
        df_time = df.copy()
        
        if datetime_col and datetime_col in df.columns:
            # Convert to datetime if not already
            df_time[datetime_col] = pd.to_datetime(df_time[datetime_col])
            
            # Extract time components
            df_time['hour'] = df_time[datetime_col].dt.hour
            df_time['day_of_week'] = df_time[datetime_col].dt.dayofweek
            df_time['month'] = df_time[datetime_col].dt.month
            df_time['is_weekend'] = df_time['day_of_week'].isin([5, 6]).astype(int)
            
            self.preprocessing_log.append(f"Created time features from {datetime_col}")
        
        return df_time
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str] = None, method: str = 'iqr') -> Dict:
        """
        Detect outliers in numerical columns.
        
        Args:
            df (pd.DataFrame): Input dataset
            columns (List[str]): Columns to check for outliers
            method (str): Method to use ('iqr' or 'zscore')
            
        Returns:
            Dict: Outlier detection results
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        outlier_results = {}
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z_scores > 3]
            
            outlier_results[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100,
                'indices': outliers.index.tolist()
            }
        
        return outlier_results
    
    def preprocess_pipeline(self, df: pd.DataFrame, 
                          clean_data: bool = True,
                          engineer_features: bool = True,
                          encode_categorical: bool = False,
                          scale_numerical: bool = False) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Input dataset
            clean_data (bool): Whether to clean the data
            engineer_features (bool): Whether to engineer new features
            encode_categorical (bool): Whether to encode categorical variables
            scale_numerical (bool): Whether to scale numerical features
            
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        self.preprocessing_log = []
        self.preprocessing_log.append("Starting preprocessing pipeline")
        
        # Validate data
        validation_results = self.validate_data(df)
        
        df_processed = df.copy()
        
        # Clean data
        if clean_data:
            df_processed = self.clean_data(df_processed)
        
        # Engineer features
        if engineer_features:
            df_processed = self.engineer_features(df_processed)
        
        # Encode categorical variables
        if encode_categorical:
            df_processed = self.encode_categorical_variables(df_processed)
        
        # Scale numerical features
        if scale_numerical:
            df_processed = self.scale_numerical_features(df_processed)
        
        self.preprocessing_log.append("Preprocessing pipeline completed")
        
        return df_processed
    
    def get_preprocessing_summary(self) -> Dict:
        """
        Get summary of preprocessing steps performed.
        
        Returns:
            Dict: Preprocessing summary
        """
        return {
            'steps_performed': self.preprocessing_log,
            'label_encoders_created': list(self.label_encoders.keys()),
            'scaler_fitted': hasattr(self.scaler, 'mean_'),
            'total_steps': len(self.preprocessing_log)
        }


def main():
    """
    Demonstrate the preprocessing pipeline.
    """
    print("FoodHub Data Preprocessing Module")
    print("=================================")
    print("Available preprocessing capabilities:")
    print("1. Data validation and quality assessment")
    print("2. Missing value handling")
    print("3. Feature engineering")
    print("4. Categorical encoding")
    print("5. Numerical scaling")
    print("6. Outlier detection")
    print("7. Complete preprocessing pipeline")


if __name__ == "__main__":
    main()