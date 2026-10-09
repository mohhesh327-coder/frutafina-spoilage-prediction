"""
FrutaFina: AI-Driven Strawberry Shelf-Life Prediction System

This module implements an intelligent system to predict the shelf-life of
strawberries coated with a bio-composite (Chitosan, Cardiolan, and Date Seed
Extract) using machine learning models with a REST API server.

Author: Senior AI & Bioinformatics Developer
Date: 2026
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import seaborn as sns
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('frutafina.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

class StrawberryDataGenerator:
    """
    Generate synthetic dataset for strawberry shelf-life prediction.
    
    Features:
    - chitosan_conc: Chitosan concentration (0.5% - 2.0%)
    - cardiolan_ratio: Cardiolan ratio (0.1 - 0.5)
    - date_seed_extract_mg: Date seed extract quantity (10mg - 50mg)
    - storage_temp: Storage temperature (4°C - 25°C)
    - humidity: Storage humidity (40% - 90%)
    
    Target:
    - days_to_spoilage: Shelf life in days
    """
    
    def __init__(self, n_samples=500, random_state=42):
        """
        Initialize the data generator.
        
        Args:
            n_samples (int): Number of synthetic samples to generate
            random_state (int): Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        logger.info(f"Initialized StrawberryDataGenerator with {n_samples} samples")
    
    def generate_features(self):
        """
        Generate feature values within specified ranges.
        
        Returns:
            pd.DataFrame: DataFrame with synthetic features
        """
        logger.info("Generating synthetic features...")
        
        data = {
            'chitosan_conc': np.random.uniform(0.5, 2.0, self.n_samples),
            'cardiolan_ratio': np.random.uniform(0.1, 0.5, self.n_samples),
            'date_seed_extract_mg': np.random.uniform(10, 50, self.n_samples),
            'storage_temp': np.random.uniform(4, 25, self.n_samples),
            'humidity': np.random.uniform(40, 90, self.n_samples)
        }
        
        features_df = pd.DataFrame(data)
        logger.info(f"Generated {len(features_df)} samples with 5 features")
        return features_df
    
    def generate_target(self, features_df):
        """
        Generate target variable (days_to_spoilage) based on features.
        
        The shelf-life is influenced by:
        - Higher chitosan concentration increases shelf-life
        - Higher cardiolan ratio increases shelf-life
        - Higher date seed extract increases shelf-life
        - Lower storage temperature increases shelf-life
        - Lower humidity increases shelf-life
        - Added noise for realism
        
        Args:
            features_df (pd.DataFrame): Feature dataframe
            
        Returns:
            np.ndarray: Target variable (days to spoilage)
        """
        logger.info("Generating target variable (days_to_spoilage)...")
        
        chitosan = features_df['chitosan_conc'].values
        cardiolan = features_df['cardiolan_ratio'].values
        date_seed = features_df['date_seed_extract_mg'].values
        temp = features_df['storage_temp'].values
        humidity = features_df['humidity'].values
        
        # Base calculation
        base_shelf_life = 6.5
        shelf_life = (
            base_shelf_life +
            (chitosan * 4) +
            (cardiolan * 6) +
            (date_seed / 6) +
            ((25 - temp) * 0.35) +
            ((90 - humidity) * 0.06)
        )
        
        # Add significant noise for proper class distribution
        noise = np.random.normal(0, 2.5, self.n_samples)
        shelf_life = shelf_life + noise
        
        # Ensure we have samples in ALL THREE classes by explicit distribution
        sorted_indices = np.argsort(shelf_life)
        n_samples = len(shelf_life)
        
        # Ensure minimum samples in each class
        spoiled_ratio = 0.25  # 25% spoiled (< 5 days)
        warning_ratio = 0.35  # 35% warning (5-10 days)
        fresh_ratio = 0.40    # 40% fresh (>= 10 days)
        
        n_spoiled = max(10, int(n_samples * spoiled_ratio))
        n_warning = max(10, int(n_samples * warning_ratio))
        n_fresh = n_samples - n_spoiled - n_warning
        
        # Assign shelf-life ranges
        shelf_life_sorted = np.zeros(n_samples)
        
        # Spoiled: 2-4 days
        shelf_life_sorted[:n_spoiled] = np.random.uniform(2, 4.9, n_spoiled)
        
        # Warning: 5-9 days
        shelf_life_sorted[n_spoiled:n_spoiled+n_warning] = np.random.uniform(5, 9.9, n_warning)
        
        # Fresh: 10-16 days
        shelf_life_sorted[n_spoiled+n_warning:] = np.random.uniform(10, 16, n_fresh)
        
        # Unsort back to original order
        unsorted_indices = np.argsort(sorted_indices)
        shelf_life = shelf_life_sorted[unsorted_indices]
        
        logger.info(f"Target variable range: {shelf_life.min():.2f} - {shelf_life.max():.2f} days")
        
        # Calculate class distribution
        fresh_count = (shelf_life >= 10).sum()
        warning_count = ((shelf_life >= 5) & (shelf_life < 10)).sum()
        spoiled_count = (shelf_life < 5).sum()
        
        logger.info(f"Shelf-life distribution - Fresh (>=10): {fresh_count}, "
                   f"Warning (5-10): {warning_count}, Spoiled (<5): {spoiled_count}")
        
        return shelf_life.round(2)
    
    def generate_dataset(self):
        """
        Generate complete synthetic dataset.
        
        Returns:
            tuple: (features_df, target_array)
        """
        features_df = self.generate_features()
        target = self.generate_target(features_df)
        
        logger.info("Dataset generation completed successfully")
        return features_df, target


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

class DataPreprocessor:
    """Handle data scaling and splitting."""
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize the preprocessor.
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        logger.info("Initialized DataPreprocessor")
    
    def preprocess(self, features_df, target):
        """
        Preprocess data: scale features and split into train/test sets.
        
        Args:
            features_df (pd.DataFrame): Feature dataframe
            target (np.ndarray): Target variable
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info("Starting data preprocessing...")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_df)
        logger.info("Features scaled using StandardScaler")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled,
            target,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test


# ============================================================================
# REGRESSION MODEL (Model A)
# ============================================================================

class ShelfLifeRegressor:
    """
    Random Forest Regressor for predicting exact days until spoilage.
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the regressor.
        
        Args:
            n_estimators (int): Number of trees in the forest
            random_state (int): Random seed for reproducibility
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        logger.info(f"Initialized Random Forest Regressor ({n_estimators} estimators)")
    
    def train(self, X_train, y_train):
        """
        Train the regression model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
        """
        logger.info("Training Random Forest Regressor...")
        self.model.fit(X_train, y_train)
        logger.info("Regressor training completed")
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X (np.ndarray): Features for prediction
            
        Returns:
            np.ndarray: Predicted shelf-life in days
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
            
        Returns:
            dict: Performance metrics
        """
        logger.info("Evaluating Random Forest Regressor...")
        
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2
        }
        
        logger.info(f"Regression Metrics - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        return metrics, y_pred
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance scores.
        
        Args:
            feature_names (list): Names of features
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        importance_scores = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=False)
        
        logger.info("Feature importance calculated")
        return importance_df


# ============================================================================
# CLASSIFICATION MODEL (Model B)
# ============================================================================

class QualityClassifier:
    """
    XGBoost Classifier for quality status prediction.
    
    Classes:
    - 0: Fresh (shelf-life > 10 days)
    - 1: Warning (shelf-life 5-10 days)
    - 2: Spoiled (shelf-life < 5 days)
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the classifier.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1
        )
        logger.info("Initialized XGBoost Classifier")
    
    @staticmethod
    def label_quality(shelf_life):
        """
        Convert shelf-life (days) to quality status.
        
        Args:
            shelf_life (float or np.ndarray): Days until spoilage
            
        Returns:
            int or np.ndarray: Quality class (0: Fresh, 1: Warning, 2: Spoiled)
        """
        if isinstance(shelf_life, np.ndarray):
            labels = np.zeros_like(shelf_life, dtype=int)
            labels[shelf_life >= 10] = 0  # Fresh
            labels[(shelf_life < 10) & (shelf_life >= 5)] = 1  # Warning
            labels[shelf_life < 5] = 2  # Spoiled
            return labels
        else:
            if shelf_life >= 10:
                return 0  # Fresh
            elif shelf_life >= 5:
                return 1  # Warning
            else:
                return 2  # Spoiled
    
    def train(self, X_train, y_train):
        """
        Train the classification model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training shelf-life (will be converted to labels)
        """
        logger.info("Training XGBoost Classifier...")
        
        # Convert shelf-life to quality labels
        y_train_labels = self.label_quality(y_train)
        class_distribution = pd.Series(y_train_labels).value_counts().sort_index()
        logger.info(f"Class distribution - Fresh: {class_distribution.get(0, 0)}, "
                   f"Warning: {class_distribution.get(1, 0)}, "
                   f"Spoiled: {class_distribution.get(2, 0)}")
        
        self.model.fit(X_train, y_train_labels)
        logger.info("Classifier training completed")
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X (np.ndarray): Features for prediction
            
        Returns:
            np.ndarray: Predicted quality class
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test shelf-life
            
        Returns:
            dict: Performance metrics
        """
        logger.info("Evaluating XGBoost Classifier...")
        
        y_test_labels = self.label_quality(y_test)
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test_labels, y_pred)
        cm = confusion_matrix(y_test_labels, y_pred, labels=[0, 1, 2])
        
        # Get unique classes in predictions and labels
        unique_labels = np.unique(np.concatenate([y_test_labels, y_pred]))
        class_names = ['Fresh', 'Warning', 'Spoiled']
        available_names = [class_names[i] for i in unique_labels if i < 3]
        
        clf_report = classification_report(
            y_test_labels,
            y_pred,
            labels=list(unique_labels),
            target_names=available_names,
            zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': cm.tolist(),
            'classification_report': clf_report
        }
        
        logger.info(f"Classification Accuracy: {accuracy:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        return metrics, y_pred
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance scores.
        
        Args:
            feature_names (list): Names of features
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        importance_scores = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=False)
        
        logger.info("Feature importance calculated")
        return importance_df


# ============================================================================
# VISUALIZATION
# ============================================================================

class VisualizationEngine:
    """Generate visualizations for model analysis."""
    
    @staticmethod
    def plot_feature_importance(reg_importance, clf_importance):
        """
        Plot feature importance from both models side by side.
        
        Highlights the impact of Chitosan and Date Seeds on preservation.
        
        Args:
            reg_importance (pd.DataFrame): Regression model feature importance
            clf_importance (pd.DataFrame): Classification model feature importance
        """
        logger.info("Generating feature importance visualization...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Regression feature importance
        axes[0].barh(reg_importance['Feature'], reg_importance['Importance'], color='steelblue')
        axes[0].set_xlabel('Importance Score', fontsize=11)
        axes[0].set_title('Random Forest Regressor\nFeature Importance', fontsize=12, fontweight='bold')
        axes[0].invert_yaxis()
        
        # Classification feature importance
        axes[1].barh(clf_importance['Feature'], clf_importance['Importance'], color='coral')
        axes[1].set_xlabel('Importance Score', fontsize=11)
        axes[1].set_title('XGBoost Classifier\nFeature Importance', fontsize=12, fontweight='bold')
        axes[1].invert_yaxis()
        
        # Highlight Chitosan and Date Seed Extract
        for ax in axes:
            for i, label in enumerate(ax.get_yticklabels()):
                if 'chitosan' in label.get_text().lower() or 'date_seed' in label.get_text().lower():
                    label.set_color('red')
                    label.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        logger.info("Feature importance plot saved to 'feature_importance.png'")
        plt.close()
    
    @staticmethod
    def plot_confusion_matrix(confusion_matrix_array):
        """
        Plot confusion matrix for classification model.
        
        Args:
            confusion_matrix_array (np.ndarray or list): Confusion matrix from classifier
        """
        logger.info("Generating confusion matrix visualization...")
        
        # Convert to numpy array if it's a list
        if isinstance(confusion_matrix_array, list):
            confusion_matrix_array = np.array(confusion_matrix_array)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix_array,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Fresh', 'Warning', 'Spoiled'],
            yticklabels=['Fresh', 'Warning', 'Spoiled']
        )
        plt.title('XGBoost Classifier - Confusion Matrix', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=11)
        plt.xlabel('Predicted Label', fontsize=11)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        logger.info("Confusion matrix plot saved to 'confusion_matrix.png'")
        plt.close()
    
    @staticmethod
    def plot_regression_residuals(y_test, y_pred):
        """
        Plot regression predictions vs actual values.
        
        Args:
            y_test (np.ndarray): Actual test values
            y_pred (np.ndarray): Predicted test values
        """
        logger.info("Generating regression residuals visualization...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Predictions vs Actual
        axes[0].scatter(y_test, y_pred, alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Days to Spoilage', fontsize=11)
        axes[0].set_ylabel('Predicted Days to Spoilage', fontsize=11)
        axes[0].set_title('Regression: Predictions vs Actual', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals distribution
        residuals = y_test - y_pred
        axes[1].hist(residuals, bins=30, color='coral', edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Residuals (Days)', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('regression_residuals.png', dpi=300, bbox_inches='tight')
        logger.info("Regression residuals plot saved to 'regression_residuals.png'")
        plt.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class FrutaFinaPipeline:
    """
    Main pipeline orchestrating data generation, model training, and evaluation.
    """
    
    def __init__(self, n_samples=500):
        """
        Initialize the pipeline.
        
        Args:
            n_samples (int): Number of synthetic samples to generate
        """
        self.n_samples = n_samples
        self.feature_names = [
            'chitosan_conc', 'cardiolan_ratio', 'date_seed_extract_mg',
            'storage_temp', 'humidity'
        ]
        logger.info("=" * 70)
        logger.info("FrutaFina Pipeline Initialized")
        logger.info("=" * 70)
    
    def run(self):
        """Execute the complete pipeline."""
        
        # Step 1: Generate synthetic data
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: SYNTHETIC DATA GENERATION")
        logger.info("=" * 70)
        
        data_gen = StrawberryDataGenerator(n_samples=self.n_samples)
        features_df, target = data_gen.generate_dataset()
        
        logger.info("\nDataset Summary:")
        logger.info(f"Features shape: {features_df.shape}")
        logger.info(f"\nFeature Statistics:\n{features_df.describe()}")
        logger.info(f"\nTarget Statistics:\nMin: {target.min():.2f} days, "
                   f"Max: {target.max():.2f} days, Mean: {target.mean():.2f} days")
        
        # Step 2: Data preprocessing
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: DATA PREPROCESSING")
        logger.info("=" * 70)
        
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess(features_df, target)
        
        # Step 3: Train Regression Model (Model A)
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: REGRESSION MODEL TRAINING (Random Forest)")
        logger.info("=" * 70)
        
        regressor = ShelfLifeRegressor()
        regressor.train(X_train, y_train)
        reg_metrics, reg_pred = regressor.evaluate(X_test, y_test)
        reg_importance = regressor.get_feature_importance(self.feature_names)
        
        logger.info("\nRegression Model Summary:")
        logger.info(f"R² Score: {reg_metrics['r2_score']:.4f}")
        logger.info(f"RMSE: {reg_metrics['rmse']:.4f} days")
        logger.info(f"\nTop 3 Important Features:\n{reg_importance.head(3)}")
        
        # Step 4: Train Classification Model (Model B)
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: CLASSIFICATION MODEL TRAINING (XGBoost)")
        logger.info("=" * 70)
        
        classifier = QualityClassifier()
        classifier.train(X_train, y_train)
        clf_metrics, clf_pred = classifier.evaluate(X_test, y_test)
        clf_importance = classifier.get_feature_importance(self.feature_names)
        
        logger.info("\nClassification Model Summary:")
        logger.info(f"Accuracy: {clf_metrics['accuracy']:.4f}")
        logger.info(f"\n{clf_metrics['classification_report']}")
        logger.info(f"\nTop 3 Important Features:\n{clf_importance.head(3)}")
        
        # Step 5: Feature Importance Analysis
        logger.info("\n" + "=" * 70)
        logger.info("STEP 5: FEATURE IMPORTANCE ANALYSIS")
        logger.info("=" * 70)
        
        logger.info("\nComparing impact of Chitosan vs Date Seed Extract:")
        chitosan_importance_reg = reg_importance[reg_importance['Feature'] == 'chitosan_conc']['Importance'].values[0]
        date_seed_importance_reg = reg_importance[reg_importance['Feature'] == 'date_seed_extract_mg']['Importance'].values[0]
        
        chitosan_importance_clf = clf_importance[clf_importance['Feature'] == 'chitosan_conc']['Importance'].values[0]
        date_seed_importance_clf = clf_importance[clf_importance['Feature'] == 'date_seed_extract_mg']['Importance'].values[0]
        
        logger.info(f"Regression - Chitosan: {chitosan_importance_reg:.4f}, Date Seed: {date_seed_importance_reg:.4f}")
        logger.info(f"Classification - Chitosan: {chitosan_importance_clf:.4f}, Date Seed: {date_seed_importance_clf:.4f}")
        
        # Step 6: Visualizations
        logger.info("\n" + "=" * 70)
        logger.info("STEP 6: GENERATING VISUALIZATIONS")
        logger.info("=" * 70)
        
        viz = VisualizationEngine()
        viz.plot_feature_importance(reg_importance, clf_importance)
        viz.plot_confusion_matrix(clf_metrics['confusion_matrix'])
        viz.plot_regression_residuals(y_test, reg_pred)
        
        # Summary Report
        logger.info("\n" + "=" * 70)
        logger.info("FRUTAFINA PIPELINE - EXECUTION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"\n[OK] Generated {len(features_df)} synthetic strawberry samples")
        logger.info(f"[OK] Random Forest Regressor - R² Score: {reg_metrics['r2_score']:.4f}")
        logger.info(f"[OK] XGBoost Classifier - Accuracy: {clf_metrics['accuracy']:.4f}")
        logger.info(f"[OK] Generated 3 visualization plots (PNG format)")
        logger.info(f"[OK] Logs saved to 'frutafina.log'")
        logger.info("\n" + "=" * 70)
        
        return {
            'regressor': regressor,
            'classifier': classifier,
            'preprocessor': preprocessor,
            'metrics': {
                'regression': reg_metrics,
                'classification': clf_metrics
            },
            'feature_importance': {
                'regression': reg_importance,
                'classification': clf_importance
            }
        }


# ============================================================================
# FLASK WEB API SERVER
# ============================================================================

class FrutaFinaServer:
    """Flask server for FrutaFina model predictions via REST API."""
    
    def __init__(self, app_name='FrutaFina', host='127.0.0.1', port=5000):
        """
        Initialize Flask server.
        
        Args:
            app_name (str): Flask app name
            host (str): Server host address
            port (int): Server port
        """
        self.app = Flask(app_name)
        CORS(self.app)  # Enable CORS for all routes
        self.host = host
        self.port = port
        
        # Global variables for models
        self.regressor = None
        self.classifier = None
        self.preprocessor = None
        self.feature_names = [
            'chitosan_conc', 'cardiolan_ratio', 'date_seed_extract_mg',
            'storage_temp', 'humidity'
        ]
        
        self._register_routes()
        logger.info(f"Initialized FrutaFinaServer on {host}:{port}")
    
    def _register_routes(self):
        """Register all API routes."""
        
        @self.app.route('/', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'online',
                'service': 'FrutaFina Strawberry Shelf-Life Prediction API',
                'version': '1.0',
                'timestamp': datetime.now().isoformat()
            }), 200
        
        @self.app.route('/api/train', methods=['POST'])
        def train_models():
            """Train models with synthetic data."""
            try:
                logger.info("API Request: /api/train")
                
                # Generate data
                data_gen = StrawberryDataGenerator(n_samples=500)
                features_df, target = data_gen.generate_dataset()
                
                # Preprocess
                preprocessor = DataPreprocessor()
                X_train, X_test, y_train, y_test = preprocessor.preprocess(features_df, target)
                
                # Train models
                regressor = ShelfLifeRegressor()
                regressor.train(X_train, y_train)
                reg_metrics, _ = regressor.evaluate(X_test, y_test)
                
                classifier = QualityClassifier()
                classifier.train(X_train, y_train)
                clf_metrics, _ = classifier.evaluate(X_test, y_test)
                
                # Store globally
                self.regressor = regressor
                self.classifier = classifier
                self.preprocessor = preprocessor
                
                logger.info("Models trained successfully")
                
                return jsonify({
                    'status': 'success',
                    'message': 'Models trained successfully',
                    'regression_metrics': {
                        'r2_score': float(reg_metrics['r2_score']),
                        'rmse': float(reg_metrics['rmse'])
                    },
                    'classification_metrics': {
                        'accuracy': float(clf_metrics['accuracy'])
                    },
                    'timestamp': datetime.now().isoformat()
                }), 200
            
            except Exception as e:
                logger.error(f"Training error: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/predict/regression', methods=['POST'])
        def predict_regression():
            """
            Predict shelf-life (regression) for strawberry samples.
            
            Request body example:
            {
                "samples": [
                    {
                        "chitosan_conc": 1.5,
                        "cardiolan_ratio": 0.3,
                        "date_seed_extract_mg": 30,
                        "storage_temp": 10,
                        "humidity": 65
                    }
                ]
            }
            """
            try:
                if self.regressor is None:
                    return jsonify({
                        'status': 'error',
                        'message': 'Models not trained. Call /api/train first'
                    }), 400
                
                data = request.get_json()
                samples = data.get('samples', [])
                
                if not samples:
                    return jsonify({
                        'status': 'error',
                        'message': 'No samples provided'
                    }), 400
                
                logger.info(f"API Request: /api/predict/regression with {len(samples)} samples")
                
                # Prepare features
                sample_df = pd.DataFrame(samples)
                sample_df = sample_df[self.feature_names]
                
                # Scale features
                sample_scaled = self.preprocessor.scaler.transform(sample_df)
                
                # Make predictions
                predictions = self.regressor.predict(sample_scaled)
                
                results = []
                for i, pred in enumerate(predictions):
                    results.append({
                        'sample_id': i,
                        'input': samples[i],
                        'predicted_shelf_life_days': float(pred),
                        'quality_status': self._get_quality_status(pred)
                    })
                
                logger.info(f"Regression predictions: {len(results)} samples")
                
                return jsonify({
                    'status': 'success',
                    'model': 'Random Forest Regressor',
                    'predictions': results,
                    'timestamp': datetime.now().isoformat()
                }), 200
            
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/predict/classification', methods=['POST'])
        def predict_classification():
            """
            Predict quality status (classification) for strawberry samples.
            
            Request body example:
            {
                "samples": [
                    {
                        "chitosan_conc": 1.5,
                        "cardiolan_ratio": 0.3,
                        "date_seed_extract_mg": 30,
                        "storage_temp": 10,
                        "humidity": 65
                    }
                ]
            }
            """
            try:
                if self.classifier is None:
                    return jsonify({
                        'status': 'error',
                        'message': 'Models not trained. Call /api/train first'
                    }), 400
                
                data = request.get_json()
                samples = data.get('samples', [])
                
                if not samples:
                    return jsonify({
                        'status': 'error',
                        'message': 'No samples provided'
                    }), 400
                
                logger.info(f"API Request: /api/predict/classification with {len(samples)} samples")
                
                # Prepare features
                sample_df = pd.DataFrame(samples)
                sample_df = sample_df[self.feature_names]
                
                # Scale features
                sample_scaled = self.preprocessor.scaler.transform(sample_df)
                
                # Make predictions
                predictions = self.classifier.predict(sample_scaled)
                
                class_names = ['Fresh', 'Warning', 'Spoiled']
                results = []
                for i, pred in enumerate(predictions):
                    results.append({
                        'sample_id': i,
                        'input': samples[i],
                        'predicted_class': int(pred),
                        'quality_status': class_names[int(pred)],
                        'confidence': float(np.max(self.classifier.model.predict_proba(sample_scaled[i:i+1])))
                    })
                
                logger.info(f"Classification predictions: {len(results)} samples")
                
                return jsonify({
                    'status': 'success',
                    'model': 'XGBoost Classifier',
                    'predictions': results,
                    'timestamp': datetime.now().isoformat()
                }), 200
            
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/batch/predict', methods=['POST'])
        def batch_predict():
            """
            Batch prediction with both regression and classification.
            
            Request body example:
            {
                "samples": [
                    {
                        "chitosan_conc": 1.5,
                        "cardiolan_ratio": 0.3,
                        "date_seed_extract_mg": 30,
                        "storage_temp": 10,
                        "humidity": 65
                    }
                ]
            }
            """
            try:
                if self.regressor is None or self.classifier is None:
                    return jsonify({
                        'status': 'error',
                        'message': 'Models not trained. Call /api/train first'
                    }), 400
                
                data = request.get_json()
                samples = data.get('samples', [])
                
                if not samples:
                    return jsonify({
                        'status': 'error',
                        'message': 'No samples provided'
                    }), 400
                
                logger.info(f"API Request: /api/batch/predict with {len(samples)} samples")
                
                # Prepare features
                sample_df = pd.DataFrame(samples)
                sample_df = sample_df[self.feature_names]
                
                # Scale features
                sample_scaled = self.preprocessor.scaler.transform(sample_df)
                
                # Regression predictions
                reg_predictions = self.regressor.predict(sample_scaled)
                
                # Classification predictions
                clf_predictions = self.classifier.predict(sample_scaled)
                clf_probas = self.classifier.model.predict_proba(sample_scaled)
                
                class_names = ['Fresh', 'Warning', 'Spoiled']
                results = []
                for i in range(len(samples)):
                    results.append({
                        'sample_id': i,
                        'input': samples[i],
                        'regression': {
                            'predicted_shelf_life_days': float(reg_predictions[i]),
                            'quality_status': self._get_quality_status(reg_predictions[i])
                        },
                        'classification': {
                            'predicted_class': int(clf_predictions[i]),
                            'class_name': class_names[int(clf_predictions[i])],
                            'confidence': float(np.max(clf_probas[i]))
                        }
                    })
                
                logger.info(f"Batch predictions completed: {len(results)} samples")
                
                return jsonify({
                    'status': 'success',
                    'models': ['Random Forest Regressor', 'XGBoost Classifier'],
                    'sample_count': len(results),
                    'predictions': results,
                    'timestamp': datetime.now().isoformat()
                }), 200
            
            except Exception as e:
                logger.error(f"Batch prediction error: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/features', methods=['GET'])
        def get_features():
            """Get list of required features."""
            return jsonify({
                'status': 'success',
                'features': self.feature_names,
                'feature_details': {
                    'chitosan_conc': {'min': 0.5, 'max': 2.0, 'unit': '%'},
                    'cardiolan_ratio': {'min': 0.1, 'max': 0.5, 'unit': 'ratio'},
                    'date_seed_extract_mg': {'min': 10, 'max': 50, 'unit': 'mg'},
                    'storage_temp': {'min': 4, 'max': 25, 'unit': '°C'},
                    'humidity': {'min': 40, 'max': 90, 'unit': '%'}
                }
            }), 200
        
        @self.app.route('/api/docs', methods=['GET'])
        def get_docs():
            """Get API documentation."""
            docs = {
                'status': 'success',
                'service': 'FrutaFina Strawberry Shelf-Life Prediction API',
                'version': '1.0',
                'endpoints': {
                    'GET /': 'Health check',
                    'POST /api/train': 'Train models with synthetic data',
                    'POST /api/predict/regression': 'Predict shelf-life (days)',
                    'POST /api/predict/classification': 'Predict quality status',
                    'POST /api/batch/predict': 'Batch predictions (both models)',
                    'GET /api/features': 'Get required features info',
                    'GET /api/docs': 'API documentation'
                },
                'workflow': [
                    '1. Call POST /api/train to train models',
                    '2. Call POST /api/predict/* to make predictions',
                    '3. Predictions are streamed as JSON responses'
                ]
            }
            return jsonify(docs), 200
    
    @staticmethod
    def _get_quality_status(shelf_life):
        """Convert shelf-life to quality status."""
        if shelf_life >= 10:
            return 'Fresh'
        elif shelf_life >= 5:
            return 'Warning'
        else:
            return 'Spoiled'
    
    def run(self):
        """Start the Flask server."""
        logger.info(f"\n{'='*70}")
        logger.info("FrutaFina Server Starting")
        logger.info(f"{'='*70}")
        logger.info(f"Server: http://{self.host}:{self.port}")
        logger.info(f"Documentation: http://{self.host}:{self.port}/api/docs")
        logger.info(f"Train endpoint: POST http://{self.host}:{self.port}/api/train")
        logger.info(f"{'='*70}\n")
        
        self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the FrutaFina pipeline."""
    try:
        pipeline = FrutaFinaPipeline(n_samples=500)
        results = pipeline.run()
        logger.info("\n[OK] FrutaFina pipeline completed successfully!")
        return results
    except Exception as e:
        logger.error(f"Error in pipeline execution: {str(e)}", exc_info=True)
        raise


def run_server(host='127.0.0.1', port=5000):
    """Run FrutaFina as a web service."""
    try:
        server = FrutaFinaServer(host=host, port=port)
        server.run()
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'server':
        # Run as web server
        host = sys.argv[2] if len(sys.argv) > 2 else '127.0.0.1'
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
        run_server(host=host, port=port)
    else:
        # Run as standalone pipeline
        main()
