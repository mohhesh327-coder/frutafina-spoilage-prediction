"""
FrutaFina Streamlit Web Application

Interactive web interface for strawberry shelf-life prediction using AI models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from frutafina import (
    StrawberryDataGenerator,
    DataPreprocessor,
    ShelfLifeRegressor,
    QualityClassifier,
    VisualizationEngine
)
import logging

# Configure Streamlit page
st.set_page_config(
    page_title="FrutaFina - Strawberry Shelf-Life Predictor",
    page_icon="🍓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 1em;
        font-weight: bold;
    }
    .section-header {
        font-size: 2em;
        color: #27ae60;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
        border-bottom: 3px solid #27ae60;
        padding-bottom: 0.5em;
    }
    .metric-box {
        background-color: #ecf0f1;
        padding: 1em;
        border-radius: 10px;
        margin: 0.5em 0;
    }
    .fresh {
        background-color: #d5f4e6;
        padding: 1em;
        border-radius: 10px;
        border-left: 5px solid #27ae60;
    }
    .warning {
        background-color: #ffeaa7;
        padding: 1em;
        border-radius: 10px;
        border-left: 5px solid #f39c12;
    }
    .spoiled {
        background-color: #fab1a0;
        padding: 1em;
        border-radius: 10px;
        border-left: 5px solid #e74c3c;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
    st.session_state.regressor = None
    st.session_state.classifier = None
    st.session_state.preprocessor = None
    st.session_state.feature_names = [
        'chitosan_conc', 'cardiolan_ratio', 'date_seed_extract_mg',
        'storage_temp', 'humidity'
    ]


def train_models():
    """Train the regression and classification models."""
    with st.spinner("Training models... This may take a moment..."):
        try:
            # Generate data
            data_gen = StrawberryDataGenerator(n_samples=500)
            features_df, target = data_gen.generate_dataset()
            
            # Preprocess
            preprocessor = DataPreprocessor()
            X_train, X_test, y_train, y_test = preprocessor.preprocess(features_df, target)
            
            # Train regressor
            regressor = ShelfLifeRegressor()
            regressor.train(X_train, y_train)
            reg_metrics, _ = regressor.evaluate(X_test, y_test)
            
            # Train classifier
            classifier = QualityClassifier()
            classifier.train(X_train, y_train)
            clf_metrics, _ = classifier.evaluate(X_test, y_test)
            
            # Store in session state
            st.session_state.regressor = regressor
            st.session_state.classifier = classifier
            st.session_state.preprocessor = preprocessor
            st.session_state.models_trained = True
            st.session_state.reg_metrics = reg_metrics
            st.session_state.clf_metrics = clf_metrics
            
            st.success("✓ Models trained successfully!")
            return True
        
        except Exception as e:
            st.error(f"Error training models: {str(e)}")
            logger.error(f"Training error: {str(e)}", exc_info=True)
            return False


def predict_sample(sample_data):
    """Make prediction for a single sample."""
    if not st.session_state.models_trained:
        st.warning("Please train models first!")
        return None
    
    try:
        # Prepare sample
        sample_df = pd.DataFrame([sample_data])
        sample_df = sample_df[st.session_state.feature_names]
        
        # Scale
        sample_scaled = st.session_state.preprocessor.scaler.transform(sample_df)
        
        # Regression prediction
        shelf_life = st.session_state.regressor.predict(sample_scaled)[0]
        
        # Classification prediction
        class_pred = st.session_state.classifier.predict(sample_scaled)[0]
        confidence = float(np.max(
            st.session_state.classifier.model.predict_proba(sample_scaled)
        ))
        
        class_names = ['Fresh', 'Warning', 'Spoiled']
        quality = class_names[int(class_pred)]
        
        return {
            'shelf_life': shelf_life,
            'quality': quality,
            'confidence': confidence
        }
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return None


# Header
st.markdown('<div class="main-header">🍓 FrutaFina</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2em; color: #555;">AI-Driven Strawberry Shelf-Life Prediction System</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    
    tab1, tab2, tab3 = st.tabs(["Models", "Prediction", "Analysis"])
    
    with tab1:
        st.subheader("Model Training")
        if st.button("🚀 Train Models", use_container_width=True, key="train_btn"):
            train_models()
        
        if st.session_state.models_trained:
            st.success("✓ Models Trained")
            st.metric("Regression R² Score", f"{st.session_state.reg_metrics['r2_score']:.4f}")
            st.metric("Classification Accuracy", f"{st.session_state.clf_metrics['accuracy']:.4f}")
    
    with tab2:
        st.subheader("Quick Settings")
        st.info("Use the main panel to make predictions with custom parameters.")
    
    with tab3:
        st.subheader("Feature Ranges")
        st.write("""
        - **Chitosan:** 0.5% - 2.0%
        - **Cardiolan:** 0.1 - 0.5 ratio
        - **Date Seed:** 10-50 mg
        - **Temperature:** 4-25°C
        - **Humidity:** 40-90%
        """)


# Main content tabs
main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs(
    ["🎯 Single Prediction", "📊 Batch Analysis", "📈 Model Performance", "ℹ️ About"]
)

# === TAB 1: Single Prediction ===
with main_tab1:
    st.markdown('<div class="section-header">Single Sample Prediction</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        chitosan = st.slider(
            "Chitosan Concentration (%)",
            min_value=0.5, max_value=2.0, value=1.5, step=0.1
        )
        date_seed = st.slider(
            "Date Seed Extract (mg)",
            min_value=10, max_value=50, value=30, step=1
        )
    
    with col2:
        cardiolan = st.slider(
            "Cardiolan Ratio",
            min_value=0.1, max_value=0.5, value=0.3, step=0.05
        )
        humidity = st.slider(
            "Humidity (%)",
            min_value=40, max_value=90, value=65, step=5
        )
    
    with col3:
        storage_temp = st.slider(
            "Storage Temperature (°C)",
            min_value=4, max_value=25, value=10, step=1
        )
    
    if st.button("🔮 Predict", use_container_width=True, key="predict_btn"):
        if not st.session_state.models_trained:
            st.warning("⚠️ Please train models first!")
        else:
            sample = {
                'chitosan_conc': chitosan,
                'cardiolan_ratio': cardiolan,
                'date_seed_extract_mg': date_seed,
                'storage_temp': storage_temp,
                'humidity': humidity
            }
            
            result = predict_sample(sample)
            if result:
                st.markdown("---")
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    if result['quality'] == 'Fresh':
                        st.markdown(f'<div class="fresh"><h3>✓ {result["quality"]}</h3><p>Shelf-life: <strong>{result["shelf_life"]:.1f}</strong> days</p></div>', unsafe_allow_html=True)
                    elif result['quality'] == 'Warning':
                        st.markdown(f'<div class="warning"><h3>⚠ {result["quality"]}</h3><p>Shelf-life: <strong>{result["shelf_life"]:.1f}</strong> days</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="spoiled"><h3>✗ {result["quality"]}</h3><p>Shelf-life: <strong>{result["shelf_life"]:.1f}</strong> days</p></div>', unsafe_allow_html=True)
                
                with col_res2:
                    st.metric("Model Confidence", f"{result['confidence']*100:.1f}%")
                
                with col_res3:
                    st.info(f"""
                    **Input Summary:**
                    - Chitosan: {chitosan}%
                    - Cardiolan: {cardiolan}
                    - Date Seed: {date_seed}mg
                    - Temp: {storage_temp}°C
                    - Humidity: {humidity}%
                    """)


# === TAB 2: Batch Analysis ===
with main_tab2:
    st.markdown('<div class="section-header">Batch Predictions</div>', unsafe_allow_html=True)
    
    batch_option = st.radio("Select batch option:", ["Upload CSV", "Predefined Samples", "Generate Random"])
    
    if batch_option == "Predefined Samples":
        if st.button("Load Sample Data", use_container_width=True):
            sample_data = {
                'chitosan_conc': [1.8, 1.2, 0.8, 1.5, 0.6],
                'cardiolan_ratio': [0.4, 0.25, 0.15, 0.3, 0.1],
                'date_seed_extract_mg': [40, 25, 15, 30, 12],
                'storage_temp': [5, 12, 22, 8, 25],
                'humidity': [50, 70, 85, 60, 88]
            }
            
            if st.session_state.models_trained:
                st.subheader("Sample Predictions")
                df = pd.DataFrame(sample_data)
                
                predictions = []
                for idx, row in df.iterrows():
                    result = predict_sample(row.to_dict())
                    if result:
                        predictions.append({
                            'Sample': idx + 1,
                            'Chitosan': row['chitosan_conc'],
                            'Cardiolan': row['cardiolan_ratio'],
                            'Date Seed (mg)': row['date_seed_extract_mg'],
                            'Temp (°C)': row['storage_temp'],
                            'Humidity': row['humidity'],
                            'Shelf-Life (days)': f"{result['shelf_life']:.2f}",
                            'Quality': result['quality'],
                            'Confidence': f"{result['confidence']*100:.1f}%"
                        })
                
                if predictions:
                    pred_df = pd.DataFrame(predictions)
                    st.dataframe(pred_df, use_container_width=True)
            else:
                st.warning("Train models first!")
    
    elif batch_option == "Generate Random":
        n_samples = st.slider("Number of random samples", 1, 20, 5)
        
        if st.button("Generate & Predict", use_container_width=True):
            if st.session_state.models_trained:
                np.random.seed(42)
                random_samples = {
                    'chitosan_conc': np.random.uniform(0.5, 2.0, n_samples),
                    'cardiolan_ratio': np.random.uniform(0.1, 0.5, n_samples),
                    'date_seed_extract_mg': np.random.uniform(10, 50, n_samples),
                    'storage_temp': np.random.uniform(4, 25, n_samples),
                    'humidity': np.random.uniform(40, 90, n_samples)
                }
                
                predictions = []
                for i in range(n_samples):
                    sample = {key: random_samples[key][i] for key in random_samples}
                    result = predict_sample(sample)
                    if result:
                        predictions.append({
                            'Sample': i + 1,
                            'Shelf-Life (days)': f"{result['shelf_life']:.2f}",
                            'Quality': result['quality'],
                            'Confidence': f"{result['confidence']*100:.1f}%"
                        })
                
                if predictions:
                    pred_df = pd.DataFrame(predictions)
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Visualization
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    
                    shelf_lives = [float(p['Shelf-Life (days)']) for p in predictions]
                    qualities = [p['Quality'] for p in predictions]
                    
                    colors = ['#27ae60' if q == 'Fresh' else '#f39c12' if q == 'Warning' else '#e74c3c' for q in qualities]
                    axes[0].bar(range(len(shelf_lives)), shelf_lives, color=colors)
                    axes[0].set_ylabel('Shelf-Life (days)')
                    axes[0].set_xlabel('Sample')
                    axes[0].set_title('Predicted Shelf-Life Distribution')
                    axes[0].axhline(y=10, color='g', linestyle='--', alpha=0.5, label='Fresh threshold')
                    axes[0].axhline(y=5, color='r', linestyle='--', alpha=0.5, label='Spoiled threshold')
                    axes[0].legend()
                    
                    quality_counts = pd.Series(qualities).value_counts()
                    axes[1].pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%',
                               colors=['#27ae60', '#f39c12', '#e74c3c'])
                    axes[1].set_title('Quality Distribution')
                    
                    st.pyplot(fig)
            else:
                st.warning("Train models first!")


# === TAB 3: Model Performance ===
with main_tab3:
    st.markdown('<div class="section-header">Model Evaluation</div>', unsafe_allow_html=True)
    
    if st.session_state.models_trained:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Regression Model (Random Forest)")
            st.metric("R² Score", f"{st.session_state.reg_metrics['r2_score']:.4f}")
            st.metric("RMSE", f"{st.session_state.reg_metrics['rmse']:.4f} days")
            st.info("Predicts exact shelf-life in days")
        
        with col2:
            st.subheader("Classification Model (XGBoost)")
            st.metric("Accuracy", f"{st.session_state.clf_metrics['accuracy']:.4f}")
            
            # Plot confusion matrix
            cm = st.session_state.clf_metrics['confusion_matrix']
            if isinstance(cm, list):
                cm = np.array(cm)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Fresh', 'Warning', 'Spoiled'],
                       yticklabels=['Fresh', 'Warning', 'Spoiled'],
                       ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            st.pyplot(fig)
            
            st.info("Classifies quality status: Fresh, Warning, or Spoiled")
    else:
        st.warning("Train models to see performance metrics")


# === TAB 4: About ===
with main_tab4:
    st.markdown('<div class="section-header">About FrutaFina</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🍓 FrutaFina: AI-Driven Strawberry Shelf-Life Prediction
    
    FrutaFina is an intelligent system that predicts the shelf-life of strawberries 
    coated with a bio-composite coating made of:
    - **Chitosan**: Natural antimicrobial polymer
    - **Cardiolan**: Biopolymer for extended preservation
    - **Date Seed Extract**: Natural antioxidant
    
    #### 🤖 Machine Learning Models
    
    **Model A - Regression (Random Forest)**
    - Predicts exact shelf-life in days
    - Uses 100 decision trees
    - Trained on 400 synthetic samples
    
    **Model B - Classification (XGBoost)**
    - Classifies quality status: Fresh, Warning, Spoiled
    - Uses gradient boosting
    - Three-class classification task
    
    #### 📊 Features
    
    The prediction system uses 5 key parameters:
    1. **Chitosan Concentration** (0.5% - 2.0%)
    2. **Cardiolan Ratio** (0.1 - 0.5)
    3. **Date Seed Extract** (10-50 mg)
    4. **Storage Temperature** (4-25°C)
    5. **Humidity Level** (40-90%)
    
    #### 🎯 Quality Classes
    
    - **Fresh**: Shelf-life ≥ 10 days ✓
    - **Warning**: Shelf-life 5-10 days ⚠
    - **Spoiled**: Shelf-life < 5 days ✗
    
    #### 💡 Use Cases
    
    - Supply chain optimization
    - Post-harvest quality assessment
    - Storage condition planning
    - Shelf-life labeling and certification
    
    ---
    
    **Version:** 1.0  
    **Author:** Senior AI & Bioinformatics Developer  
    **Year:** 2026
    """)


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; font-size: 0.9em;">
    FrutaFina © 2026 | AI-Driven Strawberry Preservation Analysis
    <br>
    Built with Streamlit | Machine Learning Models: Random Forest + XGBoost
</div>
""", unsafe_allow_html=True)
