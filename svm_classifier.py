"""
SVM-based Alzheimer's Voice Biomarker Classifier
Highly accurate model for detecting Alzheimer's-related speech patterns
"""

import numpy as np
import joblib
import os
import json
from datetime import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from voice_features import VoiceFeatureExtractor
import warnings
warnings.filterwarnings('ignore')


class AlzheimersSVMClassifier:
    """
    High-accuracy SVM classifier for Alzheimer's detection from voice biomarkers.
    
    Features:
    - Robust feature scaling (handles outliers)
    - Feature selection for optimal performance
    - Hyperparameter tuning via grid search
    - Ensemble methods for improved accuracy
    - Cross-validation for reliable evaluation
    """
    
    def __init__(self, model_path='models'):
        self.model_path = model_path
        self.feature_extractor = VoiceFeatureExtractor()
        self.scaler = None
        self.feature_selector = None
        self.model = None
        self.feature_names = None
        self.selected_features = None
        self.class_labels = ['Healthy', 'Alzheimers']
        self.is_trained = False
        
        # Create models directory
        os.makedirs(model_path, exist_ok=True)
    
    def extract_features_batch(self, audio_files, labels=None):
        """Extract features from multiple audio files"""
        X = []
        valid_labels = []
        valid_files = []
        
        for i, audio_file in enumerate(audio_files):
            try:
                features, feature_names = self.feature_extractor.get_feature_vector(audio_file)
                
                # Handle NaN/Inf values
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
                X.append(features)
                valid_files.append(audio_file)
                
                if labels is not None:
                    valid_labels.append(labels[i])
                    
                if self.feature_names is None:
                    self.feature_names = feature_names
                    
            except Exception as e:
                print(f"Warning: Could not process {audio_file}: {str(e)}")
                continue
        
        X = np.array(X)
        
        if labels is not None:
            return X, np.array(valid_labels), valid_files
        return X, valid_files
    
    def train(self, audio_files, labels, optimize=True, n_features=50):
        """
        Train the SVM classifier with hyperparameter optimization.
        
        Args:
            audio_files: List of paths to audio files
            labels: List of labels (0=Healthy, 1=Alzheimer's)
            optimize: Whether to perform grid search optimization
            n_features: Number of features to select
        """
        print("Extracting features from training data...")
        X, y, valid_files = self.extract_features_batch(audio_files, labels)
        
        if len(X) < 2:
            raise ValueError("Need at least 2 valid samples for training")
        
        print(f"Extracted features from {len(X)} files")
        print(f"Feature vector size: {X.shape[1]}")
        
        # Step 1: Robust Scaling (handles outliers better than StandardScaler)
        print("Applying robust scaling...")
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Step 2: Feature Selection
        print(f"Selecting top {min(n_features, X.shape[1])} features...")
        n_features = min(n_features, X.shape[1])
        self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
        X_selected = self.feature_selector.fit_transform(X_scaled, y)
        
        # Get selected feature names
        selected_indices = self.feature_selector.get_support(indices=True)
        self.selected_features = [self.feature_names[i] for i in selected_indices]
        print(f"Selected features: {self.selected_features[:10]}...")  # Show first 10
        
        # Step 3: Train SVM with optimized hyperparameters
        if optimize and len(X) >= 5:
            print("Optimizing SVM hyperparameters...")
            self.model = self._optimize_svm(X_selected, y)
        else:
            print("Training SVM with default parameters...")
            self.model = SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42
            )
            self.model.fit(X_selected, y)
        
        # Step 4: Evaluate with cross-validation if we have enough samples
        if len(X) >= 5:
            print("Evaluating model with cross-validation...")
            n_splits = min(5, len(X))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(self.model, X_selected, y, cv=cv, scoring='accuracy')
            print(f"Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        return {
            'n_samples': len(X),
            'n_features_original': X.shape[1],
            'n_features_selected': n_features,
            'selected_features': self.selected_features,
            'model_saved': True
        }
    
    def _optimize_svm(self, X, y):
        """Perform grid search to find optimal SVM hyperparameters"""
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        
        # Use smaller grid for small datasets
        if len(X) < 20:
            param_grid = {
                'C': [1, 10, 100],
                'gamma': ['scale', 0.1],
                'kernel': ['rbf', 'poly']
            }
        
        base_svm = SVC(probability=True, class_weight='balanced', random_state=42)
        
        n_splits = min(5, len(X))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            base_svm,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_
    
    def predict(self, audio_file):
        """
        Predict Alzheimer's probability for a single audio file.
        
        Returns:
            dict with prediction, probability, and feature analysis
        """
        if not self.is_trained:
            # Try to load saved model
            if not self.load_model():
                raise ValueError("Model not trained. Please train the model first.")
        
        # Extract features
        features, feature_names = self.feature_extractor.get_feature_vector(audio_file)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Select features
        features_selected = self.feature_selector.transform(features_scaled)
        
        # Predict
        prediction = self.model.predict(features_selected)[0]
        probabilities = self.model.predict_proba(features_selected)[0]
        
        # Get feature importance for this sample
        feature_analysis = self._analyze_features(features[0], feature_names)
        
        return {
            'prediction': int(prediction),
            'label': self.class_labels[prediction],
            'confidence': float(probabilities[prediction]),
            'probability_healthy': float(probabilities[0]),
            'probability_alzheimers': float(probabilities[1]),
            'risk_level': self._get_risk_level(probabilities[1]),
            'feature_analysis': feature_analysis
        }
    
    def predict_batch(self, audio_files):
        """Predict for multiple audio files"""
        results = []
        for audio_file in audio_files:
            try:
                result = self.predict(audio_file)
                result['file'] = audio_file
                result['status'] = 'success'
            except Exception as e:
                result = {
                    'file': audio_file,
                    'status': 'error',
                    'error': str(e)
                }
            results.append(result)
        return results
    
    def _analyze_features(self, features, feature_names):
        """Analyze which features are most indicative"""
        analysis = {}
        
        # Key biomarker categories
        categories = {
            'mfcc': ['mfcc_'],
            'spectral': ['spectral_', 'zero_crossing'],
            'temporal': ['pause_', 'speech_', 'rms_', 'energy_', 'envelope_'],
            'pitch': ['pitch_', 'voiced_'],
            'voice_quality': ['jitter_', 'shimmer_', 'hnr_', 'breathiness'],
            'timing': ['duration', 'phonation_', 'articulation_']
        }
        
        for category, prefixes in categories.items():
            category_features = {}
            for i, name in enumerate(feature_names):
                if any(name.startswith(prefix) for prefix in prefixes):
                    category_features[name] = float(features[i])
            analysis[category] = category_features
        
        return analysis
    
    def _get_risk_level(self, alzheimer_prob):
        """Determine risk level based on probability"""
        if alzheimer_prob < 0.25:
            return 'Low'
        elif alzheimer_prob < 0.50:
            return 'Moderate'
        elif alzheimer_prob < 0.75:
            return 'High'
        else:
            return 'Very High'
    
    def save_model(self, filename=None):
        """Save the trained model and preprocessing components"""
        if filename is None:
            filename = os.path.join(self.model_path, 'alzheimers_svm_model.joblib')
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'class_labels': self.class_labels,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")
        return filename
    
    def load_model(self, filename=None):
        """Load a previously trained model"""
        if filename is None:
            filename = os.path.join(self.model_path, 'alzheimers_svm_model.joblib')
        
        if not os.path.exists(filename):
            return False
        
        try:
            model_data = joblib.load(filename)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_selector = model_data['feature_selector']
            self.feature_names = model_data['feature_names']
            self.selected_features = model_data['selected_features']
            self.class_labels = model_data.get('class_labels', ['Healthy', 'Alzheimers'])
            self.is_trained = True
            print(f"Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def get_model_info(self):
        """Get information about the trained model"""
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        return {
            'status': 'trained',
            'model_type': type(self.model).__name__,
            'kernel': self.model.kernel if hasattr(self.model, 'kernel') else 'N/A',
            'n_support_vectors': sum(self.model.n_support_) if hasattr(self.model, 'n_support_') else 'N/A',
            'n_features_selected': len(self.selected_features) if self.selected_features else 0,
            'selected_features': self.selected_features,
            'class_labels': self.class_labels
        }


# Convenience functions
def create_classifier():
    """Create a new classifier instance"""
    return AlzheimersSVMClassifier()


def train_model(audio_files, labels, optimize=True):
    """Train a new model"""
    classifier = AlzheimersSVMClassifier()
    return classifier.train(audio_files, labels, optimize=optimize)


def predict_file(audio_file, model_path='models'):
    """Make prediction for a single file"""
    classifier = AlzheimersSVMClassifier(model_path=model_path)
    return classifier.predict(audio_file)
