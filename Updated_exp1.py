def save_trained_model(self, X_train: pd.DataFrame, y_train: np.ndarray,
                          label_definitions: Dict[str, str]) -> Dict[str, str]:
    """
    Train and save the best XGBoost model with all components needed for prediction.
    Uses cross-validation to find the best model.
    
    Returns:
        Dictionary with paths to saved model files
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save paths
        paths = {
            'model': os.path.join(self.model_dir, f'xgboost_model_{timestamp}.joblib'),
            'encoder': os.path.join(self.model_dir, f'label_encoder_{timestamp}.joblib'),
            'text_processor': os.path.join(self.model_dir, f'text_processor_{timestamp}.joblib'),
            'definitions': os.path.join(self.model_dir, f'label_definitions_{timestamp}.joblib')
        }
        
        self.logger.info("Training XGBoost model with cross-validation...")
        
        # Define model parameters
        params = {
            'objective': 'multi:softprob',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'tree_method': 'hist',  # For faster training
            'eval_metric': 'mlogloss',
            'early_stopping_rounds': 10,
            'n_jobs': -1  # Use all CPU cores
        }
        
        # Initialize models for cross-validation
        n_folds = 5
        models = []
        best_score = float('inf')
        best_model = None
        
        # Perform cross-validation and keep track of best model
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            self.logger.info(f"Training fold {fold}/{n_folds}")
            
            # Create DMatrix for faster training
            dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
            dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
            
            # Train model
            model = xgb.train(
                params,
                dtrain,
                evals=[(dval, 'val')],
                early_stopping_rounds=10,
                verbose_eval=False
            )
            
            # Get validation score
            val_pred = model.predict(dval)
            val_score = log_loss(y_fold_val, val_pred)
            
            self.logger.info(f"Fold {fold} validation loss: {val_score:.4f}")
            
            if val_score < best_score:
                best_score = val_score
                best_model = model
                self.logger.info(f"New best model found with validation loss: {val_score:.4f}")
        
        # Train final model on full dataset using best model's parameters
        self.logger.info("Training final model on full dataset...")
        dtrain_full = xgb.DMatrix(X_train, label=y_train)
        final_model = xgb.train(
            params,
            dtrain_full,
            num_boost_round=best_model.best_ntree_limit
        )
        
        # Save all components
        joblib.dump(final_model, paths['model'])
        joblib.dump(self.analyzer.label_encoder, paths['encoder'])
        joblib.dump(self.analyzer.text_processor, paths['text_processor'])
        joblib.dump(label_definitions, paths['definitions'])
        
        self.logger.info(f"Best model saved successfully in {self.model_dir}")
        self.logger.info(f"Best validation log loss: {best_score:.4f}")
        
        return paths
        
    except Exception as e:
        self.logger.error(f"Error saving model: {str(e)}")
        raise



import joblib
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import logging
import os
from datetime import datetime

# Enhanced logging setup
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Set up enhanced logging with file output."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"attribute_analysis_{timestamp}.log")
    
    logger = logging.getLogger("AttributeAnalyzer")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

class ModelTrainer:
    """Handles model training, saving, and prediction."""
    
    def __init__(self, analyzer: AttributeAnalyzer, model_dir: str = "models"):
        """Initialize with analyzer instance and model directory."""
        self.analyzer = analyzer
        self.model_dir = model_dir
        self.logger = setup_logging()
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
    def save_trained_model(self, X_train: pd.DataFrame, y_train: np.ndarray,
                          label_definitions: Dict[str, str]) -> Dict[str, str]:
        """
        Train and save XGBoost model with all components needed for prediction.
        
        Returns:
            Dictionary with paths to saved model files
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save paths
            paths = {
                'model': os.path.join(self.model_dir, f'xgboost_model_{timestamp}.joblib'),
                'encoder': os.path.join(self.model_dir, f'label_encoder_{timestamp}.joblib'),
                'text_processor': os.path.join(self.model_dir, f'text_processor_{timestamp}.joblib'),
                'definitions': os.path.join(self.model_dir, f'label_definitions_{timestamp}.joblib')
            }
            
            self.logger.info("Training final XGBoost model...")
            model = xgb.XGBClassifier(
                objective='multi:softprob',
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                tree_method='hist',  # For faster training
                n_jobs=-1  # Use all CPU cores
            )
            
            # Create callback for monitoring
            class ProgressCallback(xgb.callback.TrainingCallback):
                def after_iteration(self, model, epoch, evals_log):
                    if epoch % 10 == 0:
                        self.logger.info(f"Training progress: {epoch} iterations completed")
                    return False
            
            # Train model with progress monitoring
            with tqdm(total=100, desc="Training XGBoost") as pbar:
                model.fit(
                    X_train, y_train,
                    callbacks=[ProgressCallback()],
                    verbose=False
                )
                pbar.update(100)
            
            # Save all components
            joblib.dump(model, paths['model'])
            joblib.dump(self.analyzer.label_encoder, paths['encoder'])
            joblib.dump(self.analyzer.text_processor, paths['text_processor'])
            joblib.dump(label_definitions, paths['definitions'])
            
            self.logger.info(f"Model and components saved successfully in {self.model_dir}")
            return paths
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
            
    def predict_attributes(self, 
                          input_data: Union[pd.DataFrame, Tuple[str, str]], 
                          model_timestamp: str,
                          has_labels: bool = False,
                          batch_size: int = 1000) -> pd.DataFrame:
        """
        Make predictions for either a single attribute or entire DataFrame.
        
        Args:
            input_data: Either DataFrame with columns [attribute_name, description] 
                       or tuple of (attribute_name, description)
            model_timestamp: Timestamp of model to use (format: YYYYMMDD_HHMMSS)
            has_labels: Whether input data includes true labels
            batch_size: Batch size for processing large datasets
        
        Returns:
            DataFrame with predictions and probabilities
        """
        try:
            # Load saved model components
            model_paths = {
                'model': os.path.join(self.model_dir, f'xgboost_model_{model_timestamp}.joblib'),
                'encoder': os.path.join(self.model_dir, f'label_encoder_{model_timestamp}.joblib'),
                'definitions': os.path.join(self.model_dir, f'label_definitions_{model_timestamp}.joblib')
            }
            
            model = joblib.load(model_paths['model'])
            label_encoder = joblib.load(model_paths['encoder'])
            label_definitions = joblib.load(model_paths['definitions'])
            
            # Convert single attribute to DataFrame if needed
            if isinstance(input_data, tuple):
                input_data = pd.DataFrame({
                    'attribute_name': [input_data[0]],
                    'description': [input_data[1]]
                })
            
            # Prepare features in batches
            all_features = []
            total_batches = len(input_data) // batch_size + (1 if len(input_data) % batch_size else 0)
            
            with ThreadPoolExecutor() as executor:
                for i in tqdm(range(0, len(input_data), batch_size), 
                            desc="Processing batches", 
                            total=total_batches):
                    batch = input_data.iloc[i:i+batch_size]
                    batch_features, _ = self.analyzer.prepare_xgboost_data(
                        batch, label_definitions, is_training=False
                    )
                    all_features.append(batch_features)
            
            # Combine all features
            X = pd.concat(all_features, axis=0)
            
            # Make predictions
            self.logger.info("Making predictions...")
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'attribute_name': input_data['attribute_name'],
                'description': input_data['description'],
                'predicted_label': label_encoder.inverse_transform(y_pred)
            })
            
            # Add probability columns for each label
            for i, label in enumerate(label_encoder.classes_):
                results[f'probability_{label}'] = y_prob[:, i]
            
            # Add accuracy metrics if labels are present
            if has_labels and 'label' in input_data.columns:
                results['true_label'] = input_data['label']
                results['correct'] = results['predicted_label'] == results['true_label']
                accuracy = accuracy_score(results['true_label'], results['predicted_label'])
                self.logger.info(f"Overall accuracy: {accuracy:.4f}")
                
                # Add classification report
                report = classification_report(
                    results['true_label'], 
                    results['predicted_label'],
                    output_dict=True
                )
                results.attrs['classification_report'] = report
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise

def save_predictions(results: pd.DataFrame, 
                    output_file: str = None,
                    include_probabilities: bool = True) -> None:
    """Save prediction results to CSV with optional probability scores."""
    try:
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"predictions_{timestamp}.csv"
        
        # Select columns to save
        cols = ['attribute_name', 'description', 'predicted_label']
        if 'true_label' in results.columns:
            cols.extend(['true_label', 'correct'])
        if include_probabilities:
            prob_cols = [col for col in results.columns if col.startswith('probability_')]
            cols.extend(prob_cols)
            
        results[cols].to_csv(output_file, index=False)
        
        # Save classification report if available
        if 'classification_report' in results.attrs:
            report_file = output_file.replace('.csv', '_report.csv')
            pd.DataFrame(results.attrs['classification_report']).to_csv(report_file)
            
    except Exception as e:
        logging.error(f"Error saving predictions: {str(e)}")
        raise

# Example usage:
"""
# After training main pipeline:
trainer = ModelTrainer(analyzer)
model_paths = trainer.save_trained_model(X_train, y_train, label_definitions)
timestamp = model_paths['model'].split('_')[-1].split('.')[0]

# For single prediction:
prediction = trainer.predict_attributes(
    ("email_address", "User's email address for login"),
    timestamp
)

# For batch prediction with labeled data:
test_predictions = trainer.predict_attributes(
    test_data,
    timestamp,
    has_labels=True
)

# Save results:
save_predictions(test_predictions, "test_predictions.csv")
"""
def main():
    """
    Enhanced main function that includes model saving and prediction capabilities.
    """
    logger = setup_logging()
    
    try:
        logger.info("Starting attribute analysis pipeline...")
        
        # Load and validate input data
        train_data = pd.read_csv('train.csv')
        test_data = pd.read_csv('test.csv')
        
        required_columns = {'attribute_name', 'description', 'label'}
        if not required_columns.issubset(train_data.columns):
            raise ValueError(f"Training data missing required columns: {required_columns}")
            
        # Parse label definitions
        logger.info("Parsing label definitions...")
        label_parser = LabelDefinitionParser()
        label_definitions = label_parser.parse_label_definitions('label_definitions.txt')
        
        # Initialize analyzer and process data
        analyzer = AttributeAnalyzer()
        
        logger.info("Computing cosine similarities...")
        cosine_results = analyzer.analyze_cosine_similarities(train_data, label_definitions)
        
        logger.info("Preparing XGBoost data...")
        X_train, y_train = analyzer.prepare_xgboost_data(train_data, label_definitions, is_training=True)
        X_test, y_test = analyzer.prepare_xgboost_data(test_data, label_definitions, is_training=False)
        
        logger.info("Training XGBoost model...")
        xgboost_results = analyzer.train_xgboost(X_train, y_train)
        
        # Print initial classification report
        y_pred = xgboost_results['predictions']
        y_pred_labels = analyzer.label_encoder.inverse_transform(y_pred)
        y_train_labels = analyzer.label_encoder.inverse_transform(y_train)
        
        logger.info("\nInitial XGBoost Classification Report (Train):")
        print(classification_report(y_train_labels, y_pred_labels))
        
        # Save model and make predictions using enhanced functionality
        logger.info("Saving trained model...")
        trainer = ModelTrainer(analyzer)
        model_paths = trainer.save_trained_model(X_train, y_train, label_definitions)
        timestamp = model_paths['model'].split('_')[-1].split('.')[0]
        
        # Make predictions on test data
        logger.info("Making predictions on test data...")
        test_predictions = trainer.predict_attributes(
            test_data,
            timestamp,
            has_labels=True
        )
        
        # Save all results
        logger.info("Saving results...")
        cosine_results.to_csv('cosine_similarities.csv', index=False)
        save_predictions(test_predictions, f'predictions_{timestamp}.csv')
        
        # Save label mapping for reference
        pd.DataFrame({'label': analyzer.label_encoder.classes_}).to_csv('label_mapping.csv', index=False)
        
        logger.info("Pipeline completed successfully!")
        return cosine_results, xgboost_results, test_predictions
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        results, xgb_results, predictions = main()
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise
