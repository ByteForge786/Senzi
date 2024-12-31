import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from typing import Union, Dict, List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime
from tqdm import tqdm

class PredictionService:
    """Service for making predictions using saved XGBoost model."""
    
    def __init__(self, model_dir: str = "models", model_timestamp: Optional[str] = None):
        """
        Initialize prediction service.
        
        Args:
            model_dir: Directory containing saved models
            model_timestamp: Specific model timestamp to use. If None, uses latest model.
        """
        self.model_dir = model_dir
        self.setup_logging()
        
        # Load model and components
        self.load_model(model_timestamp)
        
    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def find_latest_model(self) -> str:
        """Find the most recent model timestamp in the models directory."""
        model_files = [f for f in os.listdir(self.model_dir) if f.startswith('xgboost_model_')]
        if not model_files:
            raise FileNotFoundError(f"No model files found in {self.model_dir}")
            
        # Extract timestamps and find latest
        timestamps = [f.replace('xgboost_model_', '').replace('.joblib', '') for f in model_files]
        return max(timestamps)
        
    def load_model(self, timestamp: Optional[str] = None):
        """Load model and all necessary components."""
        try:
            # Use provided timestamp or find latest
            self.timestamp = timestamp or self.find_latest_model()
            self.logger.info(f"Loading model with timestamp: {self.timestamp}")
            
            # Define component paths
            model_path = os.path.join(self.model_dir, f'xgboost_model_{self.timestamp}.joblib')
            encoder_path = os.path.join(self.model_dir, f'label_encoder_{self.timestamp}.joblib')
            text_processor_path = os.path.join(self.model_dir, f'text_processor_{self.timestamp}.joblib')
            definitions_path = os.path.join(self.model_dir, f'label_definitions_{self.timestamp}.joblib')
            
            # Load components
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            self.text_processor = joblib.load(text_processor_path)
            self.label_definitions = joblib.load(definitions_path)
            
            self.logger.info("Successfully loaded all model components")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
            
    def prepare_features(self, attribute_name: str, description: str) -> pd.DataFrame:
        """Prepare features for a single attribute."""
        try:
            # Combine attribute info with context
            text_with_context = f"Attribute name: {attribute_name} Description context: {description}"
            
            # Get embeddings
            combined_embedding = self.text_processor.chunk_and_embed(text_with_context)
            
            # Calculate similarities with label definitions
            similarities = {}
            for label, definition in self.label_definitions.items():
                def_text = f"Label {label} is defined as: {definition}"
                def_embedding = self.text_processor.chunk_and_embed(def_text)
                sim_score = cosine_similarity(combined_embedding, def_embedding)[0][0]
                similarities[f'similarity_{label}'] = sim_score
            
            # Create feature dictionary
            features = {}
            features.update({f'emb_{i}': val for i, val in enumerate(combined_embedding.flatten())})
            features.update(similarities)
            
            return pd.DataFrame([features])
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            raise
            
    def predict_single(self, attribute_name: str, description: str) -> Dict:
        """Make prediction for a single attribute."""
        try:
            # Prepare features
            features = self.prepare_features(attribute_name, description)
            
            # Create DMatrix and predict
            dtest = xgb.DMatrix(features)
            probabilities = self.model.predict(dtest)
            prediction = probabilities.argmax(axis=1)[0]
            
            # Get predicted label
            predicted_label = self.label_encoder.inverse_transform([prediction])[0]
            
            # Create result dictionary
            result = {
                'attribute_name': attribute_name,
                'description': description,
                'predicted_label': predicted_label,
            }
            
            # Add probabilities for each class
            for i, label in enumerate(self.label_encoder.classes_):
                result[f'probability_{label}'] = float(probabilities[0][i])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in single prediction: {str(e)}")
            raise
            
    def predict_batch(self, input_data: Union[pd.DataFrame, str], 
                     has_labels: bool = False,
                     output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Make predictions for multiple attributes.
        
        Args:
            input_data: DataFrame or path to CSV file
            has_labels: Whether input data includes true labels
            output_file: Optional path to save results
        """
        try:
            # Load data if CSV path provided
            if isinstance(input_data, str):
                input_data = pd.read_csv(input_data)
            
            self.logger.info(f"Processing {len(input_data)} attributes...")
            
            # Process each attribute
            results = []
            for idx, row in tqdm(input_data.iterrows(), total=len(input_data)):
                result = self.predict_single(row['attribute_name'], row['description'])
                if has_labels:
                    result['true_label'] = row['label']
                    result['correct'] = result['predicted_label'] == row['label']
                results.append(result)
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Add accuracy metrics if labels present
            if has_labels:
                accuracy = (results_df['correct']).mean()
                self.logger.info(f"Overall accuracy: {accuracy:.4f}")
            
            # Save results if output file specified
            if output_file:
                results_df.to_csv(output_file, index=False)
                self.logger.info(f"Results saved to {output_file}")
            
            return results_df
            
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize service (will use latest model by default)
    predictor = PredictionService()
    
    # Example 1: Single prediction
    result = predictor.predict_single(
        attribute_name="email_address",
        description="User's primary email for login"
    )
    print("\nSingle Prediction Result:")
    print(result)
    
    # Example 2: Batch prediction from CSV
    results_df = predictor.predict_batch(
        "test_attributes.csv",
        has_labels=True,  # Set to False for unlabeled data
        output_file="predictions_output.csv"
    )
    print("\nBatch Prediction Results:")
    print(results_df.head())
