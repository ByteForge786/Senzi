import joblib
import pandas as pd
import numpy as np
from typing import Dict, Tuple

def save_trained_model(analyzer, X_train: pd.DataFrame, y_train: np.ndarray, 
                      model_path: str = 'xgboost_model.joblib',
                      encoder_path: str = 'label_encoder.joblib') -> None:
    """
    Train and save the XGBoost model and label encoder.
    
    Args:
        analyzer: Trained AttributeAnalyzer instance
        X_train: Training features
        y_train: Training labels
        model_path: Path to save XGBoost model
        encoder_path: Path to save label encoder
    """
    # Train final model (not cross-validated)
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100
    )
    
    # Fit the model on all training data
    model.fit(X_train, y_train)
    
    # Save the model and label encoder
    joblib.dump(model, model_path)
    joblib.dump(analyzer.label_encoder, encoder_path)
    
def predict_attribute(attribute_name: str, description: str, 
                     label_definitions: Dict[str, str],
                     model_path: str = 'xgboost_model.joblib',
                     encoder_path: str = 'label_encoder.joblib') -> Dict[str, float]:
    """
    Make prediction for a single attribute using saved model.
    
    Args:
        attribute_name: Name of the attribute
        description: Description of the attribute
        label_definitions: Dictionary of label definitions
        model_path: Path to saved XGBoost model
        encoder_path: Path to saved label encoder
        
    Returns:
        Dictionary with predicted label and probabilities for each class
    """
    # Load saved model and encoder
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    
    # Initialize text processor
    text_processor = TextProcessor()
    
    # Prepare input data
    text_with_context = f"Attribute name: {attribute_name} Description context: {description}"
    combined_embedding = text_processor.chunk_and_embed(text_with_context)
    
    # Calculate similarities with label definitions
    similarities = {}
    for label, definition in label_definitions.items():
        def_text = f"Label {label} is defined as: {definition}"
        def_embedding = text_processor.chunk_and_embed(def_text)
        sim_score = cosine_similarity(combined_embedding, def_embedding.reshape(1, -1))[0][0]
        similarities[f'similarity_{label}'] = sim_score
    
    # Create feature DataFrame
    features = {}
    features.update({f'emb_{i}': val for i, val in enumerate(combined_embedding.flatten())})
    features.update(similarities)
    
    X = pd.DataFrame([features])
    
    # Get predictions and probabilities
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[0]
    
    # Create results dictionary
    predicted_label = label_encoder.inverse_transform(y_pred)[0]
    class_probabilities = {
        label: float(prob) 
        for label, prob in zip(label_encoder.classes_, y_prob)
    }
    
    return {
        'predicted_label': predicted_label,
        'probabilities': class_probabilities
    }

# Example usage:
"""
# After training, save the model
save_trained_model(analyzer, X_train, y_train)

# Later, make predictions
label_definitions = {
    'sensitive_pii': 'Definition of sensitive PII...',
    'confidential': 'Definition of confidential...',
    # ... other label definitions
}

prediction = predict_attribute(
    attribute_name="email_address",
    description="User's email address for login",
    label_definitions=label_definitions
)

print(f"Predicted label: {prediction['predicted_label']}")
print("\nProbabilities:")
for label, prob in prediction['probabilities'].items():
    print(f"{label}: {prob:.3f}")
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from typing import List, Dict, Optional, Tuple
import logging
import re
from scipy.special import softmax

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LabelDefinitionParser:
    """Parser for label definitions stored in text file."""
    
    @staticmethod
    def parse_label_definitions(filepath: str) -> Dict[str, str]:
        """
        Parse label definitions from a text file with sections.
        
        Expected format:
        Sensitive PII:
        <definition>
        
        Confidential information:
        <definition>
        
        Non sensitive PII:
        <definition>
        
        Licensed data:
        <definition>
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Define the expected labels and their text markers
            label_markers = {
                'sensitive_pii': 'Sensitive PII:',
                'confidential': 'Confidential information:',
                'non_sensitive_pii': 'Non sensitive PII:',
                'licensed': 'Licensed data:'
            }
            
            definitions = {}
            
            # Extract content between markers
            for label, marker in label_markers.items():
                pattern = f"{marker}(.*?)(?={list(label_markers.values())[0]}|$)"
                matches = re.findall(pattern, content, re.DOTALL)
                if matches:
                    # Clean up the definition text
                    definition = matches[0].strip()
                    # Normalize whitespace and clean up
                    definition = ' '.join(definition.split())
                    definitions[label] = definition
            
            if not definitions:
                raise ValueError("No valid label definitions found in file")
                
            return definitions
            
        except Exception as e:
            logger.error(f"Error parsing label definitions: {str(e)}")
            raise

class TextProcessor:
    """Handles text processing and embedding generation."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize with specified sentence transformer model."""
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
            
    def chunk_and_embed(self, text: str, chunk_size: int = 512) -> np.ndarray:
        """
        Handle long text by chunking and generating context-aware embeddings.
        """
        if not text or chunk_size <= 0:
            raise ValueError("Text must not be empty and chunk_size must be positive")
        
        # Create overlapping chunks to maintain context
        words = text.split()
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0
        overlap = 50  # Words to overlap between chunks
        
        for i, word in enumerate(words):
            if current_length + len(word) + 1 > chunk_size:
                chunks.append(' '.join(current_chunk))
                # Keep overlap words for context
                current_chunk = words[max(0, i-overlap):i]
                current_length = sum(len(w) for w in current_chunk) + len(current_chunk)
            current_chunk.append(word)
            current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Get embeddings for chunks
        chunk_embeddings = self.model.encode(chunks)
        if len(chunks) == 1:
            return chunk_embeddings.reshape(1, -1)
            
        # Weight chunks by position (later chunks get slightly higher weight)
        chunk_weights = np.linspace(0.9, 1.0, len(chunk_embeddings))
        chunk_weights = chunk_weights / chunk_weights.sum()
        
        # Compute weighted average of chunk embeddings
        weighted_embedding = np.average(chunk_embeddings, axis=0, weights=chunk_weights)
        return weighted_embedding.reshape(1, -1)

class AttributeAnalyzer:
    """Main class for analyzing attributes and making predictions."""
    
    def __init__(self):
        """Initialize with text processor, empty cache, and label encoder."""
        self.text_processor = TextProcessor()
        self.label_embeddings_cache: Dict[str, np.ndarray] = {}
        self.label_encoder = LabelEncoder()
    
    def _get_label_embedding(self, label: str, definition: str) -> np.ndarray:
        """Get cached embedding for label definition or compute if not cached."""
        cache_key = f"{label}_{hash(definition)}"
        if cache_key not in self.label_embeddings_cache:
            def_text = f"Label {label} is defined as: {definition}"
            self.label_embeddings_cache[cache_key] = self.text_processor.chunk_and_embed(def_text)
        return self.label_embeddings_cache[cache_key]
        
    def analyze_cosine_similarities(self, data: pd.DataFrame, 
                                  label_definitions: Dict[str, str]) -> pd.DataFrame:
        """
        Compute cosine similarities between attributes and label definitions.
        """
        logger.info(f"Computing similarities for {len(data)} attributes")
        results = []
        
        for idx, row in data.iterrows():
            # Combine attribute info with context
            text_with_context = (
                f"Attribute name: {row['attribute_name']} "
                f"Description context: {row['description']}"
            )
            
            # Get context-aware embedding
            combined_embedding = self.text_processor.chunk_and_embed(text_with_context)
            
            similarities = {}
            for label, definition in label_definitions.items():
                def_embedding = self._get_label_embedding(label, definition)
                sim_score = cosine_similarity(combined_embedding, def_embedding)[0][0]
                similarities[f'similarity_{label}'] = sim_score
            
            result = {
                'attribute_name': row['attribute_name'],
                'description': row['description'],
                'original_label': row['label'],
                **similarities
            }
            
            sim_scores = {k.replace('similarity_', ''): v for k, v in similarities.items()}
            result['predicted_label'] = max(sim_scores.items(), key=lambda x: x[1])[0]
            
            results.append(result)
            
        return pd.DataFrame(results)
    
    def prepare_xgboost_data(self, data: pd.DataFrame, 
                            label_definitions: Dict[str, str],
                            is_training: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare data for XGBoost training with encoded labels.
        """
        features = []
        
        for idx, row in data.iterrows():
            text_with_context = (
                f"Attribute name: {row['attribute_name']} "
                f"Description context: {row['description']}"
            )
            
            combined_embedding = self.text_processor.chunk_and_embed(text_with_context)
            
            feature_dict = {
                'text_embedding': combined_embedding.flatten()
            }
            
            for label, definition in label_definitions.items():
                def_embedding = self._get_label_embedding(label, definition)
                sim_score = cosine_similarity(combined_embedding, def_embedding)[0][0]
                feature_dict[f'similarity_{label}'] = sim_score
            
            features.append(feature_dict)
        
        # Create features DataFrame
        features_df = pd.DataFrame(features)
        
        # Process text embeddings
        embedding_cols = features_df['text_embedding'].apply(pd.Series)
        embedding_cols.columns = [f'emb_{i}' for i in range(embedding_cols.shape[1])]
        
        # Combine with similarity features
        similarity_cols = [col for col in features_df.columns if col.startswith('similarity_')]
        features_df = pd.concat([embedding_cols, features_df[similarity_cols]], axis=1)
        
        # Handle labels
        if is_training:
            labels = self.label_encoder.fit_transform(data['label'].values)
        else:
            labels = self.label_encoder.transform(data['label'].values)
            
        return features_df, labels

    def train_xgboost(self, X: pd.DataFrame, y: np.ndarray, 
                      cv_folds: int = 5,
                      params: Optional[Dict] = None) -> Dict:
        """
        Train XGBoost model with cross-validation.
        """
        default_params = {
            'objective': 'multi:softprob',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'eval_metric': 'mlogloss',
            'early_stopping_rounds': 10
        }
        
        if params:
            default_params.update(params)
            
        model = xgb.XGBClassifier(**default_params)
        
        y_pred = cross_val_predict(model, X, y, cv=cv_folds)
        y_prob = cross_val_predict(model, X, y, cv=cv_folds, method='predict_proba')
        
        return {
            'predictions': y_pred,
            'probabilities': y_prob,
            'report': classification_report(y, y_pred, output_dict=True)
        }
    
    def get_label_probabilities(self, X: pd.DataFrame, y: np.ndarray, 
                              cv_folds: int = 5) -> pd.DataFrame:
        """Get probability scores for each label."""
        xgboost_results = self.train_xgboost(X, y, cv_folds)
        y_prob = xgboost_results['probabilities']
        
        # Use original label names for columns
        label_prob_df = pd.DataFrame(y_prob, columns=self.label_encoder.classes_)
        
        return label_prob_df

def main():
    """
    Main function to run the attribute analysis pipeline.
    """
    try:
        # Load and validate input data
        train_data = pd.read_csv('train.csv')
        test_data = pd.read_csv('test.csv')
        
        required_columns = {'attribute_name', 'description', 'label'}
        if not required_columns.issubset(train_data.columns):
            raise ValueError(f"Training data missing required columns: {required_columns}")
            
        # Parse label definitions from text file
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
        
        # Print classification report with original label names
        y_pred = xgboost_results['predictions']
        y_pred_labels = analyzer.label_encoder.inverse_transform(y_pred)
        y_train_labels = analyzer.label_encoder.inverse_transform(y_train)
        
        logger.info("\nXGBoost Classification Report (Train):")
        print(classification_report(y_train_labels, y_pred_labels))
        
        logger.info("Computing label probabilities...")
        label_prob_df = analyzer.get_label_probabilities(X_train, y_train)
        
        # Save results
        cosine_results.to_csv('cosine_similarities.csv', index=False)
        label_prob_df.to_csv('label_probabilities.csv', index=False)
        
        # Save encoder classes for reference
        pd.DataFrame({'label': analyzer.label_encoder.classes_}).to_csv('label_mapping.csv', index=False)
        
        return cosine_results, xgboost_results, label_prob_df
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        results, xgb_results, label_prob_df = main()
        logger.info("Analysis completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
