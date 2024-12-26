import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
import xgboost as xgb
from typing import List, Dict, Optional, Tuple
import logging
import re

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
        Parse label definitions from a text file where each definition
        can span multiple lines.
        
        Format expected:
        label_name=definition text spanning
        multiple lines...
        another_label=another definition
        spanning lines...
        
        Args:
            filepath: Path to the text file containing label definitions
            
        Returns:
            Dictionary mapping label names to their full definitions
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split on pattern that matches label=definition
            label_pattern = r'([a-zA-Z_]+)=((?:(?!\n[a-zA-Z_]+=).)*)'
            matches = re.finditer(label_pattern, content, re.DOTALL)
            
            definitions = {}
            for match in matches:
                label = match.group(1).strip()
                definition = match.group(2).strip()
                # Clean up any excessive whitespace/newlines
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
        """
        Initialize with specified sentence transformer model.
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
            
    def chunk_and_embed(self, text: str, chunk_size: int = 512) -> np.ndarray:
        """
        Handle long text by chunking and generating embeddings.
        
        Args:
            text: Input text to be chunked and embedded
            chunk_size: Maximum size of each chunk
            
        Returns:
            np.ndarray: Concatenated embeddings of all chunks
            
        Raises:
            ValueError: If text is empty or chunk_size <= 0
        """
        if not text or chunk_size <= 0:
            raise ValueError("Text must not be empty and chunk_size must be positive")
            
        # Split text into chunks
        words = text.split()
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Process chunks in batches for better performance
        batch_size = 8
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_embeddings = self.model.encode(batch)
            if len(batch) == 1:
                batch_embeddings = [batch_embeddings]
            all_embeddings.extend(batch_embeddings)
            
        return np.concatenate(all_embeddings, axis=0)

class AttributeAnalyzer:
    """Main class for analyzing attributes and making predictions."""
    
    def __init__(self):
        """Initialize with text processor and empty cache."""
        self.text_processor = TextProcessor()
        self.label_embeddings_cache: Dict[str, np.ndarray] = {}
        
    def _get_label_embedding(self, label: str, definition: str) -> np.ndarray:
        """
        Get cached embedding for label definition or compute if not cached.
        
        Args:
            label: Label name
            definition: Label definition text
            
        Returns:
            np.ndarray: Embedding for the label definition
        """
        cache_key = f"{label}_{hash(definition)}"
        if cache_key not in self.label_embeddings_cache:
            def_text = f"the definition of label {label} is: {definition}"
            def_embedding = self.text_processor.chunk_and_embed(def_text)
            self.label_embeddings_cache[cache_key] = def_embedding
        return self.label_embeddings_cache[cache_key]
        
    def analyze_cosine_similarities(self, data: pd.DataFrame, 
                                  label_definitions: Dict[str, str]) -> pd.DataFrame:
        """
        Compute cosine similarities between attributes and label definitions.
        
        Args:
            data: DataFrame with attribute_name, description, and label columns
            label_definitions: Dictionary mapping labels to their definitions
            
        Returns:
            DataFrame with similarity scores and predictions
        """
        logger.info(f"Computing similarities for {len(data)} attributes")
        results = []
        
        for idx, row in data.iterrows():
            desc = f"here is the description: {row['description']}"
            attr_name = f"the attribute name is: {row['attribute_name']}"
            
            desc_embedding = self.text_processor.chunk_and_embed(desc)
            attr_name_embedding = self.text_processor.chunk_and_embed(attr_name)
            
            similarities = {}
            for label, definition in label_definitions.items():
                def_embedding = self._get_label_embedding(label, definition)
                sim_score = cosine_similarity([desc_embedding], [def_embedding])[0][0]
                similarities[f'similarity_{label}'] = sim_score
            
            result = {
                'attribute_name': row['attribute_name'],
                'description': row['description'],
                'original_label': row['label'],
                **similarities
            }
            
            # Get predicted label based on highest similarity
            sim_scores = {k.replace('similarity_', ''): v for k, v in similarities.items()}
            result['predicted_label'] = max(sim_scores.items(), key=lambda x: x[1])[0]
            
            results.append(result)
            
        return pd.DataFrame(results)
    
    def prepare_xgboost_data(self, data: pd.DataFrame, 
                            label_definitions: Dict[str, str]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare data for XGBoost training.
        
        Args:
            data: Input DataFrame
            label_definitions: Dictionary of label definitions
            
        Returns:
            Tuple of (feature DataFrame, label array)
        """
        feature_names = ['attribute_name', 'attribute_description'] + [f'label_{label}' for label in label_definitions.keys()]
        features = []
        
        for idx, row in data.iterrows():
            desc = f"here is the description: {row['description']}"
            attr_name = f"the attribute name is: {row['attribute_name']}"
            
            desc_embedding = self.text_processor.chunk_and_embed(desc)
            attr_name_embedding = self.text_processor.chunk_and_embed(attr_name)
            
            row_features = {
                'attribute_name': attr_name_embedding,
                'attribute_description': desc_embedding
            }
            
            for label, definition in label_definitions.items():
                def_embedding = self._get_label_embedding(label, definition)
                row_features[f'label_{label}'] = def_embedding
                
            features.append(row_features)
            
        features_df = pd.DataFrame(features)
        labels = data['label'].values
        
        return features_df, labels

    def train_xgboost(self, X: pd.DataFrame, y: np.ndarray, 
                      cv_folds: int = 5,
                      params: Optional[Dict] = None) -> Dict:
        """
        Train XGBoost model with cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Label array
            cv_folds: Number of cross-validation folds
            params: Optional XGBoost parameters
            
        Returns:
            Dictionary containing predictions, probabilities, and classification report
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
        """
        Get probability scores for each label.
        
        Args:
            X: Feature DataFrame
            y: Label array
            cv_folds: Number of cross-validation folds
            
        Returns:
            DataFrame with probability scores for each label
        """
        xgboost_results = self.train_xgboost(X, y, cv_folds)
        y_prob = xgboost_results['probabilities']
        
        label_columns = [col for col in X.columns if col.startswith('label_')]
        labels = [col.replace('label_', '') for col in label_columns]
        
        label_prob_df = pd.DataFrame(y_prob, columns=labels)
        label_prob_df['attribute_name'] = X['attribute_name'].apply(lambda x: np.array2string(x, separator=' '))
        label_prob_df['attribute_description'] = X['attribute_description'].apply(lambda x: np.array2string(x, separator=' '))
        
        return label_prob_df

def main():
    """
    Main function to run the attribute analysis pipeline.
    
    Returns:
        Tuple containing cosine results, XGBoost results, and label probabilities
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
        X_train, y_train = analyzer.prepare_xgboost_data(train_data, label_definitions)
        X_test, y_test = analyzer.prepare_xgboost_data(test_data, label_definitions)
        
        logger.info("Training XGBoost model...")
        xgboost_results = analyzer.train_xgboost(X_train, y_train)
        
        logger.info("\nXGBoost Classification Report (Train):")
        print(classification_report(y_train, xgboost_results['predictions']))
        
        logger.info("Computing label probabilities...")
        label_prob_df = analyzer.get_label_probabilities(X_train, y_train)
        
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
