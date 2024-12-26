import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
import xgboost as xgb
from typing import List, Dict

class TextProcessor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
    def chunk_and_embed(self, text: str, chunk_size: int = 512) -> np.ndarray:
        """
        Handle long text by chunking and concatenating embeddings.
        """
        words = text.split()
        chunks = []
        current_chunk = []
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
            
        # Get embeddings for all chunks
        chunk_embeddings = [self.model.encode(chunk) for chunk in chunks]
        
        # Concatenate embeddings
        return np.concatenate(chunk_embeddings, axis=0)

class AttributeAnalyzer:
    def __init__(self):
        self.text_processor = TextProcessor()
        
    def analyze_cosine_similarities(self, data: pd.DataFrame, label_definitions: Dict[str, str]) -> pd.DataFrame:
        """
        Compute cosine similarities for each attribute against each label definition.
        """
        results = []
        
        for idx, row in data.iterrows():
            desc = f"here is the description: {row['description']}"
            attr_name = f"the attribute name is: {row['attribute_name']}"
            desc_embedding = self.text_processor.chunk_and_embed(desc)
            attr_name_embedding = self.text_processor.chunk_and_embed(attr_name)
            
            similarities = {}
            for label, definition in label_definitions.items():
                label_text = f"the label is: {label}"
                def_text = f"the definition of label {label} is: {definition}"
                def_embedding = self.text_processor.chunk_and_embed(def_text)
                label_embedding = self.text_processor.chunk_and_embed(label_text)
                sim_score = cosine_similarity([desc_embedding], [def_embedding])[0][0]
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

    def prepare_xgboost_data(self, data: pd.DataFrame, label_definitions: Dict[str, str]) -> tuple:
        """
        Prepare data for XGBoost training with attribute description and label definitions as separate columns.
        """
        feature_names = ['attribute_name', 'attribute_description'] + [f'label_{label}' for label in label_definitions.keys()]
        features = pd.DataFrame(columns=feature_names)

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
                label_text = f"the label is: {label}"
                def_text = f"the definition of label {label} is: {definition}"
                def_emb = self.text_processor.chunk_and_embed(def_text)
                label_emb = self.text_processor.chunk_and_embed(label_text)
                row_features[f'label_{label}'] = def_emb
                
            features = features.append(pd.Series(row_features), ignore_index=True)
            
        labels = data['label'].values

        return features, labels

    def train_xgboost(self, X: pd.DataFrame, y: np.ndarray, cv_folds: int = 5) -> Dict:
        """
        Train XGBoost model with cross-validation and return predictions, probabilities, and classification report.
        """
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            eval_metric='mlogloss'
        )
        
        y_pred = cross_val_predict(model, X, y, cv=cv_folds)
        y_prob = cross_val_predict(model, X, y, cv=cv_folds, method='predict_proba')
        
        return {
            'predictions': y_pred,
            'probabilities': y_prob,
            'report': classification_report(y, y_pred, output_dict=True)
        }
    
    def get_label_probabilities(self, X: pd.DataFrame, y: np.ndarray, cv_folds: int = 5) -> pd.DataFrame:
        """
        Get the probability scores for each label given each attribute.
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
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    label_definitions_df = pd.read_csv('label_definitions.csv')
    label_definitions = dict(zip(label_definitions_df['label'], label_definitions_df['definition']))
    
    analyzer = AttributeAnalyzer()
    cosine_results = analyzer.analyze_cosine_similarities(train_data, label_definitions)
    X_train, y_train = analyzer.prepare_xgboost_data(train_data, label_definitions)
    X_test, y_test = analyzer.prepare_xgboost_data(test_data, label_definitions)
    
    xgboost_results = analyzer.train_xgboost(X_train, y_train)
    
    print("\nXGBoost Classification Report (Train):")
    print(classification_report(y_train, xgboost_results['predictions']))
    
    label_prob_df = analyzer.get_label_probabilities(X_train, y_train)
    print("\nLabel Probabilities:")
    print(label_prob_df)
    
    return cosine_results, xgboost_results, label_prob_df

if __name__ == "__main__":
    results, xgb_results, label_prob_df = main()
