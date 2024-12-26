def train_xgboost(self, X: pd.DataFrame, y: np.ndarray, 
                  cv_folds: int = 5,
                  params: Optional[Dict] = None) -> Dict:
    """
    Train XGBoost model with cross-validation and validation set for early stopping.
    """
    from sklearn.model_selection import train_test_split
    
    # Simple train/validation split for early stopping
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2,  # 20% validation set
        random_state=42,
        stratify=y
    )
    
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
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    # Get cross-validation predictions using the full dataset
    y_pred = cross_val_predict(model, X, y, cv=cv_folds)
    y_prob = cross_val_predict(model, X, y, cv=cv_folds, method='predict_proba')
    
    return {
        'predictions': y_pred,
        'probabilities': y_prob,
        'report': classification_report(y, y_pred, output_dict=True)
    }




class TextProcessor:
    def chunk_and_embed(self, text: str, chunk_size: int = 512) -> np.ndarray:
        """
        Handle long text by chunking and generating embeddings.
        Returns concatenated context-aware embedding.
        """
        if not text or chunk_size <= 0:
            raise ValueError("Text must not be empty and chunk_size must be positive")
            
        # Create overlapping chunks to maintain context
        words = text.split()
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0
        overlap = 50  # Number of words to overlap between chunks
        
        for i, word in enumerate(words):
            if current_length + len(word) + 1 > chunk_size:
                chunks.append(' '.join(current_chunk))
                # Keep last 'overlap' words for context
                current_chunk = words[max(0, i-overlap):i]
                current_length = sum(len(w) for w in current_chunk) + len(current_chunk)
            current_chunk.append(word)
            current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Process chunks with attention to context
        embeddings = []
        
        # Encode all chunks
        chunk_embeddings = self.model.encode(chunks)
        if len(chunks) == 1:
            chunk_embeddings = [chunk_embeddings]
            
        # Use attention mechanism to weight the importance of each chunk
        chunk_weights = np.ones(len(chunk_embeddings)) / len(chunk_embeddings)
        weighted_embedding = np.average(chunk_embeddings, axis=0, weights=chunk_weights)
        
        return weighted_embedding.reshape(1, -1)

class AttributeAnalyzer:
    def analyze_cosine_similarities(self, data: pd.DataFrame, 
                                  label_definitions: Dict[str, str]) -> pd.DataFrame:
        logger.info(f"Computing similarities for {len(data)} attributes")
        results = []
        
        for idx, row in data.iterrows():
            # Combine description and attribute name with context markers
            text_with_context = (
                f"Attribute name: {row['attribute_name']} "
                f"Description context: {row['description']}"
            )
            
            # Get context-aware embedding
            combined_embedding = self.text_processor.chunk_and_embed(text_with_context)
            
            similarities = {}
            for label, definition in label_definitions.items():
                def_text = f"Label: {label} Definition: {definition}"
                def_embedding = self._get_label_embedding(label, def_text)
                
                # Compute semantic similarity with attention to context
                attention_weights = self._compute_attention(combined_embedding, def_embedding)
                sim_score = self._compute_weighted_similarity(
                    combined_embedding, 
                    def_embedding, 
                    attention_weights
                )
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

    def _compute_attention(self, query_emb: np.ndarray, key_emb: np.ndarray) -> np.ndarray:
        """Compute attention weights between embeddings."""
        attention_scores = np.dot(query_emb, key_emb.T)
        return softmax(attention_scores, axis=-1)
    
    def _compute_weighted_similarity(self, 
                                   emb1: np.ndarray, 
                                   emb2: np.ndarray, 
                                   weights: np.ndarray) -> float:
        """Compute weighted cosine similarity with attention weights."""
        weighted_emb1 = emb1 * weights
        weighted_emb2 = emb2 * weights
        return cosine_similarity(weighted_emb1, weighted_emb2)[0][0]

    def _get_label_embedding(self, label: str, definition: str) -> np.ndarray:
        """Get cached context-aware embedding for label definition."""
        cache_key = f"{label}_{hash(definition)}"
        if cache_key not in self.label_embeddings_cache:
            self.label_embeddings_cache[cache_key] = self.text_processor.chunk_and_embed(definition)
        return self.label_embeddings_cache[cache_key]






class LabelDefinitionParser:
    """Parser for unstructured label definitions from text file."""
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize text content while preserving important whitespace."""
        # Remove leading/trailing whitespace while preserving internal structure
        lines = text.strip().split('\n')
        # Remove empty lines from start and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        return '\n'.join(lines)

    @staticmethod
    def _standardize_label(label: str) -> str:
        """Convert label header to standard format."""
        # Remove trailing colon and convert to lowercase with underscores
        label = label.strip().rstrip(':').lower()
        # Replace spaces with underscores and remove any special characters
        label = re.sub(r'[^a-z0-9\s]', '', label)
        label = re.sub(r'\s+', '_', label)
        return label

    @staticmethod
    def parse_label_definitions(filepath: str) -> Dict[str, str]:
        """
        Parse label definitions from a text file with sections.
        
        Format expected:
        Sensitive PII:
        [multi-line definition]
        
        Confidential information:
        [multi-line definition]
        ...
        
        Args:
            filepath: Path to the text file containing label definitions
            
        Returns:
            Dictionary mapping standardized label names to their full definitions
            
        Raises:
            ValueError: If no valid definitions found or parsing fails
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Known section headers to look for
            known_headers = [
                "Sensitive PII:",
                "Confidential information:",
                "Non sensitive PII:",
                "Licensed data:"
            ]
            
            # Create pattern to match any of the known headers
            header_pattern = '|'.join(map(re.escape, known_headers))
            
            # Split content at section headers
            sections = re.split(f'({header_pattern})', content)
            
            # Clean up sections and pair headers with content
            definitions = {}
            current_header = None
            
            for section in sections:
                if not section.strip():
                    continue
                    
                # Check if this is a header
                if section.strip() in known_headers:
                    current_header = section.strip()
                elif current_header:
                    # This is content for the current header
                    cleaned_content = LabelDefinitionParser._clean_text(section)
                    if cleaned_content:
                        label_key = LabelDefinitionParser._standardize_label(current_header)
                        definitions[label_key] = cleaned_content
                        current_header = None
            
            if not definitions:
                raise ValueError("No valid label definitions found in file")
            
            logger.info(f"Successfully parsed {len(definitions)} label definitions")
            for label in definitions.keys():
                logger.info(f"Found definition for: {label}")
                
            return definitions
            
        except Exception as e:
            logger.error(f"Error parsing label definitions: {str(e)}")
            raise

def test_parser():
    """Test function to demonstrate parser usage."""
    parser = LabelDefinitionParser()
    try:
        definitions = parser.parse_label_definitions('label_definitions.txt')
        
        print("\nParsed Definitions:")
        for label, definition in definitions.items():
            print(f"\n{'-'*50}")
            print(f"Label: {label}")
            print(f"{'-'*50}")
            print(f"Definition:\n{definition}\n")
            
        return definitions
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    test_parser()
