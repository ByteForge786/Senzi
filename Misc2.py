class AttributeAnalyzer:
    # ... (previous code remains the same) ...

    def predict_and_evaluate(self, unlabeled_data: pd.DataFrame, 
                           label_definitions: Dict[str, str],
                           original_labels: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Predict labels for unlabeled data and evaluate accuracy if original labels are available.
        
        Args:
            unlabeled_data: DataFrame with attributes to predict
            label_definitions: Dictionary of label definitions
            original_labels: Boolean indicating if original labels are present for comparison
            
        Returns:
            Tuple of (predictions DataFrame, metrics dictionary)
        """
        logger.info(f"Predicting labels for {len(unlabeled_data)} attributes")
        
        # Get cosine similarities and initial predictions
        predictions_df = self.analyze_cosine_similarities(unlabeled_data, label_definitions)
        
        # Prepare features for XGBoost
        X_pred, _ = self.prepare_xgboost_data(unlabeled_data, label_definitions, is_training=False)
        
        # Train XGBoost on the data
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100
        )
        model.fit(X_pred, _)
        
        # Get probabilities for each label
        label_probs = model.predict_proba(X_pred)
        
        # Add probability columns to predictions
        for i, label in enumerate(self.label_encoder.classes_):
            predictions_df[f'prob_{label}'] = label_probs[:, i]
        
        # Calculate metrics if original labels are present
        metrics = {}
        if original_labels and 'label' in unlabeled_data.columns:
            # Exact matches (case sensitive)
            exact_matches = (predictions_df['predicted_label'] == unlabeled_data['label']).sum()
            total_rows = len(unlabeled_data)
            
            # Case-insensitive matches
            case_insensitive_matches = (
                predictions_df['predicted_label'].str.lower() == 
                unlabeled_data['label'].str.lower()
            ).sum()
            
            # Handle hyphenated variations
            def normalize_label(label):
                return label.lower().replace('-', '').replace('_', '')
                
            hyphen_matches = (
                predictions_df['predicted_label'].apply(normalize_label) == 
                unlabeled_data['label'].apply(normalize_label)
            ).sum()
            
            metrics = {
                'total_rows': total_rows,
                'exact_matches': exact_matches,
                'exact_match_percentage': (exact_matches / total_rows) * 100,
                'case_insensitive_matches': case_insensitive_matches,
                'case_insensitive_percentage': (case_insensitive_matches / total_rows) * 100,
                'hyphen_aware_matches': hyphen_matches,
                'hyphen_aware_percentage': (hyphen_matches / total_rows) * 100
            }
            
            # Add original labels to the output
            predictions_df['original_label'] = unlabeled_data['label']
            
        # Sort probabilities from highest to lowest
        prob_cols = [col for col in predictions_df.columns if col.startswith('prob_')]
        predictions_df['confidence'] = predictions_df[prob_cols].max(axis=1)
        predictions_df = predictions_df.sort_values('confidence', ascending=False)
        
        return predictions_df, metrics

def main():
    """
    Main function to run the attribute analysis pipeline.
    """
    try:
        # ... (previous code remains the same until after the label_definitions) ...
        
        # Add prediction for unlabeled data
        logger.info("Predicting labels for unlabeled data...")
        unlabeled_data = pd.read_csv('unlabeled_data.csv')
        
        analyzer = AttributeAnalyzer()
        predictions_df, metrics = analyzer.predict_and_evaluate(
            unlabeled_data, 
            label_definitions,
            original_labels=True  # Set to True if original labels are available
        )
        
        # Save predictions
        predictions_df.to_csv('predictions_with_metrics.csv', index=False)
        
        # Log metrics
        logger.info("\nPrediction Metrics:")
        logger.info(f"Total Rows: {metrics['total_rows']}")
        logger.info(f"Exact Matches: {metrics['exact_matches']} ({metrics['exact_match_percentage']:.2f}%)")
        logger.info(f"Case-Insensitive Matches: {metrics['case_insensitive_matches']} ({metrics['case_insensitive_percentage']:.2f}%)")
        logger.info(f"Hyphen-Aware Matches: {metrics['hyphen_aware_matches']} ({metrics['hyphen_aware_percentage']:.2f}%)")
        
        # Save metrics to CSV
        pd.DataFrame([metrics]).to_csv('prediction_metrics.csv', index=False)
        
        return predictions_df, metrics
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        predictions_df, metrics = main()
        logger.info("Analysis and predictions completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
