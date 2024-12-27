import pandas as pd
import numpy as np

def process_csv(input_file, testing_output='testing.csv', undescribe_output='undescribe.csv'):
    """
    Process CSV file to handle descriptions and attribute names according to requirements.
    Preserves all original columns while processing specific ones.
    
    Args:
        input_file (str): Path to input CSV file
        testing_output (str): Path for output file with valid descriptions
        undescribe_output (str): Path for output file with null descriptions
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Store original columns
        original_columns = df.columns.tolist()
        
        # Create temporary columns for processing
        df['temp_description'] = df['description'].fillna('')
        df['temp_additional'] = df['additional_context'].fillna('')
        
        # Combine descriptions
        df['combined_description'] = df['temp_description'] + df['temp_additional']
        df['combined_description'] = df['combined_description'].replace('', np.nan)
        
        # Modify attribute name to include description
        df['attribute_name'] = df['temp_description'] + df['attribute_name']
        
        # Split into two dataframes based on description availability
        df_with_desc = df[df['combined_description'].notna()].copy()
        df_without_desc = df[df['combined_description'].isna()].copy()
        
        # Clean up temporary columns and rename final description
        for df_temp in [df_with_desc, df_without_desc]:
            # Remove temporary columns
            df_temp.drop(['temp_description', 'temp_additional'], axis=1, inplace=True)
            # Rename combined description to description
            df_temp.rename(columns={'combined_description': 'description'}, inplace=True)
            
            # Ensure original column order (except for modified columns)
            final_columns = [col for col in original_columns if col not in ['description', 'additional_context']]
            final_columns.insert(original_columns.index('description'), 'description')
            df_temp = df_temp[final_columns]
        
        # Save to respective files
        if not df_with_desc.empty:
            df_with_desc.to_csv(testing_output, index=False)
            print(f"Saved {len(df_with_desc)} rows to {testing_output}")
        
        if not df_without_desc.empty:
            df_without_desc.to_csv(undescribe_output, index=False)
            print(f"Saved {len(df_without_desc)} rows to {undescribe_output}")
        
        return True
        
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage
    input_file = "your_input.csv"  # Replace with your input file path
    process_csv(input_file)
