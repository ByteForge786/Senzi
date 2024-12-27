import pandas as pd
import numpy as np

def process_csv(input_file, testing_output='testing.csv', undescribe_output='undescribe.csv'):
    """
    Process CSV file to handle descriptions and attribute names according to requirements.
    
    Args:
        input_file (str): Path to input CSV file
        testing_output (str): Path for output file with valid descriptions
        undescribe_output (str): Path for output file with null descriptions
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Combine description and additional context
        # Replace NaN with empty string to avoid NaN concatenation issues
        df['description'] = df['description'].fillna('')
        df['additional_context'] = df['additional_context'].fillna('')
        
        # Combine descriptions
        df['combined_description'] = df['description'] + df['additional_context']
        # Remove empty strings (where both were null)
        df['combined_description'] = df['combined_description'].replace('', np.nan)
        
        # Modify attribute name to include description
        df['attribute_name'] = df['description'] + df['attribute_name']
        
        # Split into two dataframes based on description availability
        df_with_desc = df[df['combined_description'].notna()].copy()
        df_without_desc = df[df['combined_description'].isna()].copy()
        
        # Update column names
        df_with_desc = df_with_desc.rename(columns={'combined_description': 'description'})
        df_without_desc = df_without_desc.rename(columns={'combined_description': 'description'})
        
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
