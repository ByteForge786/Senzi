import pandas as pd
import numpy as np
from collections import defaultdict
import re

# Define stop words to remove
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
    'with', 'by', 'from', 'up', 'about', 'into', 'over', 'after', 'is', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'who', 'which', 'what', 'whose', 'where', 'when', 'why', 'how'
}

def remove_stop_words(text):
    """
    Remove stop words from text while preserving important technical terms
    """
    if pd.isna(text):
        return ""
    
    # Split by underscores first to preserve technical terms
    parts = str(text).split('_')
    cleaned_parts = []
    
    for part in parts:
        # Split into words
        words = part.split()
        # Remove stop words but keep technical terms
        cleaned_words = [word for word in words if word.lower() not in STOP_WORDS]
        cleaned_parts.append(' '.join(cleaned_words))
    
    return '_'.join(cleaned_parts)

def clean_text(text):
    """
    Clean text by removing special characters and standardizing format
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase and remove special characters except underscore
    text = re.sub(r'[^a-zA-Z0-9_\s]', '', str(text).lower())
    # Replace multiple spaces/underscores with single underscore
    text = re.sub(r'[\s_]+', '_', text)
    # Remove leading/trailing underscores
    text = text.strip('_')
    # Remove stop words
    text = remove_stop_words(text)
    return text

def clean_dataframe(df):
    """
    Standardizes and cleans dataframe columns and values
    """
    # Clean column names
    df.columns = [clean_text(col) for col in df.columns]
    
    # Standardize required column names
    column_mapping = {
        'attribute_name': ['attribute_name', 'attributename', 'attribute', 'column_name', 'columnname', 'field_name', 'fieldname'],
        'description': ['description', 'desc', 'column_description', 'field_description'],
        'additional_context': ['additional_context', 'context', 'extra_context', 'additionalcontext'],
        'data_source_name': ['data_source_name', 'datasource', 'source_name', 'sourcename', 'data_source'],
        'sensitivity_label': ['sensitivity_label', 'label', 'sensitivity', 'classification']
    }
    
    # Find and rename columns based on mapping
    for standard_name, variations in column_mapping.items():
        for col in df.columns:
            if col in variations:
                df = df.rename(columns={col: standard_name})
                break
    
    # Clean string values in each column
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(clean_text)
    
    # Standardize sensitivity labels
    standard_labels = {
        'sensitive': ['sensitive', 'personal', 'protected', 'restricted'],
        'non_sensitive': ['nonsensitive', 'non-sensitive', 'non_sensitive', 'public', 'normal'],
        'confidential': ['confidential', 'internal', 'private', 'company_confidential'],
        'licensed': ['licensed', 'third_party', 'thirdparty', 'external', 'vendor']
    }
    
    def standardize_label(label):
        if pd.isna(label):
            return label
        label = clean_text(label)
        for standard, variations in standard_labels.items():
            if label in variations or any(var in label for var in variations):
                return standard
        return label
    
    df['sensitivity_label'] = df['sensitivity_label'].apply(standardize_label)
    
    return df

def extract_patterns_from_name(name):
    """
    Extract meaningful patterns from names, handling various naming conventions
    """
    if pd.isna(name):
        return []
    
    # Clean the name first
    name = clean_text(name)
    
    # Split on underscores
    parts = name.split('_')
    
    # Handle camelCase in each part
    tokens = []
    for part in parts:
        # Split camelCase
        camel_tokens = re.findall('[a-z]+|[A-Z][a-z]*', part)
        tokens.extend([clean_text(t) for t in camel_tokens])
    
    # Check for important prefixes/suffixes
    prefixes = ['client', 'customer', 'user', 'company']
    for prefix in prefixes:
        if name.startswith(prefix):
            tokens.append(f"starts_with_{prefix}")
    
    suffixes = ['id', 'date', 'time', 'name', 'code', 'type', 'status']
    for suffix in suffixes:
        if name.endswith(suffix):
            tokens.append(f"ends_with_{suffix}")
    
    # Add special pattern indicators
    if 'date' in name or 'time' in name:
        tokens.append('temporal_field')
    if 'id' in name or 'key' in name:
        tokens.append('identifier_field')
    if 'status' in name or 'state' in name:
        tokens.append('status_field')
    
    # Remove empty tokens and duplicates
    tokens = [t for t in tokens if t]
    return list(dict.fromkeys(tokens))  # Remove duplicates while preserving order

def analyze_patterns(df):
    """
    Analyzes patterns focusing primarily on attribute names
    """
    patterns = defaultdict(lambda: defaultdict(list))
    
    for label in df['sensitivity_label'].unique():
        label_data = df[df['sensitivity_label'] == label]
        
        # Analyze attribute names
        attr_patterns = defaultdict(int)
        for attr in label_data['attribute_name']:
            tokens = extract_patterns_from_name(attr)
            for token in tokens:
                attr_patterns[token] += 1
        
        threshold = 0.3 * len(label_data)
        common_attr_patterns = {k: v for k, v in attr_patterns.items() 
                              if v >= threshold}
        patterns[label]['attribute_patterns'] = common_attr_patterns
        
        # Analyze descriptions if not all null
        if not label_data['description'].isna().all():
            desc_patterns = defaultdict(int)
            for desc in label_data['description'].dropna():
                tokens = extract_patterns_from_name(desc)
                for token in tokens:
                    desc_patterns[token] += 1
            
            valid_records = len(label_data['description'].dropna())
            threshold = 0.3 * valid_records if valid_records > 0 else 1
            
            common_desc_patterns = {k: v for k, v in desc_patterns.items() 
                                  if v >= threshold}
            patterns[label]['description_patterns'] = common_desc_patterns
        
        # Analyze data sources
        source_patterns = defaultdict(int)
        for source in label_data['data_source_name']:
            tokens = extract_patterns_from_name(source)
            for token in tokens:
                source_patterns[token] += 1
                
        threshold = 0.3 * len(label_data)
        common_source_patterns = {k: v for k, v in source_patterns.items() 
                                if v >= threshold}
        patterns[label]['source_patterns'] = common_source_patterns

def find_correlations(df):
    """
    Finds correlations between data sources and sensitivity labels
    """
    correlations = defaultdict(list)
    
    source_label_dist = pd.crosstab(df['data_source_name'], 
                                   df['sensitivity_label'], 
                                   normalize='index')
    
    for source in source_label_dist.index:
        max_label = source_label_dist.loc[source].idxmax()
        if source_label_dist.loc[source, max_label] >= 0.7:
            correlations['source_label'].append(
                f"Data source '{source}' is strongly associated with '{max_label}' "
                f"({source_label_dist.loc[source, max_label]:.1%})"
            )
    
    return correlations

def analyze_sample_data(file_path):
    """
    Main function to analyze sample data and generate insights
    """
    # Read CSV with all string columns to avoid type inference issues
    df = pd.read_csv(file_path, dtype=str)
    
    # Clean and standardize the dataframe
    df = clean_dataframe(df)
    
    # Get basic statistics
    stats = {
        'total_records': len(df),
        'label_distribution': df['sensitivity_label'].value_counts().to_dict(),
        'data_sources': df['data_source_name'].nunique(),
    }
    
    # Find patterns
    patterns = analyze_patterns(df)
    pattern_summary = generate_pattern_summary(patterns)
    
    # Find correlations
    correlations = find_correlations(df)
    
    # Generate final insights
    insights = [
        f"\nAnalyzed {stats['total_records']} records",
        "\nLabel distribution:",
        *[f"- {label}: {count} records" 
          for label, count in stats['label_distribution'].items()],
        f"\nFound {stats['data_sources']} unique data sources",
        "\nPattern Analysis:",
        pattern_summary,
        "\nCorrelations:",
        *correlations['source_label']
    ]
    
    return "\n".join(insights)

# Example usage:
# df = pd.read_csv('sample_data.csv')
# insights = analyze_sample_data('sample_data.csv')
# print(insights)
