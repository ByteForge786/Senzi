import pandas as pd
import numpy as np
from collections import defaultdict
import re

def extract_patterns_from_name(name):
    """
    Extract meaningful patterns from attribute names, handling various naming conventions
    """
    if pd.isna(name):
        return []
        
    # Convert to lowercase
    name = str(name).lower()
    
    # Handle common separators (snake_case, camelCase, etc.)
    # First split by underscores
    parts = name.split('_')
    
    # Then handle camelCase in each part
    tokens = []
    for part in parts:
        # Split camelCase
        camel_tokens = re.findall('[a-z]+|[A-Z][a-z]*', part)
        tokens.extend([t.lower() for t in camel_tokens])
        
    # Also get common prefixes/suffixes
    prefixes = ['is', 'has', 'was', 'start', 'end', 'client', 'customer', 'user', 'company']
    for prefix in prefixes:
        if name.startswith(prefix):
            tokens.append(f"starts_with_{prefix}")
            
    suffixes = ['id', 'date', 'time', 'name', 'code', 'type', 'status']
    for suffix in suffixes:
        if name.endswith(suffix):
            tokens.append(f"ends_with_{suffix}")
    
    # Check for specific patterns
    patterns = []
    if 'date' in name or 'time' in name:
        patterns.append('temporal_field')
    if 'id' in name or 'key' in name:
        patterns.append('identifier_field')
    if 'status' in name or 'state' in name:
        patterns.append('status_field')
        
    return tokens + patterns

def analyze_patterns(df):
    """
    Analyzes patterns focusing primarily on attribute names, handling null descriptions
    """
    patterns = defaultdict(lambda: defaultdict(list))
    
    # Analyze each sensitivity label
    for label in df['sensitivity_label'].unique():
        label_data = df[df['sensitivity_label'] == label]
        
        # Analyze attribute names
        attr_patterns = defaultdict(int)
        for attr in label_data['attribute_name']:
            tokens = extract_patterns_from_name(attr)
            for token in tokens:
                attr_patterns[token] += 1
        
        # Keep patterns that appear in at least 30% of records
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
                
        common_source_patterns = {k: v for k, v in source_patterns.items() 
                                if v >= threshold}
        patterns[label]['source_patterns'] = common_source_patterns
        
        # Analyze context patterns if not all null
        if 'additional_context' in label_data.columns and not label_data['additional_context'].isna().all():
            context_patterns = defaultdict(int)
            for context in label_data['additional_context'].dropna():
                tokens = extract_patterns_from_name(context)
                for token in tokens:
                    context_patterns[token] += 1
                    
            valid_context_records = len(label_data['additional_context'].dropna())
            threshold = 0.3 * valid_context_records if valid_context_records > 0 else 1
            
            common_context_patterns = {k: v for k, v in context_patterns.items() 
                                    if v >= threshold}
            patterns[label]['context_patterns'] = common_context_patterns
    
    return patterns

def generate_pattern_summary(patterns):
    """
    Generates a human-readable summary of the patterns found
    """
    summary = []
    
    for label, label_patterns in patterns.items():
        summary.append(f"\nPatterns for {label} data:")
        
        if label_patterns['attribute_patterns']:
            summary.append("\nCommon attribute name patterns:")
            sorted_attrs = sorted(label_patterns['attribute_patterns'].items(), 
                                key=lambda x: x[1], reverse=True)
            for token, count in sorted_attrs[:5]:
                summary.append(f"- '{token}' appears {count} times")
        
        if label_patterns.get('description_patterns'):
            summary.append("\nCommon description patterns:")
            sorted_desc = sorted(label_patterns['description_patterns'].items(), 
                               key=lambda x: x[1], reverse=True)
            for token, count in sorted_desc[:5]:
                summary.append(f"- '{token}' appears {count} times")
        
        if label_patterns['source_patterns']:
            summary.append("\nCommon data source patterns:")
            sorted_source = sorted(label_patterns['source_patterns'].items(), 
                                 key=lambda x: x[1], reverse=True)
            for token, count in sorted_source[:5]:
                summary.append(f"- '{token}' appears {count} times")
                
        if label_patterns.get('context_patterns'):
            summary.append("\nCommon context patterns:")
            sorted_context = sorted(label_patterns['context_patterns'].items(), 
                                  key=lambda x: x[1], reverse=True)
            for token, count in sorted_context[:5]:
                summary.append(f"- '{token}' appears {count} times")
    
    return "\n".join(summary)

def find_correlations(df):
    """
    Finds correlations between data sources and sensitivity labels
    """
    correlations = defaultdict(list)
    
    # Analyze correlation between data sources and labels
    source_label_dist = pd.crosstab(df['data_source_name'], 
                                   df['sensitivity_label'], 
                                   normalize='index')
    
    # Find sources that strongly correlate with specific labels
    for source in source_label_dist.index:
        max_label = source_label_dist.loc[source].idxmax()
        if source_label_dist.loc[source, max_label] >= 0.7:  # 70% threshold
            correlations['source_label'].append(
                f"Data source '{source}' is strongly associated with '{max_label}' "
                f"({source_label_dist.loc[source, max_label]:.1%})"
            )
    
    return correlations

def analyze_sample_data(file_path):
    """
    Main function to analyze sample data and generate insights
    """
    df = pd.read_csv(file_path)
    
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
