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
