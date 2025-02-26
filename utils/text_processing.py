import re
from typing import List, Dict, Tuple
from rapidfuzz import process
from utils.search_patterns import FlowerSearchPatterns

def clean_text(text: str) -> str:
    """Clean product text by removing HTML and special characters"""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split())

def find_closest_match(query: str, options: List[str], threshold: int = 80) -> str:
    """
    Find the closest match for a query from a list of options using rapidfuzz.
    
    Args:
        query: The query string to match.
        options: List of possible options to match against.
        threshold: Minimum similarity score to consider a match.
    Returns:
        The closest match if above threshold, otherwise None.
    """
    result = process.extractOne(query, options, score_cutoff=threshold)
    if result:
        match, score, _ = result  # Unpack the result if it's not None
        return match
    return None  # Return None if no match is found

def extract_price_constraint(query: str) -> Tuple[str, float, str]:
    """
    Extract price constraints from the query
    
    Args:
        query: Search query string
    Returns:
        tuple: (cleaned_query, price_limit, comparison_type)
    """
    # Price patterns
    price_patterns = [
        r'under\s*\$?\s*(\d+)',
        r'less than\s*\$?\s*(\d+)',
        r'below\s*\$?\s*(\d+)',
        r'\$?\s*(\d+)\s*or less',
        r'cheaper than\s*\$?\s*(\d+)',
    ]
    
    cleaned_query = query
    price_limit = None
    comparison_type = 'less'  # Could be 'less' or 'more'
    
    for pattern in price_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            price_limit = float(match.group(1))
            # Remove the price constraint from the query
            cleaned_query = re.sub(pattern, '', query, flags=re.IGNORECASE).strip()
            break
            
    return cleaned_query, price_limit, comparison_type

def parse_search_query(query: str, threshold: int = 80) -> Dict[str, str]:
    """
    Parse the search query to identify key search components with typo tolerance.
    
    Args:
        query: Raw search query string
        threshold: Matching threshold for fuzzy matching
    Returns:
        Dict containing identified search components
    """
    query = query.lower()
    components = {
        'occasion': None,
        'color': None,
        'flower_type': None,
        'size': None,
        'price_range': None,
        'arrangement': None,
        'price_limit': None,
        'cleaned_query': query
    }
    
    # Extract price limit first
    cleaned_query, price_limit, _ = extract_price_constraint(query)
    components['price_limit'] = price_limit
    components['cleaned_query'] = cleaned_query
    
    patterns = FlowerSearchPatterns()
    
    # Match patterns with typo tolerance using rapidfuzz
    for occasion, pattern_list in patterns.OCCASIONS.items():
        match = find_closest_match(cleaned_query, pattern_list, threshold)
        if match:
            components['occasion'] = occasion
            
    for color, pattern_list in patterns.COLORS.items():
        match = find_closest_match(cleaned_query, pattern_list, threshold)
        if match:
            components['color'] = color
            
    for flower, pattern_list in patterns.FLOWER_TYPES.items():
        match = find_closest_match(cleaned_query, pattern_list, threshold)
        if match:
            components['flower_type'] = flower
            
    for size, pattern_list in patterns.SIZES.items():
        match = find_closest_match(cleaned_query, pattern_list, threshold)
        if match:
            components['size'] = size
            
    for price_range, pattern_list in patterns.PRICE_RANGES.items():
        match = find_closest_match(cleaned_query, pattern_list, threshold)
        if match:
            components['price_range'] = price_range
            
    for arrangement, pattern_list in patterns.ARRANGEMENTS.items():
        match = find_closest_match(cleaned_query, pattern_list, threshold)
        if match:
            components['arrangement'] = arrangement
    
    return components

def enhance_search_query(components: Dict[str, str]) -> str:
    """
    Create an enhanced search query based on identified components
    
    Args:
        components: Dictionary of search components
    Returns:
        Enhanced search query string
    """
    query_parts = []
    
    if components['occasion']:
        query_parts.append(f"perfect for {components['occasion']}")
        
    if components['flower_type']:
        query_parts.append(components['flower_type'])
        
    if components['color']:
        query_parts.append(f"{components['color']} colored")
        
    if components['size']:
        query_parts.append(f"{components['size']} size")
        
    if components['arrangement']:
        query_parts.append(f"in {components['arrangement']}")
        
    # Add original cleaned query if it contains unique terms
    original_terms = set(components['cleaned_query'].split())
    enhanced_terms = set(' '.join(query_parts).split())
    unique_terms = original_terms - enhanced_terms
    if unique_terms:
        query_parts.append(' '.join(unique_terms))
        
    return ' '.join(query_parts)

def get_search_tips():
    """Return helpful search tips for users"""
    patterns = FlowerSearchPatterns()
    return {
        "example_searches": [
            "red roses with vase for anniversary under $75",
            "small pink birthday bouquet",
            "luxury mixed flowers for wedding",
            "sympathy lilies arrangement",
            "affordable sunflower basket"
        ],
        "search_components": {
            "occasions": list(patterns.OCCASIONS.keys()),
            "colors": list(patterns.COLORS.keys()),
            "flower_types": list(patterns.FLOWER_TYPES.keys()),
            "sizes": list(patterns.SIZES.keys()),
            "arrangements": list(patterns.ARRANGEMENTS.keys()),
            "price_ranges": list(patterns.PRICE_RANGES.keys())
        }
    }
