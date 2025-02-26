class FlowerSearchPatterns:
    """Patterns for common flower search queries"""
    
    OCCASIONS = {
        'anniversary': ['anniversary', 'wedding anniversary', 'yearly celebration'],
        'birthday': ['birthday', 'bday', 'birth day celebration'],
        'wedding': ['wedding', 'bridal', 'marriage'],
        'sympathy': ['sympathy', 'condolence', 'funeral', 'grief'],
        'romance': ['romance', 'romantic', 'love', 'valentine', "valentine's day"],
        'congratulations': ['congratulations', 'congrats', 'graduation', 'promotion'],
        'get well': ['get well', 'recovery', 'hospital', 'feel better']
    }
    
    COLORS = {
        'red': ['red', 'crimson', 'scarlet'],
        'pink': ['pink', 'rose pink', 'blush'],
        'white': ['white', 'pure white', 'snow white'],
        'yellow': ['yellow', 'golden', 'sunny'],
        'purple': ['purple', 'lavender', 'violet'],
        'orange': ['orange', 'coral', 'peach'],
        'blue': ['blue', 'azure', 'sky blue'],
        'mixed': ['mixed', 'rainbow', 'multicolor', 'colorful']
    }
    
    FLOWER_TYPES = {
        'roses': ['rose', 'roses', 'red rose', 'garden rose'],
        'lilies': ['lily', 'lilies', 'asiatic lily'],
        'carnations': ['carnation', 'carnations'],
        'chrysanthemums': ['chrysanthemum', 'mums', 'chrysanths'],
        'orchids': ['orchid', 'orchids', 'phalaenopsis'],
        'sunflowers': ['sunflower', 'sunflowers', 'sun flower'],
        'tulips': ['tulip', 'tulips', 'dutch tulip'],
        'mixed flowers': ['mixed flowers', 'mixed bouquet', 'flower mix']
    }
    
    SIZES = {
        'small': ['small', 'petit', 'tiny', 'compact'],
        'medium': ['medium', 'standard', 'regular'],
        'large': ['large', 'big', 'grand', 'deluxe'],
        'extra large': ['extra large', 'xl', 'luxury', 'premium']
    }
    
    PRICE_RANGES = {
        'budget': ['cheap', 'affordable', 'budget', 'under $50', 'inexpensive'],
        'medium': ['moderate', 'mid-range', '$50-$100'],
        'premium': ['premium', 'luxury', 'expensive', 'over $100']
    }
    
    ARRANGEMENTS = {
        'bouquet': ['bouquet', 'bunch', 'hand-tied'],
        'vase': ['vase arrangement', 'in vase', 'with vase'],
        'basket': ['basket', 'flower basket', 'gift basket'],
        'box': ['box', 'flower box', 'hat box']
    }
