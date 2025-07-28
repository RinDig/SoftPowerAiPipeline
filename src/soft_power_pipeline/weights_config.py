"""Configuration for soft power analysis weights

This module defines weights for different aspects of soft power analysis.
Weights can be customized based on:
- Item type (specific programs/ministries)
- Category (Infrastructure vs Assets)
- Country context (same item weighted differently per country)
- Value thresholds (budget-based weights)
"""

from typing import Dict, Any

# Default category weights for overall score calculation
CATEGORY_WEIGHTS = {
    'outcome': 0.5,      # 50% - actual results/rankings
    'infrastructure': 0.3,  # 30% - organizational capacity (increased from 25%)
    'assets': 0.2        # 20% - programs and initiatives (decreased from 25%)
}

# Item-specific weights (multipliers for importance)
# These are applied when calculating infrastructure/asset scores
ITEM_WEIGHTS = {
    # Infrastructure items (organizational capacity)
    'lead ministry': 2.0,
    'ministry': 2.0,
    'budget': 3.0,  # Financial commitment is critical
    'budget 2024': 3.0,
    'annual funding': 2.5,
    'funding': 2.5,
    'organizational structure': 1.5,
    'policy': 2.0,
    'strategy': 2.0,
    'governance': 1.5,
    
    # Asset items (programs and initiatives)
    'confucius institute': 2.5,  # Major cultural programs
    'british council': 2.5,
    'goethe institut': 2.5,
    'alliance française': 2.5,
    'japan foundation': 2.5,
    'instituto cervantes': 2.5,
    'cultural center': 1.5,
    'language program': 2.0,
    'exchange program': 2.0,
    'scholarship': 2.0,
    'festival': 1.0,
    'exhibition': 1.0,
    'digital platform': 1.5,
    'social media': 1.0,
    'broadcasting': 2.0,
    'news agency': 2.0
}

# Country-specific multipliers for flagship programs
# Applied on top of item weights
COUNTRY_MULTIPLIERS = {
    'China': {
        'confucius institute': 1.5,  # Extra weight for flagship
        'belt and road': 2.0,
        'bri': 2.0
    },
    'UK': {
        'british council': 1.5,
        'bbc': 1.3,
        'fcdo': 1.2
    },
    'USA': {
        'fulbright': 1.5,
        'usaid': 1.3,
        'voice of america': 1.3,
        'hollywood': 1.2
    },
    'Germany': {
        'goethe institut': 1.5,
        'daad': 1.3,
        'dw': 1.2
    },
    'Japan': {
        'japan foundation': 1.5,
        'japan house': 1.3,
        'jica': 1.2,
        'anime': 1.2,
        'manga': 1.2
    },
    'France': {
        'alliance française': 1.5,
        'institut français': 1.3,
        'tv5monde': 1.2
    },
    'South Korea': {
        'k-pop': 1.5,
        'korean wave': 1.5,
        'hallyu': 1.5,
        'korean cultural center': 1.3
    }
}

# Value-based weight thresholds
# Applied based on budget/reach metrics
VALUE_THRESHOLDS = {
    'budget': [
        {'min': 1_000_000_000, 'weight': 3.0},    # > $1B
        {'min': 500_000_000, 'weight': 2.5},      # > $500M
        {'min': 100_000_000, 'weight': 2.0},      # > $100M
        {'min': 50_000_000, 'weight': 1.5},       # > $50M
        {'min': 0, 'weight': 1.0}                 # Default
    ],
    'reach': [
        {'min': 100, 'weight': 2.0},  # 100+ countries
        {'min': 50, 'weight': 1.5},   # 50+ countries
        {'min': 20, 'weight': 1.2},   # 20+ countries
        {'min': 0, 'weight': 1.0}     # Default
    ]
}

def get_item_weight(item_name: str, country: str = None, value: Any = None) -> float:
    """
    Get the weight for a specific item
    
    Args:
        item_name: Name of the item (will be lowercased for matching)
        country: Country context (optional)
        value: Numerical value for threshold-based weights (optional)
    
    Returns:
        float: Weight multiplier (default 1.0)
    """
    item_lower = item_name.lower()
    
    # Start with base weight
    weight = 1.0
    
    # Check for item-specific weight
    for key, item_weight in ITEM_WEIGHTS.items():
        if key in item_lower:
            weight = item_weight
            break
    
    # Apply country-specific multiplier if applicable
    if country and country in COUNTRY_MULTIPLIERS:
        country_items = COUNTRY_MULTIPLIERS[country]
        for key, multiplier in country_items.items():
            if key in item_lower:
                weight *= multiplier
                break
    
    # Apply value-based threshold if applicable
    if value and isinstance(value, (int, float)):
        # Check if this is a budget item
        if any(term in item_lower for term in ['budget', 'funding', 'finance']):
            for threshold in VALUE_THRESHOLDS['budget']:
                if value >= threshold['min']:
                    weight *= threshold['weight']
                    break
    
    return weight

def get_category_weights() -> Dict[str, float]:
    """Get the current category weights for overall score calculation"""
    return CATEGORY_WEIGHTS.copy()

def update_category_weights(outcome: float = None, 
                          infrastructure: float = None, 
                          assets: float = None) -> None:
    """
    Update category weights (must sum to 1.0)
    
    Args:
        outcome: Weight for outcome scores
        infrastructure: Weight for infrastructure scores
        assets: Weight for asset scores
    """
    if outcome is not None:
        CATEGORY_WEIGHTS['outcome'] = outcome
    if infrastructure is not None:
        CATEGORY_WEIGHTS['infrastructure'] = infrastructure
    if assets is not None:
        CATEGORY_WEIGHTS['assets'] = assets
    
    # Validate weights sum to 1.0
    total = sum(CATEGORY_WEIGHTS.values())
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Category weights must sum to 1.0, got {total}")

# Reasonable default weights based on analysis goals
# Can be adjusted via configuration file or programmatically
DEFAULT_CONFIG = {
    'use_item_weights': True,
    'use_country_multipliers': True,
    'use_value_thresholds': True,
    'normalize_by_gdp': False,  # Future: normalize by country GDP
    'normalize_by_population': False  # Future: per-capita analysis
}