"""Correlation analysis for soft power data"""

import logging
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any
import pandas as pd
from .weights_config import get_item_weight, get_category_weights

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Analyzes correlations between Infrastructure, Assets, and Outcomes"""
    
    def __init__(self):
        self.outcome_indices = [
            'Brand Finance Soft Power Index',
            'Bonn Power Shift Monitor', 
            'Good Country Index',
            'GDP', 'Democracy', 'Social Progress',
            'Press Freedom', 'Academic Freedom',
            'Corruption Perceptions'
        ]
        
        # More accurate country totals for each index
        self.index_totals = {
            'Brand Finance Soft Power Index': 100,
            'Bonn Power Shift Monitor': 75,
            'Good Country Index': 169,
            'GDP': 195,
            'Democracy': 179,  # vDem
            'Social Progress': 170,
            'Press Freedom': 180,
            'Academic Freedom': 175,
            'Corruption Perceptions': 180
        }
    
    def normalize_rankings(self, rankings: List[Dict[str, Any]]) -> Dict[str, float]:
        """Normalize rankings to 0-1 scale (1 = best)"""
        normalized = {}
        
        for item in rankings:
            # Extract ranking value
            value = item.get('value', '')
            if isinstance(value, (int, float)):
                rank = float(value)
            else:
                # Try to extract number from string
                try:
                    rank = float(str(value).split()[0])
                except:
                    continue
            
            # Find appropriate total countries for this index
            total_countries = 100  # Default fallback
            item_name = item['item']
            
            for index_name, total in self.index_totals.items():
                if index_name.lower() in item_name.lower():
                    total_countries = total
                    break
            
            # Normalize: rank 1 = score 1.0, rank total = score 0.0
            normalized_value = 1.0 - (rank - 1) / (total_countries - 1)
            normalized_value = max(0, min(1, normalized_value))  # Clamp to 0-1
            
            normalized[item['item']] = normalized_value
        
        return normalized
    
    def calculate_country_scores(self, country_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate aggregated scores for each country by category"""
        country_scores = {}
        
        # Group data by country
        for data_point in country_data:
            country = data_point['country']
            category = data_point['category']
            
            if country not in country_scores:
                country_scores[country] = {
                    'Outcome': [],
                    'Infrastructure': [],
                    'Assets': [],
                    'outcome_details': {},
                    'infrastructure_count': 0,
                    'assets_count': 0
                }
            
            # Handle Outcome indices specially
            if category == 'Outcome':
                # Check if it's a known index
                is_index = any(index_name in data_point['item'] for index_name in self.outcome_indices)
                if is_index:
                    try:
                        value = data_point['value']
                        if isinstance(value, (int, float)):
                            rank = float(value)
                        else:
                            rank = float(str(value).split()[0])
                        
                        # Find appropriate total countries for this index
                        total_countries = 100  # Default fallback
                        item_name = data_point['item']
                        
                        for index_name, total in self.index_totals.items():
                            if index_name.lower() in item_name.lower():
                                total_countries = total
                                break
                        
                        # Normalize ranking (1 = best)
                        normalized = 1.0 - (rank - 1) / (total_countries - 1)
                        normalized = max(0, min(1, normalized))
                        
                        country_scores[country]['Outcome'].append(normalized)
                        country_scores[country]['outcome_details'][data_point['item']] = {
                            'raw': rank,
                            'normalized': normalized
                        }
                    except:
                        pass
            
            # Weight Infrastructure and Assets
            elif category == 'Infrastructure':
                # Get weight for this specific item
                weight = get_item_weight(data_point['item'], country, data_point.get('value'))
                country_scores[country]['infrastructure_count'] += 1
                country_scores[country]['Infrastructure'].append(weight)
                
                # Store weighted item for reporting
                if 'infrastructure_items' not in country_scores[country]:
                    country_scores[country]['infrastructure_items'] = []
                country_scores[country]['infrastructure_items'].append({
                    'item': data_point['item'],
                    'weight': weight,
                    'value': data_point.get('value', '')
                })
            
            elif category == 'Assets':
                # Get weight for this specific item
                weight = get_item_weight(data_point['item'], country, data_point.get('value'))
                country_scores[country]['assets_count'] += 1
                country_scores[country]['Assets'].append(weight)
                
                # Store weighted item for reporting
                if 'asset_items' not in country_scores[country]:
                    country_scores[country]['asset_items'] = []
                country_scores[country]['asset_items'].append({
                    'item': data_point['item'],
                    'weight': weight,
                    'value': data_point.get('value', '')
                })
        
        # Calculate weighted scores
        final_scores = {}
        category_weights = get_category_weights()
        
        for country, scores in country_scores.items():
            # Calculate weighted infrastructure score
            infra_weighted_sum = sum(scores['Infrastructure']) if scores['Infrastructure'] else 0
            infra_max_possible = scores['infrastructure_count'] * 3.0  # Max weight is 3.0
            infrastructure_score = infra_weighted_sum / infra_max_possible if infra_max_possible > 0 else 0
            
            # Calculate weighted assets score
            assets_weighted_sum = sum(scores['Assets']) if scores['Assets'] else 0
            assets_max_possible = scores['assets_count'] * 3.0  # Max weight is 3.0
            assets_score = assets_weighted_sum / assets_max_possible if assets_max_possible > 0 else 0
            
            final_scores[country] = {
                'outcome_score': np.mean(scores['Outcome']) if scores['Outcome'] else 0,
                'infrastructure_score': min(1.0, infrastructure_score),
                'assets_score': min(1.0, assets_score),
                'outcome_details': scores['outcome_details'],
                'infrastructure_count': scores['infrastructure_count'],
                'assets_count': scores['assets_count'],
                'infrastructure_weighted_sum': round(infra_weighted_sum, 2),
                'assets_weighted_sum': round(assets_weighted_sum, 2),
                'overall_score': 0  # Will calculate this
            }
            
            # Add weighted items details
            if 'infrastructure_items' in scores:
                final_scores[country]['infrastructure_items'] = scores['infrastructure_items']
            if 'asset_items' in scores:
                final_scores[country]['asset_items'] = scores['asset_items']
            
            # Calculate overall score using configured category weights
            final_scores[country]['overall_score'] = (
                final_scores[country]['outcome_score'] * category_weights['outcome'] +
                final_scores[country]['infrastructure_score'] * category_weights['infrastructure'] +
                final_scores[country]['assets_score'] * category_weights['assets']
            )
        
        return final_scores
    
    def calculate_correlations(self, country_scores: Dict[str, Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
        """Calculate Pearson correlations between categories"""
        
        if len(country_scores) < 3:
            logger.warning("Not enough countries for meaningful correlation analysis")
            return {}
        
        # Extract arrays for correlation
        outcomes = []
        infrastructure = []
        assets = []
        
        for country, scores in country_scores.items():
            outcomes.append(scores['outcome_score'])
            infrastructure.append(scores['infrastructure_score'])
            assets.append(scores['assets_score'])
        
        # Calculate correlations
        correlations = {}
        
        # Assets vs Outcomes
        r_assets_outcomes, p_assets_outcomes = stats.pearsonr(assets, outcomes)
        correlations['Assets vs Outcomes'] = (r_assets_outcomes, p_assets_outcomes)
        
        # Infrastructure vs Outcomes
        r_infra_outcomes, p_infra_outcomes = stats.pearsonr(infrastructure, outcomes)
        correlations['Infrastructure vs Outcomes'] = (r_infra_outcomes, p_infra_outcomes)
        
        # Assets vs Infrastructure
        r_assets_infra, p_assets_infra = stats.pearsonr(assets, infrastructure)
        correlations['Assets vs Infrastructure'] = (r_assets_infra, p_assets_infra)
        
        return correlations
    
    def generate_insights(self, country_scores: Dict[str, Dict[str, float]], 
                         correlations: Dict[str, Tuple[float, float]]) -> List[str]:
        """Generate insights from the analysis"""
        insights = []
        
        # Find top performers
        sorted_countries = sorted(country_scores.items(), 
                                key=lambda x: x[1]['overall_score'], 
                                reverse=True)
        
        if sorted_countries:
            top_country = sorted_countries[0][0]
            top_score = sorted_countries[0][1]['overall_score']
            insights.append(f"Top performer: {top_country} with overall score of {top_score:.2f}")
        
        # Correlation insights
        for corr_name, (r, p) in correlations.items():
            if abs(r) > 0.7:
                strength = "strong"
            elif abs(r) > 0.4:
                strength = "moderate"
            else:
                strength = "weak"
            
            direction = "positive" if r > 0 else "negative"
            
            insights.append(f"{corr_name}: {strength} {direction} correlation (r={r:.2f}, p={p:.2f})")
            
            # Special insights
            if "Assets vs Outcomes" in corr_name and r < -0.5:
                insights.append("High asset investments don't always yield high outcomes - other factors like governance may be at play")
        
        return insights
    
    def create_correlation_summary(self, country_scores: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Create a summary dataframe for the correlation analysis"""
        data = []
        
        for country, scores in country_scores.items():
            row = {
                'Country': country,
                'Outcome Score': round(scores['outcome_score'], 2),
                'Infrastructure Score': round(scores['infrastructure_score'], 2),
                'Assets Score': round(scores['assets_score'], 2),
                'Overall Score': round(scores['overall_score'], 2),
                'Infrastructure Count': scores['infrastructure_count'],
                'Assets Count': scores['assets_count']
            }
            data.append(row)
        
        return pd.DataFrame(data).sort_values('Overall Score', ascending=False)