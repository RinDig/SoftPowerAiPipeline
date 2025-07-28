"""Impact analysis for individual assets and infrastructure on outcomes"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class ImpactAnalyzer:
    """Analyzes the impact of individual assets/infrastructure on outcome scores"""
    
    def __init__(self):
        self.impact_scores = {}
        self.presence_analysis = {}
        
    def calculate_item_impacts(self, country_scores: Dict[str, Dict[str, Any]], 
                              all_data_points: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate impact scores for each unique asset/infrastructure item
        
        Returns:
            Dict mapping item names to their impact metrics
        """
        
        # Step 1: Build presence matrix (which countries have which items)
        presence_matrix = self._build_presence_matrix(all_data_points)
        
        # Step 2: Calculate presence impact for each item
        presence_impacts = self._calculate_presence_impacts(presence_matrix, country_scores)
        
        # Step 3: Calculate weighted impacts considering scale
        weighted_impacts = self._calculate_weighted_impacts(all_data_points, country_scores)
        
        # Step 4: Calculate co-occurrence bonuses
        cooccurrence_impacts = self._calculate_cooccurrence_impacts(presence_matrix, country_scores)
        
        # Step 5: Combine all impact measures
        final_impacts = self._combine_impact_measures(
            presence_impacts, weighted_impacts, cooccurrence_impacts
        )
        
        return final_impacts
    
    def _build_presence_matrix(self, all_data_points: List[Dict[str, Any]]) -> Dict[str, Dict[str, bool]]:
        """Build matrix of which countries have which items"""
        matrix = defaultdict(lambda: defaultdict(bool))
        
        for dp in all_data_points:
            if dp['category'] in ['Infrastructure', 'Assets']:
                country = dp['country']
                item = dp['item']
                matrix[item][country] = True
        
        return dict(matrix)
    
    def _calculate_presence_impacts(self, presence_matrix: Dict[str, Dict[str, bool]], 
                                  country_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate impact based on presence/absence analysis"""
        impacts = {}
        
        for item, country_presence in presence_matrix.items():
            # Get outcome scores for countries WITH and WITHOUT this item
            scores_with = []
            scores_without = []
            
            for country, scores in country_scores.items():
                outcome_score = scores.get('outcome_score', 0)
                
                if country_presence.get(country, False):
                    scores_with.append(outcome_score)
                else:
                    scores_without.append(outcome_score)
            
            if scores_with and scores_without:
                # Calculate impact metrics
                avg_with = np.mean(scores_with)
                avg_without = np.mean(scores_without)
                impact_diff = avg_with - avg_without
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt((np.std(scores_with)**2 + np.std(scores_without)**2) / 2)
                effect_size = impact_diff / pooled_std if pooled_std > 0 else 0
                
                impacts[item] = {
                    'avg_outcome_with': round(avg_with, 3),
                    'avg_outcome_without': round(avg_without, 3),
                    'impact_difference': round(impact_diff, 3),
                    'effect_size': round(effect_size, 3),
                    'countries_with': len(scores_with),
                    'countries_without': len(scores_without)
                }
            else:
                # All countries have it or none have it
                impacts[item] = {
                    'avg_outcome_with': round(np.mean(scores_with), 3) if scores_with else 0,
                    'avg_outcome_without': round(np.mean(scores_without), 3) if scores_without else 0,
                    'impact_difference': 0,
                    'effect_size': 0,
                    'countries_with': len(scores_with),
                    'countries_without': len(scores_without)
                }
        
        return impacts
    
    def _calculate_weighted_impacts(self, all_data_points: List[Dict[str, Any]], 
                                  country_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate impacts weighted by program scale/importance"""
        weighted_impacts = defaultdict(lambda: {'weighted_contribution': 0, 'appearances': 0})
        
        # Group by country and category
        country_items = defaultdict(lambda: {'infrastructure': [], 'assets': []})
        
        for dp in all_data_points:
            if dp['category'] == 'Infrastructure':
                country_items[dp['country']]['infrastructure'].append(dp)
            elif dp['category'] == 'Assets':
                country_items[dp['country']]['assets'].append(dp)
        
        # Calculate weighted contribution for each item
        for country, scores in country_scores.items():
            outcome_score = scores.get('outcome_score', 0)
            infra_items = scores.get('infrastructure_items', [])
            asset_items = scores.get('asset_items', [])
            
            # Calculate contribution based on weight share
            total_infra_weight = sum(item['weight'] for item in infra_items) if infra_items else 1
            total_asset_weight = sum(item['weight'] for item in asset_items) if asset_items else 1
            
            # Infrastructure contributions
            for item in infra_items:
                weight_share = item['weight'] / total_infra_weight
                contribution = outcome_score * weight_share * 0.3  # 30% category weight
                weighted_impacts[item['item']]['weighted_contribution'] += contribution
                weighted_impacts[item['item']]['appearances'] += 1
            
            # Asset contributions
            for item in asset_items:
                weight_share = item['weight'] / total_asset_weight
                contribution = outcome_score * weight_share * 0.2  # 20% category weight
                weighted_impacts[item['item']]['weighted_contribution'] += contribution
                weighted_impacts[item['item']]['appearances'] += 1
        
        # Calculate average weighted contribution
        for item, data in weighted_impacts.items():
            if data['appearances'] > 0:
                data['avg_weighted_contribution'] = round(
                    data['weighted_contribution'] / data['appearances'], 3
                )
            else:
                data['avg_weighted_contribution'] = 0
        
        return dict(weighted_impacts)
    
    def _calculate_cooccurrence_impacts(self, presence_matrix: Dict[str, Dict[str, bool]], 
                                      country_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate synergy effects when items appear together"""
        cooccurrence_impacts = {}
        
        # Find commonly co-occurring items
        items = list(presence_matrix.keys())
        
        for item in items:
            synergy_scores = []
            
            # Check which other items commonly appear with this one
            cooccurring_items = []
            for other_item in items:
                if item != other_item:
                    # Count co-occurrences
                    cooccur_count = sum(1 for country in presence_matrix[item]
                                      if presence_matrix[item].get(country, False) 
                                      and presence_matrix[other_item].get(country, False))
                    
                    if cooccur_count >= 2:  # At least 2 countries have both
                        cooccurring_items.append((other_item, cooccur_count))
            
            # Calculate average outcome when item appears with frequent partners
            if cooccurring_items:
                total_score = 0
                total_count = 0
                
                for country, scores in country_scores.items():
                    if presence_matrix[item].get(country, False):
                        # Check if country has any of the common partners
                        has_partner = any(presence_matrix[partner[0]].get(country, False) 
                                        for partner in cooccurring_items)
                        if has_partner:
                            total_score += scores.get('outcome_score', 0)
                            total_count += 1
                
                avg_synergy_score = total_score / total_count if total_count > 0 else 0
                
                cooccurrence_impacts[item] = {
                    'common_partners': len(cooccurring_items),
                    'synergy_outcome_avg': round(avg_synergy_score, 3),
                    'top_partners': [p[0] for p in sorted(cooccurring_items, 
                                                         key=lambda x: x[1], 
                                                         reverse=True)[:3]]
                }
            else:
                cooccurrence_impacts[item] = {
                    'common_partners': 0,
                    'synergy_outcome_avg': 0,
                    'top_partners': []
                }
        
        return cooccurrence_impacts
    
    def _combine_impact_measures(self, presence_impacts: Dict, weighted_impacts: Dict, 
                               cooccurrence_impacts: Dict) -> Dict[str, Dict[str, Any]]:
        """Combine all impact measures into final scores"""
        final_impacts = {}
        
        all_items = set(presence_impacts.keys()) | set(weighted_impacts.keys())
        
        for item in all_items:
            presence = presence_impacts.get(item, {})
            weighted = weighted_impacts.get(item, {})
            cooccur = cooccurrence_impacts.get(item, {})
            
            # Calculate composite impact score (0-10 scale)
            impact_components = []
            
            # Presence impact (effect size converted to 0-10)
            effect_size = presence.get('effect_size', 0)
            presence_score = min(10, abs(effect_size) * 5)  # Scale effect size
            impact_components.append(presence_score * 0.4)  # 40% weight
            
            # Weighted contribution
            weighted_contrib = weighted.get('avg_weighted_contribution', 0)
            weighted_score = min(10, weighted_contrib * 20)  # Scale to 0-10
            impact_components.append(weighted_score * 0.4)  # 40% weight
            
            # Synergy bonus
            synergy = cooccur.get('synergy_outcome_avg', 0)
            synergy_score = min(10, synergy * 10)  # Scale to 0-10
            impact_components.append(synergy_score * 0.2)  # 20% weight
            
            composite_score = sum(impact_components)
            
            final_impacts[item] = {
                'composite_impact_score': round(composite_score, 2),
                'presence_impact': round(presence_score, 2),
                'weighted_impact': round(weighted_score, 2),
                'synergy_impact': round(synergy_score, 2),
                'countries_with': presence.get('countries_with', 0),
                'avg_outcome_with': presence.get('avg_outcome_with', 0),
                'avg_outcome_without': presence.get('avg_outcome_without', 0),
                'impact_difference': presence.get('impact_difference', 0),
                'top_partners': cooccur.get('top_partners', [])
            }
        
        return final_impacts
    
    def create_impact_summary_df(self, impact_scores: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Create a DataFrame summary of impact scores"""
        data = []
        
        for item, scores in impact_scores.items():
            data.append({
                'Asset/Infrastructure': item,
                'Impact Score (0-10)': scores['composite_impact_score'],
                'Countries With': scores['countries_with'],
                'Avg Outcome (With)': scores['avg_outcome_with'],
                'Avg Outcome (Without)': scores['avg_outcome_without'],
                'Impact Difference': scores['impact_difference'],
                'Synergy Partners': ', '.join(scores['top_partners'][:2]) if scores['top_partners'] else 'None'
            })
        
        df = pd.DataFrame(data).sort_values('Impact Score (0-10)', ascending=False)
        return df