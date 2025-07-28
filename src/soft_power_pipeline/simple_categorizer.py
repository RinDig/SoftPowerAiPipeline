"""Simple AI-powered categorizer for data points"""

import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DataPointCategory(BaseModel):
    """Single data point with AI-determined category"""
    data_point: str = Field(description="Name/description of the data point")
    category: str = Field(description="Infrastructure, Assets, or Outcome")
    reasoning: str = Field(description="Brief reasoning for categorization")


class SimpleCategorizer:
    """Simple AI categorizer that determines if data points are Infrastructure, Assets, or Outcomes"""
    
    def __init__(self, openai_client):
        self.openai = openai_client
    
    def categorize_data_points(self, data_points: List[Dict[str, Any]]) -> List[DataPointCategory]:
        """Categorize a batch of data points"""
        
        if not data_points:
            return []
        
        # Create batch categorization prompt
        prompt = self._build_categorization_prompt(data_points)
        
        try:
            # Use OpenAI structured output for batch categorization
            response = self.openai.client.beta.chat.completions.parse(
                model=self.openai.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format=CategorizedDataPoints
            )
            
            # Handle the response properly
            if hasattr(response, 'parsed'):
                return response.parsed.categories
            else:  
                # Try accessing the response directly
                return response.choices[0].message.parsed.categories
            
        except Exception as e:
            logger.error(f"Error categorizing data points: {str(e)}")
            # Fallback to default categorization
            return [
                DataPointCategory(
                    data_point=dp.get('item', 'Unknown'),
                    category=self._default_category(dp.get('item', ''), dp.get('notes', '')),
                    reasoning="Default categorization due to AI error"
                )
                for dp in data_points
            ]
    
    def _get_system_prompt(self) -> str:
        return """You are an expert in soft power analysis and data categorization.

Your task is to categorize data points into exactly one of these three categories:

**Infrastructure**: Organizational capacity, budgets, ministries, institutional frameworks, funding
- Examples: "Lead Ministry", "Budget 2024", "Annual Funding", "Organizational Structure"

**Assets**: Specific programs, initiatives, institutions, activities, cultural centers
- Examples: "Belt and Road exchanges", "Cultural festivals", "Language institutes", "Educational programs"

**Outcome**: Performance measures, rankings, indices, reputation indicators
- Examples: "Brand Finance Soft Power Index", "GDP ranking", "Press Freedom Index"

For each data point, provide:
1. The exact category (Infrastructure, Assets, or Outcome)
2. Brief reasoning for your choice

Be consistent and precise in your categorizations."""

    def _build_categorization_prompt(self, data_points: List[Dict[str, Any]]) -> str:
        """Build prompt for batch categorization"""
        
        prompt_parts = ["Categorize each of these data points:\n"]
        
        for i, dp in enumerate(data_points, 1):
            item_name = dp.get('item', 'Unknown')
            notes = dp.get('notes', '')
            value = dp.get('value', '')
            
            context = f"{item_name}"
            if value:
                context += f": {value}"
            if notes:
                context += f" ({notes})"
            
            prompt_parts.append(f"{i}. {context}")
        
        return "\n".join(prompt_parts)
    
    def _default_category(self, item_name: str, notes: str) -> str:
        """Default categorization logic for fallback"""
        item_lower = item_name.lower()
        notes_lower = notes.lower()
        
        # Infrastructure keywords
        if any(keyword in item_lower + notes_lower for keyword in [
            'ministry', 'budget', 'funding', 'finance', 'organizational', 'structure'
        ]):
            return 'Infrastructure'
        
        # Asset keywords
        if any(keyword in item_lower + notes_lower for keyword in [
            'program', 'initiative', 'festival', 'center', 'institute', 'exchange'
        ]):
            return 'Assets'
        
        # Default to outcome for indices and rankings
        if any(keyword in item_lower for keyword in [
            'index', 'ranking', 'rank', 'score', 'gdp', 'democracy'
        ]):
            return 'Outcome'
        
        return 'Outcome'  # Default fallback


class CategorizedDataPoints(BaseModel):
    """Response model for batch categorization"""
    categories: List[DataPointCategory] = Field(description="List of categorized data points")