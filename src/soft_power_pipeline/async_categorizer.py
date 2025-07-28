"""Async AI-powered categorizer for parallel processing"""

import asyncio
import logging
from typing import Dict, Any, List, Tuple
from pydantic import BaseModel, Field
import openai
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class DataPointCategory(BaseModel):
    """Single data point with AI-determined category"""
    data_point: str = Field(description="Name/description of the data point")
    category: str = Field(description="Infrastructure, Assets, or Outcome")
    reasoning: str = Field(description="Brief reasoning for categorization")


class CategorizedDataPoints(BaseModel):
    """Response model for batch categorization"""
    categories: List[DataPointCategory] = Field(description="List of categorized data points")


class AsyncCategorizer:
    """Async AI categorizer for parallel processing of multiple countries"""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = model
        self.max_concurrent = 4  # Limit concurrent requests
        self.chunk_size = 25  # Max data points per API call
    
    async def categorize_countries_parallel(self, 
                                          countries_data: List[Tuple[str, List[Dict[str, Any]]]]) -> Dict[str, List[DataPointCategory]]:
        """
        Categorize data points for multiple countries in parallel
        
        Args:
            countries_data: List of (country_name, data_points) tuples
            
        Returns:
            Dict mapping country names to categorized data points
        """
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Process each country with chunking
        tasks = []
        for country_name, data_points in countries_data:
            if not data_points:
                continue
                
            # Split large countries into chunks to manage context windows
            chunks = self._chunk_data_points(data_points, self.chunk_size)
            
            for i, chunk in enumerate(chunks):
                task = self._categorize_chunk_with_semaphore(
                    semaphore, country_name, chunk, i
                )
                tasks.append(task)
        
        if not tasks:
            return {}
        
        # Execute all tasks in parallel
        logger.info(f"Starting {len(tasks)} parallel categorization tasks...")
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results by country
        country_results = {}
        for result in chunk_results:
            if isinstance(result, Exception):
                logger.error(f"Categorization task failed: {result}")
                continue
                
            country_name, chunk_categories = result
            
            if country_name not in country_results:
                country_results[country_name] = []
            country_results[country_name].extend(chunk_categories)
        
        logger.info(f"✓ Completed parallel categorization for {len(country_results)} countries")
        return country_results
    
    async def _categorize_chunk_with_semaphore(self, 
                                             semaphore: asyncio.Semaphore,
                                             country_name: str, 
                                             data_points: List[Dict[str, Any]], 
                                             chunk_idx: int) -> Tuple[str, List[DataPointCategory]]:
        """Categorize a chunk of data points with concurrency control"""
        
        async with semaphore:
            try:
                categories = await self._categorize_chunk(country_name, data_points, chunk_idx)
                logger.debug(f"✓ {country_name} chunk {chunk_idx}: {len(categories)} points categorized")
                return country_name, categories
                
            except Exception as e:
                logger.error(f"Error categorizing {country_name} chunk {chunk_idx}: {str(e)}")
                # Return fallback categorization
                fallback_categories = [
                    DataPointCategory(
                        data_point=dp.get('item', 'Unknown'),
                        category=self._fallback_category(dp.get('item', ''), dp.get('notes', '')),
                        reasoning="Fallback categorization due to API error"
                    )
                    for dp in data_points
                ]
                return country_name, fallback_categories
    
    async def _categorize_chunk(self, 
                              country_name: str, 
                              data_points: List[Dict[str, Any]], 
                              chunk_idx: int) -> List[DataPointCategory]:
        """Categorize a single chunk of data points"""
        
        if not data_points:
            return []
        
        # Build prompt for this chunk
        prompt = self._build_categorization_prompt(data_points, country_name)
        
        # Make async API call
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            response_format=CategorizedDataPoints,
            temperature=0.1
        )
        
        # Handle the response
        if hasattr(response, 'parsed'):
            return response.parsed.categories
        else:
            return response.choices[0].message.parsed.categories
    
    def _chunk_data_points(self, data_points: List[Dict[str, Any]], chunk_size: int) -> List[List[Dict[str, Any]]]:
        """Split data points into manageable chunks"""
        chunks = []
        for i in range(0, len(data_points), chunk_size):
            chunks.append(data_points[i:i + chunk_size])
        return chunks
    
    def _build_categorization_prompt(self, data_points: List[Dict[str, Any]], country_name: str) -> str:
        """Build prompt for batch categorization"""
        
        prompt_parts = [f"Categorize these data points for {country_name}:\\n"]
        
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
        
        return "\\n".join(prompt_parts)
    
    def _get_system_prompt(self) -> str:
        """System prompt for categorization"""
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
    
    def _fallback_category(self, item_name: str, notes: str) -> str:
        """Fallback categorization logic"""
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


def run_async_categorization(openai_api_key: str, 
                           countries_data: List[Tuple[str, List[Dict[str, Any]]]]) -> Dict[str, List[DataPointCategory]]:
    """
    Synchronous wrapper for async categorization
    
    Args:
        openai_api_key: OpenAI API key
        countries_data: List of (country_name, data_points) tuples
        
    Returns:
        Dict mapping country names to categorized data points
    """
    
    categorizer = AsyncCategorizer(openai_api_key)
    
    # Run the async function
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        categorizer.categorize_countries_parallel(countries_data)
    )