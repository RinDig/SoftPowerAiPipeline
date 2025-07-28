"""OpenAI client with native structured output support"""

import os
import logging
from typing import Optional, Type, TypeVar
from pydantic import BaseModel
import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class OpenAIClient:
    """Client for OpenAI with native structured output support"""
    
    DEFAULT_MODEL = "gpt-4o-mini"  # GPT-4 Turbo with structured output
    DEFAULT_TEMPERATURE = 0.1  # Lower for more consistent structured output
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ):
        """Initialize OpenAI client
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (defaults to gpt-4o-mini)
            temperature: Temperature for response generation
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature or self.DEFAULT_TEMPERATURE
        
        logger.info(f"Initialized OpenAI client with model: {self.model}")
    
    def generate_structured(
        self,
        request_data: dict,
        response_model: Type[T],
        system_prompt: str,
        temperature: Optional[float] = None,
        **kwargs
    ) -> T:
        """Generate a structured response using OpenAI's native structured output
        
        Args:
            request_data: Dictionary containing the request data
            response_model: Pydantic model class for parsing the response
            system_prompt: System prompt for context
            temperature: Override default temperature
            **kwargs: Additional arguments
            
        Returns:
            Parsed response as instance of response_model
        """
        try:
            # Format the user prompt with request data
            user_prompt = self._format_request_prompt(request_data)
            
            # Use OpenAI's structured output with response_format
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=response_model,
                temperature=temperature or self.temperature,
                **kwargs
            )
            
            # Return the parsed structured response
            return completion.choices[0].message.parsed
            
        except Exception as e:
            logger.error(f"Failed to generate structured response: {str(e)}")
            raise ValueError(f"Failed to generate response with {response_model.__name__}: {str(e)}")
    
    def _format_request_prompt(self, request_data: dict) -> str:
        """Format request data into a user prompt"""
        prompt_parts = []
        
        for key, value in request_data.items():
            if isinstance(value, (list, dict)):
                prompt_parts.append(f"{key}:\n{self._format_complex_data(value)}")
            else:
                prompt_parts.append(f"{key}: {value}")
        
        return "\n\n".join(prompt_parts)
    
    def _format_complex_data(self, data, indent_level: int = 0) -> str:
        """Format complex data structures for readability"""
        indent = "  " * indent_level
        
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    lines.append(f"{indent}- {key}:")
                    lines.append(self._format_complex_data(value, indent_level + 1))
                else:
                    lines.append(f"{indent}- {key}: {value}")
            return "\n".join(lines)
        
        elif isinstance(data, list):
            lines = []
            for i, item in enumerate(data):
                if isinstance(item, (list, dict)):
                    lines.append(f"{indent}{i+1}.")
                    lines.append(self._format_complex_data(item, indent_level + 1))
                else:
                    lines.append(f"{indent}- {item}")
            return "\n".join(lines)
        
        else:
            return f"{indent}{data}"