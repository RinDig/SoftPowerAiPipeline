#!/usr/bin/env python3
"""
Simple Soft Power Analysis Pipeline

Usage:
    python simple_main.py --countries Canada Germany Japan
    python simple_main.py --all-countries
"""

import argparse
import logging
from pathlib import Path
from typing import List
import os
from dotenv import load_dotenv

from src.soft_power_pipeline.ai.openai_client import OpenAIClient
from src.soft_power_pipeline.simple_pipeline import SimplePipeline

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_country_files(data_dir: Path, countries: List[str] = None) -> List[Path]:
    """Find Excel files for specified countries"""
    excel_files = []
    
    if countries:
        # Look for specific countries
        for country in countries:
            # Try different filename patterns
            patterns = [
                f"*{country}*.xlsx",
                f"*{country.lower()}*.xlsx", 
                f"*{country.upper()}*.xlsx"
            ]
            
            found = False
            for pattern in patterns:
                matches = list(data_dir.glob(pattern))
                if matches:
                    excel_files.extend(matches)
                    found = True
                    break
            
            if not found:
                logger.warning(f"No Excel file found for {country}")
    else:
        # Get all Excel files
        excel_files = list(data_dir.glob("*.xlsx"))
    
    return excel_files


def main():
    parser = argparse.ArgumentParser(description="Simple Soft Power Analysis Pipeline")
    parser.add_argument("--countries", nargs="+", help="Country names to analyze")
    parser.add_argument("--all-countries", action="store_true", help="Analyze all available countries")
    parser.add_argument("--data-dir", default="data", help="Directory containing Excel files")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1
    
    # Find Excel files
    if args.all_countries:
        excel_files = find_country_files(data_dir)
    elif args.countries:
        excel_files = find_country_files(data_dir, args.countries)
    else:
        logger.error("Must specify either --countries or --all-countries")
        return 1
    
    if not excel_files:
        logger.error("No Excel files found to process")
        return 1
    
    logger.info(f"Found {len(excel_files)} Excel files to process")
    for file in excel_files:
        logger.info(f"  - {file.name}")
    
    try:
        # Initialize pipeline
        openai_client = OpenAIClient()
        pipeline = SimplePipeline(openai_client, output_dir)
        
        # Process countries
        logger.info("Starting simple pipeline...")
        output_file = pipeline.process_countries(excel_files)
        
        if output_file:
            logger.info(f"✅ Analysis complete! Results saved to: {output_file}")
            logger.info(f"📊 Open {output_file} to view the data table")
        else:
            logger.error("❌ Pipeline failed to generate output")
            return 1
            
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())