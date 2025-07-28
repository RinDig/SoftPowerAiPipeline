"""Simplified Soft Power Analysis Pipeline

Focus: Extract data → Categorize with AI → Output clean table
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime

from .extractors.excel_extractor import ExcelExtractor
from .simple_categorizer import SimpleCategorizer
from .async_categorizer import run_async_categorization
from .ai.openai_client import OpenAIClient
from .correlation_analyzer import CorrelationAnalyzer
from .impact_analyzer import ImpactAnalyzer
from .excel_styler import ExcelStyler

logger = logging.getLogger(__name__)


class DataPoint:
    """Simple data point representation"""
    def __init__(self, country: str, item: str, raw_value: Any, category: str, source_note: str):
        self.country = country
        self.data_point = item
        self.raw_value = str(raw_value)
        self.category = category
        self.source_note = source_note


class SimplePipeline:
    """Simplified pipeline: Extract → Categorize → Table Output"""
    
    def __init__(self, openai_client: OpenAIClient, output_dir: Path):
        self.openai = openai_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.extractor = ExcelExtractor()
        self.categorizer = SimpleCategorizer(openai_client)
        self.correlation_analyzer = CorrelationAnalyzer()
        self.impact_analyzer = ImpactAnalyzer()
        self.styler = ExcelStyler()
        
        # Results storage
        self.all_data_points: List[DataPoint] = []
        self.country_scores = {}
        self.correlations = {}
        self.insights = []
        self.impact_scores = {}
    
    def process_countries(self, excel_files: List[Path]) -> Path:
        """Process multiple countries and create final table with async categorization"""
        
        self.all_data_points = []
        
        # Step 1: Extract all data first (Python - fast and doesn't need API)
        logger.info("Extracting data from all Excel files...")
        countries_raw_data = []
        
        for excel_file in excel_files:
            country_name = self._extract_country_name(excel_file)
            logger.info(f"Extracting data from {excel_file.name}")
            
            try:
                # Extract structured data
                country_data = self.extractor.extract_country_data(excel_file)
                if not country_data:
                    logger.warning(f"No data extracted for {country_name}")
                    continue
                
                # Collect all data points
                raw_data_points = self._collect_all_data_points(country_data)
                
                if not raw_data_points:
                    logger.warning(f"No data points found for {country_name}")
                    continue
                
                # Store for async processing
                countries_raw_data.append((country_name, raw_data_points, country_data))
                logger.info(f"✓ {country_name}: {len(raw_data_points)} data points extracted")
                
            except Exception as e:
                logger.error(f"Error extracting {country_name}: {str(e)}")
                continue
        
        if not countries_raw_data:
            logger.error("No data extracted from any countries")
            return None
        
        # Step 2: Parallel AI categorization
        logger.info(f"Starting parallel categorization for {len(countries_raw_data)} countries...")
        
        # Prepare data for async categorization
        countries_for_categorization = [
            (country_name, raw_data_points) 
            for country_name, raw_data_points, _ in countries_raw_data
        ]
        
        # Run async categorization
        import os
        openai_api_key = os.getenv("OPENAI_API_KEY")
        categorized_results = run_async_categorization(openai_api_key, countries_for_categorization)
        
        # Step 3: Combine results
        logger.info("Combining categorization results...")
        
        for country_name, raw_data_points, country_data in countries_raw_data:
            if country_name not in categorized_results:
                logger.warning(f"No categorization results for {country_name}")
                continue
                
            categorized = categorized_results[country_name]
            
            # Create structured data points
            for raw_dp, categorized_dp in zip(raw_data_points, categorized):
                data_point = DataPoint(
                    country=country_name,
                    item=raw_dp.get('item', 'Unknown'),
                    raw_value=raw_dp.get('value', ''),
                    category=categorized_dp.category,
                    source_note=self._format_source_note(raw_dp, country_data)
                )
                self.all_data_points.append(data_point)
            
            logger.info(f"✓ {country_name}: {len(raw_data_points)} data points categorized")
        
        # Step 5: Perform correlation analysis
        if len(self.all_data_points) > 0:
            logger.info("Performing correlation analysis...")
            
            # Convert data points to format for correlation analysis
            data_for_analysis = []
            for dp in self.all_data_points:
                data_for_analysis.append({
                    'country': dp.country,
                    'item': dp.data_point,
                    'value': dp.raw_value,
                    'category': dp.category,
                    'source': dp.source_note
                })
            
            # Calculate scores and correlations
            self.country_scores = self.correlation_analyzer.calculate_country_scores(data_for_analysis)
            
            if len(self.country_scores) >= 3:
                self.correlations = self.correlation_analyzer.calculate_correlations(self.country_scores)
                self.insights = self.correlation_analyzer.generate_insights(
                    self.country_scores, self.correlations
                )
                
                # Calculate impact scores for individual items
                self.impact_scores = self.impact_analyzer.calculate_item_impacts(
                    self.country_scores, data_for_analysis
                )
        
        # Step 6: Output comprehensive analysis (Python handles formatting)
        return self._create_excel_table()
    
    def _extract_country_name(self, excel_file: Path) -> str:
        """Extract country name from filename"""
        name = excel_file.stem
        # Remove date prefix if present
        if name.startswith('240312 ') or name.startswith('240313 '):
            name = name[7:]
        return name
    
    def _collect_all_data_points(self, country_data) -> List[Dict[str, Any]]:
        """Collect all data points from extracted country data"""
        data_points = []
        
        # From national report
        if country_data.national_report:
            for data_block in country_data.national_report.data_blocks:
                for data_point in data_block.data_points:
                    data_points.append({
                        'item': data_point.item,
                        'value': data_point.value,
                        'notes': data_point.notes,
                        'source': data_point.source,
                        'block_category': data_block.category,
                        'context': 'national'
                    })
        
        # From organizations
        for org_profile in country_data.organization_profiles:
            for data_block in org_profile.data_blocks:
                for data_point in data_block.data_points:
                    data_points.append({
                        'item': data_point.item,
                        'value': data_point.value,
                        'notes': data_point.notes,
                        'source': data_point.source,
                        'block_category': data_block.category,
                        'context': org_profile.organization_name
                    })
        
        return data_points
    
    def _format_source_note(self, raw_dp: Dict[str, Any], country_data) -> str:
        """Format source note in user's preferred style"""
        parts = []
        
        # Add context
        context = raw_dp.get('context', '')
        if context and context != 'national':
            parts.append(f"[{context}]")
        
        # Add block category info
        block_cat = raw_dp.get('block_category', '')
        if block_cat:
            parts.append(f"{block_cat}")
        
        # Add notes if available
        notes = raw_dp.get('notes')
        if notes:
            parts.append(f"{notes}")
        
        # Add source URL if available
        source = raw_dp.get('source')
        if source:
            parts.append(f"{source}")
        
        return '; '.join(parts) if parts else 'Direct from Excel document'
    
    def _create_excel_table(self) -> Path:
        """Create comprehensive Excel analysis with multiple tabs"""
        
        if not self.all_data_points:
            logger.warning("No data points to export")
            return None
        
        # Create Excel file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"soft_power_analysis_{timestamp}.xlsx"
        output_path = self.output_dir / filename
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Tab 1: Rankings and Sources (Raw Data)
            self._create_raw_data_tab(writer)
            
            # Tab 2: Infrastructure and Assets
            self._create_infrastructure_assets_tab(writer)
            
            # Tab 3: Calculations and Correlations
            self._create_correlations_tab(writer)
            
            # Tab 4: Impact Analysis
            self._create_impact_analysis_tab(writer)
            
            # Tab 5: Synthesis and Summary
            self._create_synthesis_tab(writer)
            
            # All sheets are now styled individually in their respective methods
        
        logger.info(f"✓ Comprehensive analysis created: {output_path}")
        logger.info(f"✓ Total data points: {len(self.all_data_points)}")
        logger.info(f"✓ Countries analyzed: {len(self.country_scores)}")
        if self.correlations:
            logger.info("✓ Correlation analysis completed")
        
        return output_path
    
    def _create_raw_data_tab(self, writer):
        """Tab 1: Raw data with rankings and sources"""
        data = []
        for dp in self.all_data_points:
            data.append({
                'Country': dp.country,
                'Data Point': dp.data_point,
                'Raw Value': dp.raw_value,
                'Category': dp.category,
                'Source/Notes': dp.source_note
            })
        
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name='1. Rankings and Sources', index=False)
        
        # Apply styling
        ws = writer.sheets['1. Rankings and Sources']
        self.styler.style_header_row(ws, 1, df.shape[1])
        self.styler.style_data_rows(ws, 2, df.shape[0] + 1, category_col=4)  # Category is column 4
        self.styler.auto_adjust_columns(ws)
        self.styler.freeze_header_row(ws)
    
    def _create_infrastructure_assets_tab(self, writer):
        """Tab 2: Infrastructure and Assets breakdown"""
        # Filter for Infrastructure and Assets only
        infra_assets_data = []
        for dp in self.all_data_points:
            if dp.category in ['Infrastructure', 'Assets']:
                infra_assets_data.append({
                    'Country': dp.country,
                    'Category': dp.category,
                    'Item': dp.data_point,
                    'Details': dp.raw_value,
                    'Context': dp.source_note
                })
        
        df = pd.DataFrame(infra_assets_data)
        
        # Add summary by country
        summary_data = []
        for country in df['Country'].unique():
            country_df = df[df['Country'] == country]
            summary_data.append({
                'Country': country,
                'Infrastructure Count': len(country_df[country_df['Category'] == 'Infrastructure']),
                'Assets Count': len(country_df[country_df['Category'] == 'Assets']),
                'Total': len(country_df)
            })
        
        # Write main data
        main_start_row = len(summary_data) + 4  # +4 for spacing
        df.to_excel(writer, sheet_name='2. Infrastructure & Assets', index=False, startrow=main_start_row)
        
        # Write summary at top
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='2. Infrastructure & Assets', index=False)
        
        # Apply styling
        ws = writer.sheets['2. Infrastructure & Assets']
        
        # Style summary section
        self.styler.style_header_row(ws, 1, summary_df.shape[1])
        self.styler.style_data_rows(ws, 2, len(summary_data) + 1)
        
        # Add section header
        self.styler.add_section_headers(ws, main_start_row, 1, "Detailed Infrastructure & Assets Breakdown")
        
        # Style main data section
        self.styler.style_header_row(ws, main_start_row + 1, df.shape[1])
        self.styler.style_data_rows(ws, main_start_row + 2, main_start_row + df.shape[0] + 1, category_col=2)  # Category is column 2
        
        self.styler.auto_adjust_columns(ws)
        self.styler.freeze_header_row(ws)
    
    def _create_correlations_tab(self, writer):
        """Tab 3: Calculations and Correlations"""
        if not self.country_scores:
            return
        
        # Create correlation summary
        corr_summary = self.correlation_analyzer.create_correlation_summary(self.country_scores)
        
        # Add weighted sum columns to the summary
        for idx, row in corr_summary.iterrows():
            country = row['Country']
            if country in self.country_scores:
                corr_summary.at[idx, 'Infrastructure Weighted Sum'] = self.country_scores[country].get('infrastructure_weighted_sum', 0)
                corr_summary.at[idx, 'Assets Weighted Sum'] = self.country_scores[country].get('assets_weighted_sum', 0)
        
        # Write main scores table
        corr_summary.to_excel(writer, sheet_name='3. Calculations & Correlations', index=False)
        
        ws = writer.sheets['3. Calculations & Correlations']
        
        # Style main scores table
        self.styler.style_header_row(ws, 1, corr_summary.shape[1])
        self.styler.style_data_rows(ws, 2, corr_summary.shape[0] + 1)
        
        # Apply score-based conditional formatting to score columns
        score_columns = ['B', 'C', 'D', 'E']  # Infrastructure, Assets, Overall Score columns
        self.styler.style_score_columns(ws, score_columns, 2, corr_summary.shape[0] + 1)
        
        # Add correlation results
        if self.correlations:
            corr_data = []
            for corr_name, (r, p) in self.correlations.items():
                corr_data.append({
                    'Correlation': corr_name,
                    'Pearson r': round(r, 3),
                    'p-value': round(p, 3),
                    'Significance': 'Significant' if p < 0.05 else 'Not significant',
                    'Interpretation': self._interpret_correlation(r, p)
                })
            
            corr_df = pd.DataFrame(corr_data)
            corr_start_row = len(corr_summary) + 4
            
            # Add correlation section header
            self.styler.add_section_headers(ws, corr_start_row, 1, "Correlation Analysis")
            
            corr_df.to_excel(writer, sheet_name='3. Calculations & Correlations', 
                           index=False, startrow=corr_start_row + 1)
            
            # Style correlation table
            self.styler.style_correlation_table(ws, corr_start_row + 2, corr_start_row + len(corr_df) + 1)
        
        # Add outcome details for each country
        outcome_details_row = len(corr_summary) + len(self.correlations) + 6 if self.correlations else len(corr_summary) + 3
        
        # Write outcome calculation details
        writer.sheets['3. Calculations & Correlations'].cell(row=outcome_details_row, column=1, value='Outcome Score Calculations:')
        
        details_data = []
        for country, scores in self.country_scores.items():
            for index_name, index_data in scores['outcome_details'].items():
                details_data.append({
                    'Country': country,
                    'Index': index_name,
                    'Raw Rank': index_data['raw'],
                    'Normalized (0-1)': round(index_data['normalized'], 3)
                })
        
        if details_data:
            details_df = pd.DataFrame(details_data)
            details_df.to_excel(writer, sheet_name='3. Calculations & Correlations',
                              index=False, startrow=outcome_details_row+1)
        
        # Add weighted items breakdown
        weights_row = outcome_details_row + len(details_data) + 4 if details_data else outcome_details_row + 2
        writer.sheets['3. Calculations & Correlations'].cell(row=weights_row, column=1, value='Weighted Items Breakdown:')
        
        # Collect all weighted items
        weighted_items_data = []
        for country, scores in self.country_scores.items():
            # Infrastructure items
            for item in scores.get('infrastructure_items', []):
                weighted_items_data.append({
                    'Country': country,
                    'Category': 'Infrastructure',
                    'Item': item['item'],
                    'Weight': item['weight'],
                    'Value': item['value']
                })
            # Asset items
            for item in scores.get('asset_items', []):
                weighted_items_data.append({
                    'Country': country,
                    'Category': 'Assets',
                    'Item': item['item'],
                    'Weight': item['weight'],
                    'Value': item['value']
                })
        
        if weighted_items_data:
            weighted_df = pd.DataFrame(weighted_items_data)
            weighted_df.to_excel(writer, sheet_name='3. Calculations & Correlations',
                               index=False, startrow=weights_row+1)
            
            # Style weighted items table
            self.styler.style_header_row(ws, weights_row + 2, weighted_df.shape[1])
            self.styler.style_data_rows(ws, weights_row + 3, weights_row + len(weighted_df) + 2, category_col=2)
        
        # Final touches for correlations tab
        self.styler.auto_adjust_columns(ws)
        self.styler.freeze_header_row(ws)
    
    def _create_impact_analysis_tab(self, writer):
        """Tab 4: Impact Analysis of individual items on outcomes"""
        if not self.impact_scores:
            return
        
        # Create impact summary
        impact_df = self.impact_analyzer.create_impact_summary_df(self.impact_scores)
        
        # Write main impact table
        impact_df.to_excel(writer, sheet_name='4. Impact Analysis', index=False)
        
        # Apply styling
        ws = writer.sheets['4. Impact Analysis']
        
        # Style main impact table
        self.styler.style_header_row(ws, 1, impact_df.shape[1])
        self.styler.style_data_rows(ws, 2, impact_df.shape[0] + 1)
        
        # Special styling for impact scores (0-10 scale)
        self.styler.style_impact_scores(ws, 2, 2, impact_df.shape[0] + 1)  # Impact Score column is 2nd
        
        # Add explanation
        explanation_row = len(impact_df) + 4
        writer.sheets['4. Impact Analysis'].cell(row=explanation_row, column=1, value='Impact Score Methodology:')
        writer.sheets['4. Impact Analysis'].cell(row=explanation_row+1, column=1, 
            value='• Impact Score (0-10): Composite of presence impact, weighted contribution, and synergy effects')
        writer.sheets['4. Impact Analysis'].cell(row=explanation_row+2, column=1,
            value='• Presence Impact: Difference in outcomes for countries WITH vs WITHOUT the item')
        writer.sheets['4. Impact Analysis'].cell(row=explanation_row+3, column=1,
            value='• Weighted Contribution: Share of outcome score based on item weight')
        writer.sheets['4. Impact Analysis'].cell(row=explanation_row+4, column=1,
            value='• Synergy Effects: Bonus when item appears with common partners')
        
        # Add top insights
        insights_row = explanation_row + 7
        writer.sheets['4. Impact Analysis'].cell(row=insights_row, column=1, value='Key Impact Findings:')
        
        # Get top 5 impact items
        top_items = impact_df.head(5)
        for i, (idx, row) in enumerate(top_items.iterrows()):
            insight = f"• {row['Asset/Infrastructure']}: Impact score {row['Impact Score (0-10)']} "
            insight += f"(+{row['Impact Difference']:.3f} outcome difference)"
            writer.sheets['4. Impact Analysis'].cell(row=insights_row+i+1, column=1, value=insight)
        
        # Add items with negative impact
        negative_items = impact_df[impact_df['Impact Difference'] < 0].head(3)
        if not negative_items.empty:
            neg_row = insights_row + 7
            writer.sheets['4. Impact Analysis'].cell(row=neg_row, column=1, value='Items with Negative Correlation:')
            for i, (idx, row) in enumerate(negative_items.iterrows()):
                insight = f"• {row['Asset/Infrastructure']}: {row['Impact Difference']:.3f} outcome difference"
                writer.sheets['4. Impact Analysis'].cell(row=neg_row+i+1, column=1, value=insight)
        
        # Final touches for impact analysis tab
        self.styler.auto_adjust_columns(ws)
        self.styler.freeze_header_row(ws)
    
    def _create_synthesis_tab(self, writer):
        """Tab 5: Synthesis and Summary"""
        # Create synthesis data
        synthesis_data = []
        
        # Country scores summary
        for country, scores in self.country_scores.items():
            synthesis_data.append({
                'Country': country,
                'Synthesized Outcomes Score': round(scores['outcome_score'], 2),
                'Infrastructure Score': round(scores['infrastructure_score'], 2),
                'Assets Score': round(scores['assets_score'], 2),
                'Overall Score': round(scores['overall_score'], 2)
            })
        
        synthesis_df = pd.DataFrame(synthesis_data).sort_values('Overall Score', ascending=False)
        synthesis_df.to_excel(writer, sheet_name='5. Synthesis & Summary', index=False)
        
        # Apply styling
        ws = writer.sheets['5. Synthesis & Summary']
        
        # Style main synthesis table
        self.styler.style_header_row(ws, 1, synthesis_df.shape[1])
        self.styler.style_data_rows(ws, 2, synthesis_df.shape[0] + 1)
        
        # Apply conditional formatting to score columns
        score_columns = ['B', 'C', 'D', 'E']  # All score columns
        self.styler.style_score_columns(ws, score_columns, 2, synthesis_df.shape[0] + 1)
        
        # Add insights
        if self.insights:
            insights_row = len(synthesis_df) + 3
            writer.sheets['5. Synthesis & Summary'].cell(row=insights_row, column=1, value='Key Insights:')
            
            for i, insight in enumerate(self.insights):
                writer.sheets['5. Synthesis & Summary'].cell(row=insights_row+i+1, column=1, value=f"• {insight}")
        
        # Add methodology note
        method_row = len(synthesis_df) + len(self.insights) + 6 if self.insights else len(synthesis_df) + 3
        writer.sheets['5. Synthesis & Summary'].cell(row=method_row, column=1, value='Methodology:')
        writer.sheets['5. Synthesis & Summary'].cell(row=method_row+1, column=1, 
            value='• Outcome scores: Average of normalized index rankings (0-1 scale, 1=best)')
        writer.sheets['5. Synthesis & Summary'].cell(row=method_row+2, column=1,
            value='• Infrastructure/Assets scores: Weighted sum normalized by max possible weight')
        
        # Import weights for display
        from .weights_config import get_category_weights
        cat_weights = get_category_weights()
        writer.sheets['5. Synthesis & Summary'].cell(row=method_row+3, column=1,
            value=f'• Overall score: {cat_weights["outcome"]*100:.0f}% Outcomes, {cat_weights["infrastructure"]*100:.0f}% Infrastructure, {cat_weights["assets"]*100:.0f}% Assets')
        
        # Add weight information
        writer.sheets['5. Synthesis & Summary'].cell(row=method_row+5, column=1, value='Weight System:')
        writer.sheets['5. Synthesis & Summary'].cell(row=method_row+6, column=1,
            value='• Budget items: 3.0x weight (critical importance)')
        writer.sheets['5. Synthesis & Summary'].cell(row=method_row+7, column=1,
            value='• Lead ministries: 2.0x weight (organizational capacity)')
        writer.sheets['5. Synthesis & Summary'].cell(row=method_row+8, column=1,
            value='• Flagship programs: 2.5x weight + country multipliers')
        writer.sheets['5. Synthesis & Summary'].cell(row=method_row+9, column=1,
            value='• Standard programs: 1.0-1.5x weight')
        
        # Final touches for synthesis tab
        self.styler.auto_adjust_columns(ws)
        self.styler.freeze_header_row(ws)
        
        # Add color legend at the bottom
        legend_row = method_row + 12
        self.styler.add_legend(ws, legend_row)
    
    def _interpret_correlation(self, r: float, p: float) -> str:
        """Interpret correlation coefficient"""
        if p > 0.05:
            return "No significant relationship"
        
        strength = ""
        if abs(r) < 0.3:
            strength = "Weak"
        elif abs(r) < 0.7:
            strength = "Moderate"
        else:
            strength = "Strong"
        
        direction = "positive" if r > 0 else "negative"
        
        if "Assets" in str(r) and r < -0.5:
            return f"{strength} {direction} - high inputs don't guarantee high outcomes"
        
        return f"{strength} {direction} relationship"
    
