#  Soft Power Analysis Pipeline
<img width="1840" height="563" alt="image" src="https://github.com/user-attachments/assets/04e150f6-a948-4ffe-8b0a-65dda50c4fa8" />

*An AI-powered pipeline for analyzing relationships between cultural investments and international reputation outcomes across countries*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green.svg)](https://openai.com)

##  What This Does

This pipeline analyzes **soft power effectiveness** by examining how cultural investments (infrastructure & assets) correlate with international reputation outcomes. It processes country data through a sophisticated AI-powered workflow to answer the critical question: *"Which soft power investments actually improve international standing?"*

**Key Insight Example**: The analysis revealed that high cultural investment doesn't always guarantee better outcomes - some countries with extensive programs ranked lower on international indices, suggesting that *strategy matters more than scale*.

##  Key Features

- **AI-Powered Categorization**: Automatically sorts 1000+ data points into Infrastructure, Assets, and Outcomes
- **Advanced Analytics**: Weighted scoring, correlation analysis, and impact measurement  
- **Professional Reports**: Multi-tab Excel outputs with color-coded insights
- **High Performance**: Async processing handles large datasets in ~4 minutes
- **Audit Trail**: Complete traceability of all AI decisions and calculations

## 🚀 Quick Start

Requires ICR Research Data Tables of 2024 Country Analysis

### 1. Install & Setup
```bash
git clone [[repository-url](https://github.com/RinDig/SoftPowerAiPipeline)]
cd soft-power-pipeline
pip install -r requirements.txt

# Add your OpenAI API key to .env
echo "OPENAI_API_KEY=your-key-here" > .env
```

### 2. Run Analysis
```bash
# Analyze specific countries
python simple_main.py --countries Canada Germany Japan --data-dir "2024 Data tables for BC"

# Analyze all available countries  
python simple_main.py --all-countries --data-dir "2024 Data tables for BC"
```

### 3. View Results
Open the generated Excel file in `output/` to explore the comprehensive analysis across 5 professional tabs.

## 📈 How It Works

### The Three-Phase Analysis

#### Phase 1: Data Extraction & AI Categorization
- **Excel Processing**: Extracts structured data from country spreadsheets
- **AI Classification**: OpenAI GPT-4 categorizes each data point as:
  - 🏛️ **Infrastructure**: Budgets, ministries, organizational capacity
  - 🎯 **Assets**: Cultural programs, institutions, initiatives  
  - 📊 **Outcomes**: International rankings and reputation indices

#### Phase 2: Advanced Analytics
- **Score Normalization**: Rankings converted to 0-1 scale using *accurate* country totals per index
- **Weighted Scoring**: Critical items (budgets, lead ministries) receive higher weights
- **Impact Analysis**: Measures individual contribution of each asset/infrastructure item

#### Phase 3: Correlation & Synthesis  
- **Statistical Analysis**: Pearson correlations reveal relationships between investment and outcomes
- **Professional Reporting**: 5-tab Excel output with color-coded insights and audit trails
- **Strategic Insights**: Identifies most impactful soft power approaches

### Sample Analysis Pipeline
```
Country Excel Files → AI Categorization → Weighted Scoring → Statistical Analysis → Strategic Insights
     (Raw Data)      →    (792 items)    →   (Normalized)  →   (Correlations)   →  (Excel Report)
```

## 📋 Output Structure

The pipeline generates a comprehensive Excel report with 5 professionally styled tabs:

### 🗃️ Tab 1: Rankings and Sources
- All raw data with AI-assigned categories
- Complete audit trail with source citations
- Color-coded by category (Infrastructure/Assets/Outcomes)

### 🏗️ Tab 2: Infrastructure & Assets  
- Filtered breakdown of soft power resources
- Country summary statistics
- Detailed organizational capacity analysis

### 📊 Tab 3: Calculations & Correlations
- **Country Scores**: Normalized 0-1 scale for comparison
- **Statistical Analysis**: Pearson correlations with significance testing
- **Weighted Breakdown**: Individual item contributions with multipliers

### 🎯 Tab 4: Impact Analysis
- **Impact Scores (0-10)**: Which specific assets/infrastructure drive outcomes
- **Presence Analysis**: Comparative effectiveness across countries  
- **Synergy Effects**: Items that work better in combination

### 📋 Tab 5: Synthesis & Summary
- Final country rankings with overall scores
- Key strategic insights and patterns
- Methodology documentation and weight explanations

## 🧮 Methodology

### Accurate Score Normalization
Rankings are normalized using **actual country totals** per index (not assumed values):
- Brand Finance Soft Power Index: 100 countries → Rank 13 = 0.879 score
- Good Country Index: 169 countries → Rank 18 = 0.899 score  
- Social Progress Index: 170 countries → Rank 8 = 0.959 score

### Weighted Scoring System
- **Budget items**: 3.0x weight (financial commitment is critical)
- **Lead ministries**: 2.0x weight (organizational capacity)
- **Flagship programs**: 2.5x weight + country multipliers
- **Standard programs**: 1.0-1.5x weight

### Statistical Analysis
- **Correlation Analysis**: Examines Assets↔Outcomes, Infrastructure↔Outcomes, Assets↔Infrastructure
- **Impact Measurement**: Presence/absence comparison shows individual item effectiveness
- **Significance Testing**: p-values and confidence intervals for all relationships

## 📂 Data Categories

The AI automatically classifies data points into three strategic categories:

- **🏛️ Infrastructure**: Organizational capacity, budgets, ministries, institutional frameworks
- **🎯 Assets**: Specific programs, initiatives, institutions, cultural centers  
- **📊 Outcomes**: Performance measures, rankings, indices, reputation indicators

## 💡 Example Insights Generated

Real findings from the analysis:

- **"High investment ≠ high outcomes"**: Some countries with extensive cultural programs ranked lower on international indices
- **Budget correlation**: Countries with larger soft power budgets showed 0.65 correlation with outcomes  
- **Flagship program effect**: Signature initiatives (like Confucius Institutes) had 2.5x impact vs standard programs
- **Regional patterns**: European countries showed different investment→outcome patterns than Asian countries

## ⚙️ Technical Requirements

- **Python**: 3.8+ with async/await support
- **OpenAI API**: GPT-4-mini for structured output parsing
- **Excel Processing**: openpyxl for professional styling and formatting
- **Statistics**: scipy for Pearson correlation analysis

## 🚀 Performance

- **Processing Speed**: ~4 minutes for 18 countries (792 data points)
- **Concurrent Processing**: 4 parallel API calls with chunked data
- **Memory Efficient**: Streaming Excel processing for large datasets
- **Error Resilient**: Graceful handling of malformed data and API failures

## 🏗️ Project Structure

```
soft-power-pipeline/
├── simple_main.py              # 🚀 Main entry point
├── src/soft_power_pipeline/
│   ├── simple_pipeline.py      # 🔧 Core pipeline orchestration
│   ├── async_categorizer.py    # ⚡ Parallel AI processing  
│   ├── correlation_analyzer.py # 📊 Statistical analysis engine
│   ├── impact_analyzer.py      # 🎯 Individual item impact scoring
│   ├── excel_styler.py        # 🎨 Professional Excel formatting
│   ├── weights_config.py       # ⚖️ Configurable weight system
│   ├── ai/
│   │   └── openai_client.py    # 🤖 OpenAI API integration
│   ├── extractors/
│   │   └── excel_extractor.py  # 📑 Excel data extraction
│   └── models/
│       └── excel_models.py     # 📋 Pydantic data models
├── 2024 Data tables for BC/    # 📁 Input Excel files
├── output/                     # 📊 Generated analysis reports
├── requirements.txt            # 📦 Python dependencies
├── CLAUDE.md                  # 🧠 AI assistant project context
└── README.md                  # 📖 This documentation
```

## 🔧 Configuration & Advanced Usage

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your-openai-api-key

# Optional performance tuning
MAX_CONCURRENT_REQUESTS=4
CHUNK_SIZE=25
REQUEST_TIMEOUT=60
```

### Custom Weight Configuration
Edit `weights_config.py` to adjust item importance:
```python
ITEM_WEIGHTS = {
    'budget': 3.0,              # Critical financial indicators
    'lead ministry': 2.0,       # Organizational capacity 
    'confucius institute': 2.5, # Flagship programs
    'cultural center': 1.5      # Standard programs
}
```

### Command Line Options
```bash
# Full parameter list
python simple_main.py \
  --countries Canada France Germany \
  --data-dir "custom/data/path" \
  --output-dir "custom/output" \
  --async-workers 6 \
  --chunk-size 30
```

## 🔍 Troubleshooting

### Common Issues
- **API Rate Limits**: Reduce `MAX_CONCURRENT_REQUESTS` to 2-3
- **Memory Issues**: Lower `CHUNK_SIZE` to 15-20 for large datasets  
- **Excel Errors**: Ensure input files match expected format (see CLAUDE.md)
- **Missing Data**: Pipeline gracefully handles incomplete data with warnings

### Debug Mode
```bash
export LOG_LEVEL=DEBUG
python simple_main.py --countries Canada --verbose
```

## 📜 License & Citation

This project is licensed under the MIT License. If you use this pipeline in academic research, please cite:

```bibtxt
@software{soft_power_pipeline_2024,
  title={AI-Powered Soft Power Analysis Pipeline},
  author={[J.E. Van Clief]},
  year={2024},
  description={Automated analysis of cultural investment effectiveness using OpenAI GPT-4},
  url={[repository-url]}
}
```

*Built with ❤️ for international relations researchers and policy analysts*
