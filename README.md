# Funnel Decomposition Analysis Package

A comprehensive Python package for analyzing booking funnel changes using multiple decomposition methodologies. Decompose booking changes into interpretable effects including volume, mix, approval rates, and booking rates across multiple dimensions.

## Features

### 🔄 Multiple Decomposition Methodologies

1. **Symmetric Decomposition** (Order-Independent)
   - Uses midpoint methodology with average values from both periods
   - Order-independent: results don't depend on effect calculation sequence
   - Balanced view with no bias toward either period
   - Includes interaction effect for perfect reconciliation

2. **Hierarchical Decomposition** (Sequential Waterfall)
   - Sequential waterfall where each effect builds on previous steps
   - Intuitive step-by-step transformation logic
   - Traditional approach familiar to business users
   - Perfect reconciliation without residual

### 📊 Six Core Effects

All methodologies decompose booking changes into:

1. **Volume Effect**: Change in total application volume
2. **Mix Effect**: Change in segment distribution (dimensional mix)
3. **Straight Approval Effect**: Change in straight approval rates
4. **Conditional Approval Effect**: Change in conditional approval rates
5. **Straight Booking Effect**: Change in straight booking rates (given straight approval)
6. **Conditional Booking Effect**: Change in conditional booking rates (given conditional approval)

### 🎯 Advanced Capabilities

- **Multi-Lender Analysis**: Aggregate and lender-attributed decomposition
- **Weekly/Monthly Granularity**: Flexible `date_column` parameter for any time frequency
- **Dimensional Breakdowns**: Analyze by FICO bands, offer competition tier, product line
- **Segment-Level Detail**: Deep-dive into specific dimension combinations
- **Perfect Reconciliation**: All methods guarantee exact reconciliation to actual booking changes

### 📈 Rich Visualizations

Shared visualization engine across all methodologies:

- **Waterfall Grids** (2×2 layout): Overall + 3 dimensional breakdowns
- **Dimensional Stacked Waterfalls**: Effects broken down by dimension values
- **Dimension Drilldowns**: Horizontal bar charts for detailed analysis
- **Multi-Lender Comparisons**: Side-by-side aggregate vs by-lender views
- **Chart Export Utilities**: Extract and export individual charts as PNG

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd funnel_decomposition
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import sys; sys.path.insert(0, 'src'); import symmetric_decomposition_calculator; print('Installation successful!')"
```

## Quick Start

### 🚀 Start Here: Comprehensive Demo

The **`funnel_decomposition_demo.ipynb`** notebook provides a complete walkthrough of all package capabilities:

```bash
jupyter notebook funnel_decomposition_demo.ipynb
```

This single notebook demonstrates:
- Symmetric and hierarchical decomposition
- Weekly analysis with flexible date columns
- Multi-lender analysis
- Dimension drilldowns
- Segment-level detail
- Side-by-side methodology comparisons

### 📝 Basic Usage Example

```python
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, 'src')

import symmetric_decomposition_calculator
import visualization_engine

# Load your data
df = pd.read_csv('data/funnel_data.csv')
df['month_begin_date'] = pd.to_datetime(df['month_begin_date'])

# Calculate symmetric decomposition
results = symmetric_decomposition_calculator.calculate_decomposition(
    df=df,
    date_a='2023-06-01',
    date_b='2024-06-01',
    lender='ACA'
)

# View summary
print(results.summary)

# Create visualizations
fig = visualization_engine.create_waterfall_grid(
    summary=results.summary,
    segment_detail=results.segment_detail,
    lender='ACA'
)
```

## Notebooks

### 🌟 Main Demo
- **`funnel_decomposition_demo.ipynb`** - Comprehensive demo showcasing all features

### 📋 Method-Specific Templates
- **`symmetric_funnel_decomp.ipynb`** - Symmetric decomposition template
- **`hier_funnel_decomp.ipynb`** - Hierarchical decomposition template
- **`symmetric_funnel_decomp_multi_lender.ipynb`** - Multi-lender analysis template
- **`symmetric_funnel_decomp_weekly.ipynb`** - Weekly analysis template

### 🛠️ Utilities
- **`chart_export.ipynb`** - Chart extraction and PNG export utilities

## Project Structure

```
funnel_decomposition/
├── src/                                    # Core source code
│   ├── __init__.py
│   ├── symmetric_decomposition_calculator.py  # Symmetric methodology
│   ├── hier_decomposition_calculator.py       # Hierarchical methodology
│   ├── model_decomposition_calculator.py      # Model-based approach
│   ├── visualization_engine.py                # All chart types
│   ├── utils.py                              # Shared utilities
│   └── data_transformation.py                # Data prep utilities
│
├── funnel_decomposition_demo.ipynb        # ⭐ START HERE
├── chart_export.ipynb                     # Chart export utilities
├── symmetric_funnel_decomp.ipynb          # Templates
├── symmetric_funnel_decomp_multi_lender.ipynb
├── symmetric_funnel_decomp_weekly.ipynb
├── hier_funnel_decomp.ipynb
│
├── requirements.txt                       # Python dependencies
└── README.md                             # This file
```

## Data Requirements

### Input Data Format

Your DataFrame must contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| `lender` | str | Lender identifier |
| `month_begin_date` or custom | datetime | Date identifier (configurable via `date_column`) |
| `fico_bands` | str | FICO score bands (e.g., 'High_FICO', 'Med_FICO', 'Low_FICO', 'Null_FICO') |
| `offer_comp_tier` | str | Offer competition tier (e.g., 'solo_offer', 'multi_best', 'multi_other') |
| `prod_line` | str | Product line (e.g., 'Used', 'VMax') |
| `num_tot_apps` | int | Total applications for this lender/period |
| `num_tot_bks` | float | Total bookings for this lender/period |
| `pct_of_total_apps` | float | Segment's % of total apps (e.g., 0.15 = 15%) |
| `str_apprv_rate` | float | Straight approval rate (e.g., 0.45 = 45%) |
| `str_bk_rate` | float | Straight booking rate (e.g., 0.60 = 60%) |
| `cond_apprv_rate` | float | Conditional approval rate |
| `cond_bk_rate` | float | Conditional booking rate |

### Data Example

```python
   lender month_begin_date fico_bands offer_comp_tier prod_line  num_tot_apps  num_tot_bks  pct_of_total_apps  str_apprv_rate  str_bk_rate  cond_apprv_rate  cond_bk_rate
0    ACA       2023-06-01  High_FICO      multi_best      Used         14573         5092           0.119658        0.505422     0.644748         0.244578      0.293948
1    ACA       2023-06-01  High_FICO      multi_best      VMax         14573         5092           0.076923        0.556486     0.714620         0.225156      0.238149
...
```

## Key Features in Detail

### 1. Flexible Date Column Support

All calculators support a `date_column` parameter for flexible time granularity:

```python
# Monthly analysis (default)
results = calculate_decomposition(
    df=df,
    date_a='2023-06-01',
    date_b='2024-06-01',
    lender='ACA'
)

# Weekly analysis
results = calculate_decomposition(
    df=df_weekly,
    date_a=week_1,
    date_b=week_2,
    lender='ACA',
    date_column='week_begin_date'  # Specify date column
)

# Custom date column
results = calculate_decomposition(
    df=df_custom,
    date_a='2023-Q2',
    date_b='2024-Q2',
    lender='ACA',
    date_column='quarter_begin_date'
)
```

### 2. Multi-Lender Analysis

Analyze multiple lenders simultaneously:

```python
results = symmetric_decomposition_calculator.calculate_multi_lender_decomposition(
    df=df,
    date_a='2023-06-01',
    date_b='2024-06-01'
)

# View aggregate summary
print(results.aggregate_summary)

# View lender-attributed summaries
print(results.lender_summaries)

# Create lender comparison visualizations
fig = visualization_engine.create_lender_waterfall_grid(
    lender_summaries=results.lender_summaries,
    aggregate_summary=results.aggregate_summary,
    metadata=results.metadata
)
```

### 3. Dimensional Analysis

Create detailed breakdowns by any dimension:

```python
# FICO band drilldown
fig_fico = visualization_engine.create_dimension_drilldown(
    segment_detail=results.segment_detail,
    dimension='fico_bands',
    lender='ACA'
)

# Offer competition tier drilldown
fig_comp = visualization_engine.create_dimension_drilldown(
    segment_detail=results.segment_detail,
    dimension='offer_comp_tier',
    lender='ACA'
)

# Product line drilldown
fig_prod = visualization_engine.create_dimension_drilldown(
    segment_detail=results.segment_detail,
    dimension='prod_line',
    lender='ACA'
)
```

### 4. Chart Export

Extract and export individual charts:

```python
# See chart_export.ipynb for detailed examples

# Extract individual charts from grid
charts = extract_individual_charts_from_grid(fig)
# Returns: [Overall, FICO, Comp Tier, Product Line]

# Export individual chart
export_chart(charts[0], 'output/overall_waterfall.png', dpi=300)

# Batch export
export_chart_list(charts, 'output/waterfall', prefix='chart')
```

## Results Structure

All decomposition methods return a `DecompositionResults` named tuple:

```python
results = calculate_decomposition(...)

# Access components
results.summary              # DataFrame: Aggregate effect impacts
results.segment_detail       # DataFrame: Segment-level breakdown
results.metadata            # dict: Calculation metadata

# Summary columns
# - effect_type: str (e.g., 'volume_effect', 'mix_effect', ...)
# - booking_impact: float (booking change attributed to this effect)

# Segment detail columns
# - Dimension identifiers: fico_bands, offer_comp_tier, prod_line
# - Period 1 metrics: period_1_date, period_1_total_apps, period_1_pct_of_total, ...
# - Period 2 metrics: period_2_date, period_2_total_apps, period_2_pct_of_total, ...
# - Delta metrics: delta_total_apps, delta_pct_of_total, delta_str_apprv_rate, ...
# - Effects: volume_effect, mix_effect, str_approval_effect, cond_approval_effect,
#           str_booking_effect, cond_booking_effect, total_effect
```

## Methodology Comparison

### When to Use Symmetric Decomposition

✅ **Best for:**
- Order-independent results are required
- Comparing multiple decompositions
- Academic or research settings
- Consistent methodology across analyses
- Balanced view with no period bias

### When to Use Hierarchical Decomposition

✅ **Best for:**
- Sequential logic matches business process
- Explaining step-by-step transformations
- Traditional waterfall approach is preferred
- Business stakeholders expect incremental impacts

### Key Differences

| Aspect | Symmetric | Hierarchical |
|--------|-----------|--------------|
| **Order** | Independent | Dependent |
| **Base values** | Averages | Period-specific |
| **Interaction effect** | Yes (residual) | No |
| **Sequential logic** | No | Yes |
| **Reconciliation** | Perfect | Perfect |

**Note:** Both methods guarantee perfect reconciliation to actual booking changes.

## Advanced Usage

### Segment-Level Analysis

```python
# Find top 5 segments by impact
top_segments = results.segment_detail.nlargest(5, 'total_effect')[[
    'fico_bands', 'offer_comp_tier', 'prod_line', 'total_effect'
]]
print(top_segments)

# Filter to specific dimension
high_fico = results.segment_detail[
    results.segment_detail['fico_bands'] == 'High_FICO'
]
```

### Export Results

```python
# Export summaries
results.summary.to_csv('summary.csv', index=False)
results.segment_detail.to_csv('segment_detail.csv', index=False)

# Export with metadata
import json
with open('metadata.json', 'w') as f:
    json.dump(results.metadata, f, indent=2)
```

### Custom Visualizations

```python
import matplotlib.pyplot as plt

# Use visualization engine components
from visualization_engine import _create_aggregate_waterfall

fig, ax = plt.subplots(figsize=(12, 8))
_create_aggregate_waterfall(
    ax=ax,
    summary=results.summary,
    lender='ACA'
)
plt.tight_layout()
plt.savefig('custom_waterfall.png', dpi=300, bbox_inches='tight')
```

## Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError: No module named 'symmetric_decomposition_calculator'`**

Solution:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))  # Add src to path
```

**Issue: `ValueError: No data found for [lender] on [date]`**

Solution:
- Verify date format matches your data
- Check lender name spelling
- Ensure date exists in dataset
- Verify date column name with `date_column` parameter

**Issue: `KeyError: 'month_begin_date'`**

Solution:
- Use `date_column` parameter to specify your date column name:
```python
results = calculate_decomposition(..., date_column='week_begin_date')
```

**Issue: Waterfall doesn't reconcile**

Solution:
- Verify all required columns are present
- Check for NaN values in rate columns
- Ensure `num_tot_apps` and `num_tot_bks` are consistent
- Contact support if issue persists (all methods guarantee reconciliation)

## Dependencies

- **pandas** ≥ 2.0.0 - Data manipulation
- **numpy** ≥ 1.24.0 - Numerical computing
- **matplotlib** ≥ 3.7.0 - Plotting
- **seaborn** ≥ 0.12.0 - Statistical visualizations
- **jupyter** ≥ 1.0.0 - Notebook interface
- **ipykernel** ≥ 6.25.0 - Jupyter kernel
- **python-dateutil** ≥ 2.8.0 - Date parsing
- **openpyxl** ≥ 3.1.0 - Excel I/O (optional)

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is proprietary and confidential.

## Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Contact the development team
- Review the comprehensive demo notebook for usage examples

## Changelog

### Latest Release (v2.0.0)

#### New Features
- ✅ Flexible `date_column` parameter for weekly/custom time granularities
- ✅ Comprehensive demo notebook combining all methodologies
- ✅ Chart export utilities with PNG export
- ✅ Multi-lender analysis with lender attribution
- ✅ Enhanced visualization engine with dimensional drilldowns

#### Improvements
- ✅ Cleaned template notebooks (outputs removed)
- ✅ Updated all calculators with date_column support
- ✅ Perfect reconciliation validation
- ✅ Comprehensive documentation

#### Bug Fixes
- ✅ Fixed date column validation in utils.py
- ✅ Resolved backward compatibility issues

---

**Package Version:** 2.0.0
**Last Updated:** November 2024
**Python Version:** 3.8+
