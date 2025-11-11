# Funnel Decomposition Analysis - Production Code

A hierarchical decomposition system to explain booking changes across multi-stage funnels with multiple segmentation dimensions.

## Overview

This repository contains production-ready code for **waterfall decomposition** analysis to answer questions like:
- "Bookings increased by 300 between Jan and Feb. What drove this change?"
- "How much was due to volume vs. rate improvements?"
- "Which FICO bands contributed most to the change?"

### Key Features

✅ **Exact Reconciliation**: All effects sum precisely to actual booking changes
✅ **6 Effect Types**: Volume, Mix, Str Approval Rate, Cond Approval Rate, Str Booking Rate, Cond Booking Rate
✅ **Multi-Dimensional Analysis**: Drill down by FICO band, offer comp tier, or product line
✅ **Rich Visualizations**: Waterfall charts and dimensional drilldowns
✅ **Excel/CSV Exports**: Stakeholder-ready summary tables
✅ **Data Transformation**: Convert between long and wide formats for ML modeling

---

## Repository Structure

```
prod_code/
├── src/
│   ├── __init__.py                            # Package exports
│   ├── hier_decomposition_calculator.py       # Hierarchical decomposition engine
│   ├── model_decomposition_calculator.py      # Model-based decomposition
│   ├── symmetric_decomposition_calculator.py  # Symmetric decomposition approach
│   ├── visualization_engine.py                # Charts and exports
│   ├── data_transformation.py                 # Pivot/unpivot utilities
│   └── utils.py                               # Helper functions
├── hier_funnel_decomp_template.ipynb          # Hierarchical decomposition template
├── model_funnel_decomp_template.ipynb         # Model-based decomposition template
├── symmetric_funnel_decomp.ipynb              # Symmetric decomposition template
├── symmetric_funnel_decomp_weekly.ipynb       # Weekly symmetric decomposition
├── symmetric_funnel_decomp_multi_lender.ipynb # Multi-lender symmetric decomposition
├── requirements.txt                           # Python dependencies
├── .gitignore                                 # Git ignore rules
└── README.md                                  # This file
```

---

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Use Template Notebooks

Choose from the following template notebooks to get started:

**Core Templates:**
- **hier_funnel_decomp_template.ipynb**: Hierarchical waterfall decomposition
- **model_funnel_decomp_template.ipynb**: Model-based decomposition

**Symmetric Decomposition Templates:**
- **symmetric_funnel_decomp.ipynb**: Symmetric decomposition approach
- **symmetric_funnel_decomp_weekly.ipynb**: Weekly period-over-period symmetric analysis
- **symmetric_funnel_decomp_multi_lender.ipynb**: Multi-lender comparison with symmetric decomposition

Each notebook includes:
- Data loading examples
- Complete analysis workflow
- Visualization generation
- Export functionality

### 3. Basic Usage Example

```python
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))

from hier_decomposition_calculator import calculate_decomposition
from visualization_engine import create_waterfall_grid, create_dimension_drilldown

# Load your data (must match required schema)
df = pd.read_csv('your_funnel_data.csv')
df['month_begin_date'] = pd.to_datetime(df['month_begin_date'])

# Calculate decomposition
results = calculate_decomposition(
    df=df,
    date_a='2023-01-01',
    date_b='2024-01-01',  # Year-over-Year
    lender='ACA'
)

# View summary
print(results.summary)

# Create visualizations
fig = create_waterfall_grid(
    summary=results.summary,
    segment_detail=results.segment_detail,
    lender='ACA',
    output_path='outputs/waterfalls/my_waterfall.png'
)

fig_fico = create_dimension_drilldown(
    segment_detail=results.segment_detail,
    dimension='fico_bands',
    lender='ACA',
    output_path='outputs/drilldowns/fico_drilldown.png'
)
```

---

## Data Schema

### Required Input Columns

| Column | Type | Description |
|--------|------|-------------|
| `lender` | str | Lender identifier (e.g., "ACA") |
| `month_begin_date` | datetime | First day of month |
| `fico_bands` | str | "High_FICO", "Med_FICO", "Low_FICO", "Null_FICO" |
| `offer_comp_tier` | str | "solo_offer", "multi_best", "multi_other" |
| `prod_line` | str | "Used", "VMax" |
| `num_tot_bks` | int | Total bookings for lender-month |
| `num_tot_apps` | int | Total applications for lender-month |
| `pct_of_total_apps` | float | % of total apps in this segment (0-1) |
| `str_apprv_rate` | float | % of segment apps with straight approval (0-1) |
| `str_bk_rate` | float | % of straight approvals that booked (0-1) |
| `cond_apprv_rate` | float | % of segment apps with conditional approval (0-1) |
| `cond_bk_rate` | float | % of conditional approvals that booked (0-1) |

### Data Constraints

1. **Segment Coverage**: Must have 24 segments per lender-month (4 FICO × 3 comp × 2 product)
2. **Mix Constraint**: `Σ pct_of_total_apps = 1.0` for each lender-month
3. **Booking Constraint**: `Σ segment_bookings = num_tot_bks` for each lender-month
4. **Rate Bounds**: All rates must be between 0 and 1

---

## How It Works

### Decomposition Formula

For each segment *s*, bookings change is decomposed as:

```
Δ Bookings[s] = Volume Effect
              + Mix Effect
              + Str Approval Rate Effect
              + Cond Approval Rate Effect
              + Str Booking Rate Effect
              + Cond Booking Rate Effect
```

### Effect Definitions

1. **Volume Effect**: Change due to total applications changing
   ```
   Δ Apps × Base Mix × Base Rates
   ```

2. **Mix Effect**: Change due to segment distribution shifting
   ```
   New Apps × Δ Mix × Base Rates
   ```

3. **Str Approval Rate Effect**: Change due to straight approval rates
   ```
   New Apps × New Mix × Δ Str Apprv Rate × Base Str Bk Rate
   ```

4. **Cond Approval Rate Effect**: Change due to conditional approval rates
   ```
   New Apps × New Mix × Δ Cond Apprv Rate × Base Cond Bk Rate
   ```

5. **Str Booking Rate Effect**: Change due to straight booking rates
   ```
   New Apps × New Mix × New Str Apprv Rate × Δ Str Bk Rate
   ```

6. **Cond Booking Rate Effect**: Change due to conditional booking rates
   ```
   New Apps × New Mix × New Cond Apprv Rate × Δ Cond Bk Rate
   ```

**Key Property**: Effects are calculated **sequentially** (waterfall) and **sum exactly** to actual change.

---

## API Reference

### Module 1: Hierarchical Decomposition Calculator

```python
from hier_decomposition_calculator import calculate_decomposition

results = calculate_decomposition(df, date_a, date_b, lender='ACA')
```

**Parameters**:
- `df`: Full dataset with all months
- `date_a`: Base period (Period 1)
- `date_b`: Current period (Period 2)
- `lender`: Lender to analyze (default 'ACA')

**Returns**: `DecompositionResults` named tuple with:
- `summary`: Lender-level aggregate impacts (DataFrame)
- `segment_detail`: Segment-level breakdown (DataFrame)
- `metadata`: Calculation metadata (dict)

**Summary DataFrame Columns**:
- `effect_type`: Type of effect (volume, mix, rates, etc.)
- `booking_impact`: Numeric impact on bookings
- `pct_of_total_change`: Percentage of total change

**Segment Detail DataFrame**: 24 rows (one per segment) with:
- Segment identifiers (fico_bands, offer_comp_tier, prod_line)
- Period 1 values (apps, rates, bookings)
- Period 2 values (apps, rates, bookings)
- Deltas (Δ for each metric)
- Effects (6 effect types + total)

### Module 2: Model Decomposition Calculator

```python
from model_decomposition_calculator import ModelDecompositionCalculator

calculator = ModelDecompositionCalculator()
results = calculator.calculate_decomposition(df, date_a, date_b, lender='ACA')
```

Model-based approach using regression to attribute changes.

### Module 3: Symmetric Decomposition Calculator

```python
from symmetric_decomposition_calculator import SymmetricDecompositionCalculator

calculator = SymmetricDecompositionCalculator()
results = calculator.calculate_decomposition(df, date_a, date_b, lender='ACA')
```

Symmetric decomposition approach for balanced attribution.

### Module 4: Visualization Engine

#### Waterfall Grid

```python
from visualization_engine import create_waterfall_grid

fig = create_waterfall_grid(
    summary=results.summary,
    segment_detail=results.segment_detail,
    lender='ACA',
    output_path='outputs/waterfalls/my_chart.png'  # Optional
)
```

Creates 2×2 grid:
- [0,0]: Overall aggregate waterfall
- [0,1]: By FICO band (stacked)
- [1,0]: By offer comp tier (stacked)
- [1,1]: By product line (stacked)

**Color Scheme**:
- Green: Positive contributions
- Red: Negative contributions
- Gray: Start/End bars

#### Dimensional Drilldown

```python
from visualization_engine import create_dimension_drilldown

fig = create_dimension_drilldown(
    segment_detail=results.segment_detail,
    dimension='fico_bands',  # or 'offer_comp_tier', 'prod_line'
    lender='ACA',
    output_path='outputs/drilldowns/fico.png'  # Optional
)
```

Creates 7-panel bar chart showing each effect by dimension values.

#### Export Summary Tables

```python
from visualization_engine import export_summary_tables

export_summary_tables(
    summary=results.summary,
    segment_detail=results.segment_detail,
    output_dir='outputs/tables',
    lender='ACA'
)
```

Exports:
- Excel file with multiple sheets (Summary, Segment Detail, By FICO, By Comp, By Product)
- CSV file with summary

### Module 5: Data Transformation

#### Pivot Funnel Data

```python
from data_transformation import pivot_funnel_data

# Transform from long format to wide format
pivoted_df = pivot_funnel_data(df)

# Result: One row per date-lender with segment metrics as columns
# Example column: High_FICO_solo_offer_Used_str_apprv_rate
```

**Use Case**: Convert hierarchical segment data to wide format for ML modeling.

#### Unpivot Funnel Data

```python
from data_transformation import unpivot_funnel_data

# Transform back from wide to long format
long_df = unpivot_funnel_data(pivoted_df)
```

**Use Case**: Round-trip transformation or convert ML predictions back to analysis format.

---

## Example Analyses

### Find Top Drivers

```python
# Which segments drove the largest increases?
top_segments = (results.segment_detail
                .sort_values('total_effect', ascending=False)
                [['fico_bands', 'offer_comp_tier', 'prod_line', 'total_effect']]
                .head(5))
print(top_segments)
```

### Analyze Mix Shift

```python
# Which segments gained/lost share?
mix_drivers = (results.segment_detail
               .sort_values('mix_effect', ascending=False)
               [['fico_bands', 'offer_comp_tier', 'prod_line',
                 'delta_pct_of_total', 'mix_effect']]
               .head(10))
print(mix_drivers)
```

### Dimensional Aggregation

```python
# How did each FICO band perform?
fico_summary = results.segment_detail.groupby('fico_bands')[
    ['volume_effect', 'mix_effect', 'str_approval_effect',
     'cond_approval_effect', 'str_booking_effect', 'cond_booking_effect',
     'total_effect']
].sum()

print(fico_summary)
```

---

## Module Summary

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `hier_decomposition_calculator.py` | Core hierarchical decomposition | `calculate_decomposition()` |
| `model_decomposition_calculator.py` | Model-based decomposition | `ModelDecompositionCalculator` |
| `symmetric_decomposition_calculator.py` | Symmetric decomposition | `SymmetricDecompositionCalculator` |
| `visualization_engine.py` | Charts and exports | `create_waterfall_grid()`, `create_dimension_drilldown()`, `export_summary_tables()` |
| `data_transformation.py` | Format conversion | `pivot_funnel_data()`, `unpivot_funnel_data()` |
| `utils.py` | Helper functions | `validate_dataframe()`, `calculate_segment_bookings()` |

---

## Troubleshooting

### Import Errors

If you see `ImportError: attempted relative import with no known parent package`:

**Solution**: Add `src/` to your Python path:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
```

### Reconciliation Failures

If reconciliation validation fails:

**Cause**: Input data doesn't satisfy constraints
**Check**:
1. `pct_of_total_apps` sums to 1.0 for each lender-month
2. Segment bookings sum to `num_tot_bks`
3. All 24 segments present for each lender-month (4 FICO × 3 comp × 2 product)

### Missing Segments

If you get `ValueError: Expected 24 segments`:

**Cause**: Data is missing segments for a given lender-month
**Solution**: Ensure you have all combinations of:
- 4 FICO bands × 3 offer comp tiers × 2 product lines = 24 segments

---

## Dependencies

See `requirements.txt` for complete list. Key dependencies:
- pandas
- numpy
- plotly
- openpyxl (for Excel export)

---

## License

This project is for internal use.

---

## Contributing

To extend the analysis:

1. Add new metrics to `utils.py`
2. Create new effect types in decomposition calculators
3. Add visualizations to `visualization_engine.py`
4. Extend transformations in `data_transformation.py`
5. Update template notebooks with new examples

---

## Contact

For questions or issues, please contact the maintainer or refer to the template notebooks for examples.
