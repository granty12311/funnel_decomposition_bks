# Funnel Decomposition Analysis

A hierarchical decomposition system to explain booking changes across multi-stage funnels with multiple segmentation dimensions.

## Overview

This project implements a **waterfall decomposition** approach to answer questions like:
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

## Project Structure

```
funnel_decomposition/
├── .venv/                              # Virtual environment
├── data/
│   ├── funnel_data_mock_v2.csv         # Mock data (576 rows, 24 months)
│   ├── create_mock_data_v2.py          # Data generation script
│   └── create_mock_data_pivoted.py     # Pivoted data generator
├── src/
│   ├── __init__.py                     # Package exports
│   ├── hier_decomposition_calculator.py  # Hierarchical decomposition engine
│   ├── visualization_engine.py         # Charts and exports
│   ├── data_transformation.py          # Pivot/unpivot utilities
│   └── utils.py                        # Helper functions
├── notebooks/
│   ├── funnel_decomposition_demo.ipynb        # Full demonstration
│   ├── funnel_decomposition_demo_yoy_final.ipynb  # YoY analysis demo
│   └── model_funnel_decomposition_executed.ipynb # Model integration
├── examples/
│   └── example_pivot_transformation.py # Data transformation demo
├── tests/
│   ├── test_integration.py             # Integration tests
│   ├── test_visualization_improvements.py  # Visualization tests
│   ├── test_data_transformation.py     # Transformation tests
│   └── test_yoy_comparison.py          # YoY comparison test
├── outputs/
│   ├── waterfalls/                     # Waterfall chart grids
│   ├── drilldowns/                     # Dimensional drilldown charts
│   └── tables/                         # Excel/CSV exports
├── docs/
│   └── DATA_TRANSFORMATION.md          # Data transformation guide
├── requirements.txt
└── README.md                           # This file
```

---

## Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Run Integration Tests

Validate that everything works:

```bash
# Run all tests
python tests/test_integration.py
python tests/test_visualization_improvements.py
python tests/test_yoy_comparison.py
```

Expected output:
```
✓✓✓ ALL INTEGRATION TESTS PASSED ✓✓✓
```

### 3. Basic Usage Example

```python
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))

from hier_decomposition_calculator import calculate_decomposition
from visualization_engine import create_waterfall_grid, create_dimension_drilldown

# Load data
df = pd.read_csv('data/funnel_data_mock_v2.csv')
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

### Input Data Structure

The analysis requires data with the following columns:

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

**Summary DataFrame Example**:
```
effect_type              booking_impact    pct_of_total_change
volume_effect                  -1141.1               237.2
mix_effect                         0.0                -0.0
str_approval_effect              -45.8                 9.5
cond_approval_effect             -36.2                 7.5
str_booking_effect               613.4              -127.5
cond_booking_effect              128.7               -26.8
total_change                    -481.0               100.0
```

**Segment Detail DataFrame**: 24 rows (one per segment) with:
- Segment identifiers (fico_bands, offer_comp_tier, prod_line)
- Period 1 values (apps, rates, bookings)
- Period 2 values (apps, rates, bookings)
- Deltas (Δ for each metric)
- Effects (6 effect types + total)

### Module 2: Visualization Engine

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

#### Print Waterfall Breakdowns

```python
from visualization_engine import print_waterfall_breakdowns

# After creating waterfall grid
fig = create_waterfall_grid(summary, segment_detail, lender='ACA')
print_waterfall_breakdowns(fig)
```

Displays detailed breakdown tables showing contribution of each dimension value to each effect.

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

### Module 3: Data Transformation

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

### Year-over-Year Comparison

```python
# Compare same month year-over-year
results_yoy = calculate_decomposition(
    df=df,
    date_a='2023-01-01',
    date_b='2024-01-01',
    lender='ACA'
)

print(f"YoY Change: {results_yoy.metadata['delta_total_bookings']:,.0f}")
print(results_yoy.summary)
```

---

## Testing

### Run All Tests

```bash
# Integration tests
python tests/test_integration.py

# Visualization tests
python tests/test_visualization_improvements.py

# Data transformation tests
python tests/test_data_transformation.py

# YoY comparison
python tests/test_yoy_comparison.py
```

### Test Coverage

The test suite validates:
1. ✅ Decomposition calculation for multiple date pairs
2. ✅ Exact reconciliation (all effects sum to actual change)
3. ✅ Waterfall chart generation with proper colors
4. ✅ All 3 dimensional drilldowns (FICO, comp tier, product line)
5. ✅ Excel and CSV table exports
6. ✅ Data transformation round-trip (pivot → unpivot)
7. ✅ Year-over-Year analysis

---

## Notebooks

### Interactive Demonstrations

1. **funnel_decomposition_demo.ipynb**: Complete walkthrough with Year-over-Year analysis
   - Load and explore data
   - Calculate decomposition
   - Create visualizations
   - Export results

2. **model_funnel_decomposition_executed.ipynb**: Integration with ML models
   - Shows how to use decomposition insights for modeling

### Running Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or execute from command line
jupyter nbconvert --to notebook --execute notebooks/funnel_decomposition_demo.ipynb
```

---

## Example Output

### Sample YoY Analysis (Jan 2023 → Jan 2024)

```
Period 1 bookings: 4,480
Period 2 bookings: 3,999
Delta bookings: -481 (-10.7%)

Key Findings:
- Volume declined significantly (-1,141) - largest negative contributor
- Booking rates improved (+613 straight, +129 conditional)
- High FICO drove most improvements (+440 str booking, +68 cond booking)
- Despite 25% volume decline, conversion improvements offset some impact
```

### Generated Files

All outputs are saved to:
- `outputs/waterfalls/` - PNG waterfall chart grids
- `outputs/drilldowns/` - PNG dimensional drilldown charts
- `outputs/tables/` - XLSX and CSV summary tables

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

## Documentation

- **[DATA_TRANSFORMATION.md](docs/DATA_TRANSFORMATION.md)**: Guide to pivot/unpivot transformations
- **[PLANNING.md](PLANNING.md)**: Original design document with ML approach

---

## Next Steps

### Immediate Use

1. ✅ Run integration tests to validate setup
2. ✅ Load your own data (matching the schema)
3. ✅ Run decomposition on relevant date pairs
4. ✅ Generate waterfalls and drilldowns for stakeholders

### Future Enhancements

- **Automated Reports**: Schedule monthly decomposition reports
- **Interactive Dashboard**: Build Streamlit/Dash app for self-service
- **Statistical Testing**: Add confidence intervals for effects
- **ML Attribution**: Implement complementary ML approach (see PLANNING.md)
- **Multi-Lender Support**: Compare decompositions across lenders
- **Additional Metrics**: Extend to other funnel metrics beyond bookings

---

## Contributing

To extend the analysis:

1. Add new metrics to `utils.py`
2. Create new effect types in `hier_decomposition_calculator.py`
3. Add visualizations to `visualization_engine.py`
4. Extend transformations in `data_transformation.py`
5. Update tests in `tests/` directory

---

## Module Summary

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `hier_decomposition_calculator.py` | Core decomposition engine | `calculate_decomposition()` |
| `visualization_engine.py` | Charts and exports | `create_waterfall_grid()`, `create_dimension_drilldown()`, `export_summary_tables()` |
| `data_transformation.py` | Format conversion | `pivot_funnel_data()`, `unpivot_funnel_data()` |
| `utils.py` | Helper functions | `validate_dataframe()`, `calculate_segment_bookings()` |

---

## License

This project is for internal use.

---

## Contact

For questions or issues, please refer to the documentation files in the `docs/` directory.
