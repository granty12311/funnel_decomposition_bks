# LMDI Funnel Decomposition Package

A Python package for analyzing booking funnel changes using LMDI (Logarithmic Mean Divisia Index) decomposition methodology. Decomposes booking changes into interpretable driver effects and provides penetration analysis for market share insights.

## Features

### LMDI Decomposition

Implements the LMDI methodology with split mix effects for precise attribution:

- **Volume Effect**: Total application volume change
- **Customer Mix Effect**: Customer segment distribution change (marginal mix)
- **Offer Comp Mix Effect**: Offer competitiveness distribution change (conditional mix)
- **Straight Approval Effect**: Change in straight approval rates
- **Conditional Approval Effect**: Change in conditional approval rates
- **Straight Booking Effect**: Change in straight booking rates
- **Conditional Booking Effect**: Change in conditional booking rates

**Key Benefits**:
- Perfect reconciliation (effects sum exactly to actual change)
- Order-independent decomposition
- Split mix effects for granular insight into customer vs. offer dynamics

### Penetration Analysis

Analyzes market share (lender bookings / total market bookings) with decomposed effects:

- **Gross Lender Effects**: Numerator impact (7 effects)
- **Self-Adjustment**: Lender's contribution to denominator growth
- **Net Lender Effects**: Gross minus self-adjustment
- **Competitor Effects**: Rest of market impact
- **Net Effects**: Net lender + competitor for each driver

All outputs in basis points (bps). 100 bps = 1 percentage point.

### Non-Financed Accounts Support

Handles cash buyers who don't use lender financing:
- `NON_FINANCED` rows contribute to total market bookings
- Excluded from funnel decomposition (no approval/booking rates)
- Properly included in penetration calculations

## Project Structure

```
funnel_decomposition/
├── src/
│   ├── __init__.py
│   ├── lmdi_decomposition_calculator.py    # Core LMDI decomposition
│   ├── lmdi_penetration_calculator.py      # Penetration analysis
│   ├── visualization_engine.py             # Main visualization module
│   ├── visualization_utils.py              # Chart utilities
│   ├── visualization_summary.py            # Summary visualizations
│   ├── visualization_lender.py             # Lender comparison charts
│   ├── visualization_penetration.py        # Penetration-specific charts
│   ├── dimension_config.py                 # Dimension ordering config
│   └── utils.py                            # Shared utilities
├── notebooks/
│   ├── lmdi_decomposition_demo.ipynb       # LMDI decomposition demo
│   ├── lmdi_penetration_demo.ipynb         # Penetration analysis demo
│   └── summary_demo.ipynb                  # YoY/MoM summary analysis
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/granty12311/funnel_decomposition_bks.git
cd funnel_decomposition_bks

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### LMDI Decomposition

```python
import sys
sys.path.insert(0, 'src')

import pandas as pd
from lmdi_decomposition_calculator import calculate_decomposition

# Load your data
df = pd.read_csv('data/funnel_data.csv')
df['month_begin_date'] = pd.to_datetime(df['month_begin_date'])

# Calculate decomposition
results = calculate_decomposition(
    df=df,
    date_a='2023-06-01',
    date_b='2024-06-01',
    lender='ACA'
)

# View summary (7 effects + total)
print(results.summary)

# Access segment-level detail
print(results.segment_detail)

# Metadata with totals
print(results.metadata)
```

### Penetration Analysis

```python
from lmdi_penetration_calculator import calculate_penetration_decomposition

# Calculate penetration decomposition
pen_results = calculate_penetration_decomposition(
    df=df,
    date_a='2023-06-01',
    date_b='2024-06-01',
    lender='ACA'
)

# View summary with gross/net/competitor effects (in bps)
print(pen_results.summary)

# Print formatted output
from lmdi_penetration_calculator import print_penetration_decomposition
print_penetration_decomposition(pen_results.summary, pen_results.metadata)
```

### Multi-Lender Analysis

```python
from lmdi_decomposition_calculator import calculate_multi_lender_decomposition

# Decomposition across all lenders
multi_results = calculate_multi_lender_decomposition(
    df=df,
    date_a='2023-06-01',
    date_b='2024-06-01'
)

# Aggregate summary
print(multi_results.aggregate_summary)

# Per-lender summaries
print(multi_results.lender_summaries)
```

## Data Requirements

### Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `lender` | str | Lender identifier (e.g., 'ACA', 'ALY', 'CAP') |
| `month_begin_date` | datetime | Date identifier (configurable via `date_column`) |
| `customer_segment` | str | Customer credit segment |
| `offer_comp_tier` | str | Offer competitiveness tier |
| `num_tot_apps` | int | Total applications for this lender/period |
| `num_tot_bks` | float | Total bookings for this lender/period |
| `pct_of_total_apps` | float | Segment's % of total apps (must sum to 1.0) |
| `str_apprv_rate` | float | Straight approval rate [0, 1] |
| `str_bk_rate` | float | Straight booking rate [0, 1] |
| `cond_apprv_rate` | float | Conditional approval rate [0, 1] |
| `cond_bk_rate` | float | Conditional booking rate [0, 1] |

### Dimension Values

**Customer Segment** (ordered by credit quality):
- `Super_Prime`
- `Prime`
- `Near_Prime`
- `Subprime`
- `Deep_Subprime`
- `New_To_Credit`

**Offer Comp Tier**:
- `solo_offer`: Only offer presented
- `multi_best`: Best offer among multiple
- `multi_other`: Not the best among multiple

### Segment Count

Each lender-period must have exactly 18 segments (6 customer segments × 3 offer comp tiers).

### Non-Financed Rows

For `NON_FINANCED` lender rows:
- Only `lender`, date column, and `num_tot_bks` are required
- Other columns can be NaN
- These rows are included in total market for penetration but excluded from decomposition

## Demo Notebooks

### lmdi_decomposition_demo.ipynb
Comprehensive demo of LMDI booking decomposition:
- Single lender analysis
- 7-effect breakdown with visualizations
- Segment-level detail exploration
- Multi-lender comparison

### lmdi_penetration_demo.ipynb
Penetration (market share) analysis:
- Gross vs net lender effects
- Self-adjustment calculation
- Competitor impact decomposition
- Penetration waterfall charts

### summary_demo.ipynb
High-level summary analysis:
- Year-over-Year (YoY) comparisons
- Month-over-Month (MoM) analysis
- Trend visualization

## API Reference

### DecompositionResults

```python
from lmdi_decomposition_calculator import DecompositionResults

results = calculate_decomposition(...)
results.summary          # DataFrame: 8 rows (7 effects + total)
results.segment_detail   # DataFrame: 18 rows (segment breakdown)
results.metadata         # dict: dates, totals, method info
```

**Summary columns**: `effect_type`, `booking_impact`

### PenetrationResults

```python
from lmdi_penetration_calculator import PenetrationResults

results = calculate_penetration_decomposition(...)
results.summary          # DataFrame: 8 rows with multiple effect columns
results.segment_detail   # DataFrame: segment-level penetration effects
results.metadata         # dict: penetration metrics, market totals
```

**Summary columns**:
- `effect_type`
- `gross_lender_effect_bps`
- `self_adjustment_bps`
- `net_lender_effect_bps`
- `competitor_effect_bps`
- `net_effect_bps`

### Key Functions

```python
# Single lender decomposition
calculate_decomposition(df, date_a, date_b, lender, date_column='month_begin_date')

# Multi-lender decomposition
calculate_multi_lender_decomposition(df, date_a, date_b, lenders=None, date_column='month_begin_date')

# Single lender penetration
calculate_penetration_decomposition(df, date_a, date_b, lender, date_column='month_begin_date')

# Multi-lender penetration
calculate_multi_lender_penetration_decomposition(df, date_a, date_b, lenders=None, date_column='month_begin_date')
```

## Methodology

### LMDI Decomposition

The LMDI approach uses logarithmic mean weights for exact decomposition:

```
L(x₀, xₜ) = (xₜ - x₀) / ln(xₜ / x₀)
```

Bookings are decomposed as:
```
ΔB = Σᵢ wᵢ × [ln(V₂/V₁) + ln(CS₂/CS₁) + ln(OC₂/OC₁) + ln(rates...)]
```

Where:
- `wᵢ` = logarithmic mean of segment bookings
- `V` = total applications (volume)
- `CS` = customer segment share (marginal mix)
- `OC` = offer comp share (conditional mix)

### Split Mix Effects

The package implements a hierarchical mix decomposition:
1. **Customer Mix**: Changes in customer segment distribution (e.g., shift to Prime)
2. **Offer Comp Mix**: Changes in offer competitiveness within each customer segment

This provides more actionable insights than a single combined mix effect.

### Penetration Self-Adjustment

When analyzing penetration (P = L/M):
- **Gross Effect**: Impact on numerator (lender bookings)
- **Self-Adjustment**: Lender's contribution to denominator growth
- **Net Effect**: Gross - Self-Adjustment

This eliminates double-counting when lender growth affects both numerator and denominator.

## Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## License

This project is proprietary and confidential.

## Changelog

### v3.0.0 (December 2024)

**Major Changes**:
- Replaced symmetric/hierarchical decomposition with LMDI methodology
- Added penetration analysis with self-adjusted lender effects
- New dimension structure: customer_segment + offer_comp_tier
- Added non-financed accounts support

**New Features**:
- 7-effect decomposition with split mix (customer vs offer comp)
- Penetration decomposition with gross/net/competitor effects
- Multi-lender analysis for both booking and penetration
- YoY/MoM summary analysis in demo notebooks

**Improvements**:
- Perfect reconciliation guaranteed by LMDI methodology
- Cleaner separation of visualization modules
- Configurable dimension ordering via dimension_config.py
