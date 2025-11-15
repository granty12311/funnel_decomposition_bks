# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Funnel Decomposition Analysis Package** that implements waterfall decomposition methodologies to explain booking changes in multi-stage lending funnels. The package breaks down booking changes between two time periods into six interpretable effects: volume, mix, straight approval rate, conditional approval rate, straight booking rate, and conditional booking rate.

## Architecture

### Core Calculation Methodologies

The package implements three distinct decomposition approaches, each in its own calculator module:

1. **Symmetric Decomposition** (`symmetric_decomposition_calculator.py`)
   - Order-independent methodology using midpoint values
   - Uses average values from both periods for all effects
   - Includes an interaction effect for perfect reconciliation
   - Best for unbiased, reproducible analysis

2. **Hierarchical Decomposition** (`hier_decomposition_calculator.py`)
   - Sequential waterfall approach where effects build on previous steps
   - Volume effect uses Period 1 values, then incrementally updates
   - Traditional approach familiar to business stakeholders
   - Perfect reconciliation without residual

3. **Model Decomposition** (`model_decomposition_calculator.py`)
   - ML-based approach (less commonly used)
   - Consider using symmetric or hierarchical instead

**Key Insight**: All three calculators share the same function signature and return structure (`DecompositionResults` named tuple). They can be used interchangeably by importing from different modules.

### Data Flow

```
Raw Data (Long Format)
    ↓
validate_dataframe() [utils.py]
    ↓
calculate_decomposition() [calculator modules]
    ↓
DecompositionResults (summary + segment_detail + metadata)
    ↓
Visualization Engine
    ↓
Waterfall Charts + Drilldowns + Excel/CSV Exports
```

### Module Responsibilities

- **`utils.py`**: Shared validation, date normalization, segment booking calculations
- **Calculator modules**: Core decomposition math (6 effects + reconciliation)
- **`visualization_engine.py`**: All chart types and exports (shared across methodologies)
- **`data_transformation.py`**: Pivot/unpivot for ML modeling (converts between long/wide format)

### Data Constraints

The package enforces strict data integrity:
- **24 segments required** per lender-period (4 FICO × 3 comp × 2 product)
- **Mix constraint**: `Σ pct_of_total_apps = 1.0` for each lender-period
- **Booking constraint**: Calculated segment bookings must match `num_tot_bks`
- **Rate bounds**: All rates must be [0, 1]

These constraints are validated in `utils.py:validate_period_data()`.

## Development Commands

### Environment Setup

```bash
# Activate virtual environment (use this before any work)
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Tests

```bash
# Run all integration tests
python tests/test_integration.py

# Run specific test suites
python tests/test_visualization_improvements.py
python tests/test_yoy_comparison.py

# Expected output: "✓✓✓ ALL INTEGRATION TESTS PASSED ✓✓✓"
```

**Important**: Tests use `sys.path.insert(0, 'src')` to import modules. This pattern is required for all scripts outside the `src/` directory.

### Running Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or execute notebook from command line
jupyter nbconvert --to notebook --execute notebooks/funnel_decomposition_demo.ipynb
```

### Common Import Pattern

All notebooks and scripts must add `src/` to Python path:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

# Now can import modules
from hier_decomposition_calculator import calculate_decomposition
from visualization_engine import create_waterfall_grid
```

## Code Structure Details

### DecompositionResults Structure

All calculators return a `NamedTuple` with three components:

```python
DecompositionResults(
    summary=pd.DataFrame,        # Aggregate effects (7 rows: 6 effects + total)
    segment_detail=pd.DataFrame, # Segment-level breakdown (24 rows)
    metadata=dict               # Calculation context (dates, lender, totals)
)
```

**Summary DataFrame columns**:
- `effect_type`: str (volume_effect, mix_effect, str_approval_effect, etc.)
- `booking_impact`: float (booking change attributed to this effect)
- `pct_of_total_change`: float (percentage contribution)

**Segment Detail DataFrame** contains ~30 columns including:
- Dimensions: `fico_bands`, `offer_comp_tier`, `prod_line`
- Period 1 metrics: `period_1_total_apps`, `period_1_pct_of_total`, all rates
- Period 2 metrics: Same structure with `period_2_` prefix
- Deltas: `delta_` prefix for all metrics
- Effects: 6 effect columns + `total_effect`

### Flexible Date Column Support

All calculators support custom date columns via `date_column` parameter:

```python
# Monthly analysis (default)
calculate_decomposition(df, date_a, date_b, lender='ACA')

# Weekly analysis
calculate_decomposition(df, date_a, date_b, lender='ACA',
                       date_column='week_begin_date')

# Quarterly analysis
calculate_decomposition(df, date_a, date_b, lender='ACA',
                       date_column='quarter_begin_date')
```

This parameter was added to support different time granularities without changing the data schema.

### Multi-Lender Analysis

The symmetric calculator includes multi-lender support:

```python
from symmetric_decomposition_calculator import calculate_multi_lender_decomposition

results = calculate_multi_lender_decomposition(df, date_a, date_b)
# Returns: MultiLenderResults with aggregate_summary and lender_summaries
```

This aggregates effects across all lenders and provides lender-attributed breakdowns.

## Visualization Engine

### Chart Types

1. **Waterfall Grid** (2×2 layout): `create_waterfall_grid()`
   - Top-left: Overall aggregate waterfall
   - Top-right: Stacked by FICO band
   - Bottom-left: Stacked by offer comp tier
   - Bottom-right: Stacked by product line

2. **Dimension Drilldown**: `create_dimension_drilldown(dimension='fico_bands')`
   - 7-panel horizontal bar chart
   - One panel per effect showing dimension breakdown

3. **Multi-Lender Waterfall Grid**: `create_lender_waterfall_grid()`
   - Side-by-side comparison: aggregate vs by-lender
   - Used with multi-lender results

### Export Utilities

```python
# Export summary tables to Excel
export_summary_tables(summary, segment_detail, output_dir='outputs/tables', lender='ACA')
# Creates: Excel file with multiple sheets + CSV summary

# Print waterfall breakdowns (console output)
print_waterfall_breakdowns(fig)
# Shows contribution of each dimension value to each effect

# Chart extraction and PNG export (see notebooks/chart_export.ipynb)
charts = extract_individual_charts_from_grid(fig)
export_chart(charts[0], 'output/chart.png', dpi=300)
```

## Repository Structure

### Active Development Directories

- **`src/`**: Primary source code (use this for all imports and development)
- **`notebooks/`**: Cleaned demo notebooks (no executed outputs)
- **`tests/`**: Integration and validation tests
- **`data/`**: Mock data and data generation scripts
- **`outputs/` and `output/`**: Chart and table exports

### Reference/Archive Directories

- **`prod_code/`**: Contains duplicated code from earlier development (appears to be older version)
- **`archive/`**: Older notebooks and code versions
- **`docs/`**: Documentation (DATA_TRANSFORMATION.md)

**Important**: When making changes, update code in `src/` only. The `prod_code/` directory appears to be stale/legacy.

## Common Development Scenarios

### Adding a New Effect Type

1. Modify the calculator module (`hier_decomposition_calculator.py` or `symmetric_decomposition_calculator.py`)
2. Add effect calculation in `_calculate_all_effects()` function
3. Update the summary aggregation to include new effect
4. Update `visualization_engine.py` to visualize the new effect
5. Add validation in `_validate_reconciliation()`
6. Update tests to verify new effect

### Supporting a New Dimension

1. Update schema in `utils.py:validate_dataframe()`
2. Modify pivot logic in `data_transformation.py`
3. Add visualization support in `visualization_engine.py:create_dimension_drilldown()`
4. Update segment count validation (currently hardcoded to 24)
5. Create test data with new dimension

### Working with Different Time Granularities

Use the `date_column` parameter rather than modifying code:
- Pass `date_column='week_begin_date'` for weekly data
- Pass `date_column='quarter_begin_date'` for quarterly data
- Default is `'month_begin_date'`

## Testing Strategy

**IMPORTANT**: Do NOT run the Python test files in `tests/` directory for validation.

### Testing Workflow

All testing should be done by executing relevant Jupyter notebooks and letting the user provide feedback:

1. **Make code changes** in `src/` directory
2. **Execute relevant notebook** (e.g., `notebooks/funnel_decomposition_demo.ipynb`)
3. **Review outputs** with the user - charts, tables, console output
4. **User provides feedback** on whether results are correct
5. **Iterate** based on feedback

### Which Notebook to Use for Testing

- **General functionality**: `notebooks/funnel_decomposition_demo.ipynb`
- **Symmetric decomposition**: `notebooks/symmetric_funnel_decomp.ipynb`
- **Hierarchical decomposition**: `notebooks/hier_funnel_decomp.ipynb`
- **Multi-lender analysis**: `notebooks/symmetric_funnel_decomp_multi_lender.ipynb`
- **Weekly analysis**: `notebooks/symmetric_funnel_decomp_weekly.ipynb`
- **Chart exports**: `notebooks/chart_export.ipynb`

### Executing Notebooks

```bash
# Start Jupyter and manually run cells
jupyter notebook notebooks/funnel_decomposition_demo.ipynb

# Or execute entire notebook from command line
jupyter nbconvert --to notebook --execute --inplace notebooks/funnel_decomposition_demo.ipynb
```

### What to Validate

When executing notebooks for testing:
- Charts render correctly
- No error messages or warnings
- Reconciliation is exact (effects sum to actual booking change)
- Output tables have expected structure
- User confirms results match expectations

## Key Files

- **Demo notebook**: `notebooks/funnel_decomposition_demo.ipynb` - Start here for understanding package
- **Main calculator**: `src/hier_decomposition_calculator.py` - Most commonly used methodology
- **Visualization**: `src/visualization_engine.py` - All chart generation logic
- **Utilities**: `src/utils.py` - Validation and helper functions

## Development Notes

### Date Handling

All calculators use `normalize_date()` from `utils.py` to convert strings/datetimes to pandas Timestamps. This ensures consistent date comparison regardless of input format.

### Reconciliation Math

The hierarchical approach uses sequential logic:
1. Volume effect uses Period 1 mix and rates
2. Mix effect uses Period 2 apps and Period 1 rates
3. Rate effects use Period 2 apps, mix, and incrementally update each rate

The symmetric approach calculates all effects independently using midpoint values, then adds an interaction term.

**Both approaches guarantee exact reconciliation** - this is validated in every test.

### Performance Considerations

- Decomposition calculations are fast (< 1 second for typical data)
- Chart generation can be slower for high-DPI exports
- Pivot transformations scale linearly with segment count

### Data Transformation for ML

Use `data_transformation.py` to convert from hierarchical format (one row per segment) to wide format (one row per lender-period with 120+ feature columns). This is required for ML models like LightGBM or XGBoost.

The `unpivot_funnel_data()` function reverses this transformation.

## Git Workflow

**CRITICAL**: This repository only publishes the `prod_code/` folder to GitHub. The `src/` directory is for local development only.

### Repository Configuration

- **GitHub URL**: https://github.com/granty12311/funnel_decomposition_bks
- **Authentication**: SSH only
- **Published folder**: `prod_code/` only (not the entire repository)

### Development Workflow

1. **Make all changes in `src/` directory** during development
2. **Test using notebooks** (see Testing Strategy section)
3. **Get user approval** on changes
4. **Sync to prod_code** before pushing to GitHub (see below)
5. **Push only prod_code folder** to remote

### Syncing to prod_code Before Git Push

Before pushing to GitHub, you MUST sync the following to `prod_code/`:

```bash
# 1. Sync source code
cp -r src/* prod_code/src/

# 2. Sync README
cp README.md prod_code/

# 3. Sync sample notebooks (select relevant ones)
cp notebooks/funnel_decomposition_demo.ipynb prod_code/
cp notebooks/symmetric_funnel_decomp.ipynb prod_code/
cp notebooks/hier_funnel_decomp.ipynb prod_code/
cp notebooks/symmetric_funnel_decomp_multi_lender.ipynb prod_code/
cp notebooks/symmetric_funnel_decomp_weekly.ipynb prod_code/
# Add other notebooks as needed
```

**Important**: Do NOT copy executed notebooks with outputs. Strip outputs first or copy clean versions.

### Git Commands for Pushing

```bash
# Navigate to prod_code directory
cd prod_code/

# Initialize git if not already done
git init

# Add remote (SSH) - only needed once
git remote add origin git@github.com:granty12311/funnel_decomposition_bks.git

# Or update remote if already exists
git remote set-url origin git@github.com:granty12311/funnel_decomposition_bks.git

# Stage all changes in prod_code
git add .

# Commit with descriptive message
git commit -m "Update: [describe changes here]"

# Push to GitHub using SSH
git push -u origin main
# Or if branch is named differently:
# git push -u origin master
```

### Pre-Push Checklist

Before pushing to GitHub, verify:
- [ ] All changes tested via notebooks with user approval
- [ ] Source code synced: `src/` → `prod_code/src/`
- [ ] README updated and synced to `prod_code/`
- [ ] Relevant sample notebooks synced to `prod_code/`
- [ ] No notebook outputs committed (notebooks should be clean)
- [ ] No sensitive data or large data files included
- [ ] Currently in `prod_code/` directory when running git commands

### GitHub Repository Structure

The GitHub repository (prod_code folder) should contain:
```
funnel_decomposition_bks/  (GitHub repo)
├── src/
│   ├── __init__.py
│   ├── symmetric_decomposition_calculator.py
│   ├── hier_decomposition_calculator.py
│   ├── model_decomposition_calculator.py
│   ├── visualization_engine.py
│   ├── utils.py
│   └── data_transformation.py
├── README.md
├── requirements.txt  (if in prod_code)
├── funnel_decomposition_demo.ipynb
├── symmetric_funnel_decomp.ipynb
├── hier_funnel_decomp.ipynb
└── [other sample notebooks]
```

### Common Git Issues

**Issue**: "Permission denied (publickey)"
- **Solution**: Ensure SSH keys are configured for GitHub: `ssh -T git@github.com`

**Issue**: Accidentally pushed from wrong directory
- **Solution**: Only run git commands from within `prod_code/` directory

**Issue**: Need to update what gets synced to prod_code
- **Solution**: Modify the cp commands in "Syncing to prod_code" section above
