"""
Data transformation utilities for funnel decomposition analysis.

This module provides functions to transform funnel data between different formats:
- Long format (hierarchical): One row per date-lender-segment
- Wide format (pivoted): One row per date-lender with segment metrics as columns
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def pivot_funnel_data(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    dimension_cols: Optional[List[str]] = None,
    metric_cols: Optional[List[str]] = None,
    total_cols: Optional[List[str]] = None,
    validate: bool = True
) -> pd.DataFrame:
    """
    Transform funnel data from long format to wide format for ML modeling.

    Converts hierarchical segment data (one row per date-lender-segment) into
    a pivoted format (one row per date-lender) where each segment's metrics
    become separate columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe in long format with columns:
        - lender, month_begin_date (grouping keys)
        - fico_bands, offer_comp_tier, prod_line (segment dimensions)
        - num_tot_bks, num_tot_apps (totals)
        - pct_of_total_apps, str_apprv_rate, str_bk_rate,
          cond_apprv_rate, cond_bk_rate (segment metrics)

    group_cols : List[str], optional
        Columns to group by (default: ['lender', 'month_begin_date'])

    dimension_cols : List[str], optional
        Columns defining segment hierarchy (default: ['fico_bands', 'offer_comp_tier', 'prod_line'])

    metric_cols : List[str], optional
        Segment-level metrics to pivot (default: ['pct_of_total_apps', 'str_apprv_rate',
        'str_bk_rate', 'cond_apprv_rate', 'cond_bk_rate'])

    total_cols : List[str], optional
        Aggregate columns to preserve (default: ['num_tot_bks', 'num_tot_apps'])

    validate : bool, default True
        Whether to validate the output data structure

    Returns
    -------
    pd.DataFrame
        Pivoted dataframe with:
        - One row per unique combination of group_cols
        - Columns: group_cols + total_cols + pivoted segment-metric columns
        - Pivoted columns named: {dim1}_{dim2}_{dim3}_{metric}

    Examples
    --------
    >>> import pandas as pd
    >>> from src.data_transformation import pivot_funnel_data
    >>>
    >>> # Load long format data
    >>> df = pd.read_csv('data/funnel_data_mock_v2.csv')
    >>>
    >>> # Transform to wide format
    >>> pivoted_df = pivot_funnel_data(df)
    >>>
    >>> # Result has one row per month with all segment metrics as columns
    >>> print(pivoted_df.shape)  # (24, 124)

    Notes
    -----
    - The function preserves all segments in the input data
    - Missing segments will result in NaN values (validated if validate=True)
    - Column naming follows pattern: {dim1}_{dim2}_{...}_{metric}
    - For funnel data, pct_of_total_apps columns should sum to 1.0 per row
    """
    # Set defaults
    if group_cols is None:
        group_cols = ['lender', 'month_begin_date']
    if dimension_cols is None:
        dimension_cols = ['fico_bands', 'offer_comp_tier', 'prod_line']
    if metric_cols is None:
        metric_cols = [
            'pct_of_total_apps',
            'str_apprv_rate',
            'str_bk_rate',
            'cond_apprv_rate',
            'cond_bk_rate'
        ]
    if total_cols is None:
        total_cols = ['num_tot_bks', 'num_tot_apps']

    # Validate input columns exist
    required_cols = group_cols + dimension_cols + metric_cols + total_cols
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in input dataframe: {missing_cols}")

    # Initialize result dataframe with grouping keys and totals
    result_df = df.groupby(group_cols)[total_cols].first().reset_index()

    # Get unique values for each dimension (sorted for consistent column ordering)
    dimension_values = {
        dim: sorted(df[dim].unique())
        for dim in dimension_cols
    }

    # Build pivoted columns for each segment combination
    pivot_column_map = {}  # Track column names for validation

    # Generate all combinations of dimension values
    def generate_combinations(dims, values_dict):
        """Recursively generate all combinations of dimension values."""
        if not dims:
            return [[]]
        first_dim = dims[0]
        rest_dims = dims[1:]
        rest_combinations = generate_combinations(rest_dims, values_dict)
        return [
            [val] + rest_combo
            for val in values_dict[first_dim]
            for rest_combo in rest_combinations
        ]

    dimension_combinations = generate_combinations(dimension_cols, dimension_values)

    # For each combination, pivot the metrics
    for combo in dimension_combinations:
        # Create filter mask for this segment
        segment_mask = pd.Series([True] * len(df))
        for dim_col, dim_val in zip(dimension_cols, combo):
            segment_mask &= (df[dim_col] == dim_val)

        segment_data = df[segment_mask].copy()

        # Create column prefix from dimension values
        col_prefix = '_'.join(str(val) for val in combo)

        # Pivot each metric for this segment
        for metric in metric_cols:
            col_name = f"{col_prefix}_{metric}"
            pivot_column_map[col_name] = {'segment': combo, 'metric': metric}

            # Extract segment-metric data
            metric_data = segment_data[group_cols + [metric]].rename(
                columns={metric: col_name}
            )

            # Merge into result
            result_df = result_df.merge(
                metric_data,
                on=group_cols,
                how='left'
            )

    # Validation
    if validate:
        _validate_pivoted_data(
            result_df,
            group_cols,
            pivot_column_map,
            metric_cols
        )

    return result_df


def _validate_pivoted_data(
    df: pd.DataFrame,
    group_cols: List[str],
    pivot_column_map: dict,
    metric_cols: List[str]
) -> None:
    """
    Validate pivoted dataframe structure and data quality.

    Parameters
    ----------
    df : pd.DataFrame
        Pivoted dataframe to validate
    group_cols : List[str]
        Grouping columns
    pivot_column_map : dict
        Mapping of pivoted column names to segment/metric info
    metric_cols : List[str]
        Original metric column names

    Raises
    ------
    ValueError
        If validation checks fail
    """
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        missing_cols = df.isnull().sum()[df.isnull().sum() > 0]
        raise ValueError(
            f"Found {missing_count} missing values in pivoted data:\n{missing_cols}"
        )

    # Validate pct_of_total_apps sums to 1.0 for each row
    if 'pct_of_total_apps' in metric_cols:
        pct_cols = [col for col in df.columns if col.endswith('_pct_of_total_apps')]
        pct_sums = df[pct_cols].sum(axis=1)

        if not np.allclose(pct_sums, 1.0, rtol=1e-5):
            invalid_rows = df[~np.isclose(pct_sums, 1.0, rtol=1e-5)]
            raise ValueError(
                f"pct_of_total_apps does not sum to 1.0 for {len(invalid_rows)} rows.\n"
                f"Sum range: [{pct_sums.min():.6f}, {pct_sums.max():.6f}]\n"
                f"Invalid rows: {invalid_rows[group_cols].to_dict('records')}"
            )

    # Validate rate columns are between 0 and 1
    rate_metrics = ['str_apprv_rate', 'str_bk_rate', 'cond_apprv_rate', 'cond_bk_rate']
    for rate_metric in rate_metrics:
        if rate_metric in metric_cols:
            rate_cols = [col for col in df.columns if col.endswith(f'_{rate_metric}')]
            for col in rate_cols:
                if not ((df[col] >= 0) & (df[col] <= 1)).all():
                    invalid_values = df[~((df[col] >= 0) & (df[col] <= 1))][col]
                    raise ValueError(
                        f"Rate column {col} contains values outside [0, 1]:\n{invalid_values}"
                    )


def unpivot_funnel_data(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    dimension_cols: Optional[List[str]] = None,
    total_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Transform funnel data from wide format back to long format.

    Converts pivoted data (one row per date-lender) back to hierarchical format
    (one row per date-lender-segment).

    Parameters
    ----------
    df : pd.DataFrame
        Pivoted dataframe with segment-metric columns

    group_cols : List[str], optional
        Grouping columns (default: ['lender', 'month_begin_date'])

    dimension_cols : List[str], optional
        Segment dimension column names (default: ['fico_bands', 'offer_comp_tier', 'prod_line'])

    total_cols : List[str], optional
        Total columns to preserve (default: ['num_tot_bks', 'num_tot_apps'])

    Returns
    -------
    pd.DataFrame
        Long format dataframe with segment rows

    Examples
    --------
    >>> # Convert pivoted data back to long format
    >>> long_df = unpivot_funnel_data(pivoted_df)
    """
    # Set defaults
    if group_cols is None:
        group_cols = ['lender', 'month_begin_date']
    if dimension_cols is None:
        dimension_cols = ['fico_bands', 'offer_comp_tier', 'prod_line']
    if total_cols is None:
        total_cols = ['num_tot_bks', 'num_tot_apps']

    # Identify pivoted columns (exclude group and total columns)
    base_cols = set(group_cols + total_cols)
    pivoted_cols = [col for col in df.columns if col not in base_cols]

    # Parse pivoted column names to extract dimension values and metrics
    rows = []
    for _, row in df.iterrows():
        # Group by segment (columns with same prefix)
        segments = {}
        for col in pivoted_cols:
            # Split column name: {dim1}_{dim2}_{dim3}_{metric}
            parts = col.split('_')

            # Last part is metric, rest are dimension values
            metric = parts[-1]

            # Handle multi-word metrics (e.g., pct_of_total_apps has 4 parts)
            if len(parts) > len(dimension_cols) + 1:
                # Find where dimensions end and metric begins
                # Metric patterns: pct_of_total_apps, str_apprv_rate, etc.
                if 'pct_of_total_apps' in col:
                    dim_parts = parts[:-4]
                    metric = '_'.join(parts[-4:])
                elif any(m in col for m in ['apprv_rate', 'bk_rate']):
                    dim_parts = parts[:-3]
                    metric = '_'.join(parts[-3:])
                else:
                    # Default: last part is metric
                    dim_parts = parts[:-1]
                    metric = parts[-1]
            else:
                dim_parts = parts[:-1]

            segment_key = '_'.join(dim_parts)

            if segment_key not in segments:
                segments[segment_key] = {
                    'dimensions': dim_parts,
                    'metrics': {}
                }
            segments[segment_key]['metrics'][metric] = row[col]

        # Create a row for each segment
        for segment_key, segment_data in segments.items():
            segment_row = {}

            # Add group columns
            for col in group_cols:
                segment_row[col] = row[col]

            # Add total columns
            for col in total_cols:
                segment_row[col] = row[col]

            # Add dimension columns
            for i, dim_col in enumerate(dimension_cols):
                segment_row[dim_col] = segment_data['dimensions'][i]

            # Add metric columns
            for metric, value in segment_data['metrics'].items():
                segment_row[metric] = value

            rows.append(segment_row)

    result_df = pd.DataFrame(rows)

    # Reorder columns to match original format
    column_order = group_cols + dimension_cols + total_cols + list(
        set(result_df.columns) - set(group_cols) - set(dimension_cols) - set(total_cols)
    )
    result_df = result_df[column_order]

    return result_df


if __name__ == "__main__":
    # Test the transformation functions
    import sys
    from pathlib import Path

    # Load test data
    data_path = Path(__file__).parent.parent / "data" / "funnel_data_mock_v2.csv"
    if not data_path.exists():
        print(f"Test data not found at: {data_path}")
        sys.exit(1)

    print("Loading test data...")
    df = pd.read_csv(data_path)
    print(f"Input shape: {df.shape}")

    print("\nTransforming to pivoted format...")
    pivoted = pivot_funnel_data(df)
    print(f"Pivoted shape: {pivoted.shape}")
    print(f"✓ Transformation successful!")

    print("\nSample pivoted columns:")
    sample_cols = [col for col in pivoted.columns if 'High_FICO_solo_offer' in col][:5]
    print(pivoted[['lender', 'month_begin_date'] + sample_cols].head(3))

    print("\n" + "="*80)
    print("Testing unpivot transformation...")
    unpivoted = unpivot_funnel_data(pivoted)
    print(f"Unpivoted shape: {unpivoted.shape}")
    print(f"✓ Unpivot successful!")

    # Validate round-trip
    print("\nValidating round-trip transformation...")
    assert df.shape == unpivoted.shape, "Shape mismatch after round-trip"
    print("✓ Round-trip validation passed!")
