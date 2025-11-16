"""
Symmetric decomposition calculator module.

Implements symmetric (order-independent) matrix decomposition to explain booking changes
between two time periods using midpoint methodology.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Union, NamedTuple
from datetime import datetime

try:
    from .utils import (
        validate_dataframe,
        normalize_date,
        calculate_segment_bookings,
        validate_period_data
    )
    from .dimension_config import get_dimension_columns
except ImportError:
    from utils import (
        validate_dataframe,
        normalize_date,
        calculate_segment_bookings,
        validate_period_data
    )
    from dimension_config import get_dimension_columns


class DecompositionResults(NamedTuple):
    """Container for decomposition results."""
    summary: pd.DataFrame
    segment_detail: pd.DataFrame
    metadata: dict


def calculate_decomposition(
    df: pd.DataFrame,
    date_a: Union[str, datetime, pd.Timestamp],
    date_b: Union[str, datetime, pd.Timestamp],
    lender: str = 'ACA',
    date_column: str = 'month_begin_date'
) -> DecompositionResults:
    """
    Calculate symmetric decomposition between two dates.

    Uses midpoint methodology where all effects are calculated independently
    using average values from both periods. This makes the decomposition
    order-independent.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with all months
    date_a : str, datetime, or pd.Timestamp
        Base period (Period 1)
    date_b : str, datetime, or pd.Timestamp
        Current period (Period 2)
    lender : str
        Lender to analyze (default 'ACA')
    date_column : str
        Name of the date column in the DataFrame (default 'month_begin_date').
        Use 'week_begin_date' for weekly analysis or any other date column name.

    Returns
    -------
    DecompositionResults
        Named tuple containing:
        - summary: Aggregate lender-level impacts
        - segment_detail: Segment-level breakdown
        - metadata: Calculation metadata
    """
    # Validate input (including date column)
    validate_dataframe(df, date_column=date_column)

    # Normalize dates
    date_a = normalize_date(date_a)
    date_b = normalize_date(date_b)

    # Ensure date column is datetime
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    # Extract periods
    df_1 = df[(df['lender'] == lender) & (df[date_column] == date_a)].copy()
    df_2 = df[(df['lender'] == lender) & (df[date_column] == date_b)].copy()

    if len(df_1) == 0:
        raise ValueError(f"No data found for {lender} on {date_a.date()}")
    if len(df_2) == 0:
        raise ValueError(f"No data found for {lender} on {date_b.date()}")

    # Calculate derived metrics
    df_1 = calculate_segment_bookings(df_1)
    df_2 = calculate_segment_bookings(df_2)

    # Validate periods
    validate_period_data(df_1, date_a, lender)
    validate_period_data(df_2, date_b, lender)

    # Calculate effects using symmetric approach
    segment_detail = _calculate_all_effects(df_1, df_2, date_a, date_b)

    # Validate reconciliation
    _validate_reconciliation(segment_detail, df_1, df_2)

    # Aggregate summary
    summary = _aggregate_summary(segment_detail)

    # Prepare metadata
    metadata = {
        'lender': lender,
        'date_a': str(date_a.date()),
        'date_b': str(date_b.date()),
        'period_1_total_apps': int(df_1['num_tot_apps'].iloc[0]),
        'period_2_total_apps': int(df_2['num_tot_apps'].iloc[0]),
        'period_1_total_bookings': float(df_1['num_tot_bks'].iloc[0]),
        'period_2_total_bookings': float(df_2['num_tot_bks'].iloc[0]),
        'delta_total_bookings': float(df_2['num_tot_bks'].iloc[0] - df_1['num_tot_bks'].iloc[0]),
        'num_segments': len(segment_detail),
        'calculation_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'symmetric'
    }

    return DecompositionResults(
        summary=summary,
        segment_detail=segment_detail,
        metadata=metadata
    )


def _calculate_all_effects(
    df_1: pd.DataFrame,
    df_2: pd.DataFrame,
    date_a: pd.Timestamp,
    date_b: pd.Timestamp
) -> pd.DataFrame:
    """
    Calculate all effects for each segment using symmetric (midpoint) methodology.

    All effects are calculated independently using average values from both periods,
    making the decomposition order-independent.

    Parameters
    ----------
    df_1 : pd.DataFrame
        Period 1 data with derived metrics
    df_2 : pd.DataFrame
        Period 2 data with derived metrics
    date_a : pd.Timestamp
        Period 1 date
    date_b : pd.Timestamp
        Period 2 date

    Returns
    -------
    pd.DataFrame
        Segment-level detail with all effects
    """
    # Merge periods on segment identifiers (dynamic dimension columns from config)
    merge_cols = get_dimension_columns()

    df_merged = df_1[merge_cols + [
        'num_tot_apps', 'pct_of_total_apps', 'segment_apps',
        'str_apprv_rate', 'str_bk_rate', 'cond_apprv_rate', 'cond_bk_rate',
        'segment_bookings'
    ]].merge(
        df_2[merge_cols + [
            'num_tot_apps', 'pct_of_total_apps', 'segment_apps',
            'str_apprv_rate', 'str_bk_rate', 'cond_apprv_rate', 'cond_bk_rate',
            'segment_bookings'
        ]],
        on=merge_cols,
        suffixes=('_1', '_2')
    )

    # Calculate average values for symmetric decomposition
    df_merged['avg_tot_apps'] = (df_merged['num_tot_apps_1'] + df_merged['num_tot_apps_2']) / 2
    df_merged['avg_pct_of_total'] = (df_merged['pct_of_total_apps_1'] + df_merged['pct_of_total_apps_2']) / 2
    df_merged['avg_str_apprv_rate'] = (df_merged['str_apprv_rate_1'] + df_merged['str_apprv_rate_2']) / 2
    df_merged['avg_str_bk_rate'] = (df_merged['str_bk_rate_1'] + df_merged['str_bk_rate_2']) / 2
    df_merged['avg_cond_apprv_rate'] = (df_merged['cond_apprv_rate_1'] + df_merged['cond_apprv_rate_2']) / 2
    df_merged['avg_cond_bk_rate'] = (df_merged['cond_bk_rate_1'] + df_merged['cond_bk_rate_2']) / 2

    # Calculate effects using symmetric approach (order-independent)
    df_merged['volume_effect'] = _calculate_volume_effect(df_merged)
    df_merged['mix_effect'] = _calculate_mix_effect(df_merged)
    df_merged['str_approval_effect'] = _calculate_str_approval_effect(df_merged)
    df_merged['cond_approval_effect'] = _calculate_cond_approval_effect(df_merged)
    df_merged['str_booking_effect'] = _calculate_str_booking_effect(df_merged)
    df_merged['cond_booking_effect'] = _calculate_cond_booking_effect(df_merged)

    # Calculate deltas first (needed for interaction effect)
    df_merged['delta_segment_bookings'] = df_merged['segment_bookings_2'] - df_merged['segment_bookings_1']

    # Calculate main effects sum
    main_effects_sum = (
        df_merged['volume_effect'] +
        df_merged['mix_effect'] +
        df_merged['str_approval_effect'] +
        df_merged['cond_approval_effect'] +
        df_merged['str_booking_effect'] +
        df_merged['cond_booking_effect']
    )

    # Calculate interaction effect (residual) to ensure perfect reconciliation
    # This captures the joint effect of multiple simultaneous changes
    df_merged['interaction_effect'] = df_merged['delta_segment_bookings'] - main_effects_sum

    # Calculate total effect
    df_merged['total_effect'] = (
        df_merged['volume_effect'] +
        df_merged['mix_effect'] +
        df_merged['str_approval_effect'] +
        df_merged['cond_approval_effect'] +
        df_merged['str_booking_effect'] +
        df_merged['cond_booking_effect'] +
        df_merged['interaction_effect']
    )

    # Calculate remaining deltas
    df_merged['delta_total_apps'] = df_merged['num_tot_apps_2'] - df_merged['num_tot_apps_1']
    df_merged['delta_pct_of_total'] = df_merged['pct_of_total_apps_2'] - df_merged['pct_of_total_apps_1']
    df_merged['delta_str_apprv_rate'] = df_merged['str_apprv_rate_2'] - df_merged['str_apprv_rate_1']
    df_merged['delta_str_bk_rate'] = df_merged['str_bk_rate_2'] - df_merged['str_bk_rate_1']
    df_merged['delta_cond_apprv_rate'] = df_merged['cond_apprv_rate_2'] - df_merged['cond_apprv_rate_1']
    df_merged['delta_cond_bk_rate'] = df_merged['cond_bk_rate_2'] - df_merged['cond_bk_rate_1']

    # Rename columns for clarity
    df_merged = df_merged.rename(columns={
        'num_tot_apps_1': 'period_1_total_apps',
        'num_tot_apps_2': 'period_2_total_apps',
        'pct_of_total_apps_1': 'period_1_pct_of_total',
        'pct_of_total_apps_2': 'period_2_pct_of_total',
        'segment_apps_1': 'period_1_segment_apps',
        'segment_apps_2': 'period_2_segment_apps',
        'str_apprv_rate_1': 'period_1_str_apprv_rate',
        'str_apprv_rate_2': 'period_2_str_apprv_rate',
        'str_bk_rate_1': 'period_1_str_bk_rate',
        'str_bk_rate_2': 'period_2_str_bk_rate',
        'cond_apprv_rate_1': 'period_1_cond_apprv_rate',
        'cond_apprv_rate_2': 'period_2_cond_apprv_rate',
        'cond_bk_rate_1': 'period_1_cond_bk_rate',
        'cond_bk_rate_2': 'period_2_cond_bk_rate',
        'segment_bookings_1': 'period_1_segment_bookings',
        'segment_bookings_2': 'period_2_segment_bookings',
    })

    # Add period dates
    df_merged['period_1_date'] = str(date_a.date())
    df_merged['period_2_date'] = str(date_b.date())

    # Reorder columns - dimension columns come first
    dimension_cols = get_dimension_columns()
    col_order = [
        # Identifiers (dynamic dimension columns)
        *dimension_cols,

        # Period 1
        'period_1_date', 'period_1_total_apps', 'period_1_pct_of_total',
        'period_1_segment_apps', 'period_1_str_apprv_rate', 'period_1_str_bk_rate',
        'period_1_cond_apprv_rate', 'period_1_cond_bk_rate', 'period_1_segment_bookings',

        # Period 2
        'period_2_date', 'period_2_total_apps', 'period_2_pct_of_total',
        'period_2_segment_apps', 'period_2_str_apprv_rate', 'period_2_str_bk_rate',
        'period_2_cond_apprv_rate', 'period_2_cond_bk_rate', 'period_2_segment_bookings',

        # Deltas
        'delta_total_apps', 'delta_pct_of_total', 'delta_str_apprv_rate',
        'delta_str_bk_rate', 'delta_cond_apprv_rate', 'delta_cond_bk_rate',
        'delta_segment_bookings',

        # Effects
        'volume_effect', 'mix_effect', 'str_approval_effect', 'cond_approval_effect',
        'str_booking_effect', 'cond_booking_effect', 'interaction_effect', 'total_effect'
    ]

    return df_merged[col_order]


def _calculate_volume_effect(df: pd.DataFrame) -> pd.Series:
    """
    Calculate volume effect using symmetric approach.

    Formula:
    Volume_Effect = ΔA × p_avg × (r_str_avg × b_str_avg + r_cond_avg × b_cond_avg)

    Where _avg means (value_1 + value_2) / 2
    """
    delta_apps = df['num_tot_apps_2'] - df['num_tot_apps_1']
    avg_mix = df['avg_pct_of_total']
    avg_conversion = (
        df['avg_str_apprv_rate'] * df['avg_str_bk_rate'] +
        df['avg_cond_apprv_rate'] * df['avg_cond_bk_rate']
    )

    return delta_apps * avg_mix * avg_conversion


def _calculate_mix_effect(df: pd.DataFrame) -> pd.Series:
    """
    Calculate mix effect using symmetric approach.

    Formula:
    Mix_Effect = A_avg × Δp × (r_str_avg × b_str_avg + r_cond_avg × b_cond_avg)
    """
    avg_apps = df['avg_tot_apps']
    delta_mix = df['pct_of_total_apps_2'] - df['pct_of_total_apps_1']
    avg_conversion = (
        df['avg_str_apprv_rate'] * df['avg_str_bk_rate'] +
        df['avg_cond_apprv_rate'] * df['avg_cond_bk_rate']
    )

    return avg_apps * delta_mix * avg_conversion


def _calculate_str_approval_effect(df: pd.DataFrame) -> pd.Series:
    """
    Calculate straight approval rate effect using symmetric approach.

    Formula:
    Str_Approval_Effect = A_avg × p_avg × Δr_str × b_str_avg
    """
    avg_apps = df['avg_tot_apps']
    avg_mix = df['avg_pct_of_total']
    delta_str_apprv = df['str_apprv_rate_2'] - df['str_apprv_rate_1']
    avg_str_bk = df['avg_str_bk_rate']

    return avg_apps * avg_mix * delta_str_apprv * avg_str_bk


def _calculate_cond_approval_effect(df: pd.DataFrame) -> pd.Series:
    """
    Calculate conditional approval rate effect using symmetric approach.

    Formula:
    Cond_Approval_Effect = A_avg × p_avg × Δr_cond × b_cond_avg
    """
    avg_apps = df['avg_tot_apps']
    avg_mix = df['avg_pct_of_total']
    delta_cond_apprv = df['cond_apprv_rate_2'] - df['cond_apprv_rate_1']
    avg_cond_bk = df['avg_cond_bk_rate']

    return avg_apps * avg_mix * delta_cond_apprv * avg_cond_bk


def _calculate_str_booking_effect(df: pd.DataFrame) -> pd.Series:
    """
    Calculate straight booking rate effect using symmetric approach.

    Formula:
    Str_Booking_Effect = A_avg × p_avg × r_str_avg × Δb_str
    """
    avg_apps = df['avg_tot_apps']
    avg_mix = df['avg_pct_of_total']
    avg_str_apprv = df['avg_str_apprv_rate']
    delta_str_bk = df['str_bk_rate_2'] - df['str_bk_rate_1']

    return avg_apps * avg_mix * avg_str_apprv * delta_str_bk


def _calculate_cond_booking_effect(df: pd.DataFrame) -> pd.Series:
    """
    Calculate conditional booking rate effect using symmetric approach.

    Formula:
    Cond_Booking_Effect = A_avg × p_avg × r_cond_avg × Δb_cond
    """
    avg_apps = df['avg_tot_apps']
    avg_mix = df['avg_pct_of_total']
    avg_cond_apprv = df['avg_cond_apprv_rate']
    delta_cond_bk = df['cond_bk_rate_2'] - df['cond_bk_rate_1']

    return avg_apps * avg_mix * avg_cond_apprv * delta_cond_bk


def _validate_reconciliation(
    segment_detail: pd.DataFrame,
    df_1: pd.DataFrame,
    df_2: pd.DataFrame
) -> None:
    """
    Validate that effects reconcile to actual booking changes.

    Parameters
    ----------
    segment_detail : pd.DataFrame
        Segment-level detail with effects
    df_1 : pd.DataFrame
        Period 1 data
    df_2 : pd.DataFrame
        Period 2 data

    Warnings
    --------
    UserWarning
        If reconciliation difference exceeds tolerance
    """
    # Check segment-level reconciliation
    max_segment_diff = (segment_detail['total_effect'] -
                        segment_detail['delta_segment_bookings']).abs().max()

    if max_segment_diff > 0.05:
        warnings.warn(
            f"Segment-level reconciliation difference detected. Max difference: {max_segment_diff:.4f}",
            UserWarning
        )

    # Check aggregate reconciliation
    total_effect = segment_detail['total_effect'].sum()
    actual_change = df_2['num_tot_bks'].iloc[0] - df_1['num_tot_bks'].iloc[0]

    if not np.isclose(total_effect, actual_change, atol=0.05):
        warnings.warn(
            f"Aggregate reconciliation difference detected. "
            f"Total effect: {total_effect:.2f}, Actual change: {actual_change:.2f}",
            UserWarning
        )


def _aggregate_summary(segment_detail: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate effects across segments for lender-level summary.

    Parameters
    ----------
    segment_detail : pd.DataFrame
        Segment-level detail

    Returns
    -------
    pd.DataFrame
        Summary with effect type, booking impact, and pct of total
    """
    effects = {
        'volume_effect': segment_detail['volume_effect'].sum(),
        'mix_effect': segment_detail['mix_effect'].sum(),
        'str_approval_effect': segment_detail['str_approval_effect'].sum(),
        'cond_approval_effect': segment_detail['cond_approval_effect'].sum(),
        'str_booking_effect': segment_detail['str_booking_effect'].sum(),
        'cond_booking_effect': segment_detail['cond_booking_effect'].sum(),
        'interaction_effect': segment_detail['interaction_effect'].sum(),
    }

    total_change = sum(effects.values())

    summary = pd.DataFrame({
        'effect_type': list(effects.keys()) + ['total_change'],
        'booking_impact': list(effects.values()) + [total_change]
    })

    return summary


class MultiLenderDecompositionResults(NamedTuple):
    """Container for multi-lender decomposition results."""
    lender_summaries: pd.DataFrame  # Summary for each lender
    aggregate_summary: pd.DataFrame  # Combined summary across all lenders
    lender_details: dict  # Dict of lender -> DecompositionResults
    metadata: dict


def calculate_multi_lender_decomposition(
    df: pd.DataFrame,
    date_a: Union[str, datetime, pd.Timestamp],
    date_b: Union[str, datetime, pd.Timestamp],
    lenders: list = None,
    date_column: str = 'month_begin_date'
) -> MultiLenderDecompositionResults:
    """
    Calculate symmetric decomposition across multiple lenders.

    Handles cases where lenders appear in only one time period by treating
    the missing period as having zero values for all metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with all months and all lenders
    date_a : str, datetime, or pd.Timestamp
        Base period (Period 1)
    date_b : str, datetime, or pd.Timestamp
        Current period (Period 2)
    lenders : list, optional
        List of lenders to analyze. If None, uses all lenders in data.
    date_column : str
        Name of the date column in the DataFrame (default 'month_begin_date').
        Use 'week_begin_date' for weekly analysis or any other date column name.

    Returns
    -------
    MultiLenderDecompositionResults
        Named tuple containing:
        - lender_summaries: Effect summaries for each lender
        - aggregate_summary: Combined summary across all lenders
        - lender_details: Full decomposition results for each lender
        - metadata: Calculation metadata
    """
    # Normalize dates
    date_a = normalize_date(date_a)
    date_b = normalize_date(date_b)

    # Ensure date column is datetime
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    # Get list of lenders across both periods
    if lenders is None:
        lenders_a = set(df[df[date_column] == date_a]['lender'].unique())
        lenders_b = set(df[df[date_column] == date_b]['lender'].unique())
        lenders = sorted(lenders_a | lenders_b)  # Union of both periods
    else:
        lenders_a = set(df[df[date_column] == date_a]['lender'].unique())
        lenders_b = set(df[df[date_column] == date_b]['lender'].unique())

    # Check for lenders missing in each period
    missing_in_a = [l for l in lenders if l not in lenders_a]
    missing_in_b = [l for l in lenders if l not in lenders_b]

    # Warn user about missing lenders
    if missing_in_a:
        warnings.warn(
            f"The following lenders are missing in Period 1 ({date_a.date()}) and will be treated as zeros: {', '.join(missing_in_a)}",
            UserWarning
        )
    if missing_in_b:
        warnings.warn(
            f"The following lenders are missing in Period 2 ({date_b.date()}) and will be treated as zeros: {', '.join(missing_in_b)}",
            UserWarning
        )

    # Calculate decomposition for each lender
    lender_results = {}
    for lender in lenders:
        print(f"Calculating decomposition for {lender}...")

        # Check if data exists for this lender in both periods
        has_data_a = lender in lenders_a
        has_data_b = lender in lenders_b

        if has_data_a and has_data_b:
            # Normal case - data exists in both periods
            results = calculate_decomposition(
                df=df,
                date_a=date_a,
                date_b=date_b,
                lender=lender,
                date_column=date_column
            )
        else:
            # Special case - create decomposition with zeros for missing period
            results = _calculate_decomposition_with_missing_period(
                df=df,
                date_a=date_a,
                date_b=date_b,
                lender=lender,
                date_column=date_column,
                has_data_a=has_data_a,
                has_data_b=has_data_b
            )

        lender_results[lender] = results

    # Aggregate results
    lender_summaries = _create_lender_summaries(lender_results)
    aggregate_summary = _create_aggregate_summary(lender_results)

    # Create metadata
    metadata = {
        'date_a': str(normalize_date(date_a).date()),
        'date_b': str(normalize_date(date_b).date()),
        'lenders': lenders,
        'num_lenders': len(lenders),
        'calculation_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'symmetric_multi_lender'
    }

    # Add aggregate totals to metadata
    total_bookings_1 = sum(r.metadata['period_1_total_bookings'] for r in lender_results.values())
    total_bookings_2 = sum(r.metadata['period_2_total_bookings'] for r in lender_results.values())

    metadata.update({
        'aggregate_period_1_bookings': float(total_bookings_1),
        'aggregate_period_2_bookings': float(total_bookings_2),
        'aggregate_delta_bookings': float(total_bookings_2 - total_bookings_1)
    })

    return MultiLenderDecompositionResults(
        lender_summaries=lender_summaries,
        aggregate_summary=aggregate_summary,
        lender_details=lender_results,
        metadata=metadata
    )


def _calculate_decomposition_with_missing_period(
    df: pd.DataFrame,
    date_a: pd.Timestamp,
    date_b: pd.Timestamp,
    lender: str,
    date_column: str,
    has_data_a: bool,
    has_data_b: bool
) -> DecompositionResults:
    """
    Calculate decomposition when one period is missing data for a lender.

    For lenders that appear or disappear, the entire change is attributed to
    'lender_addition' or 'lender_removal' rather than breaking it down into
    volume/mix/rate effects (which would create artificial interaction effects).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    date_a : pd.Timestamp
        Period 1 date
    date_b : pd.Timestamp
        Period 2 date
    lender : str
        Lender name
    date_column : str
        Date column name
    has_data_a : bool
        Whether data exists for period A
    has_data_b : bool
        Whether data exists for period B

    Returns
    -------
    DecompositionResults
        Decomposition results with entire change as lender addition/removal
    """
    # Determine which period has data
    if has_data_a:
        df_real = df[(df['lender'] == lender) & (df[date_column] == date_a)].copy()
        real_period = 1
        effect_type = 'lender_removal'
    else:
        df_real = df[(df['lender'] == lender) & (df[date_column] == date_b)].copy()
        real_period = 2
        effect_type = 'lender_addition'

    # Calculate derived metrics for real period
    df_real = calculate_segment_bookings(df_real)

    # Validate the real period
    if real_period == 1:
        validate_period_data(df_real, date_a, lender)
    else:
        validate_period_data(df_real, date_b, lender)

    # Get total bookings from the real period
    total_bookings = float(df_real['num_tot_bks'].iloc[0])

    # The entire change is attributed to lender addition/removal
    # For lender_addition: 0 → total_bookings (positive)
    # For lender_removal: total_bookings → 0 (negative)
    booking_change = total_bookings if effect_type == 'lender_addition' else -total_bookings

    # Create simplified summary (no volume/mix/rate breakdown)
    summary = pd.DataFrame({
        'effect_type': [effect_type, 'total_change'],
        'booking_impact': [booking_change, booking_change]
    })

    # Create simplified segment detail (minimal information)
    # Use the real period's segments as template
    dimension_cols = get_dimension_columns()
    segment_detail = df_real[dimension_cols].copy()

    # Add period information
    segment_detail['period_1_date'] = str(date_a.date())
    segment_detail['period_2_date'] = str(date_b.date())

    # Add bookings
    if real_period == 1:
        segment_detail['period_1_total_apps'] = df_real['num_tot_apps'].iloc[0]
        segment_detail['period_2_total_apps'] = 0
        segment_detail['period_1_segment_bookings'] = df_real['segment_bookings']
        segment_detail['period_2_segment_bookings'] = 0.0
    else:
        segment_detail['period_1_total_apps'] = 0
        segment_detail['period_2_total_apps'] = df_real['num_tot_apps'].iloc[0]
        segment_detail['period_1_segment_bookings'] = 0.0
        segment_detail['period_2_segment_bookings'] = df_real['segment_bookings']

    # Add the single effect
    segment_detail[effect_type] = segment_detail['period_2_segment_bookings'] - segment_detail['period_1_segment_bookings']
    segment_detail['total_effect'] = segment_detail[effect_type]

    # Prepare metadata
    metadata = {
        'lender': lender,
        'date_a': str(date_a.date()),
        'date_b': str(date_b.date()),
        'period_1_total_apps': int(df_real['num_tot_apps'].iloc[0]) if real_period == 1 else 0,
        'period_2_total_apps': int(df_real['num_tot_apps'].iloc[0]) if real_period == 2 else 0,
        'period_1_total_bookings': total_bookings if real_period == 1 else 0.0,
        'period_2_total_bookings': total_bookings if real_period == 2 else 0.0,
        'delta_total_bookings': booking_change,
        'num_segments': len(segment_detail),
        'calculation_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'lender_structural_change',
        'missing_period': 'period_1' if not has_data_a else 'period_2'
    }

    return DecompositionResults(
        summary=summary,
        segment_detail=segment_detail,
        metadata=metadata
    )


def _create_lender_summaries(lender_results: dict) -> pd.DataFrame:
    """
    Create a summary table with effects broken down by lender.

    Returns a DataFrame with:
    - Rows: effect types
    - Columns: lender, effect_type, booking_impact
    """
    all_summaries = []

    for lender, results in lender_results.items():
        summary = results.summary.copy()
        summary['lender'] = lender
        all_summaries.append(summary)

    # Combine all summaries
    combined = pd.concat(all_summaries, ignore_index=True)

    # Reorder columns
    combined = combined[['lender', 'effect_type', 'booking_impact']]

    return combined


def _create_aggregate_summary(lender_results: dict) -> pd.DataFrame:
    """
    Create an aggregate summary combining all lenders.

    Dynamically handles standard effects (volume, mix, rates, interaction) as well as
    structural effects (lender_addition, lender_removal).

    Returns a DataFrame with total impact for each effect type.
    """
    # Collect all unique effect types across all lenders
    all_effect_types = set()
    for lender, results in lender_results.items():
        all_effect_types.update(results.summary['effect_type'].tolist())

    # Initialize aggregates for all effect types found
    aggregates = {effect_type: 0.0 for effect_type in all_effect_types}

    # Sum across all lenders
    for lender, results in lender_results.items():
        for _, row in results.summary.iterrows():
            effect_type = row['effect_type']
            impact = row['booking_impact']
            aggregates[effect_type] += impact

    # Define standard ordering (with fallback for any unexpected types)
    standard_order = [
        'volume_effect',
        'mix_effect',
        'str_approval_effect',
        'cond_approval_effect',
        'str_booking_effect',
        'cond_booking_effect',
        'lender_addition',
        'lender_removal',
        'interaction_effect',
        'total_change'
    ]

    # Sort effect types by standard order, keeping any unexpected types at the end
    sorted_effects = [et for et in standard_order if et in aggregates]
    remaining = [et for et in aggregates.keys() if et not in standard_order]
    sorted_effects.extend(remaining)

    # Create DataFrame with sorted effects
    aggregate_df = pd.DataFrame({
        'effect_type': sorted_effects,
        'booking_impact': [aggregates[et] for et in sorted_effects]
    })

    return aggregate_df


def get_lender_breakdown(lender_summaries: pd.DataFrame, effect_type: str) -> pd.DataFrame:
    """
    Get breakdown of a specific effect by lender.

    Parameters
    ----------
    lender_summaries : pd.DataFrame
        Lender summaries from MultiLenderDecompositionResults
    effect_type : str
        Effect type to break down

    Returns
    -------
    pd.DataFrame
        Breakdown showing each lender's contribution to this effect
    """
    breakdown = lender_summaries[lender_summaries['effect_type'] == effect_type].copy()
    breakdown = breakdown.sort_values('booking_impact', ascending=False)
    return breakdown[['lender', 'booking_impact']]
