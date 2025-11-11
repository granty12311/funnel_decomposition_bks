"""
Core decomposition calculator module.

Implements hierarchical matrix decomposition to explain booking changes
between two time periods.
"""

import pandas as pd
import numpy as np
from typing import Union, NamedTuple
from datetime import datetime

try:
    from .utils import (
        validate_dataframe,
        normalize_date,
        calculate_segment_bookings,
        validate_period_data
    )
except ImportError:
    from utils import (
        validate_dataframe,
        normalize_date,
        calculate_segment_bookings,
        validate_period_data
    )


class DecompositionResults(NamedTuple):
    """Container for decomposition results."""
    summary: pd.DataFrame
    segment_detail: pd.DataFrame
    metadata: dict


def calculate_decomposition(
    df: pd.DataFrame,
    date_a: Union[str, datetime, pd.Timestamp],
    date_b: Union[str, datetime, pd.Timestamp],
    lender: str = 'ACA'
) -> DecompositionResults:
    """
    Calculate hierarchical decomposition between two dates.

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

    Returns
    -------
    DecompositionResults
        Named tuple containing:
        - summary: Aggregate lender-level impacts
        - segment_detail: Segment-level breakdown
        - metadata: Calculation metadata
    """
    # Validate input
    validate_dataframe(df)

    # Normalize dates
    date_a = normalize_date(date_a)
    date_b = normalize_date(date_b)

    # Ensure month_begin_date is datetime
    df = df.copy()
    df['month_begin_date'] = pd.to_datetime(df['month_begin_date'])

    # Extract periods
    df_1 = df[(df['lender'] == lender) & (df['month_begin_date'] == date_a)].copy()
    df_2 = df[(df['lender'] == lender) & (df['month_begin_date'] == date_b)].copy()

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

    # Calculate effects
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
        'calculation_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
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
    Calculate all effects for each segment.

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
    # Merge periods on segment identifiers
    merge_cols = ['fico_bands', 'offer_comp_tier', 'prod_line']

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

    # Calculate effects using sequential waterfall approach
    df_merged['volume_effect'] = _calculate_volume_effect(df_merged)
    df_merged['mix_effect'] = _calculate_mix_effect(df_merged)
    df_merged['str_approval_effect'] = _calculate_str_approval_effect(df_merged)
    df_merged['cond_approval_effect'] = _calculate_cond_approval_effect(df_merged)
    df_merged['str_booking_effect'] = _calculate_str_booking_effect(df_merged)
    df_merged['cond_booking_effect'] = _calculate_cond_booking_effect(df_merged)

    # Calculate total effect
    df_merged['total_effect'] = (
        df_merged['volume_effect'] +
        df_merged['mix_effect'] +
        df_merged['str_approval_effect'] +
        df_merged['cond_approval_effect'] +
        df_merged['str_booking_effect'] +
        df_merged['cond_booking_effect']
    )

    # Calculate deltas
    df_merged['delta_total_apps'] = df_merged['num_tot_apps_2'] - df_merged['num_tot_apps_1']
    df_merged['delta_pct_of_total'] = df_merged['pct_of_total_apps_2'] - df_merged['pct_of_total_apps_1']
    df_merged['delta_str_apprv_rate'] = df_merged['str_apprv_rate_2'] - df_merged['str_apprv_rate_1']
    df_merged['delta_str_bk_rate'] = df_merged['str_bk_rate_2'] - df_merged['str_bk_rate_1']
    df_merged['delta_cond_apprv_rate'] = df_merged['cond_apprv_rate_2'] - df_merged['cond_apprv_rate_1']
    df_merged['delta_cond_bk_rate'] = df_merged['cond_bk_rate_2'] - df_merged['cond_bk_rate_1']
    df_merged['delta_segment_bookings'] = df_merged['segment_bookings_2'] - df_merged['segment_bookings_1']

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

    # Reorder columns
    col_order = [
        # Identifiers
        'fico_bands', 'offer_comp_tier', 'prod_line',

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
        'str_booking_effect', 'cond_booking_effect', 'total_effect'
    ]

    return df_merged[col_order]


def _calculate_volume_effect(df: pd.DataFrame) -> pd.Series:
    """
    Calculate volume effect: Change in total apps × base mix × base rates.

    Formula:
    Volume_Effect = Δ A × p[1] × (r_str[1] × b_str[1] + r_cond[1] × b_cond[1])
    """
    delta_apps = df['num_tot_apps_2'] - df['num_tot_apps_1']
    base_mix = df['pct_of_total_apps_1']
    base_conversion = (
        df['str_apprv_rate_1'] * df['str_bk_rate_1'] +
        df['cond_apprv_rate_1'] * df['cond_bk_rate_1']
    )

    return delta_apps * base_mix * base_conversion


def _calculate_mix_effect(df: pd.DataFrame) -> pd.Series:
    """
    Calculate mix effect: New apps × change in mix × base rates.

    Formula:
    Mix_Effect = A[2] × Δ p × (r_str[1] × b_str[1] + r_cond[1] × b_cond[1])
    """
    new_apps = df['num_tot_apps_2']
    delta_mix = df['pct_of_total_apps_2'] - df['pct_of_total_apps_1']
    base_conversion = (
        df['str_apprv_rate_1'] * df['str_bk_rate_1'] +
        df['cond_apprv_rate_1'] * df['cond_bk_rate_1']
    )

    return new_apps * delta_mix * base_conversion


def _calculate_str_approval_effect(df: pd.DataFrame) -> pd.Series:
    """
    Calculate straight approval rate effect.

    Formula:
    Str_Approval_Effect = A[2] × p[2] × Δ r_str × b_str[1]
    """
    new_apps = df['num_tot_apps_2']
    new_mix = df['pct_of_total_apps_2']
    delta_str_apprv = df['str_apprv_rate_2'] - df['str_apprv_rate_1']
    base_str_bk = df['str_bk_rate_1']

    return new_apps * new_mix * delta_str_apprv * base_str_bk


def _calculate_cond_approval_effect(df: pd.DataFrame) -> pd.Series:
    """
    Calculate conditional approval rate effect.

    Formula:
    Cond_Approval_Effect = A[2] × p[2] × Δ r_cond × b_cond[1]
    """
    new_apps = df['num_tot_apps_2']
    new_mix = df['pct_of_total_apps_2']
    delta_cond_apprv = df['cond_apprv_rate_2'] - df['cond_apprv_rate_1']
    base_cond_bk = df['cond_bk_rate_1']

    return new_apps * new_mix * delta_cond_apprv * base_cond_bk


def _calculate_str_booking_effect(df: pd.DataFrame) -> pd.Series:
    """
    Calculate straight booking rate effect.

    Formula:
    Str_Booking_Effect = A[2] × p[2] × r_str[2] × Δ b_str
    """
    new_apps = df['num_tot_apps_2']
    new_mix = df['pct_of_total_apps_2']
    new_str_apprv = df['str_apprv_rate_2']
    delta_str_bk = df['str_bk_rate_2'] - df['str_bk_rate_1']

    return new_apps * new_mix * new_str_apprv * delta_str_bk


def _calculate_cond_booking_effect(df: pd.DataFrame) -> pd.Series:
    """
    Calculate conditional booking rate effect.

    Formula:
    Cond_Booking_Effect = A[2] × p[2] × r_cond[2] × Δ b_cond
    """
    new_apps = df['num_tot_apps_2']
    new_mix = df['pct_of_total_apps_2']
    new_cond_apprv = df['cond_apprv_rate_2']
    delta_cond_bk = df['cond_bk_rate_2'] - df['cond_bk_rate_1']

    return new_apps * new_mix * new_cond_apprv * delta_cond_bk


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

    Raises
    ------
    AssertionError
        If reconciliation fails
    """
    # Check segment-level reconciliation
    max_segment_diff = (segment_detail['total_effect'] -
                        segment_detail['delta_segment_bookings']).abs().max()

    if max_segment_diff > 0.01:
        raise AssertionError(
            f"Segment-level reconciliation failed. Max difference: {max_segment_diff:.4f}"
        )

    # Check aggregate reconciliation
    total_effect = segment_detail['total_effect'].sum()
    actual_change = df_2['num_tot_bks'].iloc[0] - df_1['num_tot_bks'].iloc[0]

    if not np.isclose(total_effect, actual_change, atol=0.01):
        raise AssertionError(
            f"Aggregate reconciliation failed. "
            f"Total effect: {total_effect:.2f}, Actual change: {actual_change:.2f}"
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
    }

    total_change = sum(effects.values())

    summary = pd.DataFrame({
        'effect_type': list(effects.keys()) + ['total_change'],
        'booking_impact': list(effects.values()) + [total_change]
    })

    return summary
