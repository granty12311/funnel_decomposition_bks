"""
Utility functions for funnel decomposition analysis.
"""

import pandas as pd
import numpy as np
from typing import Union, List
from datetime import datetime, date

try:
    from .dimension_config import (
        get_dimension_columns, get_finance_channel_column,
        get_finance_channel_values, get_lender_tier, LENDER_TIERS
    )
except ImportError:
    from dimension_config import (
        get_dimension_columns, get_finance_channel_column,
        get_finance_channel_values, get_lender_tier, LENDER_TIERS
    )


# Non-financed lender identifier (for datasets that include non-financed accounts)
NON_FINANCED_LENDER = 'NON_FINANCED'


def is_non_financed_lender(lender: str) -> bool:
    """Check if a lender is the non-financed placeholder (no funnel data available)."""
    return lender == NON_FINANCED_LENDER


def get_financed_lenders(df: pd.DataFrame) -> list:
    """Get list of financed lenders (excludes NON_FINANCED if present)."""
    all_lenders = df['lender'].unique().tolist()
    return [l for l in all_lenders if not is_non_financed_lender(l)]


def get_all_lenders(df: pd.DataFrame) -> list:
    """Get list of all unique lenders in the dataset."""
    return df['lender'].unique().tolist()


def get_all_finance_channels(df: pd.DataFrame) -> list:
    """Get list of all unique finance channels in the dataset."""
    channel_col = get_finance_channel_column()
    return df[channel_col].unique().tolist()


def validate_dataframe(df: pd.DataFrame, date_column: str = 'month_begin_date') -> None:
    """Validate input DataFrame has required columns including finance_channel."""
    dimension_cols = get_dimension_columns()
    channel_col = get_finance_channel_column()
    required_cols = [
        'lender', channel_col, date_column, *dimension_cols,
        'num_tot_bks', 'num_tot_apps', 'pct_of_total_apps',
        'str_apprv_rate', 'str_bk_rate', 'cond_apprv_rate', 'cond_bk_rate'
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Validate finance_channel values
    valid_channels = get_finance_channel_values()
    invalid_channels = set(df[channel_col].unique()) - set(valid_channels)
    if invalid_channels:
        raise ValueError(f"Invalid finance_channel values: {invalid_channels}. Expected: {valid_channels}")


def normalize_date(date_input: Union[str, datetime, date, pd.Timestamp]) -> pd.Timestamp:
    """Normalize date input to pandas Timestamp."""
    if isinstance(date_input, pd.Timestamp):
        return date_input
    return pd.Timestamp(date_input)


def calculate_segment_bookings(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived segment-level booking metrics."""
    df = df.copy()
    df['segment_apps'] = df['num_tot_apps'] * df['pct_of_total_apps']
    df['str_approvals'] = df['segment_apps'] * df['str_apprv_rate']
    df['str_bookings'] = df['str_approvals'] * df['str_bk_rate']
    df['cond_approvals'] = df['segment_apps'] * df['cond_apprv_rate']
    df['cond_bookings'] = df['cond_approvals'] * df['cond_bk_rate']
    df['segment_bookings'] = df['str_bookings'] + df['cond_bookings']
    return df


def validate_period_data(
    df: pd.DataFrame,
    date: pd.Timestamp,
    lender: str,
    finance_channel: str = None
) -> None:
    """Validate period data: mix sums to 1.0, bookings reconcile.

    Validates for a specific (lender, finance_channel, date) combination.
    If finance_channel is None, validates aggregate across all channels.
    """
    # When validating across all channels, skip pct_of_total_apps check
    # since it sums to 1.0 per channel, not across all channels
    if finance_channel is not None:
        pct_sum = df['pct_of_total_apps'].sum()
        if not np.isclose(pct_sum, 1.0, atol=1e-6):
            raise ValueError(
                f"pct_of_total_apps={pct_sum:.6f} for {lender}/{finance_channel} on {date.date()}"
            )

    segment_bks = df['segment_bookings'].sum()
    total_bks = df['num_tot_bks'].iloc[0]
    if not np.isclose(segment_bks, total_bks, rtol=0.01):
        raise ValueError(
            f"Bookings mismatch for {lender}/{finance_channel}: {segment_bks:.0f} vs {total_bks:.0f}"
        )


def format_date_label(date: pd.Timestamp) -> str:
    """Format date for display (e.g., 'Jan 2023')."""
    return date.strftime('%b %Y')


def format_number(value: float, decimals: int = 0) -> str:
    """Format number with commas."""
    return f"{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format as percentage (0.15 -> '15.0%')."""
    return f"{value * 100:.{decimals}f}%"
