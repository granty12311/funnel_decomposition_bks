"""
Utility functions for funnel decomposition analysis.
"""

import pandas as pd
import numpy as np
from typing import Union
from datetime import datetime


def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that the input DataFrame has the required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to validate

    Raises
    ------
    ValueError
        If required columns are missing
    """
    required_cols = [
        'lender', 'month_begin_date', 'fico_bands', 'offer_comp_tier', 'prod_line',
        'num_tot_bks', 'num_tot_apps', 'pct_of_total_apps',
        'str_apprv_rate', 'str_bk_rate', 'cond_apprv_rate', 'cond_bk_rate'
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_date(date: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
    """
    Normalize date input to pandas Timestamp.

    Parameters
    ----------
    date : str, datetime, or pd.Timestamp
        Date to normalize

    Returns
    -------
    pd.Timestamp
        Normalized timestamp
    """
    if isinstance(date, str):
        return pd.to_datetime(date)
    elif isinstance(date, datetime):
        return pd.Timestamp(date)
    elif isinstance(date, pd.Timestamp):
        return date
    else:
        raise ValueError(f"Unsupported date type: {type(date)}")


def calculate_segment_bookings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived metrics for segment-level bookings.

    Adds columns:
    - segment_apps: Applications in this segment
    - str_approvals: Straight approvals in this segment
    - str_bookings: Bookings from straight approvals
    - cond_approvals: Conditional approvals in this segment
    - cond_bookings: Bookings from conditional approvals
    - segment_bookings: Total bookings in this segment

    Parameters
    ----------
    df : pd.DataFrame
        Segment-level data

    Returns
    -------
    pd.DataFrame
        DataFrame with derived metrics added
    """
    df = df.copy()

    # Calculate segment applications
    df['segment_apps'] = df['num_tot_apps'] * df['pct_of_total_apps']

    # Straight approval funnel
    df['str_approvals'] = df['segment_apps'] * df['str_apprv_rate']
    df['str_bookings'] = df['str_approvals'] * df['str_bk_rate']

    # Conditional approval funnel
    df['cond_approvals'] = df['segment_apps'] * df['cond_apprv_rate']
    df['cond_bookings'] = df['cond_approvals'] * df['cond_bk_rate']

    # Total segment bookings
    df['segment_bookings'] = df['str_bookings'] + df['cond_bookings']

    return df


def validate_period_data(df: pd.DataFrame, date: pd.Timestamp, lender: str) -> None:
    """
    Validate data for a specific period.

    Checks:
    - Has exactly 18 segments
    - pct_of_total_apps sums to ~1.0
    - segment_bookings sum to num_tot_bks

    Parameters
    ----------
    df : pd.DataFrame
        Period data (should be filtered to single lender-date)
    date : pd.Timestamp
        Date being validated
    lender : str
        Lender being validated

    Raises
    ------
    ValueError
        If validation checks fail
    """
    # Check segment count (18 for 3 FICO bands, 24 for 4 FICO bands)
    n_segments = len(df)
    if n_segments not in [18, 24]:
        raise ValueError(
            f"Expected 18 or 24 segments for {lender} on {date.date()}, found {n_segments}"
        )

    # Check pct_of_total_apps sums to 1.0
    pct_sum = df['pct_of_total_apps'].sum()
    if not np.isclose(pct_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"pct_of_total_apps should sum to 1.0 for {lender} on {date.date()}, "
            f"found {pct_sum:.6f}"
        )

    # Check segment bookings sum to total bookings
    segment_bks_sum = df['segment_bookings'].sum()
    num_tot_bks = df['num_tot_bks'].iloc[0]

    if not np.isclose(segment_bks_sum, num_tot_bks, rtol=0.01):
        raise ValueError(
            f"Segment bookings ({segment_bks_sum:.2f}) don't match num_tot_bks "
            f"({num_tot_bks:.2f}) for {lender} on {date.date()}"
        )


def format_date_label(date: pd.Timestamp) -> str:
    """
    Format date for display in charts and tables.

    Parameters
    ----------
    date : pd.Timestamp
        Date to format

    Returns
    -------
    str
        Formatted date string (e.g., "Jan 2023")
    """
    return date.strftime('%b %Y')


def format_number(value: float, decimals: int = 1) -> str:
    """
    Format number for display with commas and specified decimals.

    Parameters
    ----------
    value : float
        Number to format
    decimals : int
        Number of decimal places (default 1)

    Returns
    -------
    str
        Formatted number string
    """
    return f"{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format percentage for display.

    Parameters
    ----------
    value : float
        Percentage value (0.15 = 15%)
    decimals : int
        Number of decimal places (default 1)

    Returns
    -------
    str
        Formatted percentage string (e.g., "15.0%")
    """
    return f"{value * 100:.{decimals}f}%"
