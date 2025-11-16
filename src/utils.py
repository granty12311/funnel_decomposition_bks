"""
Utility functions for funnel decomposition analysis.
"""

import pandas as pd
import numpy as np
from typing import Union
from datetime import datetime, date

try:
    from .dimension_config import get_dimension_columns
except ImportError:
    from dimension_config import get_dimension_columns


def validate_dataframe(df: pd.DataFrame, date_column: str = 'month_begin_date') -> None:
    """
    Validate that the input DataFrame has the required columns.

    Uses dynamic dimension column names from dimension_config.py.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to validate
    date_column : str
        Name of the date column to check for (default 'month_begin_date')

    Raises
    ------
    ValueError
        If required columns are missing
    """
    # Get dimension column names dynamically from config
    dimension_cols = get_dimension_columns()

    required_cols = [
        'lender', date_column, *dimension_cols,
        'num_tot_bks', 'num_tot_apps', 'pct_of_total_apps',
        'str_apprv_rate', 'str_bk_rate', 'cond_apprv_rate', 'cond_bk_rate'
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_date(date_input: Union[str, datetime, date, pd.Timestamp]) -> pd.Timestamp:
    """
    Normalize date input to pandas Timestamp.

    Parameters
    ----------
    date_input : str, datetime, date, or pd.Timestamp
        Date to normalize

    Returns
    -------
    pd.Timestamp
        Normalized timestamp
    """
    if isinstance(date_input, str):
        return pd.to_datetime(date_input)
    elif isinstance(date_input, pd.Timestamp):
        return date_input
    elif isinstance(date_input, datetime):
        return pd.Timestamp(date_input)
    elif isinstance(date_input, date):
        return pd.Timestamp(date_input)
    else:
        raise ValueError(f"Unsupported date type: {type(date_input)}")


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
    - pct_of_total_apps sums to ~1.0
    - segment_bookings sum to num_tot_bks

    Note: Segment count check has been removed as certain segments
    could be missing due to lack of volume.

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
