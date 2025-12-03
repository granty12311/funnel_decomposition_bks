"""
LMDI decomposition calculator for booking analysis.

Implements LMDI (Logarithmic Mean Divisia Index) decomposition with split mix effects:
- Volume effect: Total application volume change
- Customer mix effect: Customer segment distribution change
- Offer comp mix effect: Offer competitiveness distribution change
- Approval effects: Straight and conditional approval rate changes
- Booking effects: Straight and conditional booking rate changes
"""

import pandas as pd
import numpy as np
import warnings
from typing import Union, NamedTuple
from datetime import datetime

try:
    from .utils import validate_dataframe, normalize_date, calculate_segment_bookings, validate_period_data, is_non_financed_lender
    from .dimension_config import get_dimension_columns, NON_FINANCED_LENDER
except ImportError:
    from utils import validate_dataframe, normalize_date, calculate_segment_bookings, validate_period_data, is_non_financed_lender
    from dimension_config import get_dimension_columns, NON_FINANCED_LENDER

# Dimension hierarchy for split mix calculation
PRIMARY_DIMENSION = 'customer_segment'
SECONDARY_DIMENSION = 'offer_comp_tier'


class DecompositionResults(NamedTuple):
    """Container for decomposition results."""
    summary: pd.DataFrame
    segment_detail: pd.DataFrame
    metadata: dict


class MultiLenderResults(NamedTuple):
    """Container for multi-lender decomposition results."""
    lender_summaries: pd.DataFrame
    aggregate_summary: pd.DataFrame
    lender_details: dict
    metadata: dict


def logarithmic_mean(x0: float, xt: float, eps: float = 1e-10) -> float:
    """Calculate logarithmic mean L(x0, xt) = (xt - x0) / ln(xt / x0)."""
    if abs(x0) < eps and abs(xt) < eps:
        return 0.0
    if abs(xt - x0) < eps:
        return (x0 + xt) / 2
    if abs(x0) < eps:
        return xt / np.log(1 + xt / eps)
    if abs(xt) < eps:
        return x0 / np.log(1 + x0 / eps)
    try:
        result = (xt - x0) / np.log(xt / x0)
        return (x0 + xt) / 2 if np.isnan(result) or np.isinf(result) else result
    except (ZeroDivisionError, ValueError):
        return (x0 + xt) / 2


def safe_log_ratio(num: Union[float, pd.Series], den: Union[float, pd.Series], eps: float = 1e-10):
    """Calculate ln(num / den) with safe zero handling."""
    if isinstance(num, pd.Series):
        n, d = num.copy(), den.copy()
        both_zero = (n.abs() < eps) & (d.abs() < eps)
        n = n.where(n.abs() >= eps, eps)
        d = d.where(d.abs() >= eps, eps)
        result = np.log(n / d)
        result[both_zero] = 0.0
        return result
    else:
        if abs(num) < eps and abs(den) < eps:
            return 0.0
        return np.log(max(abs(num), eps) / max(abs(den), eps)) * (1 if num >= 0 else -1)


def calculate_decomposition(
    df: pd.DataFrame,
    date_a: Union[str, datetime, pd.Timestamp],
    date_b: Union[str, datetime, pd.Timestamp],
    lender: str = 'ACA',
    date_column: str = 'month_begin_date'
) -> DecompositionResults:
    """
    Calculate LMDI decomposition between two periods.

    Returns 7 effects: volume, customer_mix, offer_comp_mix,
    str_approval, cond_approval, str_booking, cond_booking.

    Note: NON_FINANCED lender cannot be decomposed (no funnel data).
    """
    # Validate lender is not NON_FINANCED
    if is_non_financed_lender(lender):
        raise ValueError(f"Cannot calculate decomposition for {NON_FINANCED_LENDER} - no funnel data available")

    validate_dataframe(df, date_column=date_column)
    date_a, date_b = normalize_date(date_a), normalize_date(date_b)

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    df_1 = df[(df['lender'] == lender) & (df[date_column] == date_a)].copy()
    df_2 = df[(df['lender'] == lender) & (df[date_column] == date_b)].copy()

    if len(df_1) == 0:
        raise ValueError(f"No data for {lender} on {date_a.date()}")
    if len(df_2) == 0:
        raise ValueError(f"No data for {lender} on {date_b.date()}")

    df_1 = calculate_segment_bookings(df_1)
    df_2 = calculate_segment_bookings(df_2)

    validate_period_data(df_1, date_a, lender)
    validate_period_data(df_2, date_b, lender)

    segment_detail = _calculate_effects(df_1, df_2, date_a, date_b)
    _validate_reconciliation(segment_detail, df_1, df_2)
    summary = _aggregate_summary(segment_detail)

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
        'method': 'lmdi_split_mix'
    }

    return DecompositionResults(summary=summary, segment_detail=segment_detail, metadata=metadata)


def _calculate_effects(df_1: pd.DataFrame, df_2: pd.DataFrame, date_a, date_b) -> pd.DataFrame:
    """Calculate all LMDI effects for each segment."""
    # Calculate customer shares (marginal distribution)
    cs1 = df_1.groupby(PRIMARY_DIMENSION)['pct_of_total_apps'].sum()
    cs2 = df_2.groupby(PRIMARY_DIMENSION)['pct_of_total_apps'].sum()
    df_1['customer_share'] = df_1[PRIMARY_DIMENSION].map(cs1)
    df_2['customer_share'] = df_2[PRIMARY_DIMENSION].map(cs2)

    # Calculate offer comp shares (conditional distribution)
    df_1['offer_comp_share'] = df_1['pct_of_total_apps'] / df_1['customer_share']
    df_2['offer_comp_share'] = df_2['pct_of_total_apps'] / df_2['customer_share']

    # Merge periods
    dims = get_dimension_columns()
    cols = dims + ['num_tot_apps', 'pct_of_total_apps', 'customer_share', 'offer_comp_share',
                   'segment_apps', 'str_apprv_rate', 'str_bk_rate', 'cond_apprv_rate', 'cond_bk_rate',
                   'segment_bookings', 'str_bookings', 'cond_bookings']

    m = df_1[cols].merge(df_2[cols], on=dims, suffixes=('_1', '_2'))

    # Calculate LMDI effects
    m['volume_effect'] = _calc_volume(m)
    m['customer_mix_effect'] = _calc_mix(m, 'customer_share')
    m['offer_comp_mix_effect'] = _calc_mix(m, 'offer_comp_share')
    m['str_approval_effect'] = _calc_rate(m, 'str_apprv_rate', 'str')
    m['cond_approval_effect'] = _calc_rate(m, 'cond_apprv_rate', 'cond')
    m['str_booking_effect'] = _calc_rate(m, 'str_bk_rate', 'str')
    m['cond_booking_effect'] = _calc_rate(m, 'cond_bk_rate', 'cond')

    m['total_effect'] = (m['volume_effect'] + m['customer_mix_effect'] + m['offer_comp_mix_effect'] +
                         m['str_approval_effect'] + m['cond_approval_effect'] +
                         m['str_booking_effect'] + m['cond_booking_effect'])

    # Rename columns
    rename_map = {f'{c}_1': f'period_1_{c.replace("_1", "")}' for c in m.columns if c.endswith('_1')}
    rename_map.update({f'{c}_2': f'period_2_{c.replace("_2", "")}' for c in m.columns if c.endswith('_2')})
    rename_map = {
        'num_tot_apps_1': 'period_1_total_apps', 'num_tot_apps_2': 'period_2_total_apps',
        'pct_of_total_apps_1': 'period_1_pct_of_total', 'pct_of_total_apps_2': 'period_2_pct_of_total',
        'customer_share_1': 'period_1_customer_share', 'customer_share_2': 'period_2_customer_share',
        'offer_comp_share_1': 'period_1_offer_comp_share', 'offer_comp_share_2': 'period_2_offer_comp_share',
        'segment_apps_1': 'period_1_segment_apps', 'segment_apps_2': 'period_2_segment_apps',
        'str_apprv_rate_1': 'period_1_str_apprv_rate', 'str_apprv_rate_2': 'period_2_str_apprv_rate',
        'str_bk_rate_1': 'period_1_str_bk_rate', 'str_bk_rate_2': 'period_2_str_bk_rate',
        'cond_apprv_rate_1': 'period_1_cond_apprv_rate', 'cond_apprv_rate_2': 'period_2_cond_apprv_rate',
        'cond_bk_rate_1': 'period_1_cond_bk_rate', 'cond_bk_rate_2': 'period_2_cond_bk_rate',
        'segment_bookings_1': 'period_1_segment_bookings', 'segment_bookings_2': 'period_2_segment_bookings',
        'str_bookings_1': 'period_1_str_bookings', 'str_bookings_2': 'period_2_str_bookings',
        'cond_bookings_1': 'period_1_cond_bookings', 'cond_bookings_2': 'period_2_cond_bookings',
    }
    m = m.rename(columns=rename_map)

    m['period_1_date'] = str(date_a.date())
    m['period_2_date'] = str(date_b.date())

    return m


def _calc_volume(df: pd.DataFrame) -> pd.Series:
    """Calculate volume effect."""
    log_vol = safe_log_ratio(df['num_tot_apps_2'].iloc[0], df['num_tot_apps_1'].iloc[0])
    w_str = df.apply(lambda r: logarithmic_mean(r['str_bookings_1'], r['str_bookings_2']), axis=1)
    w_cond = df.apply(lambda r: logarithmic_mean(r['cond_bookings_1'], r['cond_bookings_2']), axis=1)
    return (w_str + w_cond) * log_vol


def _calc_mix(df: pd.DataFrame, share_col: str) -> pd.Series:
    """Calculate mix effect for given share column."""
    log_ratio = safe_log_ratio(df[f'{share_col}_2'], df[f'{share_col}_1'])
    w_str = df.apply(lambda r: logarithmic_mean(r['str_bookings_1'], r['str_bookings_2']), axis=1)
    w_cond = df.apply(lambda r: logarithmic_mean(r['cond_bookings_1'], r['cond_bookings_2']), axis=1)
    return (w_str + w_cond) * log_ratio


def _calc_rate(df: pd.DataFrame, rate_col: str, funnel: str) -> pd.Series:
    """Calculate rate effect for given rate column."""
    bks_col = f'{funnel}_bookings'
    w = df.apply(lambda r: logarithmic_mean(r[f'{bks_col}_1'], r[f'{bks_col}_2']), axis=1)
    log_ratio = safe_log_ratio(df[f'{rate_col}_2'], df[f'{rate_col}_1'])
    return w * log_ratio


def _validate_reconciliation(seg: pd.DataFrame, df_1: pd.DataFrame, df_2: pd.DataFrame) -> None:
    """Validate effects reconcile to actual changes."""
    actual = df_2['num_tot_bks'].iloc[0] - df_1['num_tot_bks'].iloc[0]
    calculated = seg['total_effect'].sum()
    if not np.isclose(calculated, actual, atol=1.0):
        warnings.warn(f"Reconciliation: calculated={calculated:.2f}, actual={actual:.2f}")


def _aggregate_summary(seg: pd.DataFrame) -> pd.DataFrame:
    """Aggregate effects for summary."""
    effects = ['volume_effect', 'customer_mix_effect', 'offer_comp_mix_effect',
               'str_approval_effect', 'cond_approval_effect', 'str_booking_effect', 'cond_booking_effect']
    vals = [seg[e].sum() for e in effects]
    total = sum(vals)
    return pd.DataFrame({
        'effect_type': effects + ['total_change'],
        'booking_impact': vals + [total]
    })


def calculate_multi_lender_decomposition(
    df: pd.DataFrame,
    date_a: Union[str, datetime, pd.Timestamp],
    date_b: Union[str, datetime, pd.Timestamp],
    lenders: list = None,
    date_column: str = 'month_begin_date'
) -> MultiLenderResults:
    """Calculate decomposition across multiple lenders.

    Note: NON_FINANCED is automatically excluded from lender iteration (no funnel data).
    """
    date_a, date_b = normalize_date(date_a), normalize_date(date_b)
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    if lenders is None:
        la = set(df[df[date_column] == date_a]['lender'].unique())
        lb = set(df[df[date_column] == date_b]['lender'].unique())
        # Exclude NON_FINANCED from lender iteration
        lenders = sorted([l for l in (la | lb) if not is_non_financed_lender(l)])

    results = {}
    for lender in lenders:
        print(f"Processing {lender}...")
        try:
            results[lender] = calculate_decomposition(df, date_a, date_b, lender, date_column)
        except Exception as e:
            warnings.warn(f"Failed {lender}: {e}")

    # Build summaries
    summaries = []
    for lender, r in results.items():
        s = r.summary.copy()
        s['lender'] = lender
        summaries.append(s)
    lender_summaries = pd.concat(summaries)[['lender', 'effect_type', 'booking_impact']]

    # Aggregate
    agg = {}
    for r in results.values():
        for _, row in r.summary.iterrows():
            agg[row['effect_type']] = agg.get(row['effect_type'], 0) + row['booking_impact']

    effect_order = ['volume_effect', 'customer_mix_effect', 'offer_comp_mix_effect',
                    'str_approval_effect', 'cond_approval_effect', 'str_booking_effect',
                    'cond_booking_effect', 'total_change']
    sorted_effects = [e for e in effect_order if e in agg]
    aggregate_summary = pd.DataFrame({
        'effect_type': sorted_effects,
        'booking_impact': [agg[e] for e in sorted_effects]
    })

    bks1 = sum(r.metadata['period_1_total_bookings'] for r in results.values())
    bks2 = sum(r.metadata['period_2_total_bookings'] for r in results.values())

    metadata = {
        'date_a': str(date_a.date()),
        'date_b': str(date_b.date()),
        'lenders': list(results.keys()),
        'aggregate_period_1_bookings': bks1,
        'aggregate_period_2_bookings': bks2,
        'aggregate_delta_bookings': bks2 - bks1,
        'method': 'lmdi_split_mix_multi_lender'
    }

    return MultiLenderResults(
        lender_summaries=lender_summaries,
        aggregate_summary=aggregate_summary,
        lender_details=results,
        metadata=metadata
    )
