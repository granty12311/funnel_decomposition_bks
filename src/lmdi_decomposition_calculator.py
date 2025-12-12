"""
LMDI decomposition calculator for booking analysis.

Implements LMDI (Logarithmic Mean Divisia Index) decomposition with split mix effects:
- Volume effect: Total application volume change
- Customer mix effect: Customer segment distribution change
- Offer comp mix effect: Offer competitiveness distribution change
- Approval effects: Straight and conditional approval rate changes
- Booking effects: Straight and conditional booking rate changes

Supports finance channel separation:
- FF and Non-FF are decomposed independently
- Results are aggregated without cross-channel mix effects
"""

import pandas as pd
import numpy as np
import warnings
from typing import Union, NamedTuple, List, Optional
from datetime import datetime

try:
    from .utils import (
        validate_dataframe, normalize_date, calculate_segment_bookings,
        validate_period_data, get_all_lenders, get_all_finance_channels
    )
    from .dimension_config import (
        get_dimension_columns, get_finance_channel_column,
        get_finance_channel_values, get_lender_tier, LENDER_TIERS
    )
except ImportError:
    from utils import (
        validate_dataframe, normalize_date, calculate_segment_bookings,
        validate_period_data, get_all_lenders, get_all_finance_channels
    )
    from dimension_config import (
        get_dimension_columns, get_finance_channel_column,
        get_finance_channel_values, get_lender_tier, LENDER_TIERS
    )

# Dimension hierarchy for split mix calculation
PRIMARY_DIMENSION = 'customer_segment'
SECONDARY_DIMENSION = 'offer_comp_tier'


class DecompositionResults(NamedTuple):
    """Container for single (lender, finance_channel) decomposition results."""
    summary: pd.DataFrame
    segment_detail: pd.DataFrame
    metadata: dict


class FinanceChannelResults(NamedTuple):
    """Container for single lender, multi-channel results."""
    aggregate_summary: pd.DataFrame      # Combined effects (FF + Non-FF)
    channel_summaries: pd.DataFrame      # Effects by finance_channel
    channel_details: dict                # {channel: DecompositionResults}
    metadata: dict


class MultiLenderResults(NamedTuple):
    """Container for multi-lender, multi-channel decomposition results."""
    aggregate_summary: pd.DataFrame       # All lenders + channels combined
    tier_summary: pd.DataFrame            # Aggregated by tier (T1/T2/T3)
    channel_summary: pd.DataFrame         # Aggregated by finance_channel
    lender_channel_summaries: pd.DataFrame  # Per lender-channel
    details: dict                         # {(lender, channel): DecompositionResults}
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
    lender: str,
    finance_channel: str,
    date_column: str = 'month_begin_date'
) -> DecompositionResults:
    """
    Calculate LMDI decomposition between two periods for a single lender and finance channel.

    Args:
        df: Input DataFrame with all required columns including finance_channel
        date_a: Period 1 date
        date_b: Period 2 date
        lender: Lender identifier
        finance_channel: Finance channel ('FF' or 'NON_FF')
        date_column: Name of the date column

    Returns:
        DecompositionResults with 7 effects: volume, customer_mix, offer_comp_mix,
        str_approval, cond_approval, str_booking, cond_booking.
    """
    validate_dataframe(df, date_column=date_column)
    date_a, date_b = normalize_date(date_a), normalize_date(date_b)
    channel_col = get_finance_channel_column()

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    # Filter for specific lender and finance channel
    df_1 = df[
        (df['lender'] == lender) &
        (df[channel_col] == finance_channel) &
        (df[date_column] == date_a)
    ].copy()
    df_2 = df[
        (df['lender'] == lender) &
        (df[channel_col] == finance_channel) &
        (df[date_column] == date_b)
    ].copy()

    if len(df_1) == 0:
        raise ValueError(f"No data for {lender}/{finance_channel} on {date_a.date()}")
    if len(df_2) == 0:
        raise ValueError(f"No data for {lender}/{finance_channel} on {date_b.date()}")

    df_1 = calculate_segment_bookings(df_1)
    df_2 = calculate_segment_bookings(df_2)

    validate_period_data(df_1, date_a, lender, finance_channel)
    validate_period_data(df_2, date_b, lender, finance_channel)

    segment_detail = _calculate_effects(df_1, df_2, date_a, date_b)
    reconciliation = _validate_reconciliation(segment_detail, df_1, df_2, lender, finance_channel)
    _emit_reconciliation_warning(reconciliation)
    summary = _aggregate_summary(segment_detail)

    metadata = {
        'lender': lender,
        'finance_channel': finance_channel,
        'date_a': str(date_a.date()),
        'date_b': str(date_b.date()),
        'period_1_total_apps': int(df_1['num_tot_apps'].iloc[0]),
        'period_2_total_apps': int(df_2['num_tot_apps'].iloc[0]),
        'period_1_total_bookings': float(df_1['num_tot_bks'].iloc[0]),
        'period_2_total_bookings': float(df_2['num_tot_bks'].iloc[0]),
        'delta_total_bookings': float(df_2['num_tot_bks'].iloc[0] - df_1['num_tot_bks'].iloc[0]),
        'num_segments': len(segment_detail),
        'method': 'lmdi_split_mix',
        'reconciliation_status': reconciliation.status,
        'reconciliation_diff': reconciliation.details.get('diff', 0)
    }

    return DecompositionResults(summary=summary, segment_detail=segment_detail, metadata=metadata)


def _calculate_effects(df_1: pd.DataFrame, df_2: pd.DataFrame, date_a, date_b) -> pd.DataFrame:
    """Calculate all LMDI effects for each segment."""
    # Calculate customer shares (marginal distribution) using pct_of_total_apps
    # Since VSA Progression is a separate effect, mix should use the original app mix
    cs1 = df_1.groupby(PRIMARY_DIMENSION)['pct_of_total_apps'].sum()
    cs2 = df_2.groupby(PRIMARY_DIMENSION)['pct_of_total_apps'].sum()
    df_1['customer_share'] = df_1[PRIMARY_DIMENSION].map(cs1)
    df_2['customer_share'] = df_2[PRIMARY_DIMENSION].map(cs2)

    # Calculate offer comp shares (conditional distribution) using pct_of_total_apps
    df_1['offer_comp_share'] = df_1['pct_of_total_apps'] / df_1['customer_share']
    df_2['offer_comp_share'] = df_2['pct_of_total_apps'] / df_2['customer_share']

    # Merge periods
    dims = get_dimension_columns()
    cols = dims + ['num_tot_apps', 'pct_of_total_apps', 'pct_of_total_vsa', 'vsa_prog_pct',
                   'customer_share', 'offer_comp_share',
                   'segment_apps', 'segment_vsa',  # VSA count for approval rate weights
                   'str_approvals', 'cond_approvals',  # Approval counts for correct LMDI weights
                   'str_apprv_rate', 'str_bk_rate', 'cond_apprv_rate', 'cond_bk_rate',
                   'segment_bookings', 'str_bookings', 'cond_bookings']

    m = df_1[cols].merge(df_2[cols], on=dims, suffixes=('_1', '_2'))

    # Calculate LMDI effects (now 8 effects with VSA Progression)
    m['volume_effect'] = _calc_volume(m)
    m['vsa_progression_effect'] = _calc_vsa_progression(m)  # NEW: VSA Progression
    m['customer_mix_effect'] = _calc_mix(m, 'customer_share')
    m['offer_comp_mix_effect'] = _calc_mix(m, 'offer_comp_share')
    m['str_approval_effect'] = _calc_rate(m, 'str_apprv_rate', 'str')
    m['cond_approval_effect'] = _calc_rate(m, 'cond_apprv_rate', 'cond')
    m['str_booking_effect'] = _calc_rate(m, 'str_bk_rate', 'str')
    m['cond_booking_effect'] = _calc_rate(m, 'cond_bk_rate', 'cond')

    m['total_effect'] = (m['volume_effect'] + m['vsa_progression_effect'] +
                         m['customer_mix_effect'] + m['offer_comp_mix_effect'] +
                         m['str_approval_effect'] + m['cond_approval_effect'] +
                         m['str_booking_effect'] + m['cond_booking_effect'])

    # Rename columns
    rename_map = {f'{c}_1': f'period_1_{c.replace("_1", "")}' for c in m.columns if c.endswith('_1')}
    rename_map.update({f'{c}_2': f'period_2_{c.replace("_2", "")}' for c in m.columns if c.endswith('_2')})
    rename_map = {
        'num_tot_apps_1': 'period_1_total_apps', 'num_tot_apps_2': 'period_2_total_apps',
        'pct_of_total_apps_1': 'period_1_pct_of_total', 'pct_of_total_apps_2': 'period_2_pct_of_total',
        'pct_of_total_vsa_1': 'period_1_pct_of_total_vsa', 'pct_of_total_vsa_2': 'period_2_pct_of_total_vsa',
        'vsa_prog_pct_1': 'period_1_vsa_prog_pct', 'vsa_prog_pct_2': 'period_2_vsa_prog_pct',
        'customer_share_1': 'period_1_customer_share', 'customer_share_2': 'period_2_customer_share',
        'offer_comp_share_1': 'period_1_offer_comp_share', 'offer_comp_share_2': 'period_2_offer_comp_share',
        'segment_apps_1': 'period_1_segment_apps', 'segment_apps_2': 'period_2_segment_apps',
        'segment_vsa_1': 'period_1_segment_vsa', 'segment_vsa_2': 'period_2_segment_vsa',
        'str_approvals_1': 'period_1_str_approvals', 'str_approvals_2': 'period_2_str_approvals',
        'cond_approvals_1': 'period_1_cond_approvals', 'cond_approvals_2': 'period_2_cond_approvals',
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


def _calc_vsa_progression(df: pd.DataFrame) -> pd.Series:
    """Calculate VSA progression effect.

    Calculated as rate effect with combined weight (str + cond bookings).
    Effect = [L(Str_Bks_1, Str_Bks_2) + L(Cond_Bks_1, Cond_Bks_2)] × ln(vsa_prog_pct_2 / vsa_prog_pct_1)
    """
    log_ratio = safe_log_ratio(df['vsa_prog_pct_2'], df['vsa_prog_pct_1'])
    w_str = df.apply(lambda r: logarithmic_mean(r['str_bookings_1'], r['str_bookings_2']), axis=1)
    w_cond = df.apply(lambda r: logarithmic_mean(r['cond_bookings_1'], r['cond_bookings_2']), axis=1)
    return (w_str + w_cond) * log_ratio


def _calc_mix(df: pd.DataFrame, share_col: str) -> pd.Series:
    """Calculate mix effect for given share column."""
    log_ratio = safe_log_ratio(df[f'{share_col}_2'], df[f'{share_col}_1'])
    w_str = df.apply(lambda r: logarithmic_mean(r['str_bookings_1'], r['str_bookings_2']), axis=1)
    w_cond = df.apply(lambda r: logarithmic_mean(r['cond_bookings_1'], r['cond_bookings_2']), axis=1)
    return (w_str + w_cond) * log_ratio


def _calc_rate(df: pd.DataFrame, rate_col: str, funnel: str) -> pd.Series:
    """Calculate rate effect for given rate column.

    For multiplicative decomposition (Y = A × B × C), all effects use L(Y) as weight.
    For str_bookings = segment_vsa × str_apprv_rate × str_bk_rate:
    - All effects use L(str_bookings) as weight
    """
    bks_col = f'{funnel}_bookings'
    w = df.apply(lambda r: logarithmic_mean(r[f'{bks_col}_1'], r[f'{bks_col}_2']), axis=1)
    log_ratio = safe_log_ratio(df[f'{rate_col}_2'], df[f'{rate_col}_1'])
    return w * log_ratio


class ReconciliationResult:
    """Container for reconciliation validation results."""
    def __init__(self, status: str, message: str, details: dict = None):
        self.status = status  # 'ok', 'info', 'warning', 'error'
        self.message = message
        self.details = details or {}


def _detect_path_shutdown(df_1: pd.DataFrame, df_2: pd.DataFrame) -> dict:
    """Detect if any funnel paths have been shut down (rates going to exactly 0)."""
    shutdowns = {
        'str_apprv_shutdown': [],
        'cond_apprv_shutdown': [],
        'str_bk_shutdown': [],
        'cond_bk_shutdown': []
    }

    rate_cols = [
        ('str_apprv_rate', 'str_apprv_shutdown'),
        ('cond_apprv_rate', 'cond_apprv_shutdown'),
        ('str_bk_rate', 'str_bk_shutdown'),
        ('cond_bk_rate', 'cond_bk_shutdown')
    ]

    dims = get_dimension_columns()
    merged = df_1[dims + [c[0] for c in rate_cols]].merge(
        df_2[dims + [c[0] for c in rate_cols]],
        on=dims, suffixes=('_1', '_2')
    )

    for rate_col, shutdown_key in rate_cols:
        # Path shutdown: rate was > 0 in P1 but is exactly 0 in P2
        shutdown_mask = (merged[f'{rate_col}_1'] > 0.01) & (merged[f'{rate_col}_2'] < 1e-10)
        if shutdown_mask.any():
            for _, row in merged[shutdown_mask].iterrows():
                seg_name = f"{row[dims[0]]}/{row[dims[1]]}"
                shutdowns[shutdown_key].append({
                    'segment': seg_name,
                    'rate_p1': row[f'{rate_col}_1'],
                    'rate_p2': row[f'{rate_col}_2']
                })

    return shutdowns


def _detect_data_rounding(df_1: pd.DataFrame, df_2: pd.DataFrame) -> dict:
    """Detect if num_tot_bks appears to be rounded (causing small discrepancies)."""
    result = {'p1_rounded': False, 'p2_rounded': False, 'p1_diff': 0, 'p2_diff': 0}

    # Check if stored num_tot_bks matches calculated segment bookings
    calc_bks_1 = df_1['segment_bookings'].sum()
    calc_bks_2 = df_2['segment_bookings'].sum()
    stored_bks_1 = df_1['num_tot_bks'].iloc[0]
    stored_bks_2 = df_2['num_tot_bks'].iloc[0]

    # Check if stored values look like rounded integers
    result['p1_diff'] = calc_bks_1 - stored_bks_1
    result['p2_diff'] = calc_bks_2 - stored_bks_2

    # Rounding detected if: stored is integer-like AND there's a small diff
    if abs(stored_bks_1 - round(stored_bks_1)) < 0.001 and abs(result['p1_diff']) > 0.01:
        result['p1_rounded'] = True
    if abs(stored_bks_2 - round(stored_bks_2)) < 0.001 and abs(result['p2_diff']) > 0.01:
        result['p2_rounded'] = True

    return result


def _validate_reconciliation(
    seg: pd.DataFrame,
    df_1: pd.DataFrame,
    df_2: pd.DataFrame,
    lender: str,
    finance_channel: str
) -> ReconciliationResult:
    """
    Validate effects reconcile to actual changes with detailed error classification.

    Returns ReconciliationResult with status:
    - 'ok': Perfect reconciliation
    - 'info': Minor issue (rounding, expected edge case)
    - 'warning': Moderate discrepancy worth investigating
    - 'error': Major problem indicating data or calculation issues
    """
    actual = df_2['num_tot_bks'].iloc[0] - df_1['num_tot_bks'].iloc[0]
    calculated = seg['total_effect'].sum()
    diff = calculated - actual
    abs_diff = abs(diff)

    # Calculate relative error (avoid division by zero)
    if abs(actual) > 1e-10:
        rel_error = abs_diff / abs(actual)
    else:
        rel_error = 0 if abs_diff < 1e-10 else float('inf')

    context = f"{lender}/{finance_channel}"

    # Perfect reconciliation
    if abs_diff < 0.01:
        return ReconciliationResult('ok', f"{context}: Exact reconciliation", {
            'actual': actual, 'calculated': calculated, 'diff': diff
        })

    # Detect specific issues
    shutdowns = _detect_path_shutdown(df_1, df_2)
    rounding = _detect_data_rounding(df_1, df_2)

    has_shutdown = any(len(v) > 0 for v in shutdowns.values())
    has_rounding = rounding['p1_rounded'] or rounding['p2_rounded']

    # Case 1: Path shutdown detected - expected LMDI limitation
    if has_shutdown and abs_diff > 1.0:
        shutdown_segments = []
        for key, segments in shutdowns.items():
            for s in segments:
                shutdown_segments.append(f"{s['segment']} ({key.replace('_shutdown', '')})")

        return ReconciliationResult('info',
            f"{context}: Path shutdown detected (known LMDI limitation). "
            f"Segments with rate->0: {', '.join(shutdown_segments[:3])}{'...' if len(shutdown_segments) > 3 else ''}. "
            f"Diff={diff:+.1f} ({rel_error*100:.1f}%)",
            {'actual': actual, 'calculated': calculated, 'diff': diff,
             'shutdowns': shutdowns, 'is_expected': True}
        )

    # Case 2: Small rounding error (< 1% or < 1 booking)
    if abs_diff < 1.0 or rel_error < 0.01:
        if has_rounding:
            return ReconciliationResult('info',
                f"{context}: Minor rounding discrepancy in stored num_tot_bks. "
                f"Diff={diff:+.2f} ({rel_error*100:.2f}%)",
                {'actual': actual, 'calculated': calculated, 'diff': diff,
                 'rounding': rounding, 'is_expected': True}
            )
        return ReconciliationResult('ok', f"{context}: Within tolerance", {
            'actual': actual, 'calculated': calculated, 'diff': diff
        })

    # Case 3: Moderate discrepancy (1-5% or 1-50 bookings) - likely data issue
    if rel_error < 0.05 or abs_diff < 50:
        cause = ""
        if has_rounding:
            cause = "Likely caused by rounded num_tot_bks in source data. "
        return ReconciliationResult('warning',
            f"{context}: Moderate reconciliation discrepancy. {cause}"
            f"Calculated={calculated:.1f}, Actual={actual:.1f}, Diff={diff:+.1f} ({rel_error*100:.1f}%)",
            {'actual': actual, 'calculated': calculated, 'diff': diff,
             'rounding': rounding}
        )

    # Case 4: Major discrepancy (> 5%) - indicates serious problem but don't error
    return ReconciliationResult('warning_major',
        f"{context}: Large reconciliation discrepancy ({rel_error*100:.1f}%). "
        f"Calculated={calculated:.1f}, Actual={actual:.1f}, Diff={diff:+.1f}. "
        f"Check data integrity: segment bookings should sum to num_tot_bks.",
        {'actual': actual, 'calculated': calculated, 'diff': diff,
         'rounding': rounding, 'shutdowns': shutdowns, 'is_major': True}
    )


def _emit_reconciliation_warning(result: ReconciliationResult) -> None:
    """Emit appropriate warning based on reconciliation result."""
    if result.status == 'ok':
        return  # No warning needed
    elif result.status == 'info':
        # Only emit info-level messages if they indicate known limitations
        if result.details.get('is_expected'):
            warnings.warn(f"[INFO] {result.message}", stacklevel=4)
    elif result.status == 'warning':
        warnings.warn(f"[WARNING] {result.message}", stacklevel=4)
    elif result.status == 'warning_major':
        warnings.warn(f"[WARNING-MAJOR] {result.message}", stacklevel=4)


def _aggregate_summary(seg: pd.DataFrame) -> pd.DataFrame:
    """Aggregate effects for summary (now 8 effects with VSA Progression)."""
    effects = ['volume_effect', 'vsa_progression_effect', 'customer_mix_effect', 'offer_comp_mix_effect',
               'str_approval_effect', 'cond_approval_effect', 'str_booking_effect', 'cond_booking_effect']
    vals = [seg[e].sum() for e in effects]
    total = sum(vals)
    return pd.DataFrame({
        'effect_type': effects + ['total_change'],
        'booking_impact': vals + [total]
    })


# Effect type ordering for aggregation (8 effects with VSA Progression)
EFFECT_ORDER = [
    'volume_effect', 'vsa_progression_effect', 'customer_mix_effect', 'offer_comp_mix_effect',
    'str_approval_effect', 'cond_approval_effect', 'str_booking_effect',
    'cond_booking_effect', 'total_change'
]


def calculate_finance_channel_decomposition(
    df: pd.DataFrame,
    date_a: Union[str, datetime, pd.Timestamp],
    date_b: Union[str, datetime, pd.Timestamp],
    lender: str,
    finance_channels: Optional[List[str]] = None,
    date_column: str = 'month_begin_date'
) -> FinanceChannelResults:
    """
    Calculate decomposition for each finance channel, then aggregate.

    Decomposition is calculated independently for each finance channel (FF, Non-FF),
    then effects are summed. This ensures no mix effects between finance channels.

    Args:
        df: Input DataFrame
        date_a: Period 1 date
        date_b: Period 2 date
        lender: Lender identifier
        finance_channels: List of channels to analyze (default: ['FF', 'NON_FF'])
        date_column: Name of date column

    Returns:
        FinanceChannelResults with aggregate and per-channel breakdowns
    """
    date_a, date_b = normalize_date(date_a), normalize_date(date_b)
    channel_col = get_finance_channel_column()

    if finance_channels is None:
        finance_channels = get_finance_channel_values()

    # Calculate decomposition for each channel
    channel_details = {}
    for channel in finance_channels:
        print(f"  Processing {lender}/{channel}...")
        try:
            channel_details[channel] = calculate_decomposition(
                df, date_a, date_b, lender, channel, date_column
            )
        except Exception as e:
            warnings.warn(f"Failed {lender}/{channel}: {e}")

    if not channel_details:
        raise ValueError(f"No successful decompositions for {lender}")

    # Build channel summaries
    summaries = []
    for channel, result in channel_details.items():
        s = result.summary.copy()
        s['finance_channel'] = channel
        summaries.append(s)
    channel_summaries = pd.concat(summaries)[['finance_channel', 'effect_type', 'booking_impact']]

    # Aggregate by effect type (simple sum across channels)
    agg = {}
    for result in channel_details.values():
        for _, row in result.summary.iterrows():
            agg[row['effect_type']] = agg.get(row['effect_type'], 0) + row['booking_impact']

    sorted_effects = [e for e in EFFECT_ORDER if e in agg]
    aggregate_summary = pd.DataFrame({
        'effect_type': sorted_effects,
        'booking_impact': [agg[e] for e in sorted_effects]
    })

    # Build metadata
    total_bks_1 = sum(r.metadata['period_1_total_bookings'] for r in channel_details.values())
    total_bks_2 = sum(r.metadata['period_2_total_bookings'] for r in channel_details.values())
    total_apps_1 = sum(r.metadata['period_1_total_apps'] for r in channel_details.values())
    total_apps_2 = sum(r.metadata['period_2_total_apps'] for r in channel_details.values())

    # Per-channel totals for display
    channel_totals = {
        channel: {
            'period_1_bookings': r.metadata['period_1_total_bookings'],
            'period_2_bookings': r.metadata['period_2_total_bookings'],
            'delta_bookings': r.metadata['delta_total_bookings']
        }
        for channel, r in channel_details.items()
    }

    metadata = {
        'lender': lender,
        'date_a': str(date_a.date()),
        'date_b': str(date_b.date()),
        'finance_channels': list(channel_details.keys()),
        'period_1_total_apps': total_apps_1,
        'period_2_total_apps': total_apps_2,
        'period_1_total_bookings': total_bks_1,
        'period_2_total_bookings': total_bks_2,
        'delta_total_bookings': total_bks_2 - total_bks_1,
        'channel_totals': channel_totals,
        'method': 'lmdi_split_mix_multi_channel'
    }

    return FinanceChannelResults(
        aggregate_summary=aggregate_summary,
        channel_summaries=channel_summaries,
        channel_details=channel_details,
        metadata=metadata
    )


def calculate_multi_lender_decomposition(
    df: pd.DataFrame,
    date_a: Union[str, datetime, pd.Timestamp],
    date_b: Union[str, datetime, pd.Timestamp],
    lenders: Optional[List[str]] = None,
    finance_channels: Optional[List[str]] = None,
    date_column: str = 'month_begin_date'
) -> MultiLenderResults:
    """
    Calculate decomposition for all lender × finance_channel combinations.

    Decomposes each (lender, channel) pair independently, then aggregates:
    - By lender-channel (raw)
    - By tier (T1, T2, T3)
    - By finance channel (FF, Non-FF)
    - Total aggregate

    Args:
        df: Input DataFrame
        date_a: Period 1 date
        date_b: Period 2 date
        lenders: List of lenders (default: all in data)
        finance_channels: List of channels (default: ['FF', 'NON_FF'])
        date_column: Date column name

    Returns:
        MultiLenderResults with tier and channel aggregations
    """
    date_a, date_b = normalize_date(date_a), normalize_date(date_b)
    channel_col = get_finance_channel_column()

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    # Determine lenders and channels
    if lenders is None:
        la = set(df[df[date_column] == date_a]['lender'].unique())
        lb = set(df[df[date_column] == date_b]['lender'].unique())
        lenders = sorted(la | lb)

    if finance_channels is None:
        finance_channels = get_finance_channel_values()

    # Calculate decomposition for each (lender, channel) pair
    details = {}
    for lender in lenders:
        for channel in finance_channels:
            print(f"Processing {lender}/{channel}...")
            try:
                details[(lender, channel)] = calculate_decomposition(
                    df, date_a, date_b, lender, channel, date_column
                )
            except Exception as e:
                warnings.warn(f"Failed {lender}/{channel}: {e}")

    if not details:
        raise ValueError("No successful decompositions")

    # Build lender-channel summaries
    lc_summaries = []
    for (lender, channel), result in details.items():
        s = result.summary.copy()
        s['lender'] = lender
        s['finance_channel'] = channel
        s['lender_tier'] = get_lender_tier(lender)
        lc_summaries.append(s)
    lender_channel_summaries = pd.concat(lc_summaries)[
        ['lender', 'lender_tier', 'finance_channel', 'effect_type', 'booking_impact']
    ]

    # Aggregate by tier
    tier_agg = {}
    for (lender, channel), result in details.items():
        tier = get_lender_tier(lender)
        if tier not in tier_agg:
            tier_agg[tier] = {}
        for _, row in result.summary.iterrows():
            tier_agg[tier][row['effect_type']] = (
                tier_agg[tier].get(row['effect_type'], 0) + row['booking_impact']
            )

    tier_rows = []
    for tier in ['T1', 'T2', 'T3']:
        if tier in tier_agg:
            for effect in EFFECT_ORDER:
                if effect in tier_agg[tier]:
                    tier_rows.append({
                        'lender_tier': tier,
                        'effect_type': effect,
                        'booking_impact': tier_agg[tier][effect]
                    })
    tier_summary = pd.DataFrame(tier_rows)

    # Aggregate by finance channel
    channel_agg = {}
    for (lender, channel), result in details.items():
        if channel not in channel_agg:
            channel_agg[channel] = {}
        for _, row in result.summary.iterrows():
            channel_agg[channel][row['effect_type']] = (
                channel_agg[channel].get(row['effect_type'], 0) + row['booking_impact']
            )

    channel_rows = []
    for channel in finance_channels:
        if channel in channel_agg:
            for effect in EFFECT_ORDER:
                if effect in channel_agg[channel]:
                    channel_rows.append({
                        'finance_channel': channel,
                        'effect_type': effect,
                        'booking_impact': channel_agg[channel][effect]
                    })
    channel_summary = pd.DataFrame(channel_rows)

    # Total aggregate
    total_agg = {}
    for result in details.values():
        for _, row in result.summary.iterrows():
            total_agg[row['effect_type']] = (
                total_agg.get(row['effect_type'], 0) + row['booking_impact']
            )

    sorted_effects = [e for e in EFFECT_ORDER if e in total_agg]
    aggregate_summary = pd.DataFrame({
        'effect_type': sorted_effects,
        'booking_impact': [total_agg[e] for e in sorted_effects]
    })

    # Build metadata
    total_bks_1 = sum(r.metadata['period_1_total_bookings'] for r in details.values())
    total_bks_2 = sum(r.metadata['period_2_total_bookings'] for r in details.values())

    # Per-channel totals
    channel_totals = {}
    for channel in finance_channels:
        channel_bks_1 = sum(
            r.metadata['period_1_total_bookings']
            for (l, c), r in details.items() if c == channel
        )
        channel_bks_2 = sum(
            r.metadata['period_2_total_bookings']
            for (l, c), r in details.items() if c == channel
        )
        channel_totals[channel] = {
            'period_1_bookings': channel_bks_1,
            'period_2_bookings': channel_bks_2,
            'delta_bookings': channel_bks_2 - channel_bks_1
        }

    # Per-tier totals
    tier_totals = {}
    for tier in ['T1', 'T2', 'T3']:
        tier_bks_1 = sum(
            r.metadata['period_1_total_bookings']
            for (l, c), r in details.items() if get_lender_tier(l) == tier
        )
        tier_bks_2 = sum(
            r.metadata['period_2_total_bookings']
            for (l, c), r in details.items() if get_lender_tier(l) == tier
        )
        if tier_bks_1 > 0 or tier_bks_2 > 0:
            tier_totals[tier] = {
                'period_1_bookings': tier_bks_1,
                'period_2_bookings': tier_bks_2,
                'delta_bookings': tier_bks_2 - tier_bks_1
            }

    metadata = {
        'date_a': str(date_a.date()),
        'date_b': str(date_b.date()),
        'lenders': sorted(set(l for l, c in details.keys())),
        'finance_channels': sorted(set(c for l, c in details.keys())),
        'period_1_total_bookings': total_bks_1,
        'period_2_total_bookings': total_bks_2,
        'delta_total_bookings': total_bks_2 - total_bks_1,
        'channel_totals': channel_totals,
        'tier_totals': tier_totals,
        'method': 'lmdi_split_mix_multi_lender_channel'
    }

    return MultiLenderResults(
        aggregate_summary=aggregate_summary,
        tier_summary=tier_summary,
        channel_summary=channel_summary,
        lender_channel_summaries=lender_channel_summaries,
        details=details,
        metadata=metadata
    )
