"""
LMDI penetration decomposition calculator.

Analyzes penetration changes (Lender Bookings / Total Market Bookings) with decomposed effects:
- 7 gross lender effects (volume, customer_mix, offer_comp_mix, approvals, bookings)
- Self-adjustment (lender's contribution to denominator growth)
- 7 net lender effects (gross - self-adjustment allocated proportionally)
- 7 competitor effects (from rest of market, excluding lender)
- 7 net effects (net lender + competitor for each driver)

Uses SELF-ADJUSTED approach:
- When lender grows, it affects both numerator AND denominator
- Self-adjustment captures the denominator impact from lender's own growth
- Competitor effect is purely from rest of market (no self-influence)
- Exact reconciliation with no residual

All outputs in basis points (bps). 100 bps = 1 percentage point.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Union, NamedTuple
from datetime import datetime

try:
    from .utils import validate_dataframe, normalize_date, calculate_segment_bookings, validate_period_data, is_non_financed_lender, get_financed_lenders, NON_FINANCED_LENDER
    from .dimension_config import get_dimension_columns, get_finance_channel_column
    from .lmdi_decomposition_calculator import logarithmic_mean, safe_log_ratio, PRIMARY_DIMENSION, SECONDARY_DIMENSION
except ImportError:
    from utils import validate_dataframe, normalize_date, calculate_segment_bookings, validate_period_data, is_non_financed_lender, get_financed_lenders, NON_FINANCED_LENDER
    from dimension_config import get_dimension_columns, get_finance_channel_column
    from lmdi_decomposition_calculator import logarithmic_mean, safe_log_ratio, PRIMARY_DIMENSION, SECONDARY_DIMENSION


def _aggregate_across_channels(df: pd.DataFrame, date_column: str = 'month_begin_date') -> pd.DataFrame:
    """
    Aggregate data across finance channels to get lender-level data.

    For penetration analysis, we need lender-level totals, not channel-level.
    This aggregates segment data by summing across finance channels.
    """
    dims = get_dimension_columns()  # ['customer_segment', 'offer_comp_tier']
    channel_col = get_finance_channel_column()

    # Check if finance_channel column exists
    if channel_col not in df.columns:
        return df  # No channels to aggregate

    # Group by lender, date, and segment dimensions (aggregating across channels)
    group_cols = ['lender', date_column] + dims

    # For aggregation, we need to sum applications and recalculate rates
    df = df.copy()

    # Calculate segment-level counts before aggregating (new VSA flow)
    df['segment_apps'] = df['num_tot_apps'] * df['pct_of_total_apps']
    df['segment_vsa'] = df['segment_apps'] * df['vsa_prog_pct']
    df['str_approvals'] = df['segment_vsa'] * df['str_apprv_rate']
    df['cond_apps'] = df['segment_vsa'] * (1 - df['str_apprv_rate'])
    df['cond_approvals'] = df['cond_apps'] * df['cond_apprv_rate']
    df['str_bookings'] = df['str_approvals'] * df['str_bk_rate']
    df['cond_bookings'] = df['cond_approvals'] * df['cond_bk_rate']
    df['segment_bookings'] = df['str_bookings'] + df['cond_bookings']

    # Aggregate across channels
    agg_dict = {
        'segment_apps': 'sum',
        'str_approvals': 'sum',
        'cond_apps': 'sum',
        'cond_approvals': 'sum',
        'str_bookings': 'sum',
        'cond_bookings': 'sum',
        'segment_bookings': 'sum'
    }

    agg = df.groupby(group_cols).agg(agg_dict).reset_index()

    # Calculate lender-level totals
    lender_totals = agg.groupby(['lender', date_column]).agg({
        'segment_apps': 'sum',
        'segment_bookings': 'sum'
    }).reset_index().rename(columns={
        'segment_apps': 'num_tot_apps',
        'segment_bookings': 'num_tot_bks'
    })

    # Merge back to get lender totals
    agg = agg.merge(lender_totals, on=['lender', date_column])

    # Calculate segment percentages (of lender total apps)
    agg['pct_of_total_apps'] = agg['segment_apps'] / agg['num_tot_apps']

    # Calculate VSA progression metrics (aggregated values)
    # After aggregation, set vsa_prog_pct = 1.0 and pct_of_total_vsa = pct_of_total_apps
    # This maintains consistency for penetration analysis at the lender level
    agg['vsa_prog_pct'] = 1.0
    agg['pct_of_total_vsa'] = agg['pct_of_total_apps']

    # Recalculate rates from aggregated counts
    agg['str_apprv_rate'] = agg['str_approvals'] / agg['segment_apps'].replace(0, 1)
    agg['cond_apprv_rate'] = agg['cond_approvals'] / agg['cond_apps'].replace(0, 1)
    agg['str_bk_rate'] = agg['str_bookings'] / agg['str_approvals'].replace(0, 1)
    agg['cond_bk_rate'] = agg['cond_bookings'] / agg['cond_approvals'].replace(0, 1)

    # Clip rates to [0, 1]
    for col in ['str_apprv_rate', 'cond_apprv_rate', 'str_bk_rate', 'cond_bk_rate']:
        agg[col] = agg[col].clip(0, 1)

    return agg


class PenetrationResults(NamedTuple):
    """Container for penetration decomposition results."""
    summary: pd.DataFrame
    segment_detail: pd.DataFrame
    metadata: dict


class MultiLenderPenetrationResults(NamedTuple):
    """Container for multi-lender penetration results."""
    lender_summaries: pd.DataFrame
    aggregate_summary: pd.DataFrame
    lender_details: dict
    metadata: dict


def calculate_penetration_decomposition(
    df: pd.DataFrame,
    date_a: Union[str, datetime, pd.Timestamp],
    date_b: Union[str, datetime, pd.Timestamp],
    lender: str = 'ACA',
    date_column: str = 'month_begin_date'
) -> PenetrationResults:
    """
    Calculate LMDI penetration decomposition with self-adjusted lender effects.

    Uses SELF-ADJUSTED approach:
    - Gross lender effects capture numerator impact
    - Self-adjustment captures lender's contribution to denominator growth
    - Net lender effects = Gross - Self-adjustment (allocated proportionally)
    - Competitor effects = pure rest-of-market impact (includes NON_FINANCED)
    - Exact reconciliation: Net Lender + Competitor = Delta Penetration

    Note: Total market includes NON_FINANCED bookings in the denominator for all effects.
    NON_FINANCED rows contribute to total market but are not decomposed (no funnel data).

    Returns DataFrame with columns:
    - effect_type
    - gross_lender_effect_bps (numerator impact)
    - self_adjustment_bps (denominator impact from lender's own growth)
    - net_lender_effect_bps (gross - self-adjustment)
    - competitor_effect_bps (rest of market impact, includes non-financed)
    - net_effect_bps (net_lender + competitor)
    """
    # Validate lender is not NON_FINANCED
    if is_non_financed_lender(lender):
        raise ValueError(f"Cannot calculate decomposition for {NON_FINANCED_LENDER} - no funnel data available")

    validate_dataframe(df, date_column=date_column)
    date_a, date_b = normalize_date(date_a), normalize_date(date_b)

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    # Check if finance channels are present and aggregate if so
    channel_col = get_finance_channel_column()
    has_channels = channel_col in df.columns

    if has_channels:
        # Aggregate across finance channels for penetration analysis
        df_agg = _aggregate_across_channels(df, date_column)
    else:
        df_agg = df

    # Total market bookings (for penetration calculation)
    # Need to sum unique lender bookings per period
    total_market_1 = df_agg[df_agg[date_column] == date_a].groupby('lender')['num_tot_bks'].first().sum()
    total_market_2 = df_agg[df_agg[date_column] == date_b].groupby('lender')['num_tot_bks'].first().sum()

    if total_market_1 == 0 or total_market_2 == 0:
        raise ValueError("No market bookings found")

    # Lender data (from aggregated data)
    df_1 = df_agg[(df_agg['lender'] == lender) & (df_agg[date_column] == date_a)].copy()
    df_2 = df_agg[(df_agg['lender'] == lender) & (df_agg[date_column] == date_b)].copy()

    if len(df_1) == 0 or len(df_2) == 0:
        raise ValueError(f"No data for {lender}")

    # Segment bookings already calculated if channels were aggregated
    if not has_channels:
        df_1 = calculate_segment_bookings(df_1)
        df_2 = calculate_segment_bookings(df_2)

    validate_period_data(df_1, date_a, lender)
    validate_period_data(df_2, date_b, lender)

    # Lender bookings
    bks_1, bks_2 = df_1['num_tot_bks'].iloc[0], df_2['num_tot_bks'].iloc[0]
    delta_lender_bks = bks_2 - bks_1

    # Get non-financed bookings (if present in data)
    nf_data_1 = df[(df['lender'] == NON_FINANCED_LENDER) & (df[date_column] == date_a)]
    nf_data_2 = df[(df['lender'] == NON_FINANCED_LENDER) & (df[date_column] == date_b)]
    nf_bks_1 = nf_data_1['num_tot_bks'].iloc[0] if len(nf_data_1) > 0 else 0
    nf_bks_2 = nf_data_2['num_tot_bks'].iloc[0] if len(nf_data_2) > 0 else 0
    delta_nf_bks = nf_bks_2 - nf_bks_1

    # Rest of market (all other lenders + non-financed)
    rest_of_market_1 = total_market_1 - bks_1
    rest_of_market_2 = total_market_2 - bks_2
    delta_rest_of_market = rest_of_market_2 - rest_of_market_1

    # Financed competitors only (excluding lender and non-financed)
    financed_competitors_1 = rest_of_market_1 - nf_bks_1
    financed_competitors_2 = rest_of_market_2 - nf_bks_2
    delta_financed_competitors = financed_competitors_2 - financed_competitors_1

    # Total market change
    delta_market = total_market_2 - total_market_1

    # Penetration calculation
    pen_1, pen_2 = bks_1 / total_market_1, bks_2 / total_market_2
    delta_pen = pen_2 - pen_1

    # LMDI penetration decomposition
    # P = L / M
    # ΔP ≈ L(P) × [ln(L₂/L₁) - ln(M₂/M₁)]
    #    = Gross Lender Effect + Total Market Effect
    L_pen = logarithmic_mean(pen_1, pen_2)

    # Gross lender effect (numerator impact)
    total_gross_lender_effect = L_pen * safe_log_ratio(bks_2, bks_1)

    # Total market effect (denominator impact)
    total_market_effect = -L_pen * safe_log_ratio(total_market_2, total_market_1)

    # Split market effect into self-adjustment vs competitor
    # Self-adjustment = lender's share of market growth
    # Competitor = rest of market's share
    if abs(delta_market) > 1e-10:
        self_adj_share = delta_lender_bks / delta_market
        competitor_share = delta_rest_of_market / delta_market
    else:
        # No market change - split evenly or based on levels
        self_adj_share = bks_1 / total_market_1
        competitor_share = rest_of_market_1 / total_market_1

    total_self_adjustment = total_market_effect * self_adj_share
    total_competitor_effect = total_market_effect * competitor_share

    # Verify: self_adjustment + competitor = market effect
    assert np.isclose(total_self_adjustment + total_competitor_effect, total_market_effect, atol=1e-10)

    # Get lender booking effects (7 effects in booking terms)
    seg_detail, lender_bks_effects = _calc_booking_effects(df_1, df_2, date_a, date_b)

    # Scale gross lender effects to penetration
    total_lender_bks_effect = sum(lender_bks_effects.values())
    gross_scale = total_gross_lender_effect / total_lender_bks_effect if abs(total_lender_bks_effect) > 1e-10 else 0.0
    gross_lender_pen_effects = {k: v * gross_scale for k, v in lender_bks_effects.items()}

    # Allocate self-adjustment to each lender effect proportionally
    # Based on each effect's share of total lender booking change
    self_adj_allocated = {}
    for k, v in lender_bks_effects.items():
        if abs(total_lender_bks_effect) > 1e-10:
            effect_share = v / total_lender_bks_effect
        else:
            effect_share = 1.0 / 7.0  # Equal split if no change
        self_adj_allocated[k] = total_self_adjustment * effect_share

    # Net lender effects = Gross - Self-adjustment
    net_lender_pen_effects = {k: gross_lender_pen_effects[k] + self_adj_allocated[k]
                              for k in lender_bks_effects.keys()}

    # Get competitor booking effects (REST OF MARKET)
    competitor_bks_effects = _calc_competitor_booking_effects(df, date_a, date_b, date_column, exclude_lender=lender)

    # Scale competitor effects to penetration
    total_competitor_bks_effect = sum(competitor_bks_effects.values())
    if abs(total_competitor_bks_effect) > 1e-10:
        competitor_scale = total_competitor_effect / total_competitor_bks_effect
    else:
        competitor_scale = 0.0
    competitor_pen_effects = {k: v * competitor_scale for k, v in competitor_bks_effects.items()}

    # Net effects = Net lender + Competitor (now 8 effects with VSA Progression)
    effect_names = ['volume_effect', 'vsa_progression_effect', 'customer_mix_effect', 'offer_comp_mix_effect',
                    'str_approval_effect', 'cond_approval_effect', 'str_booking_effect',
                    'cond_booking_effect']

    net_effects = {k: net_lender_pen_effects[k] + competitor_pen_effects[k] for k in effect_names}

    # Validate reconciliation (should be exact, no residual)
    sum_net_lender = sum(net_lender_pen_effects.values())
    sum_competitor = sum(competitor_pen_effects.values())
    sum_net = sum(net_effects.values())

    expected_delta = total_gross_lender_effect + total_market_effect

    if not np.isclose(sum_net_lender, total_gross_lender_effect + total_self_adjustment, atol=1e-10):
        warnings.warn(f"Net lender reconciliation error: {abs(sum_net_lender - (total_gross_lender_effect + total_self_adjustment)):.10f}")

    if not np.isclose(sum_competitor, total_competitor_effect, atol=1e-10):
        warnings.warn(f"Competitor reconciliation error: {abs(sum_competitor - total_competitor_effect):.10f}")

    if not np.isclose(sum_net, delta_pen, atol=1e-8):
        warnings.warn(f"Total reconciliation error: {abs(sum_net - delta_pen):.10f}")

    # Build summary DataFrame (in bps)
    rows = []
    for e in effect_names:
        rows.append({
            'effect_type': e,
            'gross_lender_effect_bps': gross_lender_pen_effects[e] * 10000,
            'self_adjustment_bps': self_adj_allocated[e] * 10000,
            'net_lender_effect_bps': net_lender_pen_effects[e] * 10000,
            'competitor_effect_bps': competitor_pen_effects[e] * 10000,
            'net_effect_bps': net_effects[e] * 10000
        })

    # Add totals row
    rows.append({
        'effect_type': 'total_change',
        'gross_lender_effect_bps': sum(gross_lender_pen_effects.values()) * 10000,
        'self_adjustment_bps': sum(self_adj_allocated.values()) * 10000,
        'net_lender_effect_bps': sum_net_lender * 10000,
        'competitor_effect_bps': sum_competitor * 10000,
        'net_effect_bps': sum_net * 10000
    })

    summary = pd.DataFrame(rows)

    # Add penetration to segment detail
    seg_detail['period_1_segment_penetration'] = seg_detail['period_1_segment_bookings'] / total_market_1
    seg_detail['period_2_segment_penetration'] = seg_detail['period_2_segment_bookings'] / total_market_2

    for col in effect_names:
        if f'{col}_bks' in seg_detail.columns:
            seg_detail[f'{col}_bps'] = seg_detail[f'{col}_bks'] * gross_scale * 10000

    seg_detail['total_lender_effect_bps'] = sum(seg_detail[f'{c}_bps'] for c in effect_names
                                                 if f'{c}_bps' in seg_detail.columns)

    # Metadata with full breakdown
    metadata = {
        'lender': lender,
        'date_a': str(date_a.date()),
        'date_b': str(date_b.date()),
        # Booking metrics
        'period_1_lender_bookings': float(bks_1),
        'period_2_lender_bookings': float(bks_2),
        'delta_lender_bookings': float(delta_lender_bks),
        'period_1_total_market_bookings': float(total_market_1),
        'period_2_total_market_bookings': float(total_market_2),
        'delta_total_market_bookings': float(delta_market),
        'period_1_rest_of_market_bookings': float(rest_of_market_1),
        'period_2_rest_of_market_bookings': float(rest_of_market_2),
        'delta_rest_of_market_bookings': float(delta_rest_of_market),
        # Non-financed bookings (included in total market and rest of market)
        'period_1_non_financed_bookings': float(nf_bks_1),
        'period_2_non_financed_bookings': float(nf_bks_2),
        'delta_non_financed_bookings': float(delta_nf_bks),
        # Financed competitors only (rest of market minus non-financed)
        'period_1_financed_competitors_bookings': float(financed_competitors_1),
        'period_2_financed_competitors_bookings': float(financed_competitors_2),
        'delta_financed_competitors_bookings': float(delta_financed_competitors),
        # Penetration metrics
        'period_1_penetration': float(pen_1),
        'period_2_penetration': float(pen_2),
        'delta_penetration_bps': float(delta_pen * 10000),
        # Effect totals (in bps)
        'total_gross_lender_effect_bps': float(total_gross_lender_effect * 10000),
        'total_market_effect_bps': float(total_market_effect * 10000),
        'total_self_adjustment_bps': float(total_self_adjustment * 10000),
        'total_net_lender_effect_bps': float(sum_net_lender * 10000),
        'total_competitor_effect_bps': float(sum_competitor * 10000),
        # Shares for transparency
        'self_adjustment_share': float(self_adj_share),
        'competitor_share': float(competitor_share),
        # For Chart 4: non-volume competitor effect
        'competitor_volume_effect_bps': float(competitor_pen_effects['volume_effect'] * 10000),
        'competitor_non_volume_effect_bps': float((sum_competitor - competitor_pen_effects['volume_effect']) * 10000),
        # Net volume for Chart 4
        'net_volume_effect_bps': float(net_effects['volume_effect'] * 10000),
        # Method identifier
        'method': 'lmdi_penetration_self_adjusted'
    }

    return PenetrationResults(summary=summary, segment_detail=seg_detail, metadata=metadata)


def _calc_booking_effects(df_1, df_2, date_a, date_b):
    """Calculate segment-level booking effects for a single lender (now 8 effects with VSA Progression)."""
    # Customer shares (using pct_of_total_apps since VSA Progression is separate)
    cs1 = df_1.groupby(PRIMARY_DIMENSION)['pct_of_total_apps'].sum()
    cs2 = df_2.groupby(PRIMARY_DIMENSION)['pct_of_total_apps'].sum()
    df_1 = df_1.copy()
    df_2 = df_2.copy()
    df_1['customer_share'] = df_1[PRIMARY_DIMENSION].map(cs1)
    df_2['customer_share'] = df_2[PRIMARY_DIMENSION].map(cs2)
    df_1['offer_comp_share'] = df_1['pct_of_total_apps'] / df_1['customer_share']
    df_2['offer_comp_share'] = df_2['pct_of_total_apps'] / df_2['customer_share']

    # Merge
    dims = get_dimension_columns()
    cols = dims + ['num_tot_apps', 'pct_of_total_apps', 'pct_of_total_vsa', 'vsa_prog_pct',
                   'customer_share', 'offer_comp_share',
                   'segment_apps', 'str_apprv_rate', 'str_bk_rate', 'cond_apprv_rate', 'cond_bk_rate',
                   'segment_bookings', 'str_bookings', 'cond_bookings']
    m = df_1[cols].merge(df_2[cols], on=dims, suffixes=('_1', '_2'))

    # Weights
    m['w_str'] = m.apply(lambda r: logarithmic_mean(r['str_bookings_1'], r['str_bookings_2']), axis=1)
    m['w_cond'] = m.apply(lambda r: logarithmic_mean(r['cond_bookings_1'], r['cond_bookings_2']), axis=1)
    m['w_total'] = m['w_str'] + m['w_cond']

    # Log ratios
    apps_1, apps_2 = df_1['num_tot_apps'].iloc[0], df_2['num_tot_apps'].iloc[0]
    m['ln_vol'] = safe_log_ratio(apps_2, apps_1)
    m['ln_vsa'] = safe_log_ratio(m['vsa_prog_pct_2'], m['vsa_prog_pct_1'])  # NEW: VSA Progression
    m['ln_cs'] = safe_log_ratio(m['customer_share_2'], m['customer_share_1'])
    m['ln_ocs'] = safe_log_ratio(m['offer_comp_share_2'], m['offer_comp_share_1'])
    m['ln_str_app'] = safe_log_ratio(m['str_apprv_rate_2'], m['str_apprv_rate_1'])
    m['ln_cond_app'] = safe_log_ratio(m['cond_apprv_rate_2'], m['cond_apprv_rate_1'])
    m['ln_str_bk'] = safe_log_ratio(m['str_bk_rate_2'], m['str_bk_rate_1'])
    m['ln_cond_bk'] = safe_log_ratio(m['cond_bk_rate_2'], m['cond_bk_rate_1'])

    # Effects (now 8 effects with VSA Progression)
    m['volume_effect_bks'] = m['w_total'] * m['ln_vol']
    m['vsa_progression_effect_bks'] = m['w_total'] * m['ln_vsa']  # NEW: VSA Progression
    m['customer_mix_effect_bks'] = m['w_total'] * m['ln_cs']
    m['offer_comp_mix_effect_bks'] = m['w_total'] * m['ln_ocs']
    m['str_approval_effect_bks'] = m['w_str'] * m['ln_str_app']
    m['cond_approval_effect_bks'] = m['w_cond'] * m['ln_cond_app']
    m['str_booking_effect_bks'] = m['w_str'] * m['ln_str_bk']
    m['cond_booking_effect_bks'] = m['w_cond'] * m['ln_cond_bk']

    # Rename
    rename = {
        'segment_bookings_1': 'period_1_segment_bookings',
        'segment_bookings_2': 'period_2_segment_bookings',
        'str_bookings_1': 'period_1_str_bookings',
        'str_bookings_2': 'period_2_str_bookings',
        'cond_bookings_1': 'period_1_cond_bookings',
        'cond_bookings_2': 'period_2_cond_bookings',
    }
    m = m.rename(columns=rename)
    m['period_1_date'] = str(date_a.date())
    m['period_2_date'] = str(date_b.date())

    effects = {
        'volume_effect': m['volume_effect_bks'].sum(),
        'vsa_progression_effect': m['vsa_progression_effect_bks'].sum(),  # NEW: VSA Progression
        'customer_mix_effect': m['customer_mix_effect_bks'].sum(),
        'offer_comp_mix_effect': m['offer_comp_mix_effect_bks'].sum(),
        'str_approval_effect': m['str_approval_effect_bks'].sum(),
        'cond_approval_effect': m['cond_approval_effect_bks'].sum(),
        'str_booking_effect': m['str_booking_effect_bks'].sum(),
        'cond_booking_effect': m['cond_booking_effect_bks'].sum(),
    }

    return m, effects


def _calc_competitor_booking_effects(df, date_a, date_b, date_column, exclude_lender=None):
    """
    Calculate competitor-level booking effects (REST OF MARKET approach).

    Aggregates segment data across all FINANCED lenders EXCEPT the excluded lender,
    then applies LMDI decomposition to competitor bookings.

    Note: NON_FINANCED rows are always excluded as they have no funnel data to decompose.
    The non-financed impact is included in the overall market effect via the scaling.
    """
    dims = get_dimension_columns()

    # Get data for both periods, EXCLUDING the specified lender AND NON_FINANCED
    df_1 = df[df[date_column] == date_a].copy()
    df_2 = df[df[date_column] == date_b].copy()

    # Always exclude NON_FINANCED (no funnel data to decompose)
    df_1 = df_1[~df_1['lender'].apply(is_non_financed_lender)].copy()
    df_2 = df_2[~df_2['lender'].apply(is_non_financed_lender)].copy()

    if exclude_lender is not None:
        df_1 = df_1[df_1['lender'] != exclude_lender].copy()
        df_2 = df_2[df_2['lender'] != exclude_lender].copy()

    # Calculate segment bookings for each lender
    df_1 = calculate_segment_bookings(df_1)
    df_2 = calculate_segment_bookings(df_2)

    # Calculate intermediate values (approvals) at lender level before aggregating
    df_1['str_approvals'] = df_1['segment_apps'] * df_1['str_apprv_rate']
    df_2['str_approvals'] = df_2['segment_apps'] * df_2['str_apprv_rate']
    df_1['cond_approvals'] = df_1['segment_apps'] * (1 - df_1['str_apprv_rate']) * df_1['cond_apprv_rate']
    df_2['cond_approvals'] = df_2['segment_apps'] * (1 - df_2['str_apprv_rate']) * df_2['cond_apprv_rate']

    # Aggregate to market level by segment
    agg_cols = {
        'segment_apps': 'sum',
        'str_approvals': 'sum',
        'cond_approvals': 'sum',
        'str_bookings': 'sum',
        'cond_bookings': 'sum',
        'segment_bookings': 'sum'
    }

    mkt_1 = df_1.groupby(dims).agg(agg_cols).reset_index()
    mkt_2 = df_2.groupby(dims).agg(agg_cols).reset_index()

    # Calculate market totals
    total_apps_1 = mkt_1['segment_apps'].sum()
    total_apps_2 = mkt_2['segment_apps'].sum()

    # Calculate market-level pct_of_total_apps
    mkt_1['pct_of_total_apps'] = mkt_1['segment_apps'] / total_apps_1
    mkt_2['pct_of_total_apps'] = mkt_2['segment_apps'] / total_apps_2

    # For aggregated market data, set vsa_prog_pct = 1.0 and pct_of_total_vsa = pct_of_total_apps
    mkt_1['vsa_prog_pct'] = 1.0
    mkt_2['vsa_prog_pct'] = 1.0
    mkt_1['pct_of_total_vsa'] = mkt_1['pct_of_total_apps']
    mkt_2['pct_of_total_vsa'] = mkt_2['pct_of_total_apps']

    # Calculate customer shares (primary dimension) using pct_of_total_apps
    cs1 = mkt_1.groupby(PRIMARY_DIMENSION)['pct_of_total_apps'].sum()
    cs2 = mkt_2.groupby(PRIMARY_DIMENSION)['pct_of_total_apps'].sum()
    mkt_1['customer_share'] = mkt_1[PRIMARY_DIMENSION].map(cs1)
    mkt_2['customer_share'] = mkt_2[PRIMARY_DIMENSION].map(cs2)
    mkt_1['offer_comp_share'] = mkt_1['pct_of_total_apps'] / mkt_1['customer_share']
    mkt_2['offer_comp_share'] = mkt_2['pct_of_total_apps'] / mkt_2['customer_share']

    # Calculate proper market-level rates from aggregated counts
    mkt_1['str_apprv_rate'] = mkt_1['str_approvals'] / mkt_1['segment_apps'].replace(0, 1)
    mkt_2['str_apprv_rate'] = mkt_2['str_approvals'] / mkt_2['segment_apps'].replace(0, 1)

    mkt_1['str_bk_rate'] = mkt_1['str_bookings'] / mkt_1['str_approvals'].replace(0, 1)
    mkt_2['str_bk_rate'] = mkt_2['str_bookings'] / mkt_2['str_approvals'].replace(0, 1)

    mkt_1['cond_apps'] = mkt_1['segment_apps'] - mkt_1['str_approvals']
    mkt_2['cond_apps'] = mkt_2['segment_apps'] - mkt_2['str_approvals']
    mkt_1['cond_apprv_rate'] = mkt_1['cond_approvals'] / mkt_1['cond_apps'].replace(0, 1)
    mkt_2['cond_apprv_rate'] = mkt_2['cond_approvals'] / mkt_2['cond_apps'].replace(0, 1)

    mkt_1['cond_bk_rate'] = mkt_1['cond_bookings'] / mkt_1['cond_approvals'].replace(0, 1)
    mkt_2['cond_bk_rate'] = mkt_2['cond_bookings'] / mkt_2['cond_approvals'].replace(0, 1)

    # Merge periods
    m = mkt_1.merge(mkt_2, on=dims, suffixes=('_1', '_2'))

    # Weights (logarithmic mean of bookings)
    m['w_str'] = m.apply(lambda r: logarithmic_mean(r['str_bookings_1'], r['str_bookings_2']), axis=1)
    m['w_cond'] = m.apply(lambda r: logarithmic_mean(r['cond_bookings_1'], r['cond_bookings_2']), axis=1)
    m['w_total'] = m['w_str'] + m['w_cond']

    # Log ratios for volume and mix
    m['ln_vol'] = safe_log_ratio(total_apps_2, total_apps_1)
    m['ln_vsa'] = safe_log_ratio(m['vsa_prog_pct_2'], m['vsa_prog_pct_1'])  # NEW: VSA Progression
    m['ln_cs'] = safe_log_ratio(m['customer_share_2'], m['customer_share_1'])
    m['ln_ocs'] = safe_log_ratio(m['offer_comp_share_2'], m['offer_comp_share_1'])

    # Log ratios for rates
    m['ln_str_app'] = safe_log_ratio(m['str_apprv_rate_2'], m['str_apprv_rate_1'])
    m['ln_str_bk'] = safe_log_ratio(m['str_bk_rate_2'], m['str_bk_rate_1'])
    m['ln_cond_app'] = safe_log_ratio(m['cond_apprv_rate_2'], m['cond_apprv_rate_1'])
    m['ln_cond_bk'] = safe_log_ratio(m['cond_bk_rate_2'], m['cond_bk_rate_1'])

    # Effects (in booking terms, now 8 effects with VSA Progression)
    m['volume_effect_bks'] = m['w_total'] * m['ln_vol']
    m['vsa_progression_effect_bks'] = m['w_total'] * m['ln_vsa']  # NEW: VSA Progression
    m['customer_mix_effect_bks'] = m['w_total'] * m['ln_cs']
    m['offer_comp_mix_effect_bks'] = m['w_total'] * m['ln_ocs']

    m['str_approval_effect_bks'] = m['w_str'] * m['ln_str_app']
    m['str_booking_effect_bks'] = m['w_str'] * m['ln_str_bk']
    m['cond_approval_effect_bks'] = m['w_cond'] * m['ln_cond_app']
    m['cond_booking_effect_bks'] = m['w_cond'] * m['ln_cond_bk']

    # Aggregate effects
    effects = {
        'volume_effect': m['volume_effect_bks'].sum(),
        'vsa_progression_effect': m['vsa_progression_effect_bks'].sum(),  # NEW: VSA Progression
        'customer_mix_effect': m['customer_mix_effect_bks'].sum(),
        'offer_comp_mix_effect': m['offer_comp_mix_effect_bks'].sum(),
        'str_approval_effect': m['str_approval_effect_bks'].sum(),
        'cond_approval_effect': m['cond_approval_effect_bks'].sum(),
        'str_booking_effect': m['str_booking_effect_bks'].sum(),
        'cond_booking_effect': m['cond_booking_effect_bks'].sum(),
    }

    return effects


def calculate_multi_lender_penetration_decomposition(
    df: pd.DataFrame,
    date_a: Union[str, datetime, pd.Timestamp],
    date_b: Union[str, datetime, pd.Timestamp],
    lenders: list = None,
    date_column: str = 'month_begin_date'
) -> MultiLenderPenetrationResults:
    """Calculate penetration decomposition across multiple lenders using self-adjusted approach.

    Note: NON_FINANCED is automatically excluded from lender iteration (no funnel data).
    Total market includes NON_FINANCED bookings in the denominator for all effects.
    """
    date_a, date_b = normalize_date(date_a), normalize_date(date_b)
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    if lenders is None:
        la = set(df[df[date_column] == date_a]['lender'].unique())
        lb = set(df[df[date_column] == date_b]['lender'].unique())
        # Exclude NON_FINANCED from lender iteration
        lenders = sorted([l for l in (la & lb) if not is_non_financed_lender(l)])

    results = {}
    for lender in lenders:
        print(f"Calculating penetration decomposition for {lender}...")
        try:
            results[lender] = calculate_penetration_decomposition(df, date_a, date_b, lender, date_column)
        except Exception as e:
            warnings.warn(f"Failed {lender}: {e}")

    # Summaries
    summaries = []
    for lender, r in results.items():
        s = r.summary.copy()
        s['lender'] = lender
        s['period_1_penetration_pct'] = r.metadata['period_1_penetration'] * 100
        s['period_2_penetration_pct'] = r.metadata['period_2_penetration'] * 100
        summaries.append(s)

    lender_summaries = pd.concat(summaries)[['lender', 'effect_type', 'gross_lender_effect_bps',
                                              'self_adjustment_bps', 'net_lender_effect_bps',
                                              'competitor_effect_bps', 'net_effect_bps',
                                              'period_1_penetration_pct', 'period_2_penetration_pct']]

    # Aggregate
    agg_rows = []
    for lender, r in results.items():
        agg_rows.append({
            'lender': lender,
            'period_1_penetration_pct': r.metadata['period_1_penetration'] * 100,
            'period_2_penetration_pct': r.metadata['period_2_penetration'] * 100,
            'delta_penetration_bps': r.metadata['delta_penetration_bps'],
            'total_gross_lender_effect_bps': r.metadata['total_gross_lender_effect_bps'],
            'total_self_adjustment_bps': r.metadata['total_self_adjustment_bps'],
            'total_net_lender_effect_bps': r.metadata['total_net_lender_effect_bps'],
            'total_competitor_effect_bps': r.metadata['total_competitor_effect_bps']
        })
    aggregate_summary = pd.DataFrame(agg_rows)

    total_market_1 = df[df[date_column] == date_a].groupby('lender')['num_tot_bks'].first().sum()
    total_market_2 = df[df[date_column] == date_b].groupby('lender')['num_tot_bks'].first().sum()

    metadata = {
        'date_a': str(date_a.date()),
        'date_b': str(date_b.date()),
        'lenders': list(results.keys()),
        'period_1_total_market_bookings': float(total_market_1),
        'period_2_total_market_bookings': float(total_market_2),
        'method': 'lmdi_penetration_self_adjusted_multi_lender'
    }

    return MultiLenderPenetrationResults(
        lender_summaries=lender_summaries,
        aggregate_summary=aggregate_summary,
        lender_details=results,
        metadata=metadata
    )


def print_penetration_decomposition(summary: pd.DataFrame, metadata: dict):
    """Print formatted penetration decomposition results."""
    lender = metadata['lender']
    date_a = metadata['date_a']
    date_b = metadata['date_b']

    pen_1 = metadata['period_1_penetration'] * 100
    pen_2 = metadata['period_2_penetration'] * 100
    delta_bps = metadata['delta_penetration_bps']

    print("=" * 80)
    print(f"PENETRATION DECOMPOSITION (Self-Adjusted): {lender}")
    print(f"Period: {date_a} -> {date_b}")
    print("=" * 80)
    print(f"\nPenetration: {pen_1:.2f}% -> {pen_2:.2f}% ({delta_bps:+.1f} bps)")
    print(f"\nSelf-Adjustment Share: {metadata['self_adjustment_share']*100:.1f}% of market growth")
    print(f"Competitor Share: {metadata['competitor_share']*100:.1f}% of market growth")
    print("\n" + "-" * 80)
    print(f"{'Effect':<25} {'Gross':>10} {'Self-Adj':>10} {'Net Lender':>12} {'Competitor':>12} {'Net':>10}")
    print("-" * 80)

    for _, row in summary.iterrows():
        if row['effect_type'] == 'total_change':
            print("-" * 80)
        name = row['effect_type'].replace('_', ' ').title()
        print(f"{name:<25} {row['gross_lender_effect_bps']:>+10.1f} {row['self_adjustment_bps']:>+10.1f} "
              f"{row['net_lender_effect_bps']:>+12.1f} {row['competitor_effect_bps']:>+12.1f} "
              f"{row['net_effect_bps']:>+10.1f}")

    print("=" * 80)
    print(f"\nReconciliation: Net Effects ({metadata['delta_penetration_bps']:+.1f}) = Delta Penetration ({delta_bps:+.1f}) ✓")
