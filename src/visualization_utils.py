"""
Shared utilities and constants for visualization modules.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Union, Optional, List

try:
    from .utils import format_date_label, format_number, format_percentage
    from .dimension_config import (
        apply_dimension_order, get_dimension_columns,
        get_finance_channel_column, get_lender_tier, LENDER_TIERS
    )
except ImportError:
    from utils import format_date_label, format_number, format_percentage
    from dimension_config import (
        apply_dimension_order, get_dimension_columns,
        get_finance_channel_column, get_lender_tier, LENDER_TIERS
    )


# Color palette
COLOR_POSITIVE = '#2ecc71'  # Green
COLOR_NEGATIVE = '#e74c3c'  # Red
COLOR_TOTAL = '#95a5a6'     # Gray
COLOR_CONNECTOR = '#34495e'  # Dark gray
COLOR_MARKET = '#708090'    # Stone blue/slate gray for market effect

# Finance channel colors
CHANNEL_COLORS = {
    'FF': '#00529F',      # Blue
    'NON_FF': '#FFD520'   # Yellow/Gold
}

# Tier colors (shades of blue)
TIER_COLORS = {
    'T1': '#003366',  # Dark Navy Blue
    'T2': '#2563eb',  # Medium Blue
    'T3': '#7dd3fc'   # Light Sky Blue
}


def format_dimension_name(dim_name: str) -> str:
    """Format dimension column name for display."""
    return dim_name.replace('_', ' ').title()


def format_effect_labels(labels: List[str]) -> List[str]:
    """Format effect labels for display."""
    label_map = {
        'Start': 'Start', 'End': 'End',
        'volume_effect': 'Volume',
        'vsa_progression_effect': 'VSA Prog',  # NEW: VSA Progression
        'customer_mix_effect': 'Cust Mix',
        'offer_comp_mix_effect': 'Comp Offer Mix',
        'volume_customer_mix_effect': 'Vol + Cust Mix',
        'str_approval_effect': 'Str Apprv', 'cond_approval_effect': 'Cond Apprv',
        'str_booking_effect': 'Str Book', 'cond_booking_effect': 'Cond Book',
        'market_effect': 'Market'
    }
    return [label_map.get(l, l) for l in labels]


def detect_dimension_columns(segment_detail: pd.DataFrame) -> List[str]:
    """Auto-detect dimension columns from segment_detail DataFrame."""
    known_cols = {
        'period_1_date', 'period_2_date', 'period_1_total_apps', 'period_2_total_apps',
        'period_1_pct_of_total', 'period_2_pct_of_total', 'period_1_segment_apps',
        'period_2_segment_apps', 'period_1_str_apprv_rate', 'period_2_str_apprv_rate',
        'period_1_str_bk_rate', 'period_2_str_bk_rate', 'period_1_cond_apprv_rate',
        'period_2_cond_apprv_rate', 'period_1_cond_bk_rate', 'period_2_cond_bk_rate',
        'period_1_segment_bookings', 'period_2_segment_bookings', 'period_1_str_bookings',
        'period_2_str_bookings', 'period_1_cond_bookings', 'period_2_cond_bookings',
        # Split mix columns (for 2D split mix calculator)
        'period_1_customer_share', 'period_2_customer_share',
        'period_1_offer_comp_share', 'period_2_offer_comp_share',
        'delta_customer_share', 'delta_offer_comp_share',
        'delta_total_apps', 'delta_pct_of_total', 'delta_str_apprv_rate',
        'delta_str_bk_rate', 'delta_cond_apprv_rate', 'delta_cond_bk_rate',
        'delta_segment_bookings', 'volume_effect', 'mix_effect',
        # Split mix effects
        'customer_mix_effect', 'offer_comp_mix_effect',
        'str_approval_effect', 'cond_approval_effect',
        'str_booking_effect', 'cond_booking_effect',
        'total_effect', 'interaction_effect',
        # Penetration-specific columns
        'period_1_segment_penetration', 'period_2_segment_penetration',
        'volume_effect_bps', 'customer_mix_effect_bps', 'offer_comp_mix_effect_bps',
        'str_approval_effect_bps', 'cond_approval_effect_bps',
        'str_booking_effect_bps', 'cond_booking_effect_bps',
        'total_lender_effect_bps'
    }
    return [c for c in segment_detail.columns if c not in known_cols]


def aggregate_by_dimension(
    segment_detail: pd.DataFrame,
    dimension: str,
    exclude_volume: bool = False,
    combine_volume_mix: bool = True
) -> pd.DataFrame:
    """Aggregate effects by dimension values for dimensional waterfall charts."""
    effect_column_map = {
        'volume_effect': 'volume_effect',
        'vsa_progression_effect': 'vsa_progression_effect',  # NEW: VSA Progression
        'mix_effect': 'mix_effect',
        'customer_mix_effect': 'customer_mix_effect',
        'offer_comp_mix_effect': 'offer_comp_mix_effect',
        'str_approval_effect': 'str_approval_effect',
        'cond_approval_effect': 'cond_approval_effect',
        'str_booking_effect': 'str_booking_effect',
        'cond_booking_effect': 'cond_booking_effect'
    }

    available_effects = {k: v for k, v in effect_column_map.items() if k in segment_detail.columns}
    effect_cols = list(available_effects.keys())

    # Combine volume with customer_mix for the combined effect
    if combine_volume_mix:
        volume_cols = ['volume_effect', 'customer_mix_effect']
        other_cols = [c for c in effect_cols if c not in volume_cols]
    else:
        volume_cols = []
        other_cols = effect_cols

    if exclude_volume and 'volume_effect' in effect_cols:
        volume_total = segment_detail['volume_effect'].sum()
        effect_cols = [c for c in effect_cols if c != 'volume_effect']
        other_cols = [c for c in other_cols if c != 'volume_effect']
    else:
        volume_total = None

    results = []

    # Process combined volume+customer_mix effect
    if combine_volume_mix and volume_cols:
        vol_cols_present = [c for c in volume_cols if c in segment_detail.columns]
        if vol_cols_present:
            vol_data = segment_detail.groupby(dimension)[vol_cols_present].sum()
            combined_effect = vol_data.sum(axis=1)

            for dim_val in combined_effect.index:
                results.append({
                    dimension: dim_val,
                    'effect_type': 'volume_customer_mix_effect',
                    'impact': combined_effect[dim_val]
                })

    # Process other effects
    for effect_col in other_cols:
        if effect_col in segment_detail.columns:
            dim_effects = segment_detail.groupby(dimension)[effect_col].sum()

            for dim_val, impact in dim_effects.items():
                agg = {
                    dimension: dim_val,
                    'effect_type': effect_col,
                    'impact': impact
                }
                results.append(agg)

    aggregated = pd.concat([pd.DataFrame([r]) for r in results], ignore_index=True)

    if exclude_volume and not combine_volume_mix:
        return aggregated, volume_total
    else:
        return aggregated


def aggregate_penetration_by_dimension(
    segment_detail: pd.DataFrame,
    dimension: str
) -> pd.DataFrame:
    """Aggregate penetration effects by dimension values (in bps)."""
    effect_cols_bps = [
        'volume_effect_bps', 'vsa_progression_effect_bps', 'customer_mix_effect_bps', 'offer_comp_mix_effect_bps',
        'str_approval_effect_bps', 'cond_approval_effect_bps',
        'str_booking_effect_bps', 'cond_booking_effect_bps'
    ]

    # Filter to existing columns
    available_cols = [c for c in effect_cols_bps if c in segment_detail.columns]

    results = []

    # Combine volume + customer_mix for dimensional charts
    vol_mix_cols = ['volume_effect_bps', 'customer_mix_effect_bps']
    vol_mix_present = [c for c in vol_mix_cols if c in available_cols]

    if vol_mix_present:
        vol_mix_data = segment_detail.groupby(dimension)[vol_mix_present].sum()
        combined = vol_mix_data.sum(axis=1)

        for dim_val in combined.index:
            results.append({
                dimension: dim_val,
                'effect_type': 'volume_customer_mix_effect',
                'impact': combined[dim_val]
            })

    # Other effects (excluding volume and customer_mix which are combined)
    other_cols = [c for c in available_cols if c not in vol_mix_cols]

    for col in other_cols:
        dim_effects = segment_detail.groupby(dimension)[col].sum()

        # Map column name to effect type (remove _bps suffix)
        effect_type = col.replace('_bps', '')

        for dim_val, impact in dim_effects.items():
            results.append({
                dimension: dim_val,
                'effect_type': effect_type,
                'impact': impact
            })

    return pd.DataFrame(results)


def save_figure(fig: plt.Figure, output_path: Union[str, Path], description: str = "Figure") -> None:
    """Save figure to file with directory creation."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"{description} saved to: {output_path}")


def aggregate_by_finance_channel(
    channel_summaries: pd.DataFrame,
    combine_volume_mix: bool = True
) -> pd.DataFrame:
    """
    Transform channel_summaries for stacked visualization.

    Args:
        channel_summaries: DataFrame with finance_channel, effect_type, booking_impact
        combine_volume_mix: If True, combine volume + customer_mix into single effect

    Returns:
        DataFrame with columns: effect_type, finance_channel, impact
    """
    df = channel_summaries.copy()

    if combine_volume_mix:
        # Combine volume and customer_mix effects (vsa_progression_effect is separate)
        vol_mix = df[df['effect_type'].isin(['volume_effect', 'customer_mix_effect'])]
        combined = vol_mix.groupby('finance_channel')['booking_impact'].sum().reset_index()
        combined['effect_type'] = 'volume_customer_mix_effect'
        combined = combined.rename(columns={'booking_impact': 'impact'})

        # Get other effects (including vsa_progression_effect)
        other = df[~df['effect_type'].isin(['volume_effect', 'customer_mix_effect', 'total_change'])]
        other = other.rename(columns={'booking_impact': 'impact'})

        result = pd.concat([combined, other[['effect_type', 'finance_channel', 'impact']]])
    else:
        result = df[df['effect_type'] != 'total_change'].rename(columns={'booking_impact': 'impact'})

    return result


def aggregate_by_tier(
    tier_summary: pd.DataFrame,
    combine_volume_mix: bool = True
) -> pd.DataFrame:
    """
    Transform tier_summary for stacked visualization.

    Args:
        tier_summary: DataFrame with lender_tier, effect_type, booking_impact
        combine_volume_mix: If True, combine volume + customer_mix into single effect

    Returns:
        DataFrame with columns: effect_type, lender_tier, impact
    """
    df = tier_summary.copy()

    if combine_volume_mix:
        # Combine volume and customer_mix effects (vsa_progression_effect is separate)
        vol_mix = df[df['effect_type'].isin(['volume_effect', 'customer_mix_effect'])]
        combined = vol_mix.groupby('lender_tier')['booking_impact'].sum().reset_index()
        combined['effect_type'] = 'volume_customer_mix_effect'
        combined = combined.rename(columns={'booking_impact': 'impact'})

        # Get other effects (including vsa_progression_effect)
        other = df[~df['effect_type'].isin(['volume_effect', 'customer_mix_effect', 'total_change'])]
        other = other.rename(columns={'booking_impact': 'impact'})

        result = pd.concat([combined, other[['effect_type', 'lender_tier', 'impact']]])
    else:
        result = df[df['effect_type'] != 'total_change'].rename(columns={'booking_impact': 'impact'})

    return result


def format_channel_breakdown(metadata: dict) -> str:
    """
    Format channel breakdown for chart titles.

    Returns string like: "(FF: +350 / Non-FF: +150)"
    """
    channel_totals = metadata.get('channel_totals', {})
    if not channel_totals:
        return ""

    parts = []
    for channel in ['FF', 'NON_FF']:
        if channel in channel_totals:
            delta = channel_totals[channel].get('delta_bookings', 0)
            sign = '+' if delta >= 0 else ''
            label = 'Non-FF' if channel == 'NON_FF' else 'FF'
            parts.append(f"{label}: {sign}{delta:,.0f}")

    return f"({' / '.join(parts)})" if parts else ""


def format_tier_breakdown(metadata: dict) -> str:
    """
    Format tier breakdown for chart titles.

    Returns string like: "(T1: +100 / T2: +200 / T3: +50)"
    """
    tier_totals = metadata.get('tier_totals', {})
    if not tier_totals:
        return ""

    parts = []
    for tier in ['T1', 'T2', 'T3']:
        if tier in tier_totals:
            delta = tier_totals[tier].get('delta_bookings', 0)
            sign = '+' if delta >= 0 else ''
            parts.append(f"{tier}: {sign}{delta:,.0f}")

    return f"({' / '.join(parts)})" if parts else ""
