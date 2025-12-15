"""
Multi-lender comparison visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Optional

try:
    from .visualization_utils import (
        COLOR_POSITIVE, COLOR_NEGATIVE, COLOR_TOTAL, COLOR_CONNECTOR,
        CHANNEL_COLORS, TIER_COLORS,
        format_effect_labels, save_figure, aggregate_by_tier,
        format_channel_breakdown, format_tier_breakdown
    )
    from .visualization_summary import _create_aggregate_waterfall
    from .utils import format_number
    from .dimension_config import apply_dimension_order, get_finance_channel_values
except ImportError:
    from visualization_utils import (
        COLOR_POSITIVE, COLOR_NEGATIVE, COLOR_TOTAL, COLOR_CONNECTOR,
        CHANNEL_COLORS, TIER_COLORS,
        format_effect_labels, save_figure, aggregate_by_tier,
        format_channel_breakdown, format_tier_breakdown
    )
    from visualization_summary import _create_aggregate_waterfall
    from utils import format_number
    from dimension_config import apply_dimension_order, get_finance_channel_values


# Lender color palette
LENDER_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']


def create_lender_aggregate_waterfall(
    lender_summaries: pd.DataFrame,
    aggregate_summary: pd.DataFrame,
    date_a: str,
    date_b: str,
    output_path: str = None,
    title: str = None,
    period_1_bks: float = None,
    period_2_bks: float = None
) -> plt.Figure:
    """Create aggregate waterfall chart with lender-level breakdown."""
    # Exclude total_change and interaction_effect
    effects = aggregate_summary[
        (aggregate_summary['effect_type'] != 'total_change') &
        (aggregate_summary['effect_type'] != 'interaction_effect')
    ].copy()

    lenders = sorted(lender_summaries['lender'].unique())

    # Calculate period values if not provided
    if period_1_bks is None or period_2_bks is None:
        total_change = aggregate_summary[aggregate_summary['effect_type'] == 'total_change']['booking_impact'].iloc[0]
        period_1_bks_calc = total_change - effects['booking_impact'].sum()
        period_2_bks_calc = period_1_bks_calc + effects['booking_impact'].sum()

        if period_1_bks is None:
            period_1_bks = period_1_bks_calc
        if period_2_bks is None:
            period_2_bks = period_2_bks_calc

    fig, ax = plt.subplots(figsize=(16, 12))

    fig.suptitle(f'Multi-Lender Booking Decomposition: {date_a} → {date_b}',
                 fontsize=16, fontweight='bold', y=0.98)

    # Lender colors
    lender_colors = {}
    for i, lender in enumerate(lenders):
        lender_colors[lender] = LENDER_COLORS[i % len(LENDER_COLORS)]

    labels = ['Start'] + effects['effect_type'].tolist() + ['End']

    # Calculate cumulative positions
    positions = []
    current = period_1_bks
    positions.append(0)

    for val in effects['booking_impact']:
        positions.append(current)
        current += val

    positions.append(0)

    # Calculate y-axis limits
    all_values = [period_1_bks, period_2_bks]
    current = period_1_bks
    for val in effects['booking_impact']:
        all_values.append(current)
        current += val
        all_values.append(current)

    data_min = min(all_values)
    data_max = max(all_values)
    data_range = data_max - data_min

    y_min = data_min - (data_range * 0.50)
    y_max = data_max + (data_range * 0.20)
    y_range = y_max - y_min
    label_offset = y_range * 0.03

    x_pos = np.arange(len(labels))

    # Plot Start bar
    ax.bar(0, period_1_bks, color=COLOR_TOTAL, edgecolor='black', linewidth=1.5, width=0.78)
    ax.text(0, period_1_bks/2 + y_min/2, format_number(period_1_bks, 0),
            ha='center', va='center', fontsize=10, fontweight='bold')

    cumulative = period_1_bks

    # Plot each effect as stacked bars
    for i, (_, row) in enumerate(effects.iterrows()):
        effect_type = row['effect_type']
        total_impact = row['booking_impact']
        x_idx = i + 1

        lender_data = lender_summaries[lender_summaries['effect_type'] == effect_type].copy()

        if abs(total_impact) < 0.01:
            prev_y = positions[x_idx]
            ax.plot([x_idx-1+0.3, x_idx-0.3], [prev_y, prev_y],
                   color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)
            ax.plot(x_idx, positions[x_idx], marker='o', markersize=6, color=COLOR_CONNECTOR,
                   markeredgecolor='black', markeredgewidth=1.5, zorder=3)
            ax.text(x_idx, positions[x_idx] - label_offset, '+0',
                   ha='center', va='top', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='black', linewidth=1.5, alpha=0.95))
            cumulative += total_impact
            continue

        lender_data = lender_data.sort_values('booking_impact', ascending=True)
        bottom = cumulative

        for _, lender_row in lender_data.iterrows():
            lender = lender_row['lender']
            lender_impact = lender_row['booking_impact']

            if abs(lender_impact) < 0.01:
                continue

            color = lender_colors[lender]
            height = lender_impact

            rect = Rectangle((x_idx - 0.3, bottom), 0.6, height,
                           facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.85)
            ax.add_patch(rect)
            bottom = bottom + lender_impact

        label_y_pos = min(cumulative, cumulative + total_impact)
        sign = '+' if total_impact >= 0 else ''
        ax.text(x_idx, label_y_pos - label_offset, f'{sign}{format_number(total_impact, 0)}',
               ha='center', va='top', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='black', linewidth=1.5, alpha=0.95))

        prev_y = cumulative
        ax.plot([x_idx-1+0.3, x_idx-0.3], [prev_y, prev_y],
               color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)

        cumulative += total_impact

    # Plot End bar
    end_idx = len(labels) - 1
    ax.bar(end_idx, period_2_bks, color=COLOR_TOTAL, edgecolor='black', linewidth=1.5, width=0.78)
    ax.text(end_idx, period_2_bks/2 + y_min/2, format_number(period_2_bks, 0),
            ha='center', va='center', fontsize=10, fontweight='bold')

    ax.plot([end_idx-1+0.3, end_idx-0.3], [cumulative, cumulative],
           color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(format_effect_labels(labels), rotation=45, ha='right')
    ax.set_ylabel('Bookings', fontsize=11, fontweight='bold')
    subtitle = title if title is not None else 'By Lender'
    ax.set_title(subtitle, fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.8)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=lender_colors[lender], edgecolor='black',
                      label=lender, alpha=0.85)
        for lender in lenders
    ]
    n_legend_items = len(legend_elements)
    legend_ncol = (n_legend_items + 1) // 2 if n_legend_items >= 6 else n_legend_items

    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
             ncol=legend_ncol, frameon=True, fancybox=True, shadow=True,
             fontsize=9, title='Lenders', title_fontsize=10)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path, "Lender aggregate waterfall")

    return fig


def create_lender_drilldown(
    lender_summaries: pd.DataFrame,
    date_a: str,
    date_b: str,
    output_path: str = None
) -> plt.Figure:
    """Create lender drilldown showing each effect broken down by lender."""
    effects_to_plot = [
        'volume_effect',
        'customer_mix_effect',
        'offer_comp_mix_effect',
        'str_approval_effect',
        'cond_approval_effect',
        'str_booking_effect',
        'cond_booking_effect'
    ]

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    axes = axes.flatten()

    fig.suptitle(f'Multi-Lender Booking Impact Decomposition by Effect\n{date_a} → {date_b}',
                 fontsize=14, fontweight='bold', y=0.98)

    for idx, effect_type in enumerate(effects_to_plot):
        ax = axes[idx]

        effect_data = lender_summaries[lender_summaries['effect_type'] == effect_type].copy()
        lenders = effect_data['lender'].unique().tolist()
        ordered_lenders = apply_dimension_order('lender', lenders)

        effect_data['lender'] = pd.Categorical(effect_data['lender'], categories=ordered_lenders, ordered=True)
        effect_data = effect_data.sort_values('lender')

        positive = effect_data[effect_data['booking_impact'] > 0]
        negative = effect_data[effect_data['booking_impact'] < 0]

        if not negative.empty:
            bars_neg = ax.barh(negative['lender'], negative['booking_impact'],
                               color='#EF5350', edgecolor='black', linewidth=0.5)
            for bar in bars_neg:
                width = bar.get_width()
                if width != 0:
                    ax.text(width, bar.get_y() + bar.get_height() / 2,
                            f'{width:+.0f}', ha='right' if width < 0 else 'left',
                            va='center', fontweight='bold', fontsize=9)

        if not positive.empty:
            bars_pos = ax.barh(positive['lender'], positive['booking_impact'],
                               color='#66BB6A', edgecolor='black', linewidth=0.5)
            for bar in bars_pos:
                width = bar.get_width()
                if width != 0:
                    ax.text(width, bar.get_y() + bar.get_height() / 2,
                            f'{width:+.0f}', ha='right' if width < 0 else 'left',
                            va='center', fontweight='bold', fontsize=9)

        total_impact = effect_data['booking_impact'].sum()
        effect_label = format_effect_labels([effect_type])[0]
        ax.set_title(f'{effect_label} (Total: {total_impact:+,.0f})',
                     fontsize=11, fontweight='bold', pad=10)

        ax.set_xlabel('Booking Impact', fontsize=10, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        if not effect_data.empty:
            x_max = max(abs(effect_data['booking_impact'].min()), abs(effect_data['booking_impact'].max()))
            x_max = x_max * 1.2
            ax.set_xlim(-x_max, x_max)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path, "Lender drilldown")

    return fig


def print_lender_breakdowns(lender_summaries: pd.DataFrame) -> None:
    """Print tabular breakdown of effects by lender."""
    effects = [
        'volume_effect',
        'customer_mix_effect',
        'offer_comp_mix_effect',
        'str_approval_effect',
        'cond_approval_effect',
        'str_booking_effect',
        'cond_booking_effect'
    ]

    lenders = sorted(lender_summaries['lender'].unique())

    print("=" * 80)
    print("LENDER-LEVEL BREAKDOWN")
    print("=" * 80)

    pivot = lender_summaries[lender_summaries['effect_type'].isin(effects)].pivot(
        index='effect_type',
        columns='lender',
        values='booking_impact'
    )

    pivot['TOTAL'] = pivot.sum(axis=1)
    pivot.index = format_effect_labels(pivot.index.tolist())

    print("\n" + pivot.to_string())

    print("\n" + "=" * 80)
    total_changes = lender_summaries[lender_summaries['effect_type'] == 'total_change']
    print("\nTOTAL CHANGE BY LENDER:")
    print("-" * 40)
    for _, row in total_changes.iterrows():
        print(f"  {row['lender']}: {row['booking_impact']:+,.0f}")

    grand_total = total_changes['booking_impact'].sum()
    print(f"\n  GRAND TOTAL: {grand_total:+,.0f}")
    print("=" * 80)


def create_lender_waterfall_grid(
    lender_summaries: pd.DataFrame,
    aggregate_summary: pd.DataFrame,
    metadata: dict,
    output_path: str = None
) -> plt.Figure:
    """Create a 2-panel waterfall grid for multi-lender analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 7.5))

    date_a = metadata['date_a']
    date_b = metadata['date_b']
    period_1_bks = metadata['aggregate_period_1_bookings']
    period_2_bks = metadata['aggregate_period_2_bookings']

    lenders = metadata['lenders']
    lenders = apply_dimension_order('lender', lenders)

    fig.suptitle(f'Multi-Lender Booking Decomposition: {date_a} → {date_b}',
                 fontsize=16, fontweight='bold', y=0.96)

    effects = aggregate_summary[
        (aggregate_summary['effect_type'] != 'total_change') &
        (aggregate_summary['effect_type'] != 'interaction_effect')
    ].copy()

    # Left panel: Overall Aggregate
    _create_aggregate_waterfall(
        ax=axes[0],
        summary=aggregate_summary,
        period_1_bks=period_1_bks,
        period_2_bks=period_2_bks,
        title='Overall Aggregate'
    )

    # Right panel: Lender Breakdown
    ax = axes[1]

    lender_colors = {}
    for i, lender in enumerate(lenders):
        lender_colors[lender] = LENDER_COLORS[i % len(LENDER_COLORS)]

    labels = ['Start'] + effects['effect_type'].tolist() + ['End']

    positions = []
    current = period_1_bks
    positions.append(0)

    for val in effects['booking_impact']:
        positions.append(current)
        current += val

    positions.append(0)

    all_values = [period_1_bks, period_2_bks]
    current = period_1_bks
    for val in effects['booking_impact']:
        all_values.append(current)
        current += val
        all_values.append(current)

    data_min = min(all_values)
    data_max = max(all_values)
    data_range = data_max - data_min

    y_min = data_min - (data_range * 0.50)
    y_max = data_max + (data_range * 0.20)
    y_range = y_max - y_min
    label_offset = y_range * 0.03

    x_pos = np.arange(len(labels))

    # Plot Start bar
    ax.bar(0, period_1_bks, color=COLOR_TOTAL, edgecolor='black', linewidth=1.5, width=0.78)
    ax.text(0, period_1_bks/2 + y_min/2, format_number(period_1_bks, 0),
            ha='center', va='center', fontsize=10, fontweight='bold')

    cumulative = period_1_bks

    # Plot each effect
    for i, (_, row) in enumerate(effects.iterrows()):
        effect_type = row['effect_type']
        total_impact = row['booking_impact']
        x_idx = i + 1

        lender_data = lender_summaries[lender_summaries['effect_type'] == effect_type].copy()

        if abs(total_impact) < 0.01:
            prev_y = positions[x_idx]
            ax.plot([x_idx-1+0.3, x_idx-0.3], [prev_y, prev_y],
                   color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)
            ax.plot(x_idx, positions[x_idx], marker='o', markersize=6, color=COLOR_CONNECTOR,
                   markeredgecolor='black', markeredgewidth=1.5, zorder=3)
            ax.text(x_idx, positions[x_idx] - label_offset, '+0',
                   ha='center', va='top', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='black', linewidth=1.5, alpha=0.95))
            cumulative += total_impact
            continue

        lender_data = lender_data.sort_values('booking_impact', ascending=True)
        bottom = cumulative

        for _, lender_row in lender_data.iterrows():
            lender = lender_row['lender']
            lender_impact = lender_row['booking_impact']

            if abs(lender_impact) < 0.01:
                continue

            color = lender_colors[lender]
            height = lender_impact

            rect = Rectangle((x_idx - 0.3, bottom), 0.6, height,
                           facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.85)
            ax.add_patch(rect)
            bottom = bottom + lender_impact

        label_y_pos = min(cumulative, cumulative + total_impact)
        sign = '+' if total_impact >= 0 else ''
        ax.text(x_idx, label_y_pos - label_offset, f'{sign}{format_number(total_impact, 0)}',
               ha='center', va='top', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='black', linewidth=1.5, alpha=0.95))

        prev_y = cumulative
        ax.plot([x_idx-1+0.3, x_idx-0.3], [prev_y, prev_y],
               color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)

        cumulative += total_impact

    # Plot End bar
    end_idx = len(labels) - 1
    ax.bar(end_idx, period_2_bks, color=COLOR_TOTAL, edgecolor='black', linewidth=1.5, width=0.78)
    ax.text(end_idx, period_2_bks/2 + y_min/2, format_number(period_2_bks, 0),
            ha='center', va='center', fontsize=10, fontweight='bold')

    ax.plot([end_idx-1+0.3, end_idx-0.3], [cumulative, cumulative],
           color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(format_effect_labels(labels), rotation=45, ha='right')
    ax.set_ylabel('Bookings', fontsize=11, fontweight='bold')
    ax.set_title('By Lender', fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.8)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=lender_colors[lender], edgecolor='black',
                      label=lender, alpha=0.85)
        for lender in lenders
    ]

    n_legend_items = len(legend_elements)
    legend_ncol = (n_legend_items + 1) // 2 if n_legend_items >= 6 else n_legend_items

    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
             ncol=legend_ncol, frameon=True, fancybox=True, shadow=True,
             fontsize=9, title='Lenders', title_fontsize=10)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path, "Lender waterfall grid")

    return fig


def create_multi_lender_waterfall_grid(
    results,  # MultiLenderResults
    output_path: str = None
) -> plt.Figure:
    """
    Create waterfall grid for multi-lender, multi-channel analysis.

    Layout (2x2):
    - Top-left: Aggregate waterfall
    - Top-right: Stacked by tier (T1, T2, T3)
    - Bottom-left: Stacked by finance channel (FF, Non-FF)
    - Bottom-right: Summary panel

    Args:
        results: MultiLenderResults from calculate_multi_lender_decomposition
        output_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    metadata = results.metadata
    date_a = metadata['date_a']
    date_b = metadata['date_b']
    period_1_bks = metadata['period_1_total_bookings']
    period_2_bks = metadata['period_2_total_bookings']

    # Get breakdowns for title
    channel_breakdown = format_channel_breakdown(metadata)

    delta = period_2_bks - period_1_bks
    delta_sign = '+' if delta >= 0 else ''

    # 2x2 layout with channel breakdown
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes_flat = axes.flatten()

    fig.suptitle(
        f'Multi-Lender Booking Decomposition: {date_a} → {date_b}\n'
        f'Total Change: {delta_sign}{delta:,.0f} {channel_breakdown}',
        fontsize=14, fontweight='bold', y=0.98
    )

    # Chart 1: Aggregate waterfall
    _create_aggregate_waterfall(
        ax=axes_flat[0],
        summary=results.aggregate_summary,
        period_1_bks=period_1_bks,
        period_2_bks=period_2_bks,
        title='Overall Aggregate'
    )

    # Chart 2: Stacked by tier
    if not results.tier_summary.empty:
        _create_tier_stacked_waterfall(
            ax=axes_flat[1],
            tier_summary=results.tier_summary,
            period_1_bks=period_1_bks,
            period_2_bks=period_2_bks,
            tier_totals=metadata.get('tier_totals', {}),
            title='By Lender Tier'
        )

    # Chart 3: Stacked by channel
    if not results.channel_summary.empty:
        _create_channel_stacked_waterfall_lender(
            ax=axes_flat[2],
            channel_summary=results.channel_summary,
            period_1_bks=period_1_bks,
            period_2_bks=period_2_bks,
            channel_totals=metadata.get('channel_totals', {}),
            title='By Finance Channel'
        )

    # Chart 4: Summary panel
    _create_summary_panel(axes_flat[3], metadata, delta, delta_sign)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path, "Multi-lender waterfall grid")

    return fig


def _create_summary_panel(ax: plt.Axes, metadata: dict, delta: float, delta_sign: str) -> None:
    """Create summary panel with breakdown information."""
    ax.axis('off')

    period_1_bks = metadata['period_1_total_bookings']
    period_2_bks = metadata['period_2_total_bookings']

    summary_lines = [
        f"Period 1 Bookings: {period_1_bks:,.0f}",
        f"Period 2 Bookings: {period_2_bks:,.0f}",
        f"Total Change: {delta_sign}{delta:,.0f}",
        "",
        "Tier Breakdown:",
    ]

    tier_totals = metadata.get('tier_totals', {})
    for tier in ['T1', 'T2', 'T3']:
        if tier in tier_totals:
            t_delta = tier_totals[tier]['delta_bookings']
            t_sign = '+' if t_delta >= 0 else ''
            summary_lines.append(f"  {tier}: {t_sign}{t_delta:,.0f}")

    summary_lines.extend(["", "Channel Breakdown:"])
    channel_totals = metadata.get('channel_totals', {})
    for ch in ['FF', 'NON_FF']:
        if ch in channel_totals:
            c_delta = channel_totals[ch]['delta_bookings']
            c_sign = '+' if c_delta >= 0 else ''
            label = 'Non-FF' if ch == 'NON_FF' else 'FF'
            summary_lines.append(f"  {label}: {c_sign}{c_delta:,.0f}")

    ax.text(0.5, 0.7, "\n".join(summary_lines),
            transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa',
                     edgecolor='black', linewidth=1))

    # Add legends for tier and channel colors
    tier_elements = [
        mpatches.Patch(facecolor=TIER_COLORS.get(t, '#999999'),
                      edgecolor='black', label=t, alpha=0.85)
        for t in ['T1', 'T2', 'T3']
    ]
    channel_elements = [
        mpatches.Patch(facecolor=CHANNEL_COLORS.get(c, '#999999'),
                      edgecolor='black',
                      label='Non-FF' if c == 'NON_FF' else 'FF', alpha=0.85)
        for c in ['FF', 'NON_FF']
    ]

    ax.legend(handles=tier_elements + channel_elements,
              loc='lower center', bbox_to_anchor=(0.5, 0.05),
              ncol=5, fontsize=9, framealpha=0.95, edgecolor='black',
              title='Legend')


def _create_tier_stacked_waterfall(
    ax: plt.Axes,
    tier_summary: pd.DataFrame,
    period_1_bks: float,
    period_2_bks: float,
    tier_totals: dict,
    title: str
) -> None:
    """Create waterfall with effects stacked by lender tier."""

    # Transform data for stacking (all 8 effects separately)
    tier_agg = aggregate_by_tier(tier_summary, combine_volume_mix=False)

    # Get effect types in order (all 8 effects)
    effect_order = [
        'volume_effect',
        'vsa_progression_effect',
        'customer_mix_effect',
        'offer_comp_mix_effect',
        'str_approval_effect', 'cond_approval_effect',
        'str_booking_effect', 'cond_booking_effect'
    ]
    effects = [e for e in effect_order if e in tier_agg['effect_type'].unique()]

    labels = ['Start'] + effects + ['End']
    x_pos = np.arange(len(labels))

    tiers = ['T1', 'T2', 'T3']

    # Calculate cumulative positions and values for y-axis
    # For stacked bars, we need to track the ACTUAL extent of positive and negative stacks
    all_values = [period_1_bks, period_2_bks]
    cumulative = period_1_bks

    for effect in effects:
        effect_data = tier_agg[tier_agg['effect_type'] == effect]
        total_effect = effect_data['impact'].sum()

        # Track the actual stacked bar extents, not just cumulative + total
        pos_sum = effect_data[effect_data['impact'] > 0]['impact'].sum()
        neg_sum = effect_data[effect_data['impact'] < 0]['impact'].sum()

        # Positive bars stack upward from cumulative
        if pos_sum > 0:
            all_values.append(cumulative + pos_sum)
        # Negative bars stack downward from cumulative
        if neg_sum < 0:
            all_values.append(cumulative + neg_sum)

        all_values.extend([cumulative, cumulative + total_effect])
        cumulative += total_effect

    data_min = min(all_values)
    data_max = max(all_values)
    data_range = max(data_max - data_min, 1)

    # Use proportional padding based on data magnitude, not just range
    # This prevents over-compression when changes are small relative to totals
    padding_below = max(data_range * 0.3, data_max * 0.08)
    padding_above = max(data_range * 0.2, data_max * 0.08)
    y_min = data_min - padding_below
    y_max = data_max + padding_above
    y_range = y_max - y_min
    label_offset = y_range * 0.03

    # Plot Start bar
    ax.bar(0, period_1_bks, color=COLOR_TOTAL, edgecolor='black', linewidth=1.5, width=0.78)
    text_y_pos = (period_1_bks + y_min) / 2
    ax.text(0, text_y_pos, format_number(period_1_bks, 0),
           ha='center', va='center', fontsize=10, fontweight='bold')

    cumulative = period_1_bks

    # Plot each effect with tier stacking
    for i, effect in enumerate(effects):
        x = i + 1

        effect_data = tier_agg[tier_agg['effect_type'] == effect]
        total_effect = effect_data['impact'].sum()

        if abs(total_effect) < 0.01:
            ax.plot([x-1+0.3, x-0.3], [cumulative, cumulative],
                   color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)
            ax.plot(x, cumulative, marker='o', markersize=6, color=COLOR_CONNECTOR,
                   markeredgecolor='black', markeredgewidth=1.5, zorder=3)
            ax.text(x, cumulative - label_offset, '+0',
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='black', linewidth=1.5, alpha=0.95))
        else:
            pos_bottom = cumulative
            neg_bottom = cumulative

            for tier in tiers:
                tier_data = effect_data[effect_data['lender_tier'] == tier]
                if len(tier_data) == 0:
                    continue

                val = tier_data['impact'].iloc[0]
                color = TIER_COLORS.get(tier, '#999999')

                if val >= 0:
                    rect = Rectangle((x - 0.3, pos_bottom), 0.6, val,
                                   facecolor=color, edgecolor='black',
                                   linewidth=0.5, alpha=0.85)
                    ax.add_patch(rect)
                    pos_bottom += val
                else:
                    rect = Rectangle((x - 0.3, neg_bottom + val), 0.6, abs(val),
                                   facecolor=color, edgecolor='black',
                                   linewidth=0.5, alpha=0.85)
                    ax.add_patch(rect)
                    neg_bottom += val

            label_y_pos = min(cumulative, cumulative + total_effect, neg_bottom)
            sign = '+' if total_effect >= 0 else ''
            ax.text(x, label_y_pos - label_offset, f'{sign}{format_number(total_effect, 0)}',
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='black', linewidth=1.5, alpha=0.95))

            ax.plot([x-1+0.3, x-0.3], [cumulative, cumulative],
                   color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)

        cumulative += total_effect

    # Plot End bar
    ax.bar(len(labels)-1, period_2_bks, color=COLOR_TOTAL, edgecolor='black',
          linewidth=1.5, width=0.78)
    text_y_pos_end = (period_2_bks + y_min) / 2
    ax.text(len(labels)-1, text_y_pos_end, format_number(period_2_bks, 0),
           ha='center', va='center', fontsize=10, fontweight='bold')

    ax.plot([len(labels)-2+0.3, len(labels)-1-0.3], [cumulative, cumulative],
           color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(format_effect_labels(labels), rotation=45, ha='right')
    ax.set_ylabel('Bookings', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.8)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=TIER_COLORS.get(t, '#999999'),
                      edgecolor='black', label=t, alpha=0.85)
        for t in tiers
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
             fontsize=9, framealpha=0.95, edgecolor='black', ncol=3)


def _create_channel_stacked_waterfall_lender(
    ax: plt.Axes,
    channel_summary: pd.DataFrame,
    period_1_bks: float,
    period_2_bks: float,
    channel_totals: dict,
    title: str
) -> None:
    """Create waterfall with effects stacked by finance channel (for multi-lender)."""

    # Get effects from channel_summary (all 8 effects separately)
    df = channel_summary.copy()

    # Keep all effects separate (no combining)
    channel_agg = df[df['effect_type'] != 'total_change'].copy()
    channel_agg = channel_agg.rename(columns={'booking_impact': 'impact'})

    # Get effect types in order (all 8 effects)
    effect_order = [
        'volume_effect',
        'vsa_progression_effect',
        'customer_mix_effect',
        'offer_comp_mix_effect',
        'str_approval_effect', 'cond_approval_effect',
        'str_booking_effect', 'cond_booking_effect'
    ]
    effects = [e for e in effect_order if e in channel_agg['effect_type'].unique()]

    labels = ['Start'] + effects + ['End']
    x_pos = np.arange(len(labels))

    channels = get_finance_channel_values()

    # Calculate cumulative positions and values for y-axis
    # For stacked bars, we need to track the ACTUAL extent of positive and negative stacks
    all_values = [period_1_bks, period_2_bks]
    cumulative = period_1_bks

    for effect in effects:
        effect_data = channel_agg[channel_agg['effect_type'] == effect]
        total_effect = effect_data['impact'].sum()

        # Track the actual stacked bar extents, not just cumulative + total
        pos_sum = effect_data[effect_data['impact'] > 0]['impact'].sum()
        neg_sum = effect_data[effect_data['impact'] < 0]['impact'].sum()

        # Positive bars stack upward from cumulative
        if pos_sum > 0:
            all_values.append(cumulative + pos_sum)
        # Negative bars stack downward from cumulative
        if neg_sum < 0:
            all_values.append(cumulative + neg_sum)

        all_values.extend([cumulative, cumulative + total_effect])
        cumulative += total_effect

    data_min = min(all_values)
    data_max = max(all_values)
    data_range = max(data_max - data_min, 1)

    # Use proportional padding based on data magnitude
    padding_below = max(data_range * 0.3, data_max * 0.08)
    padding_above = max(data_range * 0.2, data_max * 0.08)
    y_min = data_min - padding_below
    y_max = data_max + padding_above
    y_range = y_max - y_min
    label_offset = y_range * 0.03

    # Plot Start bar
    ax.bar(0, period_1_bks, color=COLOR_TOTAL, edgecolor='black', linewidth=1.5, width=0.78)
    text_y_pos = (period_1_bks + y_min) / 2
    ax.text(0, text_y_pos, format_number(period_1_bks, 0),
           ha='center', va='center', fontsize=10, fontweight='bold')

    cumulative = period_1_bks

    # Plot each effect with channel stacking
    for i, effect in enumerate(effects):
        x = i + 1

        effect_data = channel_agg[channel_agg['effect_type'] == effect]
        total_effect = effect_data['impact'].sum()

        if abs(total_effect) < 0.01:
            ax.plot([x-1+0.3, x-0.3], [cumulative, cumulative],
                   color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)
            ax.plot(x, cumulative, marker='o', markersize=6, color=COLOR_CONNECTOR,
                   markeredgecolor='black', markeredgewidth=1.5, zorder=3)
            ax.text(x, cumulative - label_offset, '+0',
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='black', linewidth=1.5, alpha=0.95))
        else:
            pos_bottom = cumulative
            neg_bottom = cumulative

            for channel in channels:
                ch_data = effect_data[effect_data['finance_channel'] == channel]
                if len(ch_data) == 0:
                    continue

                val = ch_data['impact'].iloc[0]
                color = CHANNEL_COLORS.get(channel, '#999999')

                if val >= 0:
                    rect = Rectangle((x - 0.3, pos_bottom), 0.6, val,
                                   facecolor=color, edgecolor='black',
                                   linewidth=0.5, alpha=0.85)
                    ax.add_patch(rect)
                    pos_bottom += val
                else:
                    rect = Rectangle((x - 0.3, neg_bottom + val), 0.6, abs(val),
                                   facecolor=color, edgecolor='black',
                                   linewidth=0.5, alpha=0.85)
                    ax.add_patch(rect)
                    neg_bottom += val

            label_y_pos = min(cumulative, cumulative + total_effect, neg_bottom)
            sign = '+' if total_effect >= 0 else ''
            ax.text(x, label_y_pos - label_offset, f'{sign}{format_number(total_effect, 0)}',
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='black', linewidth=1.5, alpha=0.95))

            ax.plot([x-1+0.3, x-0.3], [cumulative, cumulative],
                   color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)

        cumulative += total_effect

    # Plot End bar
    ax.bar(len(labels)-1, period_2_bks, color=COLOR_TOTAL, edgecolor='black',
          linewidth=1.5, width=0.78)
    text_y_pos_end = (period_2_bks + y_min) / 2
    ax.text(len(labels)-1, text_y_pos_end, format_number(period_2_bks, 0),
           ha='center', va='center', fontsize=10, fontweight='bold')

    ax.plot([len(labels)-2+0.3, len(labels)-1-0.3], [cumulative, cumulative],
           color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(format_effect_labels(labels), rotation=45, ha='right')
    ax.set_ylabel('Bookings', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.8)

    # Legend
    legend_elements = []
    for channel in channels:
        label = 'Non-FF' if channel == 'NON_FF' else 'FF'
        legend_elements.append(
            mpatches.Patch(facecolor=CHANNEL_COLORS.get(channel, '#999999'),
                          edgecolor='black', label=label, alpha=0.85)
        )
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
             fontsize=9, framealpha=0.95, edgecolor='black', ncol=2)


# Tier × Channel combination colors
TIER_CHANNEL_COLORS = {
    ('T1', 'FF'): '#003366',      # Dark Navy Blue
    ('T1', 'NON_FF'): '#4a7fb0',  # Medium Navy Blue
    ('T2', 'FF'): '#2563eb',      # Medium Blue
    ('T2', 'NON_FF'): '#60a5fa',  # Light Medium Blue
    ('T3', 'FF'): '#7dd3fc',      # Light Sky Blue
    ('T3', 'NON_FF'): '#bae6fd',  # Very Light Blue
}


def create_tier_channel_waterfall(
    results,  # MultiLenderResults
    output_path: str = None
) -> plt.Figure:
    """
    Create waterfall chart showing effects stacked by tier×channel combination.

    This allows identification of T2 FF vs T2 NON_FF contribution to each effect.

    Args:
        results: MultiLenderResults from calculate_multi_lender_decomposition
        output_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    metadata = results.metadata
    date_a = metadata['date_a']
    date_b = metadata['date_b']
    period_1_bks = metadata['period_1_total_bookings']
    period_2_bks = metadata['period_2_total_bookings']

    delta = period_2_bks - period_1_bks
    delta_sign = '+' if delta >= 0 else ''

    fig, ax = plt.subplots(figsize=(20, 10))

    fig.suptitle(
        f'Multi-Lender Booking Decomposition: {date_a} → {date_b}\n'
        f'Total Change: {delta_sign}{delta:,.0f} (By Tier × Finance Channel)',
        fontsize=14, fontweight='bold', y=0.98
    )

    # Build tier×channel breakdown from lender_results
    tier_channel_effects = _aggregate_tier_channel_effects(results)

    # Effect order (all 8 effects)
    effect_order = [
        'volume_effect',
        'vsa_progression_effect',
        'customer_mix_effect',
        'offer_comp_mix_effect',
        'str_approval_effect', 'cond_approval_effect',
        'str_booking_effect', 'cond_booking_effect'
    ]
    effects = [e for e in effect_order if e in tier_channel_effects['effect_type'].unique()]

    labels = ['Start'] + effects + ['End']
    x_pos = np.arange(len(labels))

    # Tier × Channel combinations in order
    tier_channel_combos = [
        ('T1', 'FF'), ('T1', 'NON_FF'),
        ('T2', 'FF'), ('T2', 'NON_FF'),
        ('T3', 'FF'), ('T3', 'NON_FF')
    ]

    # Calculate y-axis limits
    all_values = [period_1_bks, period_2_bks]
    cumulative = period_1_bks

    for effect in effects:
        effect_data = tier_channel_effects[tier_channel_effects['effect_type'] == effect]
        total_effect = effect_data['impact'].sum()

        pos_sum = effect_data[effect_data['impact'] > 0]['impact'].sum()
        neg_sum = effect_data[effect_data['impact'] < 0]['impact'].sum()

        if pos_sum > 0:
            all_values.append(cumulative + pos_sum)
        if neg_sum < 0:
            all_values.append(cumulative + neg_sum)

        all_values.extend([cumulative, cumulative + total_effect])
        cumulative += total_effect

    data_min = min(all_values)
    data_max = max(all_values)
    data_range = max(data_max - data_min, 1)

    padding_below = max(data_range * 0.3, data_max * 0.08)
    padding_above = max(data_range * 0.2, data_max * 0.08)
    y_min = data_min - padding_below
    y_max = data_max + padding_above
    y_range = y_max - y_min
    label_offset = y_range * 0.03

    # Plot Start bar
    ax.bar(0, period_1_bks, color=COLOR_TOTAL, edgecolor='black', linewidth=1.5, width=0.78)
    text_y_pos = (period_1_bks + y_min) / 2
    ax.text(0, text_y_pos, format_number(period_1_bks, 0),
           ha='center', va='center', fontsize=10, fontweight='bold')

    cumulative = period_1_bks

    # Plot each effect with tier×channel stacking
    for i, effect in enumerate(effects):
        x = i + 1

        effect_data = tier_channel_effects[tier_channel_effects['effect_type'] == effect]
        total_effect = effect_data['impact'].sum()

        if abs(total_effect) < 0.01:
            ax.plot([x-1+0.3, x-0.3], [cumulative, cumulative],
                   color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)
            ax.plot(x, cumulative, marker='o', markersize=6, color=COLOR_CONNECTOR,
                   markeredgecolor='black', markeredgewidth=1.5, zorder=3)
            ax.text(x, cumulative - label_offset, '+0',
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='black', linewidth=1.5, alpha=0.95))
        else:
            pos_bottom = cumulative
            neg_bottom = cumulative

            for tier, channel in tier_channel_combos:
                combo_data = effect_data[
                    (effect_data['lender_tier'] == tier) &
                    (effect_data['finance_channel'] == channel)
                ]
                if len(combo_data) == 0:
                    continue

                val = combo_data['impact'].iloc[0]
                color = TIER_CHANNEL_COLORS.get((tier, channel), '#999999')

                if val >= 0:
                    rect = Rectangle((x - 0.3, pos_bottom), 0.6, val,
                                   facecolor=color, edgecolor='black',
                                   linewidth=0.5, alpha=0.85)
                    ax.add_patch(rect)
                    pos_bottom += val
                else:
                    rect = Rectangle((x - 0.3, neg_bottom + val), 0.6, abs(val),
                                   facecolor=color, edgecolor='black',
                                   linewidth=0.5, alpha=0.85)
                    ax.add_patch(rect)
                    neg_bottom += val

            label_y_pos = min(cumulative, cumulative + total_effect, neg_bottom)
            sign = '+' if total_effect >= 0 else ''
            ax.text(x, label_y_pos - label_offset, f'{sign}{format_number(total_effect, 0)}',
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='black', linewidth=1.5, alpha=0.95))

            ax.plot([x-1+0.3, x-0.3], [cumulative, cumulative],
                   color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)

        cumulative += total_effect

    # Plot End bar
    ax.bar(len(labels)-1, period_2_bks, color=COLOR_TOTAL, edgecolor='black',
          linewidth=1.5, width=0.78)
    text_y_pos_end = (period_2_bks + y_min) / 2
    ax.text(len(labels)-1, text_y_pos_end, format_number(period_2_bks, 0),
           ha='center', va='center', fontsize=10, fontweight='bold')

    ax.plot([len(labels)-2+0.3, len(labels)-1-0.3], [cumulative, cumulative],
           color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(format_effect_labels(labels), rotation=45, ha='right')
    ax.set_ylabel('Bookings', fontsize=11, fontweight='bold')
    ax.set_title('By Lender Tier × Finance Channel', fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.8)

    # Legend (2 rows: FF then NON_FF)
    legend_elements = []
    for tier, channel in tier_channel_combos:
        label = f'{tier} {"Non-FF" if channel == "NON_FF" else "FF"}'
        legend_elements.append(
            mpatches.Patch(facecolor=TIER_CHANNEL_COLORS.get((tier, channel), '#999999'),
                          edgecolor='black', label=label, alpha=0.85)
        )
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
             fontsize=9, framealpha=0.95, edgecolor='black', ncol=6)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path, "Tier×Channel waterfall")

    return fig


def _aggregate_tier_channel_effects(results) -> pd.DataFrame:
    """
    Aggregate effects by tier × channel combination from MultiLenderResults.

    Args:
        results: MultiLenderResults object

    Returns:
        DataFrame with columns: lender_tier, finance_channel, effect_type, impact
    """
    # Build from lender_results which has channel-level details
    all_effects = []

    for lender, lender_result in results.lender_results.items():
        tier = get_lender_tier(lender)

        # Check if lender_result has channel_summaries (FinanceChannelResults)
        if hasattr(lender_result, 'channel_summaries') and lender_result.channel_summaries is not None:
            for _, row in lender_result.channel_summaries.iterrows():
                if row['effect_type'] != 'total_change':
                    all_effects.append({
                        'lender': lender,
                        'lender_tier': tier,
                        'finance_channel': row['finance_channel'],
                        'effect_type': row['effect_type'],
                        'impact': row['booking_impact']
                    })
        else:
            # Fallback: use lender summary and split by channel from metadata
            channel_totals = results.metadata.get('channel_totals', {})
            lender_summary = lender_result.summary if hasattr(lender_result, 'summary') else None

            if lender_summary is not None:
                for _, row in lender_summary.iterrows():
                    if row['effect_type'] not in ['total_change', 'interaction_effect']:
                        # Split proportionally by channel (rough estimate)
                        for channel in ['FF', 'NON_FF']:
                            all_effects.append({
                                'lender': lender,
                                'lender_tier': tier,
                                'finance_channel': channel,
                                'effect_type': row['effect_type'],
                                'impact': row['booking_impact'] * 0.5  # Split evenly as fallback
                            })

    if not all_effects:
        return pd.DataFrame(columns=['lender_tier', 'finance_channel', 'effect_type', 'impact'])

    df = pd.DataFrame(all_effects)

    # Aggregate by tier × channel × effect
    agg = df.groupby(['lender_tier', 'finance_channel', 'effect_type'])['impact'].sum().reset_index()

    return agg


def create_tier_channel_drilldown(
    results,  # MultiLenderResults
    output_path: str = None
) -> plt.Figure:
    """
    Create horizontal bar chart showing each effect broken down by tier×channel.

    This provides a detailed view of how each tier and channel contributes to each effect.

    Args:
        results: MultiLenderResults from calculate_multi_lender_decomposition
        output_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    metadata = results.metadata
    date_a = metadata['date_a']
    date_b = metadata['date_b']

    # Build tier×channel breakdown
    tier_channel_effects = _aggregate_tier_channel_effects(results)

    # Effect order (all 8 effects)
    effect_order = [
        'volume_effect',
        'vsa_progression_effect',
        'customer_mix_effect',
        'offer_comp_mix_effect',
        'str_approval_effect', 'cond_approval_effect',
        'str_booking_effect', 'cond_booking_effect'
    ]
    effects = [e for e in effect_order if e in tier_channel_effects['effect_type'].unique()]

    # Create figure with subplots for each effect
    n_effects = len(effects)
    n_cols = 2
    n_rows = (n_effects + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes_flat = axes.flatten() if n_effects > 2 else [axes[0], axes[1]] if n_effects == 2 else [axes]

    fig.suptitle(
        f'Effect Breakdown by Tier × Finance Channel\n{date_a} → {date_b}',
        fontsize=14, fontweight='bold', y=0.98
    )

    # Tier × Channel combinations for y-axis
    tier_channel_labels = ['T1 FF', 'T1 Non-FF', 'T2 FF', 'T2 Non-FF', 'T3 FF', 'T3 Non-FF']
    tier_channel_combos = [
        ('T1', 'FF'), ('T1', 'NON_FF'),
        ('T2', 'FF'), ('T2', 'NON_FF'),
        ('T3', 'FF'), ('T3', 'NON_FF')
    ]

    for idx, effect in enumerate(effects):
        ax = axes_flat[idx]

        effect_data = tier_channel_effects[tier_channel_effects['effect_type'] == effect]

        # Get values for each tier×channel combo
        values = []
        colors = []
        for tier, channel in tier_channel_combos:
            combo_data = effect_data[
                (effect_data['lender_tier'] == tier) &
                (effect_data['finance_channel'] == channel)
            ]
            val = combo_data['impact'].iloc[0] if len(combo_data) > 0 else 0
            values.append(val)
            colors.append(TIER_CHANNEL_COLORS.get((tier, channel), '#999999'))

        # Create horizontal bar chart
        y_pos = np.arange(len(tier_channel_labels))
        bars = ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            if abs(val) > 0.1:
                sign = '+' if val >= 0 else ''
                label_x = val + (abs(val) * 0.05) if val >= 0 else val - (abs(val) * 0.05)
                ha = 'left' if val >= 0 else 'right'
                ax.text(label_x, i, f'{sign}{format_number(val, 0)}',
                       va='center', ha=ha, fontsize=9, fontweight='bold')

        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tier_channel_labels)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        total_impact = sum(values)
        effect_label = format_effect_labels([effect])[0]
        ax.set_title(f'{effect_label} (Total: {total_impact:+,.0f})',
                    fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('Booking Impact', fontsize=10)

    # Hide empty subplots
    for idx in range(len(effects), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path, "Tier×Channel drilldown")

    return fig
