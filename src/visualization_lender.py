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
        format_effect_labels, save_figure
    )
    from .visualization_summary import _create_aggregate_waterfall
    from .utils import format_number
    from .dimension_config import apply_dimension_order
except ImportError:
    from visualization_utils import (
        COLOR_POSITIVE, COLOR_NEGATIVE, COLOR_TOTAL, COLOR_CONNECTOR,
        format_effect_labels, save_figure
    )
    from visualization_summary import _create_aggregate_waterfall
    from utils import format_number
    from dimension_config import apply_dimension_order


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
