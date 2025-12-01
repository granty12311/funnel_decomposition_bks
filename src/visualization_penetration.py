"""
Penetration decomposition visualizations (Self-Adjusted approach).

Four chart types:
1. Net Lender + Competitor: 7 net lender effects + 1 total competitor effect
2. Net All Inclusive: 7 net effects (net lender + competitor combined per driver)
3. Net Lender vs Competitor Breakdown: Stacked bars showing net lender vs competitor per effect
4. Net Volume Inclusive + Split Others: Net volume (all inclusive) + 6 net lender + remaining competitor

Uses SELF-ADJUSTED approach:
- Net lender effects = Gross lender effect - Self-adjustment (denominator impact)
- Competitor effects are purely from rest of market (no self-influence)
- Exact reconciliation with no residual

All effects are expressed in basis points (bps).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Union, Optional, List

try:
    from .visualization_utils import (
        COLOR_POSITIVE, COLOR_NEGATIVE, COLOR_TOTAL, COLOR_CONNECTOR, COLOR_MARKET,
        format_dimension_name, format_effect_labels, save_figure
    )
except ImportError:
    from visualization_utils import (
        COLOR_POSITIVE, COLOR_NEGATIVE, COLOR_TOTAL, COLOR_CONNECTOR, COLOR_MARKET,
        format_dimension_name, format_effect_labels, save_figure
    )

# Colors for lender vs competitor comparison
COLOR_LENDER = '#2E86AB'  # Blue for net lender
COLOR_COMPETITOR = '#708090'  # Stone blue/slate gray for competitor (rest of market)


def _bps_to_pct_formatter(x, pos):
    """Format y-axis ticks: convert bps to percentage (100 bps = 1%)."""
    return f'{x/100:.1f}%'


def create_penetration_waterfall_grid(
    summary: pd.DataFrame,
    metadata: dict,
    lender: str = 'ACA',
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Create 2x2 grid of penetration decomposition charts (Self-Adjusted approach).

    Chart 1 (top-left): Net Lender + Competitor (7 net lender + 1 competitor)
    Chart 2 (top-right): Net All Inclusive (7 net effects)
    Chart 3 (bottom-left): Net Lender vs Competitor Breakdown (stacked)
    Chart 4 (bottom-right): Net Volume Inclusive + Split Others
    """
    date_a = metadata['date_a']
    date_b = metadata['date_b']
    period_1_pen = metadata['period_1_penetration'] * 100 * 100  # Convert to bps
    period_2_pen = metadata['period_2_penetration'] * 100 * 100  # Convert to bps

    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes_flat = axes.flatten()

    fig.suptitle(
        f'{lender} Penetration Decomposition (Self-Adjusted): {date_a} -> {date_b}\n'
        f'Penetration: {period_1_pen/100:.1f}% -> {period_2_pen/100:.1f}% '
        f'({metadata["delta_penetration_bps"]:+.0f} bps)',
        fontsize=14, fontweight='bold', y=0.995
    )

    # Chart 1: Net Lender + Competitor
    _create_net_lender_plus_competitor_waterfall(
        ax=axes_flat[0],
        summary=summary,
        period_1_pen=period_1_pen,
        period_2_pen=period_2_pen,
        title='Chart 1: Net Lender Effects + Competitor'
    )

    # Chart 2: Net All Inclusive
    _create_net_all_inclusive_waterfall(
        ax=axes_flat[1],
        summary=summary,
        period_1_pen=period_1_pen,
        period_2_pen=period_2_pen,
        title='Chart 2: Net All Inclusive'
    )

    # Chart 3: Net Lender vs Competitor Breakdown
    _create_net_lender_vs_competitor_waterfall(
        ax=axes_flat[2],
        summary=summary,
        metadata=metadata,
        period_1_pen=period_1_pen,
        period_2_pen=period_2_pen,
        title='Chart 3: Net Lender vs Competitor Breakdown'
    )

    # Chart 4: Net Volume Inclusive + Split Others
    _create_net_volume_inclusive_waterfall(
        ax=axes_flat[3],
        summary=summary,
        metadata=metadata,
        period_1_pen=period_1_pen,
        period_2_pen=period_2_pen,
        title='Chart 4: Net Volume (Inclusive) + Net Lender + Remaining Competitor'
    )

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path, "Penetration waterfall grid")

    return fig


def _create_net_lender_plus_competitor_waterfall(
    ax: plt.Axes,
    summary: pd.DataFrame,
    period_1_pen: float,
    period_2_pen: float,
    title: str
) -> None:
    """Chart 1: Net lender effects + total competitor effect (8 bars)."""

    # Get effects (excluding total_change)
    effects = summary[summary['effect_type'] != 'total_change'].copy()

    effect_types = effects['effect_type'].tolist()
    net_lender_vals = effects['net_lender_effect_bps'].tolist()

    # Total competitor effect
    total_competitor = effects['competitor_effect_bps'].sum()

    # Build labels and values: Start + 7 net lender effects + competitor + End
    labels = ['Start'] + effect_types + ['Competitor', 'End']
    values = [period_1_pen] + net_lender_vals + [total_competitor, period_2_pen]

    _draw_waterfall(ax, labels, values, period_1_pen, period_2_pen, title,
                    competitor_idx=len(effect_types) + 1)


def _create_net_all_inclusive_waterfall(
    ax: plt.Axes,
    summary: pd.DataFrame,
    period_1_pen: float,
    period_2_pen: float,
    title: str
) -> None:
    """Chart 2: Net all inclusive (7 net effects = net lender + competitor per step)."""

    # Get effects (excluding total_change)
    effects = summary[summary['effect_type'] != 'total_change'].copy()

    labels = ['Start'] + effects['effect_type'].tolist() + ['End']
    values = [period_1_pen] + effects['net_effect_bps'].tolist() + [period_2_pen]

    _draw_waterfall(ax, labels, values, period_1_pen, period_2_pen, title)


def _create_net_lender_vs_competitor_waterfall(
    ax: plt.Axes,
    summary: pd.DataFrame,
    metadata: dict,
    period_1_pen: float,
    period_2_pen: float,
    title: str
) -> None:
    """Chart 3: Stacked waterfall showing net lender vs competitor per effect."""

    # Get effects (excluding total_change)
    effects = summary[summary['effect_type'] != 'total_change'].copy()

    effect_types = effects['effect_type'].tolist()
    net_lender_vals = effects['net_lender_effect_bps'].tolist()
    competitor_vals = effects['competitor_effect_bps'].tolist()
    net_vals = effects['net_effect_bps'].tolist()

    labels = ['Start'] + effect_types + ['End']
    x_pos = np.arange(len(labels))

    # Pre-calculate cumulative values for y-axis limits
    all_values = [period_1_pen, period_2_pen]
    temp_cumulative = period_1_pen

    for net_val, lender_val, competitor_val in zip(net_vals, net_lender_vals, competitor_vals):
        all_values.append(temp_cumulative)
        # Track extremes from stacking
        all_values.append(temp_cumulative + max(0, lender_val) + max(0, competitor_val))
        all_values.append(temp_cumulative + min(0, lender_val) + min(0, competitor_val))
        temp_cumulative += net_val
        all_values.append(temp_cumulative)

    data_min = min(all_values)
    data_max = max(all_values)
    data_range = data_max - data_min if data_max != data_min else abs(data_max) * 0.1

    y_min = data_min - (data_range * 0.50)
    y_max = data_max + (data_range * 0.20)
    y_range = y_max - y_min
    label_offset = y_range * 0.03

    # Plot Start bar
    ax.bar(0, period_1_pen, color=COLOR_TOTAL, edgecolor='black', linewidth=1.5, width=0.78)
    text_y_pos = (period_1_pen + y_min) / 2
    ax.text(0, text_y_pos, f'{period_1_pen/100:.1f}%',
           ha='center', va='center', fontsize=10, fontweight='bold')

    cumulative = period_1_pen

    # Plot each effect with stacked net lender/competitor bars
    for i, (effect, lender_val, competitor_val, net_val) in enumerate(zip(effect_types, net_lender_vals, competitor_vals, net_vals)):
        x = i + 1

        # Draw connector from previous bar
        ax.plot([x-1+0.3, x-0.3], [cumulative, cumulative],
               color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)

        if abs(net_val) < 0.01:
            # Zero net impact - draw dot
            ax.plot(x, cumulative, marker='o', markersize=6, color=COLOR_CONNECTOR,
                   markeredgecolor='black', markeredgewidth=1.5, zorder=3)
            ax.text(x, cumulative - label_offset, '+0 bps',
                   ha='center', va='top', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='black', linewidth=1.5, alpha=0.95))
        else:
            bar_bottom = cumulative
            bar_top = cumulative

            # Stack positive contributions (lender positive, then competitor positive)
            pos_bottom = cumulative
            if lender_val > 0:
                rect = Rectangle((x - 0.3, pos_bottom), 0.6, lender_val,
                               facecolor=COLOR_LENDER, edgecolor='black',
                               linewidth=0.5, alpha=0.9)
                ax.add_patch(rect)
                pos_bottom += lender_val
                bar_top = max(bar_top, pos_bottom)

            if competitor_val > 0:
                rect = Rectangle((x - 0.3, pos_bottom), 0.6, competitor_val,
                               facecolor=COLOR_COMPETITOR, edgecolor='black',
                               linewidth=0.5, alpha=0.9)
                ax.add_patch(rect)
                pos_bottom += competitor_val
                bar_top = max(bar_top, pos_bottom)

            # Stack negative contributions (lender negative, then competitor negative)
            neg_bottom = cumulative
            if lender_val < 0:
                rect = Rectangle((x - 0.3, neg_bottom + lender_val), 0.6, abs(lender_val),
                               facecolor=COLOR_LENDER, edgecolor='black',
                               linewidth=0.5, alpha=0.9)
                ax.add_patch(rect)
                neg_bottom += lender_val
                bar_bottom = min(bar_bottom, neg_bottom)

            if competitor_val < 0:
                rect = Rectangle((x - 0.3, neg_bottom + competitor_val), 0.6, abs(competitor_val),
                               facecolor=COLOR_COMPETITOR, edgecolor='black',
                               linewidth=0.5, alpha=0.9)
                ax.add_patch(rect)
                neg_bottom += competitor_val
                bar_bottom = min(bar_bottom, neg_bottom)

            # Add net effect label
            new_cumulative = cumulative + net_val
            label_y_pos = min(bar_bottom, cumulative, new_cumulative)
            sign = '+' if net_val >= 0 else ''
            ax.text(x, label_y_pos - label_offset, f'{sign}{net_val:.0f} bps',
                   ha='center', va='top', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='black', linewidth=1.5, alpha=0.95))

        cumulative += net_val

    # Plot End bar
    ax.bar(len(labels)-1, period_2_pen, color=COLOR_TOTAL, edgecolor='black',
          linewidth=1.5, width=0.78)
    text_y_pos_end = (period_2_pen + y_min) / 2
    ax.text(len(labels)-1, text_y_pos_end, f'{period_2_pen/100:.1f}%',
           ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw connector to End bar
    ax.plot([len(labels)-2+0.3, len(labels)-1-0.3], [cumulative, cumulative],
           color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(format_effect_labels(labels), rotation=45, ha='right')
    ax.set_ylabel('Penetration (%)', fontsize=11, fontweight='bold')
    ax.yaxis.set_major_formatter(FuncFormatter(_bps_to_pct_formatter))
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.8)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_LENDER, edgecolor='black', label='Net Lender', alpha=0.9),
        mpatches.Patch(facecolor=COLOR_COMPETITOR, edgecolor='black', label='Competitor', alpha=0.9),
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
             fontsize=9, framealpha=0.95, edgecolor='black', ncol=2)


def _create_net_volume_inclusive_waterfall(
    ax: plt.Axes,
    summary: pd.DataFrame,
    metadata: dict,
    period_1_pen: float,
    period_2_pen: float,
    title: str
) -> None:
    """Chart 4: Net volume (all inclusive) + 6 net lender effects + remaining competitor."""

    # Get effects (excluding total_change)
    effects = summary[summary['effect_type'] != 'total_change'].copy()

    # Net volume (all inclusive = net lender volume + competitor volume)
    net_volume = metadata['net_volume_effect_bps']

    # Remaining competitor (non-volume competitor effects)
    remaining_competitor = metadata['competitor_non_volume_effect_bps']

    # Other net lender effects (excluding volume)
    other_effects = effects[effects['effect_type'] != 'volume_effect'].copy()
    other_types = other_effects['effect_type'].tolist()
    other_net_lender_vals = other_effects['net_lender_effect_bps'].tolist()

    # Build labels and values
    labels = ['Start', 'Net Volume'] + other_types + ['Remaining Competitor', 'End']
    values = [period_1_pen, net_volume] + other_net_lender_vals + [remaining_competitor, period_2_pen]

    _draw_waterfall(ax, labels, values, period_1_pen, period_2_pen, title,
                    net_volume_idx=1, remaining_competitor_idx=len(labels)-2)


def _draw_waterfall(
    ax: plt.Axes,
    labels: list,
    values: list,
    period_1_pen: float,
    period_2_pen: float,
    title: str,
    competitor_idx: int = None,
    net_volume_idx: int = None,
    remaining_competitor_idx: int = None
) -> None:
    """Generic waterfall drawing function."""

    # Calculate cumulative positions
    positions = []
    current = period_1_pen
    positions.append(0)  # Start bar

    for i, val in enumerate(values[1:-1], 1):
        positions.append(current)
        current += val

    positions.append(0)  # End bar

    # Calculate y-axis limits
    all_values = [period_1_pen, period_2_pen]
    current = period_1_pen
    for val in values[1:-1]:
        all_values.append(current)
        current += val
        all_values.append(current)

    data_min = min(all_values)
    data_max = max(all_values)
    data_range = data_max - data_min if data_max != data_min else abs(data_max) * 0.1

    y_min = data_min - (data_range * 0.50)
    y_max = data_max + (data_range * 0.20)
    y_range = y_max - y_min
    label_offset = y_range * 0.03

    x_pos = np.arange(len(labels))

    current = period_1_pen

    for i, (label, val) in enumerate(zip(labels, values)):
        if label in ['Start', 'End']:
            # Total bars (gray) - show as percentage
            ax.bar(i, val, color=COLOR_TOTAL, edgecolor='black', linewidth=1.5, width=0.78)
            ax.text(i, val/2 + y_min/2, f'{val/100:.1f}%',
                   ha='center', va='center', fontsize=10, fontweight='bold')
        else:
            effect_val = val

            # Determine color based on effect type
            if i == competitor_idx or i == remaining_competitor_idx:
                color = COLOR_COMPETITOR
            else:
                # All lender effects (including net volume) use green/red based on sign
                color = COLOR_POSITIVE if effect_val >= 0 else COLOR_NEGATIVE

            if abs(effect_val) < 0.01:
                # Zero impact - draw connector and dot
                if i > 1:
                    ax.plot([i-1+0.3, i-0.3], [current, current],
                           color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)
                ax.plot(i, current, marker='o', markersize=6, color=COLOR_CONNECTOR,
                       markeredgecolor='black', markeredgewidth=1.5, zorder=3)
                ax.text(i, current - label_offset, '+0 bps',
                       ha='center', va='top', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='black', linewidth=1.5, alpha=0.95))
            else:
                bottom = current if effect_val >= 0 else current + effect_val
                height = abs(effect_val)

                rect = Rectangle((i - 0.3, bottom), 0.6, height,
                               facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)

                label_y_pos = min(current, current + effect_val)
                sign = '+' if effect_val >= 0 else ''
                ax.text(i, label_y_pos - label_offset, f'{sign}{effect_val:.0f} bps',
                       ha='center', va='top', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='black', linewidth=1.5, alpha=0.95))

                # Draw connector from previous bar
                if i > 0:
                    ax.plot([i-1+0.3, i-0.3], [current, current],
                           color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)

            current += effect_val

    # Draw connector to End bar
    last_effect_idx = len(labels) - 2
    end_idx = len(labels) - 1
    ax.plot([last_effect_idx+0.3, end_idx-0.3], [current, current],
           color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(format_effect_labels(labels), rotation=45, ha='right')
    ax.set_ylabel('Penetration (%)', fontsize=11, fontweight='bold')
    ax.yaxis.set_major_formatter(FuncFormatter(_bps_to_pct_formatter))
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.8)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_POSITIVE, edgecolor='black', label='Positive', alpha=0.9),
        mpatches.Patch(facecolor=COLOR_NEGATIVE, edgecolor='black', label='Negative', alpha=0.9),
    ]

    if competitor_idx is not None or remaining_competitor_idx is not None:
        legend_elements.append(
            mpatches.Patch(facecolor=COLOR_COMPETITOR, edgecolor='black', label='Competitor', alpha=0.9)
        )

    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
             fontsize=9, framealpha=0.95, edgecolor='black', ncol=len(legend_elements))


# Individual chart creation functions for separate PNG export

def create_penetration_chart_net_lender_plus_competitor(
    summary: pd.DataFrame,
    metadata: dict,
    lender: str = 'ACA',
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """Create Chart 1: Net Lender Effects + Competitor."""
    date_a = metadata['date_a']
    date_b = metadata['date_b']
    period_1_pen = metadata['period_1_penetration'] * 100 * 100
    period_2_pen = metadata['period_2_penetration'] * 100 * 100

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(
        f'{lender} Penetration: {date_a} -> {date_b} | '
        f'{period_1_pen/100:.1f}% -> {period_2_pen/100:.1f}% ({metadata["delta_penetration_bps"]:+.0f} bps)',
        fontsize=12, fontweight='bold'
    )

    _create_net_lender_plus_competitor_waterfall(ax, summary, period_1_pen, period_2_pen,
                                                   'Net Lender Effects + Competitor')
    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path, "Penetration net lender + competitor")
    return fig


def create_penetration_chart_net_all_inclusive(
    summary: pd.DataFrame,
    metadata: dict,
    lender: str = 'ACA',
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """Create Chart 2: Net All Inclusive (7 net effects)."""
    date_a = metadata['date_a']
    date_b = metadata['date_b']
    period_1_pen = metadata['period_1_penetration'] * 100 * 100
    period_2_pen = metadata['period_2_penetration'] * 100 * 100

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(
        f'{lender} Penetration: {date_a} -> {date_b} | '
        f'{period_1_pen/100:.1f}% -> {period_2_pen/100:.1f}% ({metadata["delta_penetration_bps"]:+.0f} bps)',
        fontsize=12, fontweight='bold'
    )

    _create_net_all_inclusive_waterfall(ax, summary, period_1_pen, period_2_pen,
                                         'Net All Inclusive (Lender + Competitor Combined)')
    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path, "Penetration net all inclusive")
    return fig


def create_penetration_chart_net_lender_vs_competitor(
    summary: pd.DataFrame,
    metadata: dict,
    lender: str = 'ACA',
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """Create Chart 3: Net Lender vs Competitor Breakdown (stacked)."""
    date_a = metadata['date_a']
    date_b = metadata['date_b']
    period_1_pen = metadata['period_1_penetration'] * 100 * 100
    period_2_pen = metadata['period_2_penetration'] * 100 * 100

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(
        f'{lender} Penetration: {date_a} -> {date_b} | '
        f'{period_1_pen/100:.1f}% -> {period_2_pen/100:.1f}% ({metadata["delta_penetration_bps"]:+.0f} bps)',
        fontsize=12, fontweight='bold'
    )

    _create_net_lender_vs_competitor_waterfall(ax, summary, metadata, period_1_pen, period_2_pen,
                                                 'Net Lender vs Competitor Breakdown')
    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path, "Penetration net lender vs competitor")
    return fig


def create_penetration_chart_net_volume_inclusive(
    summary: pd.DataFrame,
    metadata: dict,
    lender: str = 'ACA',
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """Create Chart 4: Net Volume Inclusive + Net Lender + Remaining Competitor."""
    date_a = metadata['date_a']
    date_b = metadata['date_b']
    period_1_pen = metadata['period_1_penetration'] * 100 * 100
    period_2_pen = metadata['period_2_penetration'] * 100 * 100

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(
        f'{lender} Penetration: {date_a} -> {date_b} | '
        f'{period_1_pen/100:.1f}% -> {period_2_pen/100:.1f}% ({metadata["delta_penetration_bps"]:+.0f} bps)',
        fontsize=12, fontweight='bold'
    )

    _create_net_volume_inclusive_waterfall(ax, summary, metadata, period_1_pen, period_2_pen,
                                            'Net Volume (Inclusive) + Net Lender + Remaining Competitor')
    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path, "Penetration net volume inclusive")
    return fig


def create_penetration_summary_table(
    summary: pd.DataFrame,
    metadata: dict,
    lender: str = 'ACA'
) -> pd.DataFrame:
    """Create formatted summary table for penetration decomposition."""
    # Get effects (excluding total_change and reconciliation_residual)
    effects = summary[~summary['effect_type'].isin(['total_change', 'reconciliation_residual'])].copy()

    # Determine column names based on available columns
    if 'net_lender_effect_bps' in effects.columns:
        # Self-adjusted approach
        lender_col = 'net_lender_effect_bps'
    else:
        # Rest-of-market approach
        lender_col = 'lender_effect_bps'

    rows = []
    for _, row in effects.iterrows():
        effect_name = row['effect_type'].replace('_', ' ').title()
        rows.append({
            'Effect': effect_name,
            'Lender (bps)': f"{row[lender_col]:+.1f}",
            'Competitor (bps)': f"{row['competitor_effect_bps']:+.1f}",
            'Net (bps)': f"{row['net_effect_bps']:+.1f}"
        })

    # Add total row
    total_lender = effects[lender_col].sum()
    total_competitor = effects['competitor_effect_bps'].sum()
    total_net = effects['net_effect_bps'].sum()
    rows.append({
        'Effect': 'TOTAL',
        'Lender (bps)': f"{total_lender:+.1f}",
        'Competitor (bps)': f"{total_competitor:+.1f}",
        'Net (bps)': f"{total_net:+.1f}"
    })

    return pd.DataFrame(rows)


def print_penetration_decomposition(summary: pd.DataFrame, metadata: dict):
    """Print formatted penetration decomposition results (Self-Adjusted approach)."""

    print("=" * 90)
    print(f"PENETRATION DECOMPOSITION (Self-Adjusted): {metadata['lender']}")
    print(f"Period: {metadata['date_a']} -> {metadata['date_b']}")
    print("=" * 90)

    print(f"\nPenetration: {metadata['period_1_penetration']*100:.2f}% -> "
          f"{metadata['period_2_penetration']*100:.2f}% "
          f"({metadata['delta_penetration_bps']:+.1f} bps)")

    print(f"\nSelf-Adjustment Share: {metadata['self_adjustment_share']*100:.1f}% of market growth")
    print(f"Competitor Share: {metadata['competitor_share']*100:.1f}% of market growth")

    print("\n" + "-" * 90)
    print(f"{'Effect':<25} {'Gross':>10} {'Self-Adj':>10} {'Net Lender':>12} {'Competitor':>12} {'Net':>10}")
    print("-" * 90)

    effects = summary[summary['effect_type'] != 'total_change']
    for _, row in effects.iterrows():
        effect_name = row['effect_type'].replace('_', ' ').title()
        print(f"{effect_name:<25} {row['gross_lender_effect_bps']:>+10.1f} "
              f"{row['self_adjustment_bps']:>+10.1f} {row['net_lender_effect_bps']:>+12.1f} "
              f"{row['competitor_effect_bps']:>+12.1f} {row['net_effect_bps']:>+10.1f}")

    print("-" * 90)
    totals = summary[summary['effect_type'] == 'total_change'].iloc[0]
    print(f"{'TOTAL':<25} {totals['gross_lender_effect_bps']:>+10.1f} "
          f"{totals['self_adjustment_bps']:>+10.1f} {totals['net_lender_effect_bps']:>+12.1f} "
          f"{totals['competitor_effect_bps']:>+12.1f} {totals['net_effect_bps']:>+10.1f}")

    print("=" * 90)

    # Chart 4 breakdown
    print("\nChart 4 Breakdown:")
    print(f"  Net Volume Effect (All Inclusive): {metadata['net_volume_effect_bps']:>+10.1f} bps")
    print(f"  Remaining Competitor Effect:       {metadata['competitor_non_volume_effect_bps']:>+10.1f} bps")

    print(f"\nReconciliation: Net Effects ({totals['net_effect_bps']:+.1f}) = "
          f"Delta Penetration ({metadata['delta_penetration_bps']:+.1f}) ✓")
