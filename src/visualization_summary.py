"""
Summary decomposition visualizations for single-lender analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Union, Optional, List

try:
    from .visualization_utils import (
        COLOR_POSITIVE, COLOR_NEGATIVE, COLOR_TOTAL, COLOR_CONNECTOR,
        format_dimension_name, format_effect_labels, aggregate_by_dimension,
        detect_dimension_columns, save_figure
    )
    from .utils import format_number
    from .dimension_config import apply_dimension_order
except ImportError:
    from visualization_utils import (
        COLOR_POSITIVE, COLOR_NEGATIVE, COLOR_TOTAL, COLOR_CONNECTOR,
        format_dimension_name, format_effect_labels, aggregate_by_dimension,
        detect_dimension_columns, save_figure
    )
    from utils import format_number
    from dimension_config import apply_dimension_order


def create_waterfall_grid(
    summary: pd.DataFrame,
    segment_detail: pd.DataFrame,
    lender: str = 'ACA',
    output_path: Optional[Union[str, Path]] = None,
    dimension_columns: Optional[List[str]] = None
) -> plt.Figure:
    """Create grid of waterfall charts showing booking decomposition."""
    # Extract metadata from segment_detail
    date_a = segment_detail['period_1_date'].iloc[0]
    date_b = segment_detail['period_2_date'].iloc[0]
    period_1_bks = segment_detail['period_1_segment_bookings'].sum()
    period_2_bks = segment_detail['period_2_segment_bookings'].sum()

    # Get dimension columns
    if dimension_columns is None:
        dimension_cols = detect_dimension_columns(segment_detail)
    else:
        dimension_cols = dimension_columns

    num_dimensions = len(dimension_cols)

    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()

    if num_dimensions <= 2:
        chart_positions = [0, 1, 2]  # Use first 3 positions, 4th for legend
    else:
        chart_positions = [0, 1, 2, 3]  # All 4 positions used

    fig.suptitle(
        f'{lender} Booking Decomposition: {date_a} -> {date_b}',
        fontsize=16, fontweight='bold', y=0.995
    )

    # Chart 1: Overall aggregate
    _create_aggregate_waterfall(
        ax=axes_flat[chart_positions[0]],
        summary=summary,
        period_1_bks=period_1_bks,
        period_2_bks=period_2_bks,
        title='Overall Aggregate'
    )

    # Store breakdowns for each dimension
    dimension_breakdowns = {}
    first_dim_legend_info = None

    # Create dimensional charts for each dimension
    for i, dim_col in enumerate(dimension_cols[:3]):  # Support up to 3 dimensions
        chart_idx = i + 1  # Skip first position (aggregate)
        if chart_idx < len(chart_positions):
            show_legend_on_chart = not (num_dimensions <= 2 and i == 0)

            result = _create_dimensional_waterfall(
                ax=axes_flat[chart_positions[chart_idx]],
                summary=summary,
                segment_detail=segment_detail,
                dimension=dim_col,
                period_1_bks=period_1_bks,
                period_2_bks=period_2_bks,
                title=f'By {format_dimension_name(dim_col)}',
                show_legend=show_legend_on_chart
            )
            dimension_breakdowns[dim_col] = result['breakdown']

            if i == 0 and num_dimensions <= 2:
                first_dim_legend_info = result['legend_info']

    # For 2D data: display first dimension legend in 4th quadrant
    if num_dimensions <= 2 and first_dim_legend_info is not None:
        ax_legend = axes_flat[3]
        ax_legend.set_visible(True)
        ax_legend.axis('off')

        legend = ax_legend.legend(
            handles=first_dim_legend_info['elements'],
            loc='upper center',
            bbox_to_anchor=(0.5, 1.15),
            fontsize=9,
            framealpha=0.95,
            edgecolor='black',
            title=f"{format_dimension_name(first_dim_legend_info['dimension'])} Legend",
            title_fontsize=11,
            ncol=3
        )
        legend.get_title().set_fontweight('bold')

    # Store breakdowns in figure metadata
    fig.breakdown_details = dimension_breakdowns

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path, "Waterfall grid")

    return fig


def _create_aggregate_waterfall(
    ax: plt.Axes,
    summary: pd.DataFrame,
    period_1_bks: float,
    period_2_bks: float,
    title: str
) -> None:
    """Create simple aggregate waterfall chart."""

    # Get effects (excluding total_change and interaction_effect)
    effects = summary[
        (summary['effect_type'] != 'total_change') &
        (summary['effect_type'] != 'interaction_effect')
    ].copy()

    # Prepare data for plotting
    labels = ['Start'] + effects['effect_type'].tolist() + ['End']
    values = [period_1_bks] + effects['booking_impact'].tolist() + [period_2_bks]

    # Calculate cumulative positions
    positions = []
    current = period_1_bks
    positions.append(0)  # Start bar

    for i, val in enumerate(effects['booking_impact']):
        positions.append(current)
        current += val

    positions.append(0)  # End bar

    # Set dynamic y-axis
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

    # Plot bars
    x_pos = np.arange(len(labels))

    for i, (label, val) in enumerate(zip(labels, values)):
        if label in ['Start', 'End']:
            ax.bar(i, val, color=COLOR_TOTAL, edgecolor='black', linewidth=1.5, width=0.78)
            ax.text(i, val/2 + y_min/2, format_number(val, 0),
                   ha='center', va='center', fontsize=10, fontweight='bold')
        else:
            effect_val = effects.iloc[i-1]['booking_impact']

            if abs(effect_val) < 0.01:
                if i > 1:
                    prev_y = positions[i]
                    ax.plot([i-1+0.3, i-0.3], [prev_y, prev_y],
                           color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)
                ax.plot(i, positions[i], marker='o', markersize=6, color=COLOR_CONNECTOR,
                       markeredgecolor='black', markeredgewidth=1.5, zorder=3)
                ax.text(i, positions[i] - label_offset, '+0',
                       ha='center', va='top', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='black', linewidth=1.5, alpha=0.95))
            else:
                color = COLOR_POSITIVE if effect_val >= 0 else COLOR_NEGATIVE
                bottom = positions[i]
                height = abs(effect_val)

                if effect_val < 0:
                    bottom = positions[i] + effect_val

                rect = Rectangle((i - 0.3, bottom), 0.6, height,
                               facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)

                label_y_pos = min(positions[i], positions[i] + effect_val)
                sign = '+' if effect_val >= 0 else ''
                ax.text(i, label_y_pos - label_offset, f'{sign}{format_number(effect_val, 0)}',
                       ha='center', va='top', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='black', linewidth=1.5, alpha=0.95))

                prev_y = positions[i]
                ax.plot([i-1+0.3, i-0.3], [prev_y, prev_y],
                       color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)

    # Draw connector from last effect to End bar
    last_effect_idx = len(labels) - 2
    end_idx = len(labels) - 1
    ax.plot([last_effect_idx+0.3, end_idx-0.3], [current, current],
           color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(format_effect_labels(labels), rotation=45, ha='right')
    ax.set_ylabel('Bookings', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.8)

    legend_elements = [
        mpatches.Patch(facecolor=COLOR_POSITIVE, edgecolor='black', label='Positive Impact', alpha=0.9),
        mpatches.Patch(facecolor=COLOR_NEGATIVE, edgecolor='black', label='Negative Impact', alpha=0.9)
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
             fontsize=9, framealpha=0.95, edgecolor='black', ncol=3)


def _create_dimensional_waterfall(
    ax: plt.Axes,
    summary: pd.DataFrame,
    segment_detail: pd.DataFrame,
    dimension: str,
    period_1_bks: float,
    period_2_bks: float,
    title: str,
    show_legend: bool = True
) -> dict:
    """Create dimensional waterfall chart with combined volume+mix and stacked effects."""

    # Aggregate effects by dimension with volume and mix combined
    dim_agg = aggregate_by_dimension(segment_detail, dimension, combine_volume_mix=True)

    # Get effect types
    effects_dimensional = dim_agg['effect_type'].unique().tolist()
    effect_order = [
        'volume_mix_effect',
        'volume_customer_mix_effect',
        'offer_comp_mix_effect',
        'str_approval_effect', 'cond_approval_effect',
        'str_booking_effect', 'cond_booking_effect'
    ]
    effects_dimensional = [e for e in effect_order if e in effects_dimensional]

    labels = ['Start'] + effects_dimensional + ['End']
    x_pos = np.arange(len(labels))

    # Get unique dimension values and apply ordering
    dim_values_unsorted = segment_detail[dimension].unique().tolist()
    dim_values = apply_dimension_order(dimension, dim_values_unsorted)

    # Color scheme
    n_dims = len(dim_values)
    green_colors = plt.cm.Greens(np.linspace(0.4, 0.9, n_dims))
    red_colors = plt.cm.Reds(np.linspace(0.4, 0.9, n_dims))

    # Pre-calculate values for y-axis limits
    all_values = [period_1_bks, period_2_bks]
    temp_cumulative = period_1_bks

    for effect in effects_dimensional:
        effect_data = dim_agg[dim_agg['effect_type'] == effect]
        total_effect = effect_data['impact'].sum()
        all_values.append(temp_cumulative)
        temp_cumulative += total_effect
        all_values.append(temp_cumulative)

        positives = effect_data[effect_data['impact'] > 0]
        negatives = effect_data[effect_data['impact'] < 0]
        if len(positives) > 0:
            pos_cumulative = temp_cumulative - total_effect
            for _, row in positives.iterrows():
                pos_cumulative += row['impact']
                all_values.append(pos_cumulative)
        if len(negatives) > 0:
            neg_cumulative = temp_cumulative - total_effect
            for _, row in negatives.iterrows():
                neg_cumulative += row['impact']
                all_values.append(neg_cumulative)

    data_min = min(all_values)
    data_max = max(all_values)
    data_range = data_max - data_min

    y_min = data_min - (data_range * 0.50)
    y_max = data_max + (data_range * 0.20)
    y_range = y_max - y_min
    label_offset = y_range * 0.03

    breakdown_details = []

    # Plot Start bar
    ax.bar(0, period_1_bks, color=COLOR_TOTAL, edgecolor='black', linewidth=1.5, width=0.78)
    text_y_pos = (period_1_bks + y_min) / 2
    ax.text(0, text_y_pos, format_number(period_1_bks, 0),
           ha='center', va='center', fontsize=10, fontweight='bold')

    cumulative = period_1_bks

    # Plot each dimensional effect
    for i, effect in enumerate(effects_dimensional):
        x = i + 1

        effect_data = dim_agg[dim_agg['effect_type'] == effect]
        positives = effect_data[effect_data['impact'] > 0].sort_values('impact')
        negatives = effect_data[effect_data['impact'] < 0].sort_values('impact')
        total_effect = effect_data['impact'].sum()
        total_positive = positives['impact'].sum() if len(positives) > 0 else 0
        total_negative = negatives['impact'].sum() if len(negatives) > 0 else 0

        effect_breakdown = {
            'effect_type': effect,
            'total_impact': total_effect,
            'positive_contrib': total_positive,
            'negative_contrib': total_negative
        }

        for _, row in positives.iterrows():
            effect_breakdown[f"{row[dimension]}_positive"] = row['impact']
        for _, row in negatives.iterrows():
            effect_breakdown[f"{row[dimension]}_negative"] = row['impact']

        breakdown_details.append(effect_breakdown)

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
            bar_bottom = cumulative
            bar_top = cumulative

            if len(positives) > 0:
                bottom_pos = cumulative
                for _, row in positives.iterrows():
                    dim_val = row[dimension]
                    val = row['impact']
                    color_idx = dim_values.index(dim_val)

                    rect = Rectangle((x - 0.3, bottom_pos), 0.6, val,
                                   facecolor=green_colors[color_idx], edgecolor='black',
                                   linewidth=0.5, alpha=0.9)
                    ax.add_patch(rect)
                    bottom_pos += val
                    bar_top = max(bar_top, bottom_pos)

            if len(negatives) > 0:
                bottom_neg = cumulative
                for _, row in negatives.iterrows():
                    dim_val = row[dimension]
                    val = row['impact']
                    color_idx = dim_values.index(dim_val)

                    rect = Rectangle((x - 0.3, bottom_neg + val), 0.6, abs(val),
                                   facecolor=red_colors[color_idx], edgecolor='black',
                                   linewidth=0.5, alpha=0.9)
                    ax.add_patch(rect)
                    bottom_neg += val
                    bar_bottom = min(bar_bottom, bottom_neg)

            new_cumulative = cumulative + total_effect
            label_y_pos = min(bar_bottom, cumulative, new_cumulative)

            sign = '+' if total_effect >= 0 else ''
            label_text = f'{sign}{format_number(total_effect, 0)}'

            ax.text(x, label_y_pos - label_offset, label_text,
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

    # Build legend elements
    legend_elements = []
    for i, dim_val in enumerate(dim_values):
        legend_elements.append(mpatches.Patch(facecolor=green_colors[i], edgecolor='black',
                                             label=f'{dim_val} (+)', alpha=0.9))
    for i, dim_val in enumerate(dim_values):
        legend_elements.append(mpatches.Patch(facecolor=red_colors[i], edgecolor='black',
                                             label=f'{dim_val} (-)', alpha=0.9))

    legend_info = {
        'elements': legend_elements,
        'dim_values': dim_values,
        'dimension': dimension,
        'green_colors': green_colors,
        'red_colors': red_colors
    }

    if show_legend:
        ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.05),
                 fontsize=8, framealpha=0.95, edgecolor='black',
                 title=format_dimension_name(dimension),
                 ncol=len(dim_values))

    return {
        'breakdown': pd.DataFrame(breakdown_details),
        'legend_info': legend_info
    }


def print_waterfall_breakdowns(fig: plt.Figure) -> None:
    """Print detailed breakdowns from waterfall grid."""
    if not hasattr(fig, 'breakdown_details'):
        print("No breakdown details available.")
        return

    breakdowns = fig.breakdown_details

    for dimension_name, breakdown_df in breakdowns.items():
        print("\n" + "="*80)
        print(f"BREAKDOWN BY {dimension_name.upper().replace('_', ' ')}")
        print("="*80 + "\n")

        display_df = breakdown_df.copy()
        numeric_cols = display_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            display_df[col] = display_df[col].apply(lambda x: f"{x:,.1f}" if pd.notna(x) else "-")

        display_df['effect_type'] = display_df['effect_type'].apply(
            lambda x: format_effect_labels([x])[0] if isinstance(x, str) else x
        )

        print(display_df.to_string(index=False))
        print()


def create_dimension_drilldown(
    segment_detail: pd.DataFrame,
    dimension: str,
    lender: str = 'ACA',
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """Create bar charts showing effect breakdown by a specific dimension."""
    date_a = segment_detail['period_1_date'].iloc[0]
    date_b = segment_detail['period_2_date'].iloc[0]

    agg_cols = {
        'total_effect': 'sum',
        'volume_effect': 'sum',
        'customer_mix_effect': 'sum',
        'offer_comp_mix_effect': 'sum',
        'str_approval_effect': 'sum',
        'cond_approval_effect': 'sum',
        'str_booking_effect': 'sum',
        'cond_booking_effect': 'sum'
    }

    dim_agg = segment_detail.groupby(dimension).agg(agg_cols).reset_index()
    dim_agg['volume_customer_mix_effect'] = dim_agg['volume_effect'] + dim_agg['customer_mix_effect']
    dim_agg['total_net_impact'] = dim_agg['total_effect']

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    effects = [
        ('total_net_impact', 'Total Net Impact', True),
        ('volume_customer_mix_effect', 'Volume + Cust Mix Effect', True),
        ('offer_comp_mix_effect', 'Competitor Offer Mix Effect', True),
        ('str_approval_effect', 'Straight Approval Effect', True),
        ('cond_approval_effect', 'Conditional Approval Effect', True),
        ('str_booking_effect', 'Straight Booking Effect', True),
        ('cond_booking_effect', 'Conditional Booking Effect', True)
    ]
    axes[3, 1].set_visible(False)

    fig.suptitle(
        f'{lender} Booking Impact Decomposition by {format_dimension_name(dimension)}\n'
        f'{date_a} → {date_b}',
        fontsize=14, fontweight='bold', y=0.98
    )

    axes_flat = axes.flatten()

    for idx, (effect_col, title, by_dimension) in enumerate(effects):
        ax = axes_flat[idx]

        data = dim_agg[[dimension, effect_col]].copy()
        dim_values = data[dimension].unique().tolist()
        ordered_dims = apply_dimension_order(dimension, dim_values)

        data[dimension] = pd.Categorical(data[dimension], categories=ordered_dims, ordered=True)
        data = data.sort_values(dimension)

        colors = [COLOR_POSITIVE if val >= 0 else COLOR_NEGATIVE
                 for val in data[effect_col]]

        bars = ax.barh(data[dimension], data[effect_col], color=colors,
                      edgecolor='black', linewidth=1)

        max_val = data[effect_col].max()
        min_val = data[effect_col].min()
        val_range = max_val - min_val

        if abs(max_val) < 0.01 and abs(min_val) < 0.01:
            x_min = -10
            x_max = 10
        else:
            padding = max(val_range * 0.15, abs(max_val) * 0.1, abs(min_val) * 0.1, 10)
            x_min = min_val - padding if min_val < 0 else -padding * 0.3
            x_max = max_val + padding if max_val > 0 else padding * 0.3

        for i, (val, dim_val) in enumerate(zip(data[effect_col], data[dimension])):
            sign = '+' if val >= 0 else ''
            if val >= 0:
                label_x = val
                label_text = f'  {sign}{format_number(val, 0)}'
                h_align = 'left'
            else:
                offset = abs(x_max - x_min) * 0.01
                label_x = val - offset
                label_text = f'{sign}{format_number(val, 0)} '
                h_align = 'right'

            ax.text(label_x, i, label_text,
                   va='center', ha=h_align,
                   fontsize=10, fontweight='bold')

        ax.set_xlabel('Booking Impact', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.set_xlim(x_min, x_max)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path, "Drilldown chart")

    return fig
