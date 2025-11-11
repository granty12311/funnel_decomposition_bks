"""
Visualization engine for funnel decomposition analysis.

Generates waterfall charts, drilldown visualizations, and exports.
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
except ImportError:
    from utils import format_date_label, format_number, format_percentage


# Color palette
COLOR_POSITIVE = '#2ecc71'  # Green
COLOR_NEGATIVE = '#e74c3c'  # Red
COLOR_TOTAL = '#95a5a6'     # Gray
COLOR_CONNECTOR = '#34495e'  # Dark gray


def create_waterfall_grid(
    summary: pd.DataFrame,
    segment_detail: pd.DataFrame,
    lender: str = 'ACA',
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Create grid of waterfall charts showing booking decomposition.

    Creates a 2x2 grid:
    - [0, 0]: Overall aggregate waterfall
    - [0, 1]: By FICO band (stacked)
    - [1, 0]: By offer comp tier (stacked)
    - [1, 1]: By product line (stacked)

    Parameters
    ----------
    summary : pd.DataFrame
        Lender-level aggregate impacts
    segment_detail : pd.DataFrame
        Segment-level breakdown
    lender : str
        Lender name for chart titles
    output_path : str or Path, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with waterfall grid
    """
    # Extract metadata from segment_detail
    date_a = segment_detail['period_1_date'].iloc[0]
    date_b = segment_detail['period_2_date'].iloc[0]
    period_1_bks = segment_detail['period_1_segment_bookings'].sum()
    period_2_bks = segment_detail['period_2_segment_bookings'].sum()

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f'{lender} Booking Decomposition: {date_a} → {date_b}',
        fontsize=16, fontweight='bold', y=0.995
    )

    # Chart 1: Overall aggregate (top-left)
    _create_aggregate_waterfall(
        ax=axes[0, 0],
        summary=summary,
        period_1_bks=period_1_bks,
        period_2_bks=period_2_bks,
        title='Overall Aggregate'
    )

    # Chart 2: By FICO band (top-right)
    fico_breakdown = _create_dimensional_waterfall(
        ax=axes[0, 1],
        summary=summary,
        segment_detail=segment_detail,
        dimension='fico_bands',
        period_1_bks=period_1_bks,
        period_2_bks=period_2_bks,
        title='By FICO Band'
    )

    # Chart 3: By offer comp tier (bottom-left)
    comp_breakdown = _create_dimensional_waterfall(
        ax=axes[1, 0],
        summary=summary,
        segment_detail=segment_detail,
        dimension='offer_comp_tier',
        period_1_bks=period_1_bks,
        period_2_bks=period_2_bks,
        title='By Offer Comp Tier'
    )

    # Chart 4: By product line (bottom-right)
    prod_breakdown = _create_dimensional_waterfall(
        ax=axes[1, 1],
        summary=summary,
        segment_detail=segment_detail,
        dimension='prod_line',
        period_1_bks=period_1_bks,
        period_2_bks=period_2_bks,
        title='By Product Line'
    )

    # Store breakdowns in figure metadata for optional display
    fig.breakdown_details = {
        'fico_bands': fico_breakdown,
        'offer_comp_tier': comp_breakdown,
        'prod_line': prod_breakdown
    }

    plt.tight_layout()

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Waterfall grid saved to: {output_path}")

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

    # Set dynamic y-axis based on all cumulative values
    # Include all intermediate positions plus the actual bar heights
    all_values = [period_1_bks, period_2_bks]
    current = period_1_bks
    for val in effects['booking_impact']:
        all_values.append(current)  # Position before adding effect
        current += val
        all_values.append(current)  # Position after adding effect

    # Calculate y-axis limits based on data range for better legend placement
    # Calculate the range between min and max waterfall bar values
    data_min = min(all_values)
    data_max = max(all_values)
    data_range = data_max - data_min

    # Y-min: Start 50% of range below the minimum of Start/End bars
    start_end_min = min(period_1_bks, period_2_bks)
    y_min = start_end_min - (data_range * 0.50)  # 50% of range below

    # Y-max: 20% of range above the maximum value
    y_max = data_max + (data_range * 0.20)  # 20% of range above

    y_range = y_max - y_min
    label_offset = y_range * 0.03  # 3% of y-axis range for label spacing

    # Plot bars
    x_pos = np.arange(len(labels))

    for i, (label, val) in enumerate(zip(labels, values)):
        if label in ['Start', 'End']:
            # Total bars (gray)
            ax.bar(i, val, color=COLOR_TOTAL, edgecolor='black', linewidth=1.5, width=0.6)
            ax.text(i, val/2 + y_min/2, format_number(val, 0),
                   ha='center', va='center', fontsize=10, fontweight='bold')
        else:
            # Effect bars
            effect_val = effects.iloc[i-1]['booking_impact']

            # For zero-impact effects, draw connector line only
            if abs(effect_val) < 0.01:
                # Draw connector line at same height
                if i > 1:
                    prev_y = positions[i]
                    ax.plot([i-1+0.3, i-0.3], [prev_y, prev_y],
                           color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)
                # Draw a small circle marker to indicate zero impact
                ax.plot(i, positions[i], marker='o', markersize=6, color=COLOR_CONNECTOR,
                       markeredgecolor='black', markeredgewidth=1.5, zorder=3)
                # Add zero label
                ax.text(i, positions[i] - label_offset, '+0',
                       ha='center', va='top', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='black', linewidth=1.5, alpha=0.95))
            else:
                color = COLOR_POSITIVE if effect_val >= 0 else COLOR_NEGATIVE

                # Floating bar
                bottom = positions[i]
                height = abs(effect_val)

                if effect_val < 0:
                    bottom = positions[i] + effect_val

                rect = Rectangle((i - 0.3, bottom), 0.6, height,
                               facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)

                # Value label below the bar
                label_y_pos = min(positions[i], positions[i] + effect_val)
                sign = '+' if effect_val >= 0 else ''
                ax.text(i, label_y_pos - label_offset, f'{sign}{format_number(effect_val, 0)}',
                       ha='center', va='top', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='black', linewidth=1.5, alpha=0.95))

                # Draw connector to previous bar (including from Start to first effect)
                prev_y = positions[i]
                ax.plot([i-1+0.3, i-0.3], [prev_y, prev_y],
                       color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)

    # Draw connector from last effect to End bar
    # The last effect ends at 'current' which equals period_2_bks
    last_effect_idx = len(labels) - 2  # Index of last effect (before End)
    end_idx = len(labels) - 1  # Index of End bar
    ax.plot([last_effect_idx+0.3, end_idx-0.3], [current, current],
           color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(_format_effect_labels(labels), rotation=45, ha='right')
    ax.set_ylabel('Bookings', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.8)

    # Add legend inside chart area, just above the x-axis
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_POSITIVE, edgecolor='black', label='Positive Impact', alpha=0.9),
        mpatches.Patch(facecolor=COLOR_NEGATIVE, edgecolor='black', label='Negative Impact', alpha=0.9),
        mpatches.Patch(facecolor=COLOR_TOTAL, edgecolor='black', label='Start/End', alpha=0.9)
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
    title: str
) -> pd.DataFrame:
    """Create diverging stacked waterfall chart by dimension for mixed effects clarity.

    Returns DataFrame with breakdown details for display below chart.
    """

    # Aggregate effects by dimension
    dim_agg = _aggregate_by_dimension(segment_detail, dimension)

    # Get effect types (excluding total_change and interaction_effect)
    effects = summary[
        (summary['effect_type'] != 'total_change') &
        (summary['effect_type'] != 'interaction_effect')
    ]['effect_type'].tolist()

    # Prepare data
    labels = ['Start'] + effects + ['End']
    x_pos = np.arange(len(labels))

    # Get unique dimension values
    dim_values = sorted(segment_detail[dimension].unique())

    # Color scheme: shades of green for positives, shades of red for negatives
    # Create gradients for each dimension value
    n_dims = len(dim_values)
    green_colors = plt.cm.Greens(np.linspace(0.4, 0.9, n_dims))  # Light to dark green
    red_colors = plt.cm.Reds(np.linspace(0.4, 0.9, n_dims))      # Light to dark red

    # Pre-calculate all cumulative values to set proper y-axis limits
    all_values = [period_1_bks, period_2_bks]
    temp_cumulative = period_1_bks
    for effect in effects:
        effect_data = dim_agg[dim_agg['effect_type'] == effect]
        total_effect = effect_data['impact'].sum()
        all_values.append(temp_cumulative)  # Position before effect
        temp_cumulative += total_effect
        all_values.append(temp_cumulative)  # Position after effect

        # Also include individual stacked bar heights
        positives = effect_data[effect_data['impact'] > 0]
        negatives = effect_data[effect_data['impact'] < 0]
        if len(positives) > 0:
            pos_cumulative = temp_cumulative - total_effect  # Start position
            for _, row in positives.iterrows():
                pos_cumulative += row['impact']
                all_values.append(pos_cumulative)
        if len(negatives) > 0:
            neg_cumulative = temp_cumulative - total_effect  # Start position
            for _, row in negatives.iterrows():
                neg_cumulative += row['impact']
                all_values.append(neg_cumulative)

    # Calculate y-axis limits based on data range for better legend placement
    # Calculate the range between min and max waterfall bar values
    data_min = min(all_values)
    data_max = max(all_values)
    data_range = data_max - data_min

    # Y-min: Start 50% of range below the minimum of Start/End bars
    start_end_min = min(period_1_bks, period_2_bks)
    y_min = start_end_min - (data_range * 0.50)  # 50% of range below

    # Y-max: 20% of range above the maximum value
    y_max = data_max + (data_range * 0.20)  # 20% of range above

    y_range = y_max - y_min
    label_offset = y_range * 0.03  # 3% of y-axis range for label spacing

    # Store breakdown details for summary table
    breakdown_details = []

    # Plot Start bar
    ax.bar(0, period_1_bks, color=COLOR_TOTAL, edgecolor='black', linewidth=1.5, width=0.6)
    # Adjust text position for dynamic y-axis
    text_y_pos = (period_1_bks + y_min) / 2
    ax.text(0, text_y_pos, format_number(period_1_bks, 0),
           ha='center', va='center', fontsize=10, fontweight='bold')

    # Track cumulative position for connectors
    cumulative = period_1_bks

    # Plot each effect as diverging stacked bar
    for i, effect in enumerate(effects):
        x = i + 1

        # Get values for this effect by dimension
        effect_data = dim_agg[dim_agg['effect_type'] == effect]

        # Separate positive and negative contributions
        positives = effect_data[effect_data['impact'] > 0].sort_values('impact')
        negatives = effect_data[effect_data['impact'] < 0].sort_values('impact')

        total_effect = effect_data['impact'].sum()
        total_positive = positives['impact'].sum() if len(positives) > 0 else 0
        total_negative = negatives['impact'].sum() if len(negatives) > 0 else 0

        # Store breakdown for this effect
        effect_breakdown = {
            'effect_type': effect,
            'total_impact': total_effect,
            'positive_contrib': total_positive,
            'negative_contrib': total_negative
        }

        # Add individual dimension contributions
        for _, row in positives.iterrows():
            effect_breakdown[f"{row[dimension]}_positive"] = row['impact']
        for _, row in negatives.iterrows():
            effect_breakdown[f"{row[dimension]}_negative"] = row['impact']

        breakdown_details.append(effect_breakdown)

        # For zero-impact effects, draw connector line and marker only
        if abs(total_effect) < 0.01:
            # Draw connector line at same height
            ax.plot([x-1+0.3, x-0.3], [cumulative, cumulative],
                   color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)
            # Draw a small circle marker to indicate zero impact
            ax.plot(x, cumulative, marker='o', markersize=6, color=COLOR_CONNECTOR,
                   markeredgecolor='black', markeredgewidth=1.5, zorder=3)
            # Add zero label
            ax.text(x, cumulative - label_offset, '+0',
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='black', linewidth=1.5, alpha=0.95))
        else:
            # Plot positives stacking upward from baseline
            if len(positives) > 0:
                bottom_pos = cumulative
                for _, row in positives.iterrows():
                    dim_val = row[dimension]
                    val = row['impact']
                    color_idx = dim_values.index(dim_val)

                    # Green shades for positive contributions
                    rect = Rectangle((x - 0.3, bottom_pos), 0.6, val,
                                   facecolor=green_colors[color_idx], edgecolor='black',
                                   linewidth=0.5, alpha=0.9)
                    ax.add_patch(rect)

                    bottom_pos += val

            # Plot negatives stacking downward from baseline
            if len(negatives) > 0:
                bottom_neg = cumulative
                for _, row in negatives.iterrows():
                    dim_val = row[dimension]
                    val = row['impact']  # negative value
                    color_idx = dim_values.index(dim_val)

                    # Red shades for negative contributions
                    rect = Rectangle((x - 0.3, bottom_neg + val), 0.6, abs(val),
                                   facecolor=red_colors[color_idx], edgecolor='black',
                                   linewidth=0.5, alpha=0.9)
                    ax.add_patch(rect)

                    # Update bottom position for next negative bar (stack downward)
                    bottom_neg += val

            # Net effect label below the bar (after the effect is applied)
            # Position it below the ending point of this effect
            new_cumulative = cumulative + total_effect

            # Find the y-position for label (below the bar)
            # Use the lower of the two cumulative positions to place label below
            label_y_pos = min(cumulative, new_cumulative)

            sign = '+' if total_effect >= 0 else ''
            label_text = f'{sign}{format_number(total_effect, 0)}'

            ax.text(x, label_y_pos - label_offset, label_text,
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='black', linewidth=1.5, alpha=0.95))

            # Connector line from end of previous bar to start of current bar
            # Previous bar ended at 'cumulative', current bar starts at 'cumulative'
            # Draw for all effects including first one (connects from Start bar)
            ax.plot([x-1+0.3, x-0.3], [cumulative, cumulative],
                   color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)

        # Update cumulative for next bar
        cumulative += total_effect

    # Plot End bar
    ax.bar(len(labels)-1, period_2_bks, color=COLOR_TOTAL, edgecolor='black',
          linewidth=1.5, width=0.6)
    text_y_pos_end = (period_2_bks + y_min) / 2
    ax.text(len(labels)-1, text_y_pos_end, format_number(period_2_bks, 0),
           ha='center', va='center', fontsize=10, fontweight='bold')

    # Connector to end
    ax.plot([len(labels)-2+0.3, len(labels)-1-0.3], [cumulative, cumulative],
           color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(_format_effect_labels(labels), rotation=45, ha='right')
    ax.set_ylabel('Bookings', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.8)

    # Enhanced legend showing green/red scheme - positioned inside chart, spread horizontally
    legend_elements = []

    # Add dimension-specific green shades (positive)
    for i, dim_val in enumerate(dim_values):
        legend_elements.append(mpatches.Patch(facecolor=green_colors[i], edgecolor='black',
                                             label=f'{dim_val} (+)', alpha=0.9))

    # Add dimension-specific red shades (negative)
    for i, dim_val in enumerate(dim_values):
        legend_elements.append(mpatches.Patch(facecolor=red_colors[i], edgecolor='black',
                                             label=f'{dim_val} (-)', alpha=0.9))

    # Position legend inside chart area with extra clearance from x-axis
    # Using 0.05 (5%) from bottom to ensure no overlap with value labels
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.05),
             fontsize=8, framealpha=0.95, edgecolor='black',
             title=dimension.replace('_', ' ').title(),
             ncol=len(dim_values))

    # Return breakdown details as DataFrame
    return pd.DataFrame(breakdown_details)


def _aggregate_by_dimension(
    segment_detail: pd.DataFrame,
    dimension: str
) -> pd.DataFrame:
    """
    Aggregate effects by dimension.

    Parameters
    ----------
    segment_detail : pd.DataFrame
        Segment-level detail
    dimension : str
        Dimension to aggregate by

    Returns
    -------
    pd.DataFrame
        Aggregated effects by dimension
    """
    effect_cols = [
        'volume_effect', 'mix_effect', 'str_approval_effect',
        'cond_approval_effect', 'str_booking_effect', 'cond_booking_effect'
    ]

    results = []
    for effect in effect_cols:
        agg = segment_detail.groupby(dimension)[effect].sum().reset_index()
        agg['effect_type'] = effect
        agg = agg.rename(columns={effect: 'impact'})
        results.append(agg)

    return pd.concat(results, ignore_index=True)


def _format_effect_labels(labels: List[str]) -> List[str]:
    """Format effect labels for display."""
    label_map = {
        'Start': 'Start',
        'End': 'End',
        'volume_effect': 'Volume',
        'mix_effect': 'Mix',
        'str_approval_effect': 'Str Apprv',
        'cond_approval_effect': 'Cond Apprv',
        'str_booking_effect': 'Str Book',
        'cond_booking_effect': 'Cond Book'
    }
    return [label_map.get(l, l) for l in labels]


def print_waterfall_breakdowns(fig: plt.Figure) -> None:
    """
    Print detailed breakdowns from waterfall grid.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure returned from create_waterfall_grid()

    Example
    -------
    >>> fig = create_waterfall_grid(summary, segment_detail)
    >>> print_waterfall_breakdowns(fig)
    """
    if not hasattr(fig, 'breakdown_details'):
        print("No breakdown details available. This figure was not created with create_waterfall_grid().")
        return

    breakdowns = fig.breakdown_details

    for dimension_name, breakdown_df in breakdowns.items():
        print("\n" + "="*80)
        print(f"BREAKDOWN BY {dimension_name.upper().replace('_', ' ')}")
        print("="*80 + "\n")

        # Format the dataframe for display
        display_df = breakdown_df.copy()

        # Format numeric columns
        numeric_cols = display_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            display_df[col] = display_df[col].apply(lambda x: f"{x:,.1f}" if pd.notna(x) else "-")

        # Clean up effect names
        display_df['effect_type'] = display_df['effect_type'].apply(
            lambda x: _format_effect_labels([x])[0] if isinstance(x, str) else x
        )

        print(display_df.to_string(index=False))
        print()


def create_dimension_drilldown(
    segment_detail: pd.DataFrame,
    dimension: str,
    lender: str = 'ACA',
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Create bar charts showing effect breakdown by a specific dimension.

    Parameters
    ----------
    segment_detail : pd.DataFrame
        Segment-level breakdown
    dimension : str
        Dimension to drill into ('fico_bands', 'offer_comp_tier', 'prod_line')
    lender : str
        Lender name for chart titles
    output_path : str or Path, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        Multi-panel bar chart
    """
    # Extract metadata
    date_a = segment_detail['period_1_date'].iloc[0]
    date_b = segment_detail['period_2_date'].iloc[0]

    # Aggregate by dimension
    dim_agg = segment_detail.groupby(dimension).agg({
        'total_effect': 'sum',
        'volume_effect': 'sum',
        'mix_effect': 'sum',
        'str_approval_effect': 'sum',
        'cond_approval_effect': 'sum',
        'str_booking_effect': 'sum',
        'cond_booking_effect': 'sum'
    }).reset_index()

    # Create figure with 7 panels (2 cols × 4 rows)
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle(
        f'{lender} Booking Impact Decomposition by {dimension.replace("_", " ").title()}\n'
        f'{date_a} → {date_b}',
        fontsize=14, fontweight='bold', y=0.995
    )

    effects = [
        ('total_effect', 'Total Net Impact'),
        ('volume_effect', 'Volume Effect'),
        ('mix_effect', 'Mix Effect'),
        ('str_approval_effect', 'Straight Approval Effect'),
        ('cond_approval_effect', 'Conditional Approval Effect'),
        ('str_booking_effect', 'Straight Booking Effect'),
        ('cond_booking_effect', 'Conditional Booking Effect')
    ]

    # Plot each effect
    for idx, (effect_col, title) in enumerate(effects):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        # Get data
        data = dim_agg[[dimension, effect_col]].sort_values(effect_col, ascending=False)

        # Colors based on positive/negative
        colors = [COLOR_POSITIVE if val >= 0 else COLOR_NEGATIVE
                 for val in data[effect_col]]

        # Plot bars
        bars = ax.barh(data[dimension], data[effect_col], color=colors,
                      edgecolor='black', linewidth=1)

        # Calculate x-axis limits with padding for labels
        max_val = data[effect_col].max()
        min_val = data[effect_col].min()
        val_range = max_val - min_val

        # Handle case where all values are zero or very close to zero
        if abs(max_val) < 0.01 and abs(min_val) < 0.01:
            x_min = -10
            x_max = 10
        else:
            # Add 15% padding on each side for labels
            padding = max(val_range * 0.15, abs(max_val) * 0.1, abs(min_val) * 0.1, 10)
            x_min = min_val - padding if min_val < 0 else -padding * 0.3
            x_max = max_val + padding if max_val > 0 else padding * 0.3

        # Value labels
        for i, (val, dim_val) in enumerate(zip(data[effect_col], data[dimension])):
            sign = '+' if val >= 0 else ''
            # For negative values, offset label slightly from bar for readability
            if val >= 0:
                label_x = val
                label_text = f'  {sign}{format_number(val, 0)}'
                h_align = 'left'
            else:
                # Shift negative labels slightly left (1% of axis range, about one space)
                offset = abs(x_max - x_min) * 0.01
                label_x = val - offset
                label_text = f'{sign}{format_number(val, 0)} '
                h_align = 'right'

            ax.text(label_x, i, label_text,
                   va='center', ha=h_align,
                   fontsize=10, fontweight='bold')

        # Formatting
        ax.set_xlabel('Booking Impact', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.set_xlim(x_min, x_max)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Hide the last panel (blank)
    axes[3, 1].axis('off')

    plt.tight_layout()

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Drilldown chart saved to: {output_path}")

    return fig


# ============================================================================
# LENDER-LEVEL VISUALIZATION FUNCTIONS
# ============================================================================

def create_lender_aggregate_waterfall(
    lender_summaries: pd.DataFrame,
    aggregate_summary: pd.DataFrame,
    date_a: str,
    date_b: str,
    output_path: str = None
) -> plt.Figure:
    """
    Create aggregate waterfall chart with lender-level breakdown.

    Matches the style of create_waterfall_grid with stacked bars showing
    each lender's contribution.

    Parameters
    ----------
    lender_summaries : pd.DataFrame
        Lender-level summaries from MultiLenderDecompositionResults
    aggregate_summary : pd.DataFrame
        Aggregate summary from MultiLenderDecompositionResults
    date_a : str
        Period 1 date
    date_b : str
        Period 2 date
    output_path : str, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Exclude total_change from effects
    effects = aggregate_summary[aggregate_summary['effect_type'] != 'total_change'].copy()

    # Get lenders
    lenders = sorted(lender_summaries['lender'].unique())

    # Calculate period values
    period_1_bks = aggregate_summary[aggregate_summary['effect_type'] == 'total_change']['booking_impact'].iloc[0]
    period_1_bks = period_1_bks - effects['booking_impact'].sum()
    period_2_bks = period_1_bks + effects['booking_impact'].sum()

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Lender color palette (matching dimensional waterfall colors)
    lender_colors = {}
    base_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
    for i, lender in enumerate(lenders):
        lender_colors[lender] = base_colors[i % len(base_colors)]

    # Prepare labels
    labels = ['Start'] + effects['effect_type'].tolist() + ['End']
    
    # Calculate cumulative positions for waterfall
    positions = []
    current = period_1_bks
    positions.append(0)  # Start bar base
    
    for val in effects['booking_impact']:
        positions.append(current)
        current += val
    
    positions.append(0)  # End bar base

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

    start_end_min = min(period_1_bks, period_2_bks)
    y_min = start_end_min - (data_range * 0.50)
    y_max = data_max + (data_range * 0.20)
    y_range = y_max - y_min
    label_offset = y_range * 0.03

    # Plot positions
    x_pos = np.arange(len(labels))

    # Plot Start bar
    ax.bar(0, period_1_bks, color=COLOR_TOTAL, edgecolor='black', linewidth=1.5, width=0.6)
    ax.text(0, period_1_bks/2 + y_min/2, format_number(period_1_bks, 0),
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Track cumulative for connectors
    cumulative = period_1_bks

    # Plot each effect as stacked bars
    for i, (_, row) in enumerate(effects.iterrows()):
        effect_type = row['effect_type']
        total_impact = row['booking_impact']
        x_idx = i + 1

        # Get breakdown by lender for this effect
        lender_data = lender_summaries[lender_summaries['effect_type'] == effect_type].copy()
        
        # Handle zero impact specially
        if abs(total_impact) < 0.01:
            # Draw connector line
            prev_y = positions[x_idx]
            ax.plot([x_idx-1+0.3, x_idx-0.3], [prev_y, prev_y],
                   color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)
            # Draw circle marker
            ax.plot(x_idx, positions[x_idx], marker='o', markersize=6, color=COLOR_CONNECTOR,
                   markeredgecolor='black', markeredgewidth=1.5, zorder=3)
            # Add zero label
            ax.text(x_idx, positions[x_idx] - label_offset, '+0',
                   ha='center', va='top', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='black', linewidth=1.5, alpha=0.95))
            continue

        # Determine if positive or negative overall
        is_positive = total_impact >= 0

        # Sort lenders appropriately for stacking
        if is_positive:
            # For positive: stack from bottom to top
            lender_data = lender_data.sort_values('booking_impact', ascending=True)
        else:
            # For negative: stack from top to bottom  
            lender_data = lender_data.sort_values('booking_impact', ascending=False)

        # Draw the stacked bar using rectangles
        bottom = cumulative if is_positive else cumulative + total_impact
        
        for _, lender_row in lender_data.iterrows():
            lender = lender_row['lender']
            lender_impact = lender_row['booking_impact']
            
            if abs(lender_impact) < 0.01:
                continue
            
            # Get lender color with green/red tint
            base_color = lender_colors[lender]
            # Mix with green or red based on sign
            if lender_impact >= 0:
                color = base_color  # Keep original for positive
            else:
                color = base_color  # Keep original for negative
            
            height = abs(lender_impact)
            rect_bottom = bottom if lender_impact >= 0 else bottom - height
            
            rect = Rectangle((x_idx - 0.3, rect_bottom), 0.6, height,
                           facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.85)
            ax.add_patch(rect)
            
            bottom = bottom + lender_impact

        # Add total effect label below the bar
        label_y_pos = min(cumulative, cumulative + total_impact)
        sign = '+' if total_impact >= 0 else ''
        ax.text(x_idx, label_y_pos - label_offset, f'{sign}{format_number(total_impact, 0)}',
               ha='center', va='top', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='black', linewidth=1.5, alpha=0.95))

        # Draw connector line from previous position
        prev_y = cumulative
        ax.plot([x_idx-1+0.3, x_idx-0.3], [prev_y, prev_y],
               color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)

        cumulative += total_impact

    # Plot End bar
    end_idx = len(labels) - 1
    ax.bar(end_idx, period_2_bks, color=COLOR_TOTAL, edgecolor='black', linewidth=1.5, width=0.6)
    ax.text(end_idx, period_2_bks/2 + y_min/2, format_number(period_2_bks, 0),
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw connector from last effect to End
    ax.plot([end_idx-1+0.3, end_idx-0.3], [cumulative, cumulative],
           color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(_format_effect_labels(labels), rotation=45, ha='right')
    ax.set_ylabel('Bookings', fontsize=11, fontweight='bold')
    ax.set_title(f'Multi-Lender Booking Decomposition: {date_a} → {date_b}',
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.8)

    # Add legend with lenders
    legend_elements = [
        mpatches.Patch(facecolor=lender_colors[lender], edgecolor='black', 
                      label=lender, alpha=0.85) 
        for lender in lenders
    ]
    legend_elements.append(
        mpatches.Patch(facecolor=COLOR_TOTAL, edgecolor='black', 
                      label='Start/End', alpha=0.9)
    )
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
             ncol=len(legend_elements), frameon=True, fancybox=True, shadow=True,
             fontsize=9, title='Lenders', title_fontsize=10)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Lender aggregate waterfall saved to: {output_path}")

    return fig


def create_lender_drilldown(
    lender_summaries: pd.DataFrame,
    date_a: str,
    date_b: str,
    output_path: str = None
) -> plt.Figure:
    """
    Create lender drilldown showing each effect broken down by lender.

    Similar to dimension drilldown but for lenders.

    Parameters
    ----------
    lender_summaries : pd.DataFrame
        Lender-level summaries from MultiLenderDecompositionResults
    date_a : str
        Period 1 date
    date_b : str
        Period 2 date
    output_path : str, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Exclude total_change
    effects_to_plot = [
        'volume_effect',
        'mix_effect',
        'str_approval_effect',
        'cond_approval_effect',
        'str_booking_effect',
        'cond_booking_effect'
    ]

    # Create figure with subplots (3 rows x 2 cols for 6 effects)
    fig, axes = plt.subplots(3, 2, figsize=(16, 10))
    axes = axes.flatten()

    fig.suptitle(f'Multi-Lender Booking Impact Decomposition by Effect\n{date_a} → {date_b}',
                 fontsize=14, fontweight='bold', y=0.98)

    # Plot each effect
    for idx, effect_type in enumerate(effects_to_plot):
        ax = axes[idx]

        # Get data for this effect
        effect_data = lender_summaries[lender_summaries['effect_type'] == effect_type].copy()
        effect_data = effect_data.sort_values('booking_impact', ascending=True)

        # Separate positive and negative
        positive = effect_data[effect_data['booking_impact'] > 0]
        negative = effect_data[effect_data['booking_impact'] < 0]

        # Plot
        if not negative.empty:
            bars_neg = ax.barh(negative['lender'], negative['booking_impact'],
                               color='#EF5350', edgecolor='black', linewidth=0.5)
            # Add labels
            for bar in bars_neg:
                width = bar.get_width()
                if width != 0:
                    ax.text(width, bar.get_y() + bar.get_height() / 2,
                            f'{width:+.0f}', ha='right' if width < 0 else 'left',
                            va='center', fontweight='bold', fontsize=9)

        if not positive.empty:
            bars_pos = ax.barh(positive['lender'], positive['booking_impact'],
                               color='#66BB6A', edgecolor='black', linewidth=0.5)
            # Add labels
            for bar in bars_pos:
                width = bar.get_width()
                if width != 0:
                    ax.text(width, bar.get_y() + bar.get_height() / 2,
                            f'{width:+.0f}', ha='right' if width < 0 else 'left',
                            va='center', fontweight='bold', fontsize=9)

        # Calculate total for this effect
        total_impact = effect_data['booking_impact'].sum()

        # Title
        effect_label = _format_effect_labels([effect_type])[0]
        ax.set_title(f'{effect_label} (Total: {total_impact:+,.0f})',
                     fontsize=11, fontweight='bold', pad=10)

        # Formatting
        ax.set_xlabel('Booking Impact', fontsize=10, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        # Set x-axis limits with some padding
        if not effect_data.empty:
            x_max = max(abs(effect_data['booking_impact'].min()), abs(effect_data['booking_impact'].max()))
            x_max = x_max * 1.2
            ax.set_xlim(-x_max, x_max)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Lender drilldown saved to: {output_path}")

    return fig


def print_lender_breakdowns(lender_summaries: pd.DataFrame) -> None:
    """
    Print tabular breakdown of effects by lender.

    Parameters
    ----------
    lender_summaries : pd.DataFrame
        Lender-level summaries from MultiLenderDecompositionResults
    """
    # Get unique effects (exclude total_change for detailed view)
    effects = [
        'volume_effect',
        'mix_effect',
        'str_approval_effect',
        'cond_approval_effect',
        'str_booking_effect',
        'cond_booking_effect'
    ]

    lenders = sorted(lender_summaries['lender'].unique())

    print("=" * 80)
    print("LENDER-LEVEL BREAKDOWN")
    print("=" * 80)

    # Create pivot table
    pivot = lender_summaries[lender_summaries['effect_type'].isin(effects)].pivot(
        index='effect_type',
        columns='lender',
        values='booking_impact'
    )

    # Add total column
    pivot['TOTAL'] = pivot.sum(axis=1)

    # Format effect names
    pivot.index = _format_effect_labels(pivot.index.tolist())

    # Print
    print("\n" + pivot.to_string())

    # Print total_change separately
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
    """
    Create a 2-panel waterfall grid for multi-lender analysis.
    
    Left panel: Overall aggregate waterfall
    Right panel: Lender breakdown waterfall (stacked by lender)

    Parameters
    ----------
    lender_summaries : pd.DataFrame
        Lender-level summaries from MultiLenderDecompositionResults
    aggregate_summary : pd.DataFrame
        Aggregate summary from MultiLenderDecompositionResults
    metadata : dict
        Metadata from MultiLenderDecompositionResults
    output_path : str, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Create figure with 1x2 grid (optimized for 2 panels)
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))

    # Extract metadata
    date_a = metadata['date_a']
    date_b = metadata['date_b']
    period_1_bks = metadata['aggregate_period_1_bookings']
    period_2_bks = metadata['aggregate_period_2_bookings']
    lenders = metadata['lenders']

    # Main title
    fig.suptitle(f'Multi-Lender Booking Decomposition: {date_a} → {date_b}',
                 fontsize=16, fontweight='bold', y=0.96)

    # Exclude total_change and interaction_effect from effects
    effects = aggregate_summary[
        (aggregate_summary['effect_type'] != 'total_change') &
        (aggregate_summary['effect_type'] != 'interaction_effect')
    ].copy()
    
    # ========================================================================
    # LEFT PANEL: Overall Aggregate Waterfall
    # ========================================================================
    _create_aggregate_waterfall(
        ax=axes[0],
        summary=aggregate_summary,
        period_1_bks=period_1_bks,
        period_2_bks=period_2_bks,
        title='Overall Aggregate'
    )
    
    # ========================================================================
    # RIGHT PANEL: Lender Breakdown Waterfall
    # ========================================================================
    ax = axes[1]
    
    # Lender color palette
    lender_colors = {}
    base_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
    for i, lender in enumerate(lenders):
        lender_colors[lender] = base_colors[i % len(base_colors)]
    
    # Prepare labels
    labels = ['Start'] + effects['effect_type'].tolist() + ['End']
    
    # Calculate cumulative positions for waterfall
    positions = []
    current = period_1_bks
    positions.append(0)  # Start bar base
    
    for val in effects['booking_impact']:
        positions.append(current)
        current += val
    
    positions.append(0)  # End bar base
    
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
    
    start_end_min = min(period_1_bks, period_2_bks)
    y_min = start_end_min - (data_range * 0.50)
    y_max = data_max + (data_range * 0.20)
    y_range = y_max - y_min
    label_offset = y_range * 0.03
    
    # Plot positions
    x_pos = np.arange(len(labels))
    
    # Plot Start bar
    ax.bar(0, period_1_bks, color=COLOR_TOTAL, edgecolor='black', linewidth=1.5, width=0.6)
    ax.text(0, period_1_bks/2 + y_min/2, format_number(period_1_bks, 0),
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Track cumulative for connectors
    cumulative = period_1_bks
    
    # Plot each effect as stacked bars by lender
    for i, (_, row) in enumerate(effects.iterrows()):
        effect_type = row['effect_type']
        total_impact = row['booking_impact']
        x_idx = i + 1
        
        # Get breakdown by lender for this effect
        lender_data = lender_summaries[lender_summaries['effect_type'] == effect_type].copy()
        
        # Handle zero impact specially
        if abs(total_impact) < 0.01:
            # Draw connector line
            prev_y = positions[x_idx]
            ax.plot([x_idx-1+0.3, x_idx-0.3], [prev_y, prev_y],
                   color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)
            # Draw circle marker
            ax.plot(x_idx, positions[x_idx], marker='o', markersize=6, color=COLOR_CONNECTOR,
                   markeredgecolor='black', markeredgewidth=1.5, zorder=3)
            # Add zero label
            ax.text(x_idx, positions[x_idx] - label_offset, '+0',
                   ha='center', va='top', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='black', linewidth=1.5, alpha=0.95))
            continue
        
        # Determine if positive or negative overall
        is_positive = total_impact >= 0
        
        # Sort lenders for consistent stacking
        lender_data = lender_data.sort_values('booking_impact', ascending=True)
        
        # Draw the stacked bar using rectangles
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
        
        # Add total effect label below the bar
        label_y_pos = min(cumulative, cumulative + total_impact)
        sign = '+' if total_impact >= 0 else ''
        ax.text(x_idx, label_y_pos - label_offset, f'{sign}{format_number(total_impact, 0)}',
               ha='center', va='top', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='black', linewidth=1.5, alpha=0.95))
        
        # Draw connector line from previous position
        prev_y = cumulative
        ax.plot([x_idx-1+0.3, x_idx-0.3], [prev_y, prev_y],
               color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)
        
        cumulative += total_impact
    
    # Plot End bar
    end_idx = len(labels) - 1
    ax.bar(end_idx, period_2_bks, color=COLOR_TOTAL, edgecolor='black', linewidth=1.5, width=0.6)
    ax.text(end_idx, period_2_bks/2 + y_min/2, format_number(period_2_bks, 0),
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw connector from last effect to End
    ax.plot([end_idx-1+0.3, end_idx-0.3], [cumulative, cumulative],
           color=COLOR_CONNECTOR, linestyle='--', linewidth=1, alpha=0.6)
    
    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(_format_effect_labels(labels), rotation=45, ha='right')
    ax.set_ylabel('Bookings', fontsize=11, fontweight='bold')
    ax.set_title('By Lender', fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    # Add legend with lender colors
    legend_elements = [
        mpatches.Patch(facecolor=lender_colors[lender], edgecolor='black', 
                      label=lender, alpha=0.85) 
        for lender in lenders
    ]
    legend_elements.append(
        mpatches.Patch(facecolor=COLOR_TOTAL, edgecolor='black', 
                      label='Start/End', alpha=0.9)
    )
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
             ncol=len(legend_elements), frameon=True, fancybox=True, shadow=True,
             fontsize=9, title='Lenders', title_fontsize=10)
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Lender waterfall grid saved to: {output_path}")
    
    return fig
