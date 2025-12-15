"""
Visualization engine for funnel decomposition analysis.
"""

try:
    # Re-export from visualization_utils
    from .visualization_utils import (
        COLOR_POSITIVE, COLOR_NEGATIVE, COLOR_TOTAL, COLOR_CONNECTOR, COLOR_MARKET,
        CHANNEL_COLORS, TIER_COLORS,
        format_dimension_name, format_effect_labels, aggregate_by_dimension,
        aggregate_penetration_by_dimension, aggregate_by_finance_channel,
        aggregate_by_tier, format_channel_breakdown, format_tier_breakdown,
        detect_dimension_columns, save_figure,
    )
    # Re-export from visualization_summary
    from .visualization_summary import (
        create_waterfall_grid, create_dimension_drilldown, print_waterfall_breakdowns,
        create_channel_waterfall_grid,
    )
    # Re-export from visualization_lender
    from .visualization_lender import (
        create_lender_aggregate_waterfall, create_lender_drilldown,
        create_lender_waterfall_grid, print_lender_breakdowns,
        create_multi_lender_waterfall_grid,
        create_tier_channel_waterfall, create_tier_channel_drilldown,
        TIER_CHANNEL_COLORS,
    )
    # Re-export from visualization_penetration
    from .visualization_penetration import (
        create_penetration_waterfall_grid,
    )
except ImportError:
    from visualization_utils import (
        COLOR_POSITIVE, COLOR_NEGATIVE, COLOR_TOTAL, COLOR_CONNECTOR, COLOR_MARKET,
        CHANNEL_COLORS, TIER_COLORS,
        format_dimension_name, format_effect_labels, aggregate_by_dimension,
        aggregate_penetration_by_dimension, aggregate_by_finance_channel,
        aggregate_by_tier, format_channel_breakdown, format_tier_breakdown,
        detect_dimension_columns, save_figure,
    )
    from visualization_summary import (
        create_waterfall_grid, create_dimension_drilldown, print_waterfall_breakdowns,
        create_channel_waterfall_grid,
    )
    from visualization_lender import (
        create_lender_aggregate_waterfall, create_lender_drilldown,
        create_lender_waterfall_grid, print_lender_breakdowns,
        create_multi_lender_waterfall_grid,
        create_tier_channel_waterfall, create_tier_channel_drilldown,
        TIER_CHANNEL_COLORS,
    )
    from visualization_penetration import (
        create_penetration_waterfall_grid,
    )


__all__ = [
    'COLOR_POSITIVE', 'COLOR_NEGATIVE', 'COLOR_TOTAL', 'COLOR_CONNECTOR', 'COLOR_MARKET',
    'CHANNEL_COLORS', 'TIER_COLORS', 'TIER_CHANNEL_COLORS',
    'create_waterfall_grid', 'create_dimension_drilldown', 'print_waterfall_breakdowns',
    'create_channel_waterfall_grid',
    'create_lender_aggregate_waterfall', 'create_lender_drilldown',
    'create_lender_waterfall_grid', 'print_lender_breakdowns',
    'create_multi_lender_waterfall_grid',
    'create_tier_channel_waterfall', 'create_tier_channel_drilldown',
    'create_penetration_waterfall_grid',
    'format_dimension_name', 'format_effect_labels', 'aggregate_by_dimension',
    'aggregate_penetration_by_dimension', 'aggregate_by_finance_channel',
    'aggregate_by_tier', 'format_channel_breakdown', 'format_tier_breakdown',
    'detect_dimension_columns', 'save_figure',
]
