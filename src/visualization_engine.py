"""
Visualization engine for funnel decomposition analysis.
"""

try:
    # Re-export from visualization_utils
    from .visualization_utils import (
        COLOR_POSITIVE, COLOR_NEGATIVE, COLOR_TOTAL, COLOR_CONNECTOR, COLOR_MARKET,
        format_dimension_name, format_effect_labels, aggregate_by_dimension,
        aggregate_penetration_by_dimension, detect_dimension_columns, save_figure,
    )
    # Re-export from visualization_summary
    from .visualization_summary import (
        create_waterfall_grid, create_dimension_drilldown, print_waterfall_breakdowns,
    )
    # Re-export from visualization_lender
    from .visualization_lender import (
        create_lender_aggregate_waterfall, create_lender_drilldown,
        create_lender_waterfall_grid, print_lender_breakdowns,
    )
    # Re-export from visualization_penetration
    from .visualization_penetration import (
        create_penetration_waterfall_grid,
    )
except ImportError:
    from visualization_utils import (
        COLOR_POSITIVE, COLOR_NEGATIVE, COLOR_TOTAL, COLOR_CONNECTOR, COLOR_MARKET,
        format_dimension_name, format_effect_labels, aggregate_by_dimension,
        aggregate_penetration_by_dimension, detect_dimension_columns, save_figure,
    )
    from visualization_summary import (
        create_waterfall_grid, create_dimension_drilldown, print_waterfall_breakdowns,
    )
    from visualization_lender import (
        create_lender_aggregate_waterfall, create_lender_drilldown,
        create_lender_waterfall_grid, print_lender_breakdowns,
    )
    from visualization_penetration import (
        create_penetration_waterfall_grid,
    )


__all__ = [
    'COLOR_POSITIVE', 'COLOR_NEGATIVE', 'COLOR_TOTAL', 'COLOR_CONNECTOR', 'COLOR_MARKET',
    'create_waterfall_grid', 'create_dimension_drilldown', 'print_waterfall_breakdowns',
    'create_lender_aggregate_waterfall', 'create_lender_drilldown',
    'create_lender_waterfall_grid', 'print_lender_breakdowns',
    'create_penetration_waterfall_grid',
    'format_dimension_name', 'format_effect_labels', 'aggregate_by_dimension',
    'aggregate_penetration_by_dimension', 'detect_dimension_columns', 'save_figure',
]
