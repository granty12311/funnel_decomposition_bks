"""
Funnel Decomposition Analysis Package

LMDI decomposition for booking and penetration analysis.
"""

__version__ = "2.0.0"

from .lmdi_decomposition_calculator import (
    calculate_decomposition, calculate_multi_lender_decomposition,
    DecompositionResults, MultiLenderResults
)
from .lmdi_penetration_calculator import (
    calculate_penetration_decomposition, calculate_multi_lender_penetration_decomposition,
    PenetrationResults, MultiLenderPenetrationResults
)
from .visualization_engine import (
    create_waterfall_grid, create_dimension_drilldown, print_waterfall_breakdowns,
    create_penetration_waterfall_grid, create_penetration_dimension_drilldown,
    create_lender_waterfall_grid, create_lender_drilldown
)

__all__ = [
    'calculate_decomposition', 'calculate_multi_lender_decomposition',
    'DecompositionResults', 'MultiLenderResults',
    'calculate_penetration_decomposition', 'calculate_multi_lender_penetration_decomposition',
    'PenetrationResults', 'MultiLenderPenetrationResults',
    'create_waterfall_grid', 'create_dimension_drilldown', 'print_waterfall_breakdowns',
    'create_penetration_waterfall_grid', 'create_penetration_dimension_drilldown',
    'create_lender_waterfall_grid', 'create_lender_drilldown'
]
