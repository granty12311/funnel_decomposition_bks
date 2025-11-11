"""
Funnel Decomposition Analysis Package

This package provides tools for hierarchical decomposition of booking changes
across multi-stage funnels with multiple segmentation dimensions.
"""

__version__ = "1.0.0"

from .hier_decomposition_calculator import calculate_decomposition, DecompositionResults
from .visualization_engine import (
    create_waterfall_grid,
    create_dimension_drilldown,
    print_waterfall_breakdowns
)

__all__ = [
    'calculate_decomposition',
    'DecompositionResults',
    'create_waterfall_grid',
    'create_dimension_drilldown',
    'print_waterfall_breakdowns'
]
