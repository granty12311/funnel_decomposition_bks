"""
Dimension Configuration for Funnel Decomposition Visualizations

This module defines the ordering of dimension values used in charts and drilldowns.
Values will be displayed in the order specified below. If a value exists in the data
but not in this config, it will appear after the configured values in alphabetical order.

Usage:
    - Modify DIMENSION_ORDER to match your data's dimension values
    - Order matters: values will appear in charts in the order listed
    - Missing values (not in data) are automatically skipped
    - Extra values (in data but not in config) are appended alphabetically
"""

# ============================================================================
# DIMENSION VALUE ORDERING
# ============================================================================

DIMENSION_ORDER = {
    # Lender ordering
    # Includes both standard lenders and alternative test lenders
    'lender': [
        # Standard lenders
        'CAF1',
        'CAF2',
        'CAP',
        'ALY',
        'SAN',
        'EXE',
        'WS2',
        'CAF3',
        'ACA',
        'WST',
        # Alternative test lenders
        'BANK_A',
        'BANK_B',
        'BANK_C',
        'CREDIT_UNION_1',
        'FINTECH_CO'
    ],

    # Credit score band ordering (using alternative column name)
    # Includes both standard and alternative naming conventions
    'credit_score_band': [
        # Standard FICO bands
        'High_FICO',
        'Medium_FICO',
        'Low_FICO',
        'Null_FICO',
        # Alternative credit score bands (descriptive names)
        'Excellent',
        'Good',
        'Fair',
        'Poor'
    ],

    # Legacy support - still include old name for backward compatibility
    'fico_bands': [
        'High_FICO',
        'Medium_FICO',
        'Low_FICO',
        'Null_FICO',
        'Excellent',
        'Good',
        'Fair',
        'Poor'
    ],

    # Competitiveness tier ordering (using alternative column name)
    # Includes both standard and alternative naming conventions
    'competitiveness_tier': [
        # Standard comp tiers
        'solo_offer',
        'multi_best',
        'multi_other',
        # Alternative comp tiers (descriptive names)
        'exclusive',
        'competitive',
        'standard'
    ],

    # Legacy support - still include old name
    'offer_comp_tier': [
        'solo_offer',
        'multi_best',
        'multi_other',
        'exclusive',
        'competitive',
        'standard'
    ],

    # Product type ordering (using alternative column name)
    # Includes both standard and alternative naming conventions
    'product_type': [
        # Standard product lines
        'used',
        'vmax',
        # Alternative product types (descriptive names)
        'auto_loan',
        'personal_loan'
    ],

    # Legacy support - still include old name
    'prod_line': [
        'used',
        'vmax',
        'auto_loan',
        'personal_loan'
    ]
}


# ============================================================================
# DIMENSION COLUMN NAMES
# ============================================================================

# Define the actual column names used in your data for each dimension.
# These column names will be used throughout the codebase for calculations,
# validations, and visualizations.
#
# To use alternative column names, simply change these values:
#   Example: DIMENSION_COLUMNS = ['credit_score', 'competitiveness', 'product']
#
# The system will automatically use these names everywhere.

# Use standard names for compatibility with demo data
DIMENSION_COLUMNS = [
    'fico_bands',        # Credit score dimension
    'offer_comp_tier',   # Competitiveness dimension
    'prod_line'          # Product line dimension
]

# ALTERNATIVE: Use alternative names (comment out above and uncomment below to change)
# DIMENSION_COLUMNS = [
#     'credit_score_band',     # Credit score dimension
#     'competitiveness_tier',  # Competitiveness dimension
#     'product_type'           # Product line dimension
# ]

# Legacy support: Individual dimension name getters
# These map to the DIMENSION_COLUMNS list for backward compatibility
def get_fico_dimension_name():
    """Get the column name for the FICO/credit score dimension."""
    return DIMENSION_COLUMNS[0]

def get_comp_dimension_name():
    """Get the column name for the competitiveness dimension."""
    return DIMENSION_COLUMNS[1]

def get_product_dimension_name():
    """Get the column name for the product dimension."""
    return DIMENSION_COLUMNS[2]

# Note: 'lender' dimension is handled separately in multi-lender visualizations

# ============================================================================
# ACTIVE DIMENSIONS (DEPRECATED - use DIMENSION_COLUMNS instead)
# ============================================================================

# For backward compatibility - returns the same as DIMENSION_COLUMNS
ACTIVE_DIMENSIONS = DIMENSION_COLUMNS


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_dimension_order(dimension: str) -> list:
    """
    Get the configured ordering for a dimension.

    Parameters
    ----------
    dimension : str
        Dimension name (e.g., 'fico_bands', 'lender')

    Returns
    -------
    list
        Ordered list of dimension values, or empty list if not configured
    """
    return DIMENSION_ORDER.get(dimension, [])


def apply_dimension_order(dimension: str, values: list) -> list:
    """
    Sort dimension values according to configured ordering.

    Values in the config appear first in the specified order.
    Values not in the config appear last in alphabetical order.
    Values in the config but not in the data are skipped.

    Parameters
    ----------
    dimension : str
        Dimension name
    values : list
        List of actual values present in the data

    Returns
    -------
    list
        Sorted list of values

    Examples
    --------
    >>> apply_dimension_order('fico_bands', ['Low_FICO', 'High_FICO', 'Medium_FICO'])
    ['High_FICO', 'Medium_FICO', 'Low_FICO']

    >>> apply_dimension_order('fico_bands', ['High_FICO', 'Unknown', 'Low_FICO'])
    ['High_FICO', 'Low_FICO', 'Unknown']  # 'Unknown' not in config, appears last
    """
    config_order = get_dimension_order(dimension)

    if not config_order:
        # No ordering configured for this dimension - use alphabetical
        return sorted(values)

    # Convert to sets for efficient lookup
    values_set = set(values)
    config_set = set(config_order)

    # Values that are in both config and data, in config order
    ordered_values = [v for v in config_order if v in values_set]

    # Values in data but not in config, alphabetically sorted
    extra_values = sorted(values_set - config_set)

    return ordered_values + extra_values


def get_active_dimensions() -> list:
    """
    Get the list of active dimension column names.

    Returns
    -------
    list
        List of dimension column names to use for breakdowns
    """
    return DIMENSION_COLUMNS.copy()


def get_dimension_columns() -> list:
    """
    Get the list of dimension column names.

    Returns
    -------
    list
        List of dimension column names [fico, comp, product]
    """
    return DIMENSION_COLUMNS.copy()


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_config():
    """
    Validate the configuration for common issues.

    Prints warnings for potential problems but doesn't raise errors,
    since data may not match config exactly.
    """
    issues = []

    # Check for duplicate values in orderings
    for dimension, order in DIMENSION_ORDER.items():
        if len(order) != len(set(order)):
            duplicates = [v for v in order if order.count(v) > 1]
            issues.append(f"Dimension '{dimension}' has duplicate values: {set(duplicates)}")

    # Check that active dimensions have orderings defined
    for dimension in ACTIVE_DIMENSIONS:
        if dimension not in DIMENSION_ORDER:
            issues.append(f"Active dimension '{dimension}' has no ordering defined in DIMENSION_ORDER")

    if issues:
        print("⚠ Dimension configuration warnings:")
        for issue in issues:
            print(f"  - {issue}")

    return len(issues) == 0


# Run validation on import (non-blocking)
if __name__ != '__main__':
    validate_config()
