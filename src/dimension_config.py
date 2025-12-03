"""
Dimension configuration for funnel decomposition.

Defines dimension ordering and column names for:
- customer_segment: Consumer credit segment
- offer_comp_tier: Offer competitiveness tier
"""

# Non-financed lender identifier
# Used to represent bookings from customers who pay cash (no lender financing)
# These rows only have num_tot_bks populated and are excluded from funnel decomposition
NON_FINANCED_LENDER = 'NON_FINANCED'

# Dimension value ordering
DIMENSION_ORDER = {
    'lender': ['ACA', 'ALY', 'CAP'],
    'customer_segment': [
        'Super_Prime', 'Prime', 'Near_Prime',
        'Subprime', 'Deep_Subprime', 'New_To_Credit'
    ],
    'offer_comp_tier': ['solo_offer', 'multi_best', 'multi_other']
}

# Active dimension columns
DIMENSION_COLUMNS = ['customer_segment', 'offer_comp_tier']


def get_dimension_order(dimension: str) -> list:
    """Get configured ordering for a dimension."""
    return DIMENSION_ORDER.get(dimension, [])


def apply_dimension_order(dimension: str, values: list) -> list:
    """Sort values according to configured ordering."""
    order = get_dimension_order(dimension)
    if not order:
        return sorted(values)

    values_set = set(values)
    ordered = [v for v in order if v in values_set]
    extra = sorted(values_set - set(order))
    return ordered + extra


def get_dimension_columns() -> list:
    """Get list of dimension column names."""
    return DIMENSION_COLUMNS.copy()
