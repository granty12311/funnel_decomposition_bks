"""
Dimension configuration for funnel decomposition.

Defines dimension ordering and column names for:
- customer_segment: Consumer credit segment
- offer_comp_tier: Offer competitiveness tier
- finance_channel: FF vs Non-FF account populations
- lender_tier: Grouped lender tiers for aggregation
"""

# Finance channel configuration
# FF and Non-FF are treated as separate populations with independent decompositions
# (no mix effects between finance channels)
FINANCE_CHANNEL_COLUMN = 'finance_channel'
FINANCE_CHANNEL_VALUES = ['FF', 'NON_FF']

# Lender tier mapping for multi-lender aggregation
# Groups lenders into tiers to reduce chart complexity
LENDER_TIERS = {
    'T1': ['CAF1'],
    'T2': ['ALY', 'CAP', 'CAF2', 'SAN', 'EXE', 'WS2'],
    'T3': ['ACA', 'WST', 'CAF3']
}

# Reverse mapping: lender -> tier (for easy lookup)
LENDER_TIER_MAP = {}
for tier, lenders in LENDER_TIERS.items():
    for lender in lenders:
        LENDER_TIER_MAP[lender] = tier

# Dimension value ordering
DIMENSION_ORDER = {
    'lender': ['CAF1', 'ALY', 'CAP', 'CAF2', 'SAN', 'EXE', 'WS2', 'ACA', 'WST', 'CAF3'],
    'lender_tier': ['T1', 'T2', 'T3'],
    'finance_channel': ['FF', 'NON_FF'],
    'customer_segment': [
        'Super_Prime', 'Prime', 'Near_Prime',
        'Subprime', 'Deep_Subprime', 'New_To_Credit'
    ],
    'offer_comp_tier': ['solo_offer', 'multi_best', 'multi_other']
}

# Active dimension columns (within a finance channel)
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


def get_lender_tier(lender: str) -> str:
    """Return tier name for a lender (T1, T2, T3, or 'Unknown')."""
    return LENDER_TIER_MAP.get(lender, 'Unknown')


def get_lender_tier_map() -> dict:
    """Return the full lender -> tier mapping dictionary."""
    return LENDER_TIER_MAP.copy()


def get_finance_channel_column() -> str:
    """Get the finance channel column name."""
    return FINANCE_CHANNEL_COLUMN


def get_finance_channel_values() -> list:
    """Get list of valid finance channel values."""
    return FINANCE_CHANNEL_VALUES.copy()
