# Penetration Decomposition Methodology

## Self-Adjusted LMDI Approach

This document explains the complete calculation methodology for decomposing penetration changes into driver effects, using the **Self-Adjusted LMDI** approach.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Non-Financed Accounts](#2-non-financed-accounts)
3. [Raw Data](#3-raw-data)
4. [Penetration Definition](#4-penetration-definition)
5. [LMDI Decomposition](#5-lmdi-decomposition)
6. [Booking-Level Calculations](#6-booking-level-calculations)
7. [Translation to Penetration (BPS)](#7-translation-to-penetration-bps)
8. [Self-Adjustment Allocation](#8-self-adjustment-allocation)
9. [Volume Effect: Complete Walkthrough](#9-volume-effect-complete-walkthrough)
10. [Final Reconciliation](#10-final-reconciliation)

---

## 1. Overview

### The Problem

When analyzing market share (penetration) changes, we need to answer:
- How much did our own actions contribute?
- How much did competitor actions contribute?
- How do we avoid "self-influence" where our own growth appears as market headwind?

### The Solution: Self-Adjusted Approach

```
Penetration = Lender_Bookings / Total_Market

When lender grows by +X bookings:
  - Numerator increases by +X  → helps penetration
  - Denominator ALSO increases by +X → hurts penetration

Self-adjustment captures this denominator impact from own growth.
```

### Key Outputs

| Effect Type | Description |
|-------------|-------------|
| **Gross Lender** | Numerator impact (before self-adjustment) |
| **Self-Adjustment** | Denominator impact from lender's own growth |
| **Net Lender** | Gross + Self-Adjustment = true lender impact |
| **Competitor** | Pure rest-of-market impact |
| **Net Effect** | Net Lender + Competitor = total penetration change |

---

## 2. Non-Financed Accounts

### What Are Non-Financed Accounts?

Non-financed accounts represent customers who complete a purchase without using lender financing (e.g., cash buyers). These transactions:
- **Have bookings** but no application funnel data
- **Contribute to total market** denominator for penetration calculation
- **Cannot be decomposed** into driver effects (no volume, mix, or rate data)

### Data Format

Non-financed bookings are stored as a special lender row:

```
lender='NON_FINANCED', month_begin_date, num_tot_bks=X
```

Other fields (num_tot_apps, rates, etc.) are `NaN` or `0`.

### Impact on Penetration Calculation

**Non-financed is in the denominator for ALL effects:**

```
Penetration = Lender_Bookings / Total_Market

Where:
  Total_Market = All_Financed_Lenders + NON_FINANCED
               = ACA + ALY + CAP + NON_FINANCED
```

This affects all lender effects because they are calculated as impacts on penetration, which uses the full market denominator.

### Treatment in Decomposition

| Component | Treatment |
|-----------|-----------|
| **Total Market** | Includes non-financed bookings |
| **Lender Effects** | Calculated against full market denominator (including non-financed) |
| **Self-Adjustment** | Based on lender's share of total market change (including non-financed) |
| **Competitor/Market Effect** | Includes impact from financed competitors AND non-financed |
| **Competitor Decomposition** | Only decomposes financed competitors (non-financed has no funnel data) |

### Example with Non-Financed

```
Period 1:
  ACA Bookings:      4,649
  Other Financed:    5,288
  Non-Financed:      1,100
  Total Market:     11,037  ← includes non-financed

Period 2:
  ACA Bookings:      5,126
  Other Financed:    5,882
  Non-Financed:      1,500
  Total Market:     12,508  ← includes non-financed

ACA Penetration:
  Period 1: 4,649 / 11,037 = 42.12%
  Period 2: 5,126 / 12,508 = 40.98%
```

The non-financed growth (+400 bookings) contributes to the market headwind that reduces ACA's penetration.

### Metadata Output

The calculator provides detailed non-financed metrics:

```python
metadata = {
    'period_1_non_financed_bookings': 1100,
    'period_2_non_financed_bookings': 1500,
    'delta_non_financed_bookings': 400,
    'period_1_financed_competitors_bookings': 5288,
    'period_2_financed_competitors_bookings': 5882,
    ...
}
```

---

## 3. Raw Data

### Example: ACA, June 2023 → June 2024

| Metric | Period 1 | Period 2 | Change |
|--------|----------|----------|--------|
| **ACA Bookings** | 4,649 | 5,126 | +477 |
| **Total Market** | 11,037 | 12,508 | +1,471 |
| **Rest of Market** | 6,388 | 7,382 | +994 |
| **ACA Applications** | 13,220 | 14,816 | +1,596 |

### Market Growth Attribution

```
Total Market Growth: +1,471 bookings

  ACA contributed:         +477 (32.4%)
  Competitors contributed: +994 (67.6%)
```

---

## 3. Penetration Definition

```
Penetration = Lender_Bookings / Total_Market
```

### Calculation

| Period | Calculation | Penetration |
|--------|-------------|-------------|
| Period 1 | 4,649 / 11,037 | 42.12% |
| Period 2 | 5,126 / 12,508 | 40.98% |
| **Change** | | **-114.0 bps** |

> **Note**: Despite ACA growing bookings by +477, penetration FELL because competitors grew even more (+994).

---

## 4. LMDI Decomposition

### The LMDI Formula

LMDI (Logarithmic Mean Divisia Index) decomposes penetration change as:

```
ΔP = L(P) × [ln(L₂/L₁) - ln(M₂/M₁)]
   = L(P) × ln(L₂/L₁)  +  [-L(P) × ln(M₂/M₁)]
   = Gross Lender Effect  +  Total Market Effect
```

Where:
- `P` = Penetration
- `L` = Lender bookings
- `M` = Total market bookings
- `L(P)` = Logarithmic mean of penetration

### Logarithmic Mean

```
L(a, b) = (b - a) / ln(b/a)
```

For penetration:
```
L(P) = L(0.4212, 0.4098)
     = (0.4098 - 0.4212) / ln(0.4098/0.4212)
     = -0.0114 / -0.0274
     = 0.4155
```

### LMDI Calculation

| Component | Formula | Calculation | Result |
|-----------|---------|-------------|--------|
| **Gross Lender Effect** | L(P) × ln(L₂/L₁) | 0.4155 × ln(5126/4649) | +405.8 bps |
| **Total Market Effect** | -L(P) × ln(M₂/M₁) | -0.4155 × ln(12508/11037) | -519.8 bps |
| **Total** | | | **-114.0 bps** ✓ |

---

## 5. Booking-Level Calculations

### Booking Formula

```
Bookings = Apps × Mix × Str_Apprv × Str_Bk + Apps × Mix × Cond_Apprv × Cond_Bk
```

Where:
- `Apps` = Total applications
- `Mix` = Segment share (pct_of_total_apps)
- `Str_Apprv` = Straight approval rate
- `Str_Bk` = Straight booking rate
- `Cond_Apprv` = Conditional approval rate (independent, not on remainder)
- `Cond_Bk` = Conditional booking rate

### Seven Driver Effects (Booking Terms)

Using LMDI at segment level, we decompose booking changes into:

| Effect | Driver | Formula |
|--------|--------|---------|
| **Volume** | Applications | w_s × ln(apps₂/apps₁) |
| **Customer Mix** | Customer segment share | w_s × ln(cs₂/cs₁) |
| **Offer Comp Mix** | Offer competition share | w_s × ln(ocs₂/ocs₁) |
| **Str Approval** | Straight approval rate | w_str_s × ln(str_app₂/str_app₁) |
| **Cond Approval** | Conditional approval rate | w_cond_s × ln(cond_app₂/cond_app₁) |
| **Str Booking** | Straight booking rate | w_str_s × ln(str_bk₂/str_bk₁) |
| **Cond Booking** | Conditional booking rate | w_cond_s × ln(cond_bk₂/cond_bk₁) |

Where weights are calculated **at segment level**:
- `w_s` = L(str_bks_s₀, str_bks_sₜ) + L(cond_bks_s₀, cond_bks_sₜ)
- `w_str_s` = L(str_bks_s₀, str_bks_sₜ)
- `w_cond_s` = L(cond_bks_s₀, cond_bks_sₜ)

Effects are calculated per segment, then summed: `Effect = Σ_s (w_s × ln_ratio_s)`

### Volume Effect: Booking Calculation

```
Applications: 13,220 → 14,816 (+1,596, +12.1%)
Log ratio = ln(14816/13220) = 0.1140 (same for all segments)

Volume Effect = Σ_s [w_s × ln(apps₂/apps₁)]
              = (Σ_s w_s) × 0.1140
              = 4,869.5 × 0.1140
              = +554.7 bookings
```

> **Note**: The sum of segment-level logarithmic means (4,869.5) differs from
> the logarithmic mean of totals L(4649, 5126) = 4,883.6 because logarithmic
> means are NOT additive. The segment-level approach is used for exact LMDI
> decomposition.

---

## 6. Translation to Penetration (BPS)

### The Core Insight: Proportional Scaling

We have two parallel decompositions that must align:

| Level | What We Know | Total |
|-------|--------------|-------|
| **Penetration (Section 4)** | Gross Lender Effect from LMDI | +405.8 bps |
| **Bookings (Section 5)** | 7 booking effects from segment-level LMDI | +477.7 bookings |

**Key insight**: Each booking effect's share of the total booking change equals its share of the total bps effect.

```
Scale Factor = Total BPS / Total Bookings
             = 405.8 bps / 477.7 bookings
             = 0.8495 bps per booking
```

For any effect:
```
Effect_bps = Effect_bookings × Scale Factor
           = Effect_bookings × (405.8 / 477.7)
```

Or equivalently:
```
Effect_bps = Total_bps × (Effect_bookings / Total_bookings)
           = 405.8 × (Effect's share of booking change)
```

### Example: Volume Effect

```
Volume booking effect:  +554.7 bookings
Volume's share:         554.7 / 477.7 = 116.1%
Volume bps:             405.8 × 116.1% = +471.3 bps
```

> **Note**: Volume's share exceeds 100% because other effects (like Str Booking at -215.6)
> are negative. The shares sum to 100%, but individual effects can exceed 100% or be negative.

### All Effects Scaled

| Effect | Booking Impact | Share of Total | Scaled to BPS |
|--------|----------------|----------------|---------------|
| Volume | +554.7 | 116.1% | +471.3 bps |
| Customer Mix | +105.9 | 22.2% | +90.0 bps |
| Offer Comp Mix | -20.6 | -4.3% | -17.5 bps |
| Str Approval | +115.4 | 24.2% | +98.0 bps |
| Cond Approval | -21.7 | -4.5% | -18.5 bps |
| Str Booking | -215.6 | -45.1% | -183.2 bps |
| Cond Booking | -40.4 | -8.5% | -34.3 bps |
| **Total** | **+477.7** | **100.0%** | **+405.8 bps** |

---

## 7. Self-Adjustment Allocation

### The Same Proportional Logic

Just as we scaled booking effects to bps using proportional shares (Section 6), we allocate self-adjustment using the **same shares**.

### Step 1: Calculate Total Self-Adjustment

The Total Market Effect (-519.8 bps) is split based on who caused market growth:

```
Market Growth: +1,471 bookings

ACA contributed:        +477 bookings (32.4%)
Competitors contributed: +994 bookings (67.6%)
```

```
Self-Adjustment = Market Effect × ACA's share
                = -519.8 × 32.4%
                = -168.6 bps

Competitor Effect = Market Effect × Competitor's share
                  = -519.8 × 67.6%
                  = -351.3 bps
```

### Step 2: Allocate Self-Adjustment to Each Effect

**Key insight**: Use the same shares from Section 6. Each effect's share of gross bps = its share of self-adjustment.

```
Self_Adj_i = Total_Self_Adj × (Effect's share)
           = -168.6 × (Effect_bps / Total_bps)
           = -168.6 × (Effect_bookings / Total_bookings)  ← Same ratio!
```

### Example: Volume Self-Adjustment

```
Volume's share (from Section 6):  116.1%
Volume self-adjustment:           -168.6 × 116.1% = -195.8 bps
```

### All Effects with Self-Adjustment

| Effect | Share (from §6) | Gross (bps) | Self-Adj (bps) | Net Lender (bps) |
|--------|-----------------|-------------|----------------|------------------|
| Volume | 116.1% | +471.3 | -195.8 | +275.5 |
| Customer Mix | 22.2% | +90.0 | -37.4 | +52.6 |
| Offer Comp Mix | -4.3% | -17.5 | +7.3 | -10.2 |
| Str Approval | 24.2% | +98.0 | -40.7 | +57.3 |
| Cond Approval | -4.5% | -18.5 | +7.7 | -10.8 |
| Str Booking | -45.1% | -183.2 | +76.1 | -107.1 |
| Cond Booking | -8.5% | -34.3 | +14.3 | -20.1 |
| **Total** | **100.0%** | **+405.8** | **-168.6** | **+237.3** |

### Sign Logic

| Gross Effect | Self-Adjustment | Reason |
|--------------|-----------------|--------|
| Positive | Negative | Grew bookings → grew denominator |
| Negative | Positive | Hurt bookings → reduced denominator growth |

---

## 8. Volume Effect: Complete Walkthrough

### Step 1: Application Growth

```
Applications: 13,220 → 14,816
Change: +1,596 (+12.1%)
Log ratio: ln(14816/13220) = 0.1140
```

### Step 2: Segment-Level Booking Impact

The volume effect is calculated at segment level using segment-specific weights:

```
For each segment s:
  w_s = L(str_bks_s₀, str_bks_sₜ) + L(cond_bks_s₀, cond_bks_sₜ)
  Volume_Effect_s = w_s × ln(apps₂/apps₁)

Example segments:
  Super_Prime, solo:    w = 262.3,  effect = 262.3 × 0.1140 = +29.9
  Prime, solo:          w = 315.5,  effect = 315.5 × 0.1140 = +36.0
  ...

Total: Σ w_s = 4,869.5
Volume Effect = 4,869.5 × 0.1140 = +554.7 bookings
```

### Step 3: The Concept - Dual Impact

When ACA gains bookings from volume, the same bookings affect BOTH:
- **Numerator** (ACA bookings) → helps penetration
- **Denominator** (total market) → hurts penetration

```
Starting Point:
  Penetration = 4,649 / 11,037 = 42.12%

After +555 bookings from volume:
  New Numerator:   4,649 + 555 = 5,204   ← ACA grew
  New Denominator: 11,037 + 555 = 11,592 ← Market ALSO grew (same bookings!)
```

This is WHY we need self-adjustment: to separate the numerator benefit from the denominator cost.

### Step 4: Apply Proportional Scaling (from Sections 6-7)

Using the proportional logic from Sections 6 and 7:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ VOLUME EFFECT: APPLYING PROPORTIONAL SHARES                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ Given:                                                                  │
│   Volume booking effect:     +554.7 bookings                            │
│   Total booking effects:     +477.7 bookings                            │
│   Volume's share:            554.7 / 477.7 = 116.1%                     │
│                                                                         │
│ Apply to Gross Lender (Section 6):                                      │
│   Gross Volume = +405.8 bps × 116.1% = +471.3 bps                       │
│                                                                         │
│ Apply to Self-Adjustment (Section 7):                                   │
│   Volume Self-Adj = -168.6 bps × 116.1% = -195.8 bps                    │
│                                                                         │
│ Net Lender Volume:                                                      │
│   +471.3 + (-195.8) = +275.5 bps                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Step 5: Add Competitor Effect

Competitors' volume effect is calculated separately using the same LMDI methodology on rest-of-market data:

```
Net Lender Volume:      +275.5 bps  (ACA's true volume impact)
Competitor Volume:      -253.5 bps  (rest-of-market volume impact)
───────────────────────────────────
Total Net Volume:        +22.1 bps  (market-wide volume effect on penetration)
```

### Step 6: The Ratio

```
Gross : Self-Adj = 471.3 : 195.8 = 2.4 : 1

For every 2.4 bps of numerator gain from volume,
1.0 bps is offset by denominator growth from the same bookings.
```

### Visual Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ VOLUME EFFECT FLOW                                                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│ 1. DRIVER CHANGE                                                                │
│    Applications: 13,220 → 14,816 (+1,596, +12.1%)                               │
│                                     │                                           │
│                                     ▼                                           │
│ 2. BOOKING IMPACT (segment-level weights)                                       │
│    Volume booking effect: +555 bookings                                         │
│                                     │                                           │
│                    ┌────────────────┴────────────────┐                          │
│                    ▼                                 ▼                          │
│ 3. NUMERATOR IMPACT                    4. DENOMINATOR IMPACT                    │
│    +555 to ACA bookings                   +555 to market total                  │
│    = +471.3 bps (gross)                   = -195.8 bps (self-adj)               │
│                    │                                 │                          │
│                    └────────────────┬────────────────┘                          │
│                                     ▼                                           │
│ 5. NET LENDER VOLUME EFFECT                                                     │
│    +471.3 + (-195.8) = +275.5 bps                                               │
│                                     │                                           │
│                                     ▼                                           │
│ 6. ADD COMPETITOR VOLUME                                                        │
│    +275.5 + (-253.5) = +22.1 bps (total net volume)                             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Final Reconciliation

### Complete Summary Table (All Values in BPS)

```
┌──────────────────┬─────────┬────────────┬──────────┬────────────┬────────────┬────────────┐
│                  │  Share  │   Gross    │  Self-   │    Net     │            │    Net     │
│ Effect           │    %    │   Lender   │   Adj    │   Lender   │ Competitor │   Effect   │
├──────────────────┼─────────┼────────────┼──────────┼────────────┼────────────┼────────────┤
│ Volume           │  116.1% │    +471.3  │  -195.8  │    +275.5  │    -253.5  │     +22.1  │
│ Customer Mix     │   22.2% │     +90.0  │   -37.4  │     +52.6  │     -14.3  │     +38.3  │
│ Offer Comp Mix   │   -4.3% │     -17.5  │    +7.3  │     -10.2  │      -1.8  │     -12.1  │
│ Str Approval     │   24.2% │     +98.0  │   -40.7  │     +57.3  │     -32.3  │     +25.0  │
│ Cond Approval    │   -4.5% │     -18.5  │    +7.7  │     -10.8  │     -14.0  │     -24.8  │
│ Str Booking      │  -45.1% │    -183.2  │   +76.1  │    -107.1  │     -26.8  │    -133.9  │
│ Cond Booking     │   -8.5% │     -34.3  │   +14.3  │     -20.1  │      -8.6  │     -28.7  │
├──────────────────┼─────────┼────────────┼──────────┼────────────┼────────────┼────────────┤
│ TOTAL            │  100.0% │    +405.8  │  -168.6  │    +237.3  │    -351.3  │    -114.0  │
└──────────────────┴─────────┴────────────┴──────────┴────────────┴────────────┴────────────┘

Column Relationships:
  Share %     = Effect_bookings / Total_bookings (from Section 6)
  Gross       = +405.8 × Share %
  Self-Adj    = -168.6 × Share %
  Net Lender  = Gross + Self-Adj
  Net Effect  = Net Lender + Competitor
```

### Verification

```
Step 1: Gross Lender + Self-Adjustment = Net Lender
        +405.8 + (-168.6) = +237.3 bps ✓

Step 2: Net Lender + Competitor = Net Effect
        +237.3 + (-351.3) = -114.0 bps ✓

Step 3: Net Effect = Actual Penetration Change
        -114.0 bps = (40.98% - 42.12%) × 10000 ✓

EXACT RECONCILIATION - NO RESIDUAL
```

---

## Key Takeaways

1. **Self-Adjustment is Essential**: Without it, lender growth appears as both a positive (numerator) and negative (market headwind), double-counting the impact.

2. **Proportional Allocation**: Each effect's self-adjustment is proportional to its contribution to lender booking growth.

3. **Sign Matters**: Positive gross effects get negative self-adjustments (grew denominator); negative gross effects get positive self-adjustments (reduced denominator growth).

4. **Exact Reconciliation**: Net Lender + Competitor = Actual Penetration Change, with no residual.

5. **Competitor is Pure**: Competitor effects come only from rest-of-market changes, no self-influence.

---

## Appendix A: Edge Case Handling

### Overview

The penetration decomposition inherits edge case handling from the LMDI booking decomposition, with additional considerations for market-level calculations.

### 1. Zero or Very Small Lender Bookings

**Scenario**: Lender has zero or very few bookings in one period.

**Handling**:
- `logarithmic_mean(0, x)` uses limiting behavior
- Penetration calculation proceeds normally
- Effects proportionally allocated

### 2. No Market Growth (Delta Market ≈ 0)

**Scenario**: Total market bookings are unchanged between periods.

**Handling**:
```python
if abs(delta_market) > 1e-10:
    self_adj_share = delta_lender_bks / delta_market
    competitor_share = delta_rest_of_market / delta_market
else:
    # No market change - split based on levels
    self_adj_share = bks_1 / total_market_1
    competitor_share = rest_of_market_1 / total_market_1
```

### 3. NON_FINANCED Lender in Input

**Scenario**: User attempts to calculate decomposition for NON_FINANCED.

**Handling**:
```python
if is_non_financed_lender(lender):
    raise ValueError("Cannot calculate decomposition for NON_FINANCED - no funnel data available")
```

### 4. Missing NON_FINANCED Data

**Scenario**: Dataset doesn't include NON_FINANCED rows.

**Handling**:
- `nf_bks_1 = 0` and `nf_bks_2 = 0`
- Total market includes only financed lenders
- Calculation proceeds normally

### 5. Finance Channel Aggregation

**Scenario**: Dataset has finance channels that need aggregation for penetration.

**Handling**:
- `_aggregate_across_channels()` sums segment data across channels
- Rates are recalculated from aggregated counts
- Validates rate bounds [0, 1]

### 6. Identical Periods

**Scenario**: All values identical between periods.

**Handling**:
- Delta penetration = 0
- All effects = 0
- Exact reconciliation maintained

### 7. Extreme Penetration Changes

**Scenario**: Large penetration swings (e.g., 50% → 10%).

**Handling**:
- LMDI logarithmic mean handles smoothly
- No special treatment needed
- Exact reconciliation maintained

### Validation Checks

```python
# Market validation
if total_market_1 == 0 or total_market_2 == 0:
    raise ValueError("No market bookings found")

# Lender data validation
if len(df_1) == 0 or len(df_2) == 0:
    raise ValueError(f"No data for {lender}")

# Reconciliation validation (tolerance: 1e-8)
if not np.isclose(sum_net, delta_pen, atol=1e-8):
    warnings.warn(f"Total reconciliation error: {abs(sum_net - delta_pen):.10f}")
```

### Path Shutdown in Penetration

Penetration decomposition inherits the LMDI booking decomposition, including its path shutdown limitation:

- If a lender or competitor experiences a path shutdown (rate → 0), the underlying booking decomposition may have reconciliation issues
- These issues propagate to the penetration effects
- The system emits appropriate `[INFO]` warnings when path shutdowns are detected
- Effects remain directionally correct even if not perfectly reconciled

---

## Appendix B: Finance Channel Support

### Aggregation for Penetration

Penetration analysis requires lender-level totals, not channel-level. When finance channels are present:

1. **Aggregate Across Channels**: Sum segment data across FF and NON_FF
2. **Recalculate Rates**: Derive rates from aggregated counts
3. **Validate**: Ensure rates are within [0, 1]

```python
# Aggregation function
agg = _aggregate_across_channels(df, date_column='month_begin_date')

# Rate recalculation from counts
agg['str_apprv_rate'] = agg['str_approvals'] / agg['segment_apps']
agg['str_bk_rate'] = agg['str_bookings'] / agg['str_approvals']
# ... etc
```

### Market Definition

Total market includes all lenders:
```python
Total_Market = Σ (Financed Lenders) + NON_FINANCED
             = ACA + ALY + CAP + ... + NON_FINANCED
```

For competitor effects:
```python
Rest_of_Market = Total_Market - Lender
Financed_Competitors = Rest_of_Market - NON_FINANCED
```

### Competitor Decomposition

Only financed competitors are decomposed (NON_FINANCED has no funnel data):
```python
# Excludes specified lender AND NON_FINANCED
df_competitors = df[~df['lender'].apply(is_non_financed_lender)]
df_competitors = df_competitors[df_competitors['lender'] != exclude_lender]
```

---

## Appendix C: Multi-Lender Penetration Analysis

### Overview

The `calculate_multi_lender_penetration_decomposition()` function calculates penetration decomposition for multiple lenders in a single call.

### Usage

```python
from lmdi_penetration_calculator import calculate_multi_lender_penetration_decomposition

results = calculate_multi_lender_penetration_decomposition(
    df=df,
    date_a='2024-01-01',
    date_b='2024-02-01',
    lenders=['ACA', 'ALY', 'CAP'],  # Optional, auto-detects if None
    date_column='month_begin_date'
)
```

### Output Structure

```python
MultiLenderPenetrationResults(
    lender_summaries,    # DataFrame with all lender effects
    aggregate_summary,   # DataFrame with per-lender totals
    lender_details,      # Dict of {lender: PenetrationResults}
    metadata            # Analysis metadata
)
```

### NON_FINANCED Handling

NON_FINANCED is automatically excluded from lender iteration but included in total market:
```python
lenders = [l for l in all_lenders if not is_non_financed_lender(l)]
```
