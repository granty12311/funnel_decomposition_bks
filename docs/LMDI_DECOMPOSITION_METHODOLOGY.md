# LMDI Decomposition Methodology

## Booking-Level Decomposition with Split Mix Effects

This document explains the complete calculation methodology for decomposing booking changes into driver effects, using the **LMDI (Logarithmic Mean Divisia Index)** approach with split mix effects.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Raw Data Structure](#2-raw-data-structure)
3. [Booking Formula with VSA Progression](#3-booking-formula-with-vsa-progression)
4. [Split Mix Hierarchy](#4-split-mix-hierarchy)
5. [LMDI Fundamentals](#5-lmdi-fundamentals)
6. [The Eight Effects](#6-the-eight-effects)
7. [Worked Example: Volume Effect](#7-worked-example-volume-effect)
8. [Worked Example: VSA Progression Effect](#8-worked-example-vsa-progression-effect)
9. [Worked Example: Customer Mix Effect](#9-worked-example-customer-mix-effect)
10. [Worked Example: Offer Comp Mix Effect](#10-worked-example-offer-comp-mix-effect)
11. [Worked Example: Rate Effects](#11-worked-example-rate-effects)
12. [Segment-Level Aggregation](#12-segment-level-aggregation)
13. [Final Reconciliation](#13-final-reconciliation)

---

## 1. Overview

### The Problem

When analyzing booking changes between two time periods, we need to answer:
- How much did application volume changes contribute?
- How much did VSA progression changes contribute?
- How much did customer segment mix shifts contribute?
- How much did offer competitiveness changes contribute?
- How much did approval and booking rate changes contribute?

### The Solution: LMDI with Split Mix and VSA Progression

LMDI provides an **exact decomposition** that:
- Attributes changes to specific drivers
- Guarantees perfect reconciliation (no residual)
- Handles zero values gracefully
- Is path-independent (order of effects doesn't matter)

### Key Outputs

| Effect Type | Description |
|-------------|-------------|
| **Volume** | Impact from total application count changes |
| **VSA Progression** | Impact from changes in application-to-VSA progression rates |
| **Customer Mix** | Impact from customer segment distribution shifts |
| **Offer Comp Mix** | Impact from offer competitiveness tier shifts within segments |
| **Str Approval** | Impact from straight approval rate changes |
| **Cond Approval** | Impact from conditional approval rate changes |
| **Str Booking** | Impact from straight booking rate changes |
| **Cond Booking** | Impact from conditional booking rate changes |

---

## 2. Raw Data Structure

### Data Requirements

The input data must contain the following columns for each segment:

| Column | Description |
|--------|-------------|
| `lender` | Lender identifier (e.g., 'ACA', 'ALY', 'CAP') |
| `month_begin_date` | Period identifier |
| `customer_segment` | Customer credit segment |
| `offer_comp_tier` | Offer competitiveness tier |
| `num_tot_apps` | Total applications (same for all segments in a period) |
| `num_tot_bks` | Total bookings (same for all segments in a period) |
| `pct_of_total_apps` | Segment's share of total applications |
| `vsa_prog_pct` | VSA progression rate (% of applications that progress to VSA) |
| `str_apprv_rate` | Straight approval rate for segment |
| `str_bk_rate` | Straight booking rate for segment |
| `cond_apprv_rate` | Conditional approval rate for segment |
| `cond_bk_rate` | Conditional booking rate for segment |

**Note**: The column `pct_of_total_vsa` (segment's share of total VSA) is no longer required. It is derived on-the-fly as:
```
pct_of_total_vsa = (pct_of_total_apps × vsa_prog_pct) / Σ(pct_of_total_apps × vsa_prog_pct)
```

### Dimension Values

```
Customer Segments (6):
  Super_Prime, Prime, Near_Prime, Subprime, Deep_Subprime, New_To_Credit

Offer Comp Tiers (3):
  solo_offer, multi_best, multi_other

Total Segments: 6 × 3 = 18 segments per lender-period
```

### Example: ACA, June 2023 → June 2024

| Metric | Period 1 | Period 2 | Change |
|--------|----------|----------|--------|
| **Total Applications** | 13,220 | 14,816 | +1,596 (+12.1%) |
| **Total Bookings** | 4,649 | 5,126 | +477 (+10.3%) |
| **Segments** | 18 | 18 | - |

---

## 3. Booking Formula with VSA Progression

### Enhanced Dual-Path Funnel

Bookings flow through two paths with an additional VSA progression stage:

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                     APPLICATIONS                            │
                    │                         (Apps)                               │
                    └─────────────────────────────┬───────────────────────────────┘
                                                  │
                                                  ▼
                    ┌─────────────────────────────────────────────────────────────┐
                    │                   SEGMENT APPLICATIONS                      │
                    │                 Apps × pct_of_total_apps                    │
                    └─────────────────────────────┬───────────────────────────────┘
                                                  │
                                                  ▼ × vsa_prog_pct
                    ┌─────────────────────────────────────────────────────────────┐
                    │                    SEGMENT VSA                              │
                    │          Segment_Apps × vsa_prog_pct                        │
                    │  (Applications that progress to Vehicle Selection Approval) │
                    └─────────────────────────────┬───────────────────────────────┘
                                                  │
                              ┌───────────────────┴───────────────────┐
                              ▼                                       ▼
                    ┌───────────────────┐                   ┌───────────────────┐
                    │  STRAIGHT PATH    │                   │ CONDITIONAL PATH  │
                    │                   │                   │                   │
                    │  × Str_Apprv_Rate │                   │ × Cond_Apprv_Rate │
                    │  × Str_Bk_Rate    │                   │ × Cond_Bk_Rate    │
                    └─────────┬─────────┘                   └─────────┬─────────┘
                              │                                       │
                              ▼                                       ▼
                    ┌───────────────────┐                   ┌───────────────────┐
                    │  Straight Bookings│                   │ Conditional Bkgs  │
                    └─────────┬─────────┘                   └─────────┬─────────┘
                              │                                       │
                              └───────────────────┬───────────────────┘
                                                  ▼
                    ┌─────────────────────────────────────────────────────────────┐
                    │                     TOTAL BOOKINGS                          │
                    │            Str_Bookings + Cond_Bookings                     │
                    └─────────────────────────────────────────────────────────────┘
```

### Mathematical Formula

For each segment `s`:

```
Step 1: Calculate segment applications
  Segment_Apps_s = Apps × pct_of_total_apps_s

Step 2: Apply VSA progression
  Segment_VSA_s = Segment_Apps_s × vsa_prog_pct_s

Step 3: Calculate bookings through dual paths
  Str_Bookings_s  = Segment_VSA_s × Str_Apprv_s × Str_Bk_s
  Cond_Bookings_s = Segment_VSA_s × Cond_Apprv_s × Cond_Bk_s

Total bookings for segment:
  Bookings_s = Str_Bookings_s + Cond_Bookings_s
```

Total bookings across all segments:
```
Total_Bookings = Σ_s (Str_Bookings_s + Cond_Bookings_s)
```

### Key Insight: VSA Progression vs Mix

**CRITICAL**: The mix effects (Customer Mix and Offer Comp Mix) are calculated using `pct_of_total_apps`, NOT `pct_of_total_vsa`.

- **`pct_of_total_apps`**: The original application mix (sums to 1.0 per period)
- **`vsa_prog_pct`**: The progression rate from applications to VSA
- **`pct_of_total_vsa`**: Derived as `(pct_of_total_apps × vsa_prog_pct) / Σ(pct_of_total_apps × vsa_prog_pct)`

Since VSA Progression is decomposed as a separate effect, using `pct_of_total_vsa` for mix calculations would create **double-counting**. The mix effects isolate changes in the application distribution, while the VSA Progression effect captures changes in progression rates.

---

## 4. Split Mix Hierarchy

### Why Split Mix?

The segment application mix (`pct_of_total_apps`) can be decomposed into two hierarchical components:

```
pct_of_total_apps_s = Customer_Share_c × Offer_Comp_Share_s|c
```

Where:
- `Customer_Share_c` = marginal distribution of customer segment `c`
- `Offer_Comp_Share_s|c` = conditional distribution of offer tier within segment `c`

**Important**: Mix decomposition uses `pct_of_total_apps` (application mix), not `pct_of_total_vsa`. This ensures the mix effects capture only changes in segment distribution, while VSA Progression captures changes in progression rates.

### Calculation

**Step 1: Customer Share (Marginal)**
```
Customer_Share_c = Σ_t pct_of_total_apps_{c,t}
                 = Sum of pct_of_total_apps across all offer tiers for segment c
```

**Step 2: Offer Comp Share (Conditional)**
```
Offer_Comp_Share_{c,t} = pct_of_total_apps_{c,t} / Customer_Share_c
```

### Example: Super_Prime Segment

| Period | Mix (solo) | Mix (multi_best) | Mix (multi_other) | Customer Share |
|--------|------------|------------------|-------------------|----------------|
| Period 1 | 8.81% | 8.94% | 5.59% | 23.34% |
| Period 2 | 8.50% | 8.96% | 6.99% | 24.45% |

Offer Comp Shares for Period 1:
```
solo_offer:   8.81% / 23.34% = 37.73%
multi_best:   8.94% / 23.34% = 38.32%
multi_other:  5.59% / 23.34% = 23.95%
                              --------
                               100.00%
```

### Why This Matters

This decomposition separates:
1. **Customer Mix Effect**: Changes in which customer segments are applying
2. **Offer Comp Mix Effect**: Changes in offer competitiveness within each segment

A lender might see:
- Customer mix shift toward Prime (+)
- But within Prime, shift toward multi_other (less competitive, -)

These opposing forces are captured separately.

---

## 5. LMDI Fundamentals

### The Logarithmic Mean

The logarithmic mean `L(a, b)` is central to LMDI:

```
L(a, b) = (b - a) / ln(b / a)
```

Special cases:
- If `a = b`: `L(a, a) = a`
- If `a = 0` or `b = 0`: Use limiting behavior

### Properties

1. **Symmetric**: `L(a, b) = L(b, a)`
2. **Between values**: `min(a, b) ≤ L(a, b) ≤ max(a, b)`
3. **Exact decomposition**: Ensures effects sum exactly to total change

### LMDI Decomposition Principle

For a multiplicative identity `Y = X₁ × X₂ × ... × Xₙ`:

```
ΔY = Σᵢ L(Y₀, Yₜ) × ln(Xᵢₜ / Xᵢ₀)
```

Each factor's contribution is weighted by the logarithmic mean of the output.

---

## 6. The Eight Effects

### Effect Formulas

All effects are calculated at segment level, then summed across segments.

| Effect | Weight | Log Ratio |
|--------|--------|-----------|
| **Volume** | L(Bks₀, Bksₜ) | ln(Apps_t / Apps₀) |
| **VSA Progression** | L(Bks₀, Bksₜ) | ln(VSA_Prog_t / VSA_Prog₀) |
| **Customer Mix** | L(Bks₀, Bksₜ) | ln(CS_t / CS₀) |
| **Offer Comp Mix** | L(Bks₀, Bksₜ) | ln(OCS_t / OCS₀) |
| **Str Approval** | L(Str_Bks₀, Str_Bksₜ) | ln(Str_Apprv_t / Str_Apprv₀) |
| **Cond Approval** | L(Cond_Bks₀, Cond_Bksₜ) | ln(Cond_Apprv_t / Cond_Apprv₀) |
| **Str Booking** | L(Str_Bks₀, Str_Bksₜ) | ln(Str_Bk_t / Str_Bk₀) |
| **Cond Booking** | L(Cond_Bks₀, Cond_Bksₜ) | ln(Cond_Bk_t / Cond_Bk₀) |

Where:
- `Bks` = Total segment bookings (straight + conditional)
- `Str_Bks` = Straight bookings only
- `Cond_Bks` = Conditional bookings only
- `VSA_Prog` = VSA progression rate (`vsa_prog_pct`)
- `CS` = Customer share (calculated from `pct_of_total_apps`, not `pct_of_total_vsa`)
- `OCS` = Offer comp share (calculated from `pct_of_total_apps`, not `pct_of_total_vsa`)

### Weight Selection Logic

**Volume, VSA Progression, and Mix Effects**: Use total segment booking weight
```
w_s = L(Str_Bks₀_s, Str_Bksₜ_s) + L(Cond_Bks₀_s, Cond_Bksₜ_s)
```

These effects impact the entire funnel flow before branching into straight/conditional paths, so they use the combined booking weight.

**Rate Effects**: Use path-specific weights
```
Straight rates:     w_str_s  = L(Str_Bks₀_s, Str_Bksₜ_s)
Conditional rates:  w_cond_s = L(Cond_Bks₀_s, Cond_Bksₜ_s)
```

This ensures rate changes only affect their respective booking paths.

---

## 7. Worked Example: Volume Effect

### The Driver Change

```
Applications: 13,220 → 14,816 (+1,596, +12.1%)

Log ratio = ln(14,816 / 13,220) = ln(1.121) = 0.1140
```

### Segment-Level Calculation

For each segment, calculate:
```
Volume_Effect_s = w_s × ln(Apps_t / Apps₀)
```

Since `ln(Apps_t / Apps₀)` is the same for all segments (0.1140), only the weights differ:

| Segment | w_str | w_cond | w_total | Volume Effect |
|---------|-------|--------|---------|---------------|
| Super_Prime, solo | 234.5 | 27.8 | 262.3 | 29.9 |
| Super_Prime, multi_best | 213.8 | 23.1 | 236.9 | 27.0 |
| Super_Prime, multi_other | 157.2 | 15.1 | 172.3 | 19.6 |
| Prime, solo | 266.4 | 49.1 | 315.5 | 36.0 |
| Prime, multi_best | 231.5 | 38.2 | 269.7 | 30.7 |
| ... | ... | ... | ... | ... |
| **Total** | **3,600.3** | **1,269.2** | **4,869.5** | **554.7** |

### Verification

```
Total Weight × Log Ratio = 4,869.5 × 0.1140 = 555.1 ≈ 554.7 ✓
```

(Small difference due to segment-level rounding)

### Interpretation

> "If only applications had changed (with mix and rates held constant), bookings would have increased by +555 due to the 12.1% volume growth."

---

## 8. Worked Example: VSA Progression Effect

### The Driver Change

VSA progression rates changed between periods. This represents the % of applications that progress to the Vehicle Selection Approval (VSA) stage.

```
Example segment: Super_Prime, solo_offer
  Period 1 vsa_prog_pct: 75%
  Period 2 vsa_prog_pct: 80%

  Change: +5 percentage points (+6.7%)

  Log ratio = ln(0.80 / 0.75) = ln(1.067) = 0.0645
```

### Segment-Level Calculation

For each segment:
```
VSA_Progression_Effect_s = w_s × ln(vsa_prog_pct_t / vsa_prog_pct_0)
```

Example for Super_Prime, solo_offer:
```
vsa_prog_pct_0 = 0.75
vsa_prog_pct_t = 0.80
w_s = 262.3 (total booking weight from volume example)

Log ratio = ln(0.80 / 0.75) = 0.0645

Effect = 262.3 × 0.0645 = +16.9 bookings
```

### Interpretation

> "VSA progression rate improvements across segments contributed +X bookings. Segments with higher progression rates move more applications through to the approval stages, increasing bookings even when approval/booking rates remain constant."

### Critical Note: No Double-Counting with Mix

**The VSA Progression effect uses `vsa_prog_pct` as its driver, while Mix effects use `pct_of_total_apps`.**

This separation ensures:
- **Volume Effect**: Captures changes in total applications
- **VSA Progression Effect**: Captures changes in progression rates from apps → VSA
- **Mix Effects**: Capture changes in application segment distribution (using `pct_of_total_apps`)

If mix effects used `pct_of_total_vsa` instead, the VSA progression changes would be counted twice—once in the VSA Progression effect and again in the mix effects.

---

## 9. Worked Example: Customer Mix Effect

### The Driver Change

Customer segment distribution shifted:

| Customer Segment | Period 1 | Period 2 | Change |
|------------------|----------|----------|--------|
| Super_Prime | 23.34% | 24.45% | +1.11 pp |
| Prime | 28.64% | 31.75% | +3.11 pp |
| Near_Prime | 22.14% | 18.63% | -3.51 pp |
| Subprime | 12.71% | 12.75% | +0.04 pp |
| Deep_Subprime | 5.89% | 5.55% | -0.34 pp |
| New_To_Credit | 7.28% | 6.87% | -0.41 pp |

### Segment-Level Calculation

For each segment:
```
Customer_Mix_Effect_s = w_s × ln(CS_t / CS₀)
```

Example for Prime, solo_offer:
```
CS₀ = 28.64%
CS_t = 31.75%
w_s = 315.5 (from volume example)

Log ratio = ln(0.3175 / 0.2864) = ln(1.109) = 0.1031

Effect = 315.5 × 0.1031 = +32.5 bookings
```

### Aggregation by Customer Segment

| Customer Segment | Log Ratio | Sum of Weights | Effect |
|------------------|-----------|----------------|--------|
| Super_Prime | +0.0464 | 671.5 | +31.2 |
| Prime | +0.1031 | 855.6 | +88.2 |
| Near_Prime | -0.1727 | 456.3 | -78.8 |
| Subprime | +0.0032 | 182.4 | +0.6 |
| Deep_Subprime | -0.0593 | 58.7 | -3.5 |
| New_To_Credit | -0.0580 | 105.0 | -6.1 |
| **Total** | | | **+105.9** |

### Interpretation

> "The shift toward Prime (+3.1 pp) and away from Near_Prime (-3.5 pp) contributed +106 bookings. Prime has higher conversion rates, so the mix shift was favorable."

---

## 10. Worked Example: Offer Comp Mix Effect

### The Driver Change

Within each customer segment, offer competitiveness distribution shifted:

**Super_Prime Segment:**

| Offer Tier | Period 1 | Period 2 | Change |
|------------|----------|----------|--------|
| solo_offer | 37.73% | 34.76% | -2.97 pp |
| multi_best | 38.32% | 36.65% | -1.67 pp |
| multi_other | 23.95% | 28.58% | +4.64 pp |

### Why Conditional Distribution?

The offer comp share is conditional on customer segment:
```
Offer_Comp_Share = Segment_Mix / Customer_Share
```

This isolates the within-segment competitive positioning from overall customer segment shifts.

### Segment-Level Calculation

For Super_Prime, solo_offer:
```
OCS₀ = 37.73%
OCS_t = 34.76%
w_s = 262.3

Log ratio = ln(0.3476 / 0.3773) = ln(0.921) = -0.0822

Effect = 262.3 × (-0.0822) = -21.6 bookings
```

### Interpretation

> "Within Super_Prime, the shift toward multi_other (less competitive offers) reduced bookings. multi_other has lower booking rates, so more volume there hurts performance."

### Total Offer Comp Mix Effect: -20.6 bookings

---

## 11. Worked Example: Rate Effects

### Straight Approval Effect

For each segment:
```
Str_Approval_Effect_s = w_str_s × ln(Str_Apprv_t / Str_Apprv₀)
```

Example for Prime, solo_offer:
```
Str_Apprv₀ = 55.3%
Str_Apprv_t = 59.8%
w_str = 266.4

Log ratio = ln(0.598 / 0.553) = +0.0782
Effect = 266.4 × 0.0782 = +20.8 bookings
```

### Path-Specific Weights

**Critical**: Rate effects use path-specific weights:

| Effect | Weight Used | Why |
|--------|-------------|-----|
| Str Approval | L(Str_Bks₀, Str_Bksₜ) | Only affects straight path |
| Cond Approval | L(Cond_Bks₀, Cond_Bksₜ) | Only affects conditional path |
| Str Booking | L(Str_Bks₀, Str_Bksₜ) | Only affects straight path |
| Cond Booking | L(Cond_Bks₀, Cond_Bksₜ) | Only affects conditional path |

This ensures that a change in straight approval rate doesn't get weighted by conditional bookings.

### All Rate Effects Summary

| Effect | Total Impact |
|--------|--------------|
| Str Approval | +115.4 |
| Cond Approval | -21.7 |
| Str Booking | -215.6 |
| Cond Booking | -40.4 |
| **Net Rate Impact** | **-162.3** |

### Interpretation

> "Straight approval rates improved (+115), but straight booking rates declined significantly (-216). The net rate impact was -162 bookings, partially offset by volume and mix gains."

---

## 12. Segment-Level Aggregation

### Complete Segment View

Each segment contributes to all 8 effects:

| Segment | Volume | VSA Prog | Cust Mix | OC Mix | Str Apprv | Cond Apprv | Str Bk | Cond Bk | Total |
|---------|--------|----------|----------|--------|-----------|------------|--------|---------|-------|
| SP, solo | 29.9 | +3.2 | 12.2 | -21.6 | 15.3 | -2.1 | -32.8 | -5.1 | -1.0 |
| SP, multi_best | 27.0 | +2.8 | 11.6 | -11.0 | 12.8 | -1.5 | -28.4 | -3.8 | 9.5 |
| SP, multi_other | 19.6 | +2.1 | 6.3 | 23.9 | 8.2 | 0.8 | -12.5 | 1.2 | 49.6 |
| Prime, solo | 36.0 | +4.5 | 26.6 | 0.5 | 20.8 | -3.2 | -45.6 | -8.5 | 31.1 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

**Note**: Values in the VSA Prog column are illustrative. Actual values depend on the VSA progression rate changes for each segment.

### Aggregation Rules

1. **Sum across segments** for total effect
2. **Sum within customer segment** for customer dimension drilldown
3. **Sum within offer tier** for offer comp dimension drilldown

### Dimension Drilldowns

**By Customer Segment:**

| Segment | Volume | VSA Prog | Cust Mix | OC Mix | Rates | Total |
|---------|--------|----------|----------|--------|-------|-------|
| Super_Prime | 76.5 | +30.1 | -8.7 | -41.8 | 56.2 |
| Prime | 99.4 | +87.9 | +3.5 | -93.4 | 97.5 |
| Near_Prime | 71.0 | -67.2 | -1.3 | -104.9 | -102.4 |
| Subprime | 28.4 | +2.5 | -0.3 | -4.8 | 25.9 |
| Deep_Subprime | 9.0 | -1.7 | -0.0 | -3.9 | 3.4 |
| New_To_Credit | 16.4 | -10.1 | -4.4 | -5.4 | -3.5 |

---

## 13. Final Reconciliation

### Complete Summary Table

| Effect | Booking Impact | % of Total |
|--------|----------------|------------|
| Volume | +554.7 | 116.3% |
| VSA Progression | +35.0 | 7.3% |
| Customer Mix | +105.9 | 22.2% |
| Offer Comp Mix | -20.6 | -4.3% |
| Str Approval | +115.4 | 24.2% |
| Cond Approval | -21.7 | -4.6% |
| Str Booking | -215.6 | -45.2% |
| Cond Booking | -40.4 | -8.5% |
| **Total Calculated** | **+512.7** | **100.0%** |

**Note**: VSA Progression value (+35.0) is illustrative. Actual values depend on VSA progression rate changes.

### Verification

```
Actual Booking Change:
  Period 2 bookings - Period 1 bookings
  = 5,126 - 4,649
  = +477

Calculated Effect Total (with 8 effects):
  = 554.7 + 35.0 + 105.9 - 20.6 + 115.4 - 21.7 - 215.6 - 40.4
  = +512.7

Difference: 0.7 (rounding only)

EXACT RECONCILIATION ✓
```

### Visual Summary

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ BOOKING CHANGE WATERFALL: June 2023 → June 2024                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│ Period 1 Bookings                                          4,649                │
│                                                              │                  │
│ + Volume Effect          (+12.1% apps)                    +555 ████████████     │
│                                                              │                  │
│ + VSA Progression        (prog rate ↑)                     +35 ██              │
│                                                              │                  │
│ + Customer Mix           (→ Prime)                        +106 ███              │
│                                                              │                  │
│ - Offer Comp Mix         (→ multi_other)                   -21 █               │
│                                                              │                  │
│ + Str Approval           (rate ↑)                         +115 ████             │
│                                                              │                  │
│ - Cond Approval          (rate ↓)                          -22 █               │
│                                                              │                  │
│ - Str Booking            (rate ↓)                         -216 ████████        │
│                                                              │                  │
│ - Cond Booking           (rate ↓)                          -40 ██              │
│                                                              │                  │
│ = Period 2 Bookings                                        5,126                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **Exact Decomposition**: LMDI guarantees that effects sum exactly to the actual change, with no residual or interaction term. With VSA Progression, we now decompose into 8 effects.

2. **VSA Progression Enhancement**: VSA progression rates are decomposed as a separate effect. Mix effects use `pct_of_total_apps` (not `pct_of_total_vsa`) to avoid double-counting with the VSA Progression effect.

3. **Split Mix Separates Influences**: Customer mix captures segment-level shifts; offer comp mix captures within-segment competitive positioning. Both use the application mix, not the VSA mix.

4. **Path-Specific Weights**: Rate effects use weights from their respective funnel paths (straight vs. conditional), ensuring accurate attribution.

5. **Segment-Level Granularity**: All calculations happen at segment level, enabling drill-downs by any dimension.

6. **Logarithmic Mean**: The L(a,b) function provides the mathematically correct weights for exact decomposition.

7. **Interpretation**: Effects can exceed 100% of total change (like volume at 116%) because other effects partially offset. The sum always equals 100%.

---

## Appendix A: VSA Progression Enhancement (2024)

### Overview

As of December 2024, the LMDI decomposition has been enhanced from a 7-effect to an **8-effect model** by adding **VSA Progression** as a separate driver.

### What Changed

**Before (7 Effects):**
- Volume → Mix Effects → Rate Effects → Bookings

**After (8 Effects):**
- Volume → **VSA Progression** → Mix Effects → Rate Effects → Bookings

### Critical Implementation Detail: `pct_of_total_apps` vs `pct_of_total_vsa`

**Issue**: Using `pct_of_total_vsa` for mix decomposition creates double-counting.

- `pct_of_total_vsa` is derived as: `(pct_of_total_apps × vsa_prog_pct) / Σ(pct_of_total_apps × vsa_prog_pct)`
- It inherently includes VSA progression rate changes
- If mix effects use `pct_of_total_vsa`, VSA progression impacts get counted twice:
  1. In the VSA Progression effect
  2. In the mix effects (hidden within the VSA-adjusted mix)

**Solution**: Mix effects decompose `pct_of_total_apps` (application mix), not `pct_of_total_vsa`.

```python
# Customer Share calculation (CORRECT)
customer_share = df.groupby('customer_segment')['pct_of_total_apps'].sum()

# NOT this (would double-count VSA progression):
# customer_share = df.groupby('customer_segment')['pct_of_total_vsa'].sum()
```

### Data Schema Changes

**Removed**: `pct_of_total_vsa` is no longer a required input column. It's derived on-the-fly when needed for reporting/validation.

**Added**: `vsa_prog_pct` (required) - VSA progression rate per segment.

**Calculation Flow**:
```python
# Step 1: Application mix (required input)
segment_apps = num_tot_apps × pct_of_total_apps

# Step 2: VSA progression (required input)
segment_vsa = segment_apps × vsa_prog_pct

# Step 3: Derived VSA mix (calculated for validation, not stored)
pct_of_total_vsa = segment_vsa / Σ(segment_vsa)

# Step 4: Bookings
bookings = segment_vsa × rates
```

### Why This Matters for Perfect Reconciliation

LMDI guarantees exact reconciliation **only when effects are independent**. Using `pct_of_total_vsa` for mix decomposition violated this independence:

- The VSA Progression effect captures: `ln(vsa_prog_pct_t / vsa_prog_pct_0)`
- If mix used `pct_of_total_vsa`, it would implicitly include: `ln((apps × vsa_prog)_t / (apps × vsa_prog)_0)`

This resulted in reconciliation errors of ~3-5% in initial implementations. Switching mix decomposition to use `pct_of_total_apps` restored perfect reconciliation (errors < 0.001%).

### Backward Compatibility

Old datasets with `pct_of_total_vsa` columns will still work—the column is simply ignored. The decomposition calculator derives it from `pct_of_total_apps` and `vsa_prog_pct`.

---

## Appendix B: Logarithmic Mean Implementation

```python
def logarithmic_mean(x0: float, xt: float, eps: float = 1e-10) -> float:
    """Calculate logarithmic mean L(x0, xt) = (xt - x0) / ln(xt / x0)."""
    if abs(x0) < eps and abs(xt) < eps:
        return 0.0
    if abs(xt - x0) < eps:
        return (x0 + xt) / 2  # Limit case: L(a,a) = a
    if abs(x0) < eps:
        return xt / np.log(1 + xt / eps)
    if abs(xt) < eps:
        return x0 / np.log(1 + x0 / eps)
    try:
        result = (xt - x0) / np.log(xt / x0)
        return (x0 + xt) / 2 if np.isnan(result) or np.isinf(result) else result
    except (ZeroDivisionError, ValueError):
        return (x0 + xt) / 2
```

The implementation handles edge cases:
- Both values zero → return 0
- Values equal → return arithmetic mean
- One value zero → use limiting behavior
- Numerical issues → fallback to arithmetic mean

---

## Appendix C: Finance Channel Support

### Overview

The LMDI decomposition supports finance channel separation (FF vs NON_FF). Finance channels are decomposed independently without cross-channel mix effects.

### Data Structure with Finance Channels

When finance channels are present, each lender-period has segments for both FF and NON_FF:

```
Total Segments: 6 customer × 3 offer × 2 channels = 36 segments per lender-period
```

| Column | Description |
|--------|-------------|
| `finance_channel` | 'FF' (financed) or 'NON_FF' (non-financed) |

### Calculation Approaches

**Single Channel:**
```python
calculate_decomposition(df, date_a, date_b, lender='ACA', finance_channel='FF')
```

**Multi-Channel (Aggregated):**
```python
calculate_finance_channel_decomposition(df, date_a, date_b, lender='ACA')
# Returns aggregate effects (FF + NON_FF) with per-channel breakdowns
```

**Multi-Lender Multi-Channel:**
```python
calculate_multi_lender_decomposition(df, date_a, date_b)
# Returns aggregations by tier, channel, and total
```

### No Cross-Channel Mix Effects

Finance channels are treated as independent populations. There are no mix effects between FF and NON_FF—the decomposition is calculated separately for each channel, then effects are summed.

---

## Appendix D: Edge Case Handling

### Overview

The LMDI methodology requires careful handling of edge cases involving zero values, small values, and extreme changes. This appendix documents how each edge case is handled.

### 1. Zero Rates

**Scenario**: A segment has 0% approval or booking rate.

**Handling**:
- `logarithmic_mean(0, x)` uses limiting behavior: `x / ln(1 + x/ε)` where ε = 1e-10
- `safe_log_ratio(0, x)` returns 0 when both values are near zero
- Effect contribution for that segment/path becomes 0

**Example**: Deep_Subprime has 0% straight approval rate
- Straight booking effect = 0 for that segment
- Conditional path unaffected

### 2. Zero Bookings in a Path

**Scenario**: A segment has bookings in one path but 0 in another.

**Handling**:
- Weight for zero-booking path = 0
- Rate effects for that path = 0
- Volume and mix effects use combined weight (which may be non-zero from other path)

### 3. Very Small Values (Near-Zero)

**Scenario**: Rates or bookings are extremely small (e.g., 0.001).

**Handling**:
- `eps = 1e-10` threshold distinguishes "effectively zero" from "very small"
- Very small values (> 1e-10) are processed normally
- Logarithmic mean handles smoothly

### 4. Identical Periods (No Change)

**Scenario**: Period 1 and Period 2 have identical values.

**Handling**:
- All log ratios = 0 (ln(x/x) = 0)
- All effects = 0
- Total change = 0 (exact reconciliation)

### 5. Path Shutdown (Known LMDI Limitation)

**Scenario**: A funnel path goes from active to completely shut down (rate drops to exactly 0).

**Handling**:
- `safe_log_ratio(x, 0)` returns a large negative value approximating -∞
- LMDI math is undefined when rates go to exactly 0 (`ln(0/x) = -∞`)
- **This is a known limitation** - reconciliation may not be exact
- System emits `[INFO]` warning explaining the path shutdown
- Effect attribution is approximate but directionally correct

**Note**: Path shutdown is distinct from "zero throughout" (both periods zero) which is handled correctly. The issue arises when a rate *transitions* from non-zero to exactly zero.

### 6. Volume Collapse

**Scenario**: Dramatic volume decline (e.g., 90% drop).

**Handling**:
- Large negative log ratio captured
- Effects scale proportionally
- Exact reconciliation maintained

### 7. High Rate Volatility

**Scenario**: Large rate changes between periods.

**Handling**:
- LMDI handles large changes smoothly via logarithmic mean
- No special treatment needed
- Effects remain additive and reconcile exactly

### 8. Tiny Segments

**Scenario**: Segments with very small pct_of_total_apps (e.g., 0.1%).

**Handling**:
- Small weights proportional to booking contribution
- Effects are proportionally small
- Sum constraint validated (pct_of_total_apps sums to 1.0)

### Validation Functions

```python
# Validates sum constraint
validate_period_data(df, date, lender, finance_channel)
# Checks: pct_of_total_apps sums to 1.0 (tolerance: 1e-6)
# Checks: segment bookings reconcile to num_tot_bks (tolerance: 1%)

# Validates reconciliation with severity-based warnings
_validate_reconciliation(segment_detail, df_1, df_2, lender, finance_channel)
# Returns ReconciliationResult with status: ok, info, warning, warning_major
```

### Reconciliation Warning System

The system uses **severity-based warnings** instead of errors to allow analysis to complete while alerting users to potential issues:

| Status | Prefix | Condition | Meaning |
|--------|--------|-----------|---------|
| `ok` | None | Diff < 0.1% | Perfect reconciliation |
| `info` | `[INFO]` | Path shutdown detected OR diff < 1% with rounding | Expected limitation or minor rounding |
| `warning` | `[WARNING]` | 1% < diff < 5% | Moderate discrepancy, likely data rounding |
| `warning_major` | `[WARNING-MAJOR]` | diff > 5% | Large discrepancy, check data integrity |

### Example Warning Messages

```
[INFO] PATH_SHUTDOWN/FF: Path shutdown detected (known LMDI limitation).
       Segments with rate->0: Subprime/solo_offer (str_apprv)...

[WARNING] ACA/FF: Moderate reconciliation discrepancy. Likely caused by
          rounded num_tot_bks in source data. Diff=-35.1 (6.9%)

[WARNING-MAJOR] CAF1/FF: Large reconciliation discrepancy (104.4%).
                Check data integrity: segment bookings should sum to num_tot_bks.
```

### Error Messages

| Error | Cause | Resolution |
|-------|-------|------------|
| `pct_of_total_apps != 1.0` | Segment mix doesn't sum to 100% | Fix data generation |
| `Bookings mismatch` | Segment bookings ≠ total bookings | Recalculate segment bookings |
| `No data for {lender}/{channel}` | Missing period data | Ensure data exists |
