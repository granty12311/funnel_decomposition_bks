# Dataset Schema: Channel Funnel Data

This document describes the schema and constraints for the channel funnel decomposition dataset.

---

## Overview

The dataset captures lending funnel metrics at the segment level, enabling LMDI decomposition analysis of booking changes. Each row represents a unique combination of:
- Lender
- Finance Channel (FF / NON_FF)
- Customer Segment
- Offer Competitiveness Tier
- Time Period

---

## Funnel Structure

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                     APPLICATIONS                            │
                    │                      (num_tot_apps)                          │
                    └─────────────────────────────┬───────────────────────────────┘
                                                  │
                                                  ▼ × pct_of_total_apps
                    ┌─────────────────────────────────────────────────────────────┐
                    │                   SEGMENT APPLICATIONS                      │
                    │                      (num_seg_apps)                          │
                    │           = num_tot_apps × pct_of_total_apps                │
                    └─────────────────────────────┬───────────────────────────────┘
                                                  │
                                                  ▼ × vsa_prog_pct
                    ┌─────────────────────────────────────────────────────────────┐
                    │                    SEGMENT VSA                              │
                    │                     (num_seg_vsa)                            │
                    │  Applications that progress to Vehicle Selection Approval   │
                    │              = num_seg_apps × vsa_prog_pct                  │
                    └─────────────────────────────┬───────────────────────────────┘
                                                  │
                              ┌───────────────────┼───────────────────┐
                              ▼                   ▼                   ▼
                    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
                    │  STRAIGHT PATH  │ │ CONDITIONAL PATH│ │    DECLINED     │
                    │                 │ │                 │ │                 │
                    │ × str_apprv_rate│ │× cond_apprv_rate│ │  (remainder)    │
                    └────────┬────────┘ └────────┬────────┘ └─────────────────┘
                             │                   │
                             ▼                   ▼
                    ┌─────────────────┐ ┌─────────────────┐
                    │ STR APPROVALS   │ │ COND APPROVALS  │
                    │num_seg_str_appr │ │num_seg_cond_appr│
                    │= VSA × str_rate │ │= VSA × cond_rate│
                    └────────┬────────┘ └────────┬────────┘
                             │                   │
                             ▼ × str_bk_rate     ▼ × cond_bk_rate
                    ┌─────────────────┐ ┌─────────────────┐
                    │ STR BOOKINGS    │ │ COND BOOKINGS   │
                    │num_seg_str_bks  │ │num_seg_cond_bks │
                    │= str_appr × rate│ │= cond_appr× rate│
                    └────────┬────────┘ └────────┬────────┘
                             │                   │
                             └─────────┬─────────┘
                                       ▼
                    ┌─────────────────────────────────────────────────────────────┐
                    │                   SEGMENT BOOKINGS                          │
                    │                    (num_seg_bookings)                        │
                    │         = num_seg_str_bookings + num_seg_cond_bookings      │
                    └─────────────────────────────────────────────────────────────┘
                                       │
                                       ▼ Σ across all segments
                    ┌─────────────────────────────────────────────────────────────┐
                    │                    TOTAL BOOKINGS                           │
                    │                      (num_tot_bks)                           │
                    │              = Σ num_seg_bookings (all segments)            │
                    └─────────────────────────────────────────────────────────────┘
```

---

## Column Definitions

### Dimension Columns

| Column | Type | Description |
|--------|------|-------------|
| `lender` | string | Lender identifier (e.g., 'CAF1', 'ALY', 'CAP', 'ACA') |
| `finance_channel` | string | Finance channel: 'FF' (financed) or 'NON_FF' (non-financed) |
| `customer_segment` | string | Customer credit segment (e.g., 'Super_Prime', 'Prime', 'Near_Prime', 'Subprime', 'Deep_Subprime', 'New_To_Credit') |
| `offer_comp_tier` | string | Offer competitiveness tier: 'solo_offer', 'multi_best', 'multi_other' |
| `month_begin_date` | date | First day of the month for the period |

### Aggregate Columns (Same for all segments in a lender/channel/period)

| Column | Type | Description |
|--------|------|-------------|
| `num_tot_apps` | integer | Total applications for the lender/channel/period |
| `num_tot_bks` | float | Total bookings for the lender/channel/period |

### Segment Mix Column

| Column | Type | Description | Constraint |
|--------|------|-------------|------------|
| `pct_of_total_apps` | float | Segment's share of total applications | Σ = 1.0 per lender/channel/period |

### VSA Progression Column

| Column | Type | Description | Constraint |
|--------|------|-------------|------------|
| `vsa_prog_pct` | float | % of segment applications that progress to VSA stage | 0 ≤ value ≤ 1 |

### Approval Rate Columns (MUTUALLY EXCLUSIVE)

| Column | Type | Description | Constraint |
|--------|------|-------------|------------|
| `str_apprv_rate` | float | % of VSAs that receive straight (unconditional) approval | 0 ≤ value ≤ 1 |
| `cond_apprv_rate` | float | % of VSAs that receive conditional approval | 0 ≤ value ≤ 1 |

**Critical Constraint**: `str_apprv_rate + cond_apprv_rate ≤ 1.0`

- Straight and conditional approvals are **mutually exclusive** paths
- Each VSA can receive at most ONE type of approval (straight OR conditional)
- The remainder `(1 - str_apprv_rate - cond_apprv_rate)` represents **declined** VSAs

### Booking Rate Columns

| Column | Type | Description | Constraint |
|--------|------|-------------|------------|
| `str_bk_rate` | float | % of straight approvals that result in bookings | 0 ≤ value ≤ 1 |
| `cond_bk_rate` | float | % of conditional approvals that result in bookings | 0 ≤ value ≤ 1 |

### Derived/Validation Columns (Calculated)

| Column | Type | Calculation |
|--------|------|-------------|
| `num_seg_apps` | float | `num_tot_apps × pct_of_total_apps` |
| `num_seg_vsa` | float | `num_seg_apps × vsa_prog_pct` |
| `num_seg_str_approvals` | float | `num_seg_vsa × str_apprv_rate` |
| `num_seg_cond_approvals` | float | `num_seg_vsa × cond_apprv_rate` |
| `num_seg_str_bookings` | float | `num_seg_str_approvals × str_bk_rate` |
| `num_seg_cond_bookings` | float | `num_seg_cond_approvals × cond_bk_rate` |
| `num_seg_bookings` | float | `num_seg_str_bookings + num_seg_cond_bookings` |

---

## Data Constraints

### 1. Segment Mix Constraint
```
Σ pct_of_total_apps = 1.0  (per lender/channel/period)
```
All segment shares must sum to exactly 100% for each group.

### 2. Mutually Exclusive Approval Paths
```
str_apprv_rate + cond_apprv_rate ≤ 1.0  (per segment)
```
A VSA can only receive ONE type of approval:
- **Straight approval**: Approved without conditions
- **Conditional approval**: Approved with stipulations/conditions
- **Declined**: Neither approval type (the remainder)

### 3. Booking Reconciliation
```
Σ num_seg_bookings = num_tot_bks  (per lender/channel/period)
```
The sum of calculated segment bookings must equal the stated total bookings.

### 4. Rate Bounds
```
0 ≤ rate ≤ 1  (for all rate columns)
```
All rates must be valid percentages.

---

## Segment Dimensions

### Customer Segments (6)
| Segment | Description |
|---------|-------------|
| `Super_Prime` | Highest credit quality |
| `Prime` | Good credit quality |
| `Near_Prime` | Moderate credit quality |
| `Subprime` | Below average credit |
| `Deep_Subprime` | Poor credit quality |
| `New_To_Credit` | Limited credit history |

### Offer Competitiveness Tiers (3)
| Tier | Description |
|------|-------------|
| `solo_offer` | Only offer from this lender |
| `multi_best` | Multiple offers, this is best |
| `multi_other` | Multiple offers, this is not best |

### Total Segments per Lender/Channel/Period
```
6 customer segments × 3 offer tiers = 18 segments
```

---

## Example Calculations

For a segment with:
- `num_tot_apps = 10,000`
- `pct_of_total_apps = 0.15` (15% of apps)
- `vsa_prog_pct = 0.80` (80% progress to VSA)
- `str_apprv_rate = 0.50` (50% get straight approval)
- `cond_apprv_rate = 0.30` (30% get conditional approval)
- `str_bk_rate = 0.70` (70% of straight approvals book)
- `cond_bk_rate = 0.50` (50% of conditional approvals book)

**Funnel Flow:**
```
Segment Apps:        10,000 × 0.15 = 1,500
Segment VSA:         1,500 × 0.80 = 1,200
  ├─ Straight Appr:  1,200 × 0.50 = 600
  ├─ Cond Appr:      1,200 × 0.30 = 360
  └─ Declined:       1,200 × 0.20 = 240  (remainder)
Straight Bookings:   600 × 0.70 = 420
Cond Bookings:       360 × 0.50 = 180
SEGMENT BOOKINGS:    420 + 180 = 600
```

**Decline Rate**: 20% of VSAs are declined (1 - 0.50 - 0.30 = 0.20)

---

## File Naming Convention

| File | Description |
|------|-------------|
| `channel_mock_data.csv` | Mock data following all constraints (for testing/demos) |
| `sample_channel_data.csv` | Legacy sample data (may not follow all constraints) |

---

## Usage in LMDI Decomposition

The dataset supports decomposition of booking changes into 8 effects:

1. **Volume Effect**: Changes in total applications
2. **VSA Progression Effect**: Changes in app-to-VSA progression rates
3. **Customer Mix Effect**: Shifts in customer segment distribution
4. **Offer Comp Mix Effect**: Shifts in offer tier distribution within segments
5. **Straight Approval Effect**: Changes in straight approval rates
6. **Conditional Approval Effect**: Changes in conditional approval rates
7. **Straight Booking Effect**: Changes in straight booking rates
8. **Conditional Booking Effect**: Changes in conditional booking rates

The LMDI methodology guarantees that these 8 effects sum exactly to the actual booking change (perfect reconciliation).
