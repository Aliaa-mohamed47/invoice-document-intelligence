# Evaluation Report
## Invoice / Document Intelligence System

**Author:** Alaa Mohamed  
**Role:** AI Evaluation & JSON Output Engineer  
**Model:** bert-base-multilingual-cased fine-tuned with LoRA  
**Dataset:** SROIE + CORD-v2 (merged)  
**Date:** April 2026  

---

## 1. Objective

This report evaluates the BERT-based invoice extraction model before and after
LoRA fine-tuning. The goal is to measure field-level extraction accuracy across
four key fields: Company, Date, Total, and Address, using Precision, Recall,
and F1-score as the primary metrics.

---

## 2. Dataset

| Property       | Value                                     |
|----------------|-------------------------------------------|
| Source         | SROIE + CORD-v2 (merged)                  |
| Test Records   | ~346                                      |
| Train Split    | 80%                                       |
| Test Split     | 20%                                       |
| Fields Labeled | COMPANY, DATE, TOTAL, ADDRESS (BIO tags)  |

---

## 3. Baseline Results (Pre Fine-tuning)

| Field       | Precision | Recall | F1     |
|-------------|-----------|--------|--------|
| Company     | 0.8406    | 0.6918 | 0.7590 |
| Date        | 0.8864    | 0.7707 | 0.8245 |
| Total       | 0.8674    | 0.7363 | 0.7965 |
| Address     | 0.8051    | 0.6355 | 0.7103 |
| **Macro Avg** | **0.8499** | **0.7086** | **0.7726** |

---

## 4. Fine-tuned Results (Post Fine-tuning)

| Field       | Precision | Recall | F1     |
|-------------|-----------|--------|--------|
| Company     | 0.8899    | 0.9083 | 0.8990 |
| Date        | 0.9734    | 0.9821 | 0.9777 |
| Total       | 0.8547    | 0.8547 | 0.8547 |
| Address     | 0.7317    | 0.8191 | 0.7729 |
| **Macro Avg** | **0.8624** | **0.8911** | **0.8761** |

---

## 5. Improvement After Fine-tuning

| Field   | Baseline F1 | Fine-tuned F1 | Δ F1    | Improved? |
|---------|-------------|---------------|---------|-----------|
| Company | 0.7590      | 0.8990        | +0.1400 | ✓         |
| Date    | 0.8245      | 0.9777        | +0.1532 | ✓         |
| Total   | 0.7965      | 0.8547        | +0.0582 | ✓         |
| Address | 0.7103      | 0.7729        | +0.0626 | ✓         |

Fine-tuning produced an overall macro F1 improvement of **+10.35%**
(from 0.7726 → 0.8761).

![F1 Comparison](evaluation_results/charts/f1_comparison.png)
![Metrics Per Field](evaluation_results/charts/metrics_per_field.png)

---

## 6. JSON Output Format

Every processed invoice is returned in this structure:

```json
{
  "invoice_id": "X00016469670",
  "extracted_fields": {
    "company": "OJC MARKETING SDN BHD",
    "date": "15/01/2019",
    "total": "193.00",
    "address": "NO 2 & 4, JALAN BAYU 4, BANDAR SERI ALAM, 81750 MASAI, JOHOR"
  },
  "confidence_scores": {
    "company": 0.998,
    "date": 0.9994,
    "total": 0.9992,
    "address": 0.999
  },
  "processing_time_ms": 3889.16
}
```

---

## 7. Error Analysis

### 7.1 Common Error Types

| Error Type                  | Field   | Example |
|-----------------------------|---------|---------|
| Prefix/suffix hallucination | Company | Predicted `"TONY ROMA'S AEON TEBRAU CITY GRAND COMPANIONS SDN BHD"` instead of `"GRAND COMPANIONS SDN"` |
| Total duplication           | Total   | Predicted `"3.00 4.90"` instead of `"4.90"` (picked up both subtotal and final) |
| Address truncation          | Address | Predicted `"LOT SATMZ 23, MEZZANINE LEVEL SATELLITE BUILDING KUALA"` missing `"LUMPUR INTERNATIONAL AIRPORT"` |
| Currency format mismatch    | Total   | Predicted `"5.00"` instead of `"RM 5.00"` (dropped currency prefix) |
| Date over-extraction        | Date    | Predicted `"22/04/2018 22/04/2018"` when date appeared twice on invoice |

### 7.2 Why Address Has the Lowest F1

The address field spans multiple tokens with no fixed format or length. A
single missing or extra word counts as a full mismatch under exact-match
scoring. The model also struggles when the ground truth label cuts off
mid-address (e.g. `"NO 2 & 4, JALAN BAYU 4, BANDAR SERI ALAM,"` vs the
model correctly extracting the full address). This label inconsistency in
the SROIE dataset is a known issue and explains why address F1 (0.7729)
lags behind date F1 (0.9777).

### 7.3 Why Date Achieved the Highest F1

Date fields follow consistent, short patterns (DD/MM/YYYY, DD-MM-YY,
DD MON YYYY etc.). The model learned these patterns reliably, achieving
0.9821 recall — missing only 6 out of 335 date instances in the test set.

### 7.4 Why Total Has Equal Precision and Recall

The total field shows identical Precision and Recall (0.8547), meaning every
false positive also corresponds to a false negative. This is characteristic
of substitution errors — the model extracts a total value but the wrong one
(e.g. subtotal instead of grand total), which counts as both an FP and an FN.

---

## 8. Sample Predictions

**Correct extraction (high confidence):**

| Field   | Ground Truth               | Predicted                  | Match |
|---------|----------------------------|----------------------------|-------|
| company | RESTORAN WAN SHENG         | RESTORAN WAN SHENG         | ✓     |
| date    | 23-03-2018                 | 23-03-2018                 | ✓     |
| total   | 6.70                       | 6.70                       | ✓     |
| address | NO.2, JALAN TEMENGGUNG...  | NO.2, JALAN TEMENGGUNG...  | ✓     |

**Error case — total duplication:**

| Field | Ground Truth | Predicted | Match |
|-------|-------------|-----------|-------|
| total | 4.90        | 3.00 4.90 | ✗     |

**Error case — company hallucination:**

| Field   | Ground Truth         | Predicted                                              | Match |
|---------|----------------------|--------------------------------------------------------|-------|
| company | GRAND COMPANIONS SDN | TONY ROMA'S AEON TEBRAU CITY GRAND COMPANIONS SDN BHD | ✗     |

---

## 9. Conclusion

LoRA fine-tuning on the merged SROIE + CORD-v2 dataset significantly improved
extraction accuracy across all four fields, achieving an overall macro F1 of
**0.8761** — a **+10.35% improvement** over the baseline.

The `Date` field achieved the highest F1 (0.9777), benefiting from its
structured and consistent formatting. The `Company` field showed the largest
absolute gain (+0.14 F1), demonstrating the strongest benefit from domain
adaptation via fine-tuning.

The `Address` field remains the most challenging due to variable length,
inconsistent ground truth labeling in the SROIE dataset, and the strictness
of exact-match scoring. Future improvements could include partial-match
scoring (token-level F1) and post-processing rules to normalize currency
symbols and remove duplicate values in the Total field.
