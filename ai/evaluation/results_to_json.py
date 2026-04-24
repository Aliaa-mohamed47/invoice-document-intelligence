import time
import uuid
import json

FIELDS = ["company", "date", "total", "address"]


def format_output(invoice_id, raw_prediction, start_time):
    """
    Converts raw model output into the required structured JSON.

    Args:
        invoice_id      : str like "INV_001", or None to auto-generate
        raw_prediction  : dict from model inference
                            e.g. {"company": "ABC Corp", "date": "2024-01-15",
                                "total": "$500", "address": "Cairo"}
                            Optionally includes confidence keys:
                            {"company_score": 0.93, "date_score": 0.88, ...}
        start_time      : float from time.time() captured before inference

    Returns:
        dict matching the required JSON schema
    """
    if not invoice_id:
        invoice_id = f"INV_{uuid.uuid4().hex[:8].upper()}"

    # Extract the 4 fields
    extracted_fields = {
        field: raw_prediction.get(field) or None
        for field in FIELDS
    }

    # Confidence scores: use model scores if provided, else fallback values
    confidence_scores = {}
    for field in FIELDS:
        score_key = f"{field}_score"
        if score_key in raw_prediction:
            confidence_scores[field] = round(float(raw_prediction[score_key]), 4)
        elif extracted_fields[field]:
            confidence_scores[field] = 0.75   # field found but no score available
        else:
            confidence_scores[field] = 0.0    # field not extracted

    processing_time_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "invoice_id":         invoice_id,
        "extracted_fields":   extracted_fields,
        "confidence_scores":  confidence_scores,
        "processing_time_ms": processing_time_ms
    }


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mock = {
        "company":       "ABC Corporation",
        "date":          "2024-01-15",
        "total":         "$1,200.00",
        "address":       "123 Main Street, Cairo",
        "company_score": 0.9321,
        "date_score":    0.8814,
        "total_score":   0.9755,
        "address_score": 0.7203,
    }
    start = time.time()
    result = format_output("INV_001", mock, start)
    print(json.dumps(result, indent=2, ensure_ascii=False))
