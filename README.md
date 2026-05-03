# 🧾 Invoice Intelligence System

An AI-powered system that extracts structured information from invoices (PDF and images) using OCR and a Layout-aware deep learning model (LayoutLM).

---

## 🚀 Features

- Extracts key invoice fields:
  - Company Name
  - Date
  - Total Amount
  - Address

- Supports:
  - Images (JPG / PNG)
  - PDF invoices

- Hybrid pipeline:
  - OCR-based text extraction (Tesseract)
  - Layout-aware understanding (LayoutLM model)
  - Rule-based fallback for robustness

- Returns structured JSON with confidence scores

---

## 🧠 System Pipeline

Invoice → OCR → Token + Bounding Boxes → LayoutLM Model → Post-processing → JSON Output

---

## ⚙️ Tech Stack

- FastAPI (Backend API)
- PyTorch (Deep Learning)
- HuggingFace Transformers (LayoutLM)
- Tesseract OCR
- PIL / OpenCV
- AWS S3 (file storage)

---

## 📡 API Endpoint

### POST `/extract`

Upload an invoice file (image/PDF)

### Response Example:
```json
{
  "company": "ABC Corp",
  "date": "20 Jun 2018",
  "total": "RM 8.20",
  "address": "Johor Bahru",
  "confidence_scores": {
    "company": 0.97,
    "date": 0.95,
    "total": 0.99,
    "address": 0.88
  }
}
```

## How to run the Dashboard
1. Download the repo
2. Open `ai/Dashboard/dashboard.html` in any browser
3. That's it! Backend is hosted on AWS

---

## 👩‍💻 Team

Aliaa • Alaa • Waad • Sama • Fatma • A`laa
