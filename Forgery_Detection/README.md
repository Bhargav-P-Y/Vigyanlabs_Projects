#  Forensic Document Integrity & Extraction Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Computer Vision](https://img.shields.io/badge/Computer_Vision-OpenCV%20%7C%20ELA-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production_Ready-success?style=flat-square)

##  Module Overview

This directory contains an enterprise-grade forensic pipeline designed to detect manipulated documents (certificates, financial records, IDs) with high precision. The system integrates **Computer Vision (Error Level Analysis)** and **Statistical Profiling** to distinguish between genuine digital artifacts and surgical forgeries.

**Key Capabilities:**
* **Multi-Layer Audit:** Combines structural noise analysis (Forensics) with semantic verification (OCR).
* **Surgical Detection:** Uses **99.9th percentile peak detection** to identify single-character alterations often missed by global averages.
* **Explainable AI (XAI):** Includes a Reasoning Engine that diagnoses the *type* of forgery (e.g., "Surgical Paste" vs. "Global Re-save").

---

##  Repository Structure

### 1. `production_extraction_engine.py` (End-to-End Pipeline)
**Purpose:** Production deployment for processing incoming document streams.
* **Workflow:** Image Conversion (300 DPI) → Forensic Scoring → OCR Extraction → Report Generation.
* **Features:**
    * **Robust OCR:** Implements retry logic with exponential backoff and extended timeouts (180s) for high-res scans.
    * **Output:** Generates a Microsoft Word (`.docx`) report containing the forensic verdict, integrity score, and full text extraction.
    * **Threshold:** Calibrated **0.90 Integrity Threshold** to filter subtle forgeries.

### 2. `multi_page_forensics.py` (Forensic Reasoning Tool)
**Purpose:** High-volume triage and R&D analysis of multi-page PDFs.
* **Workflow:** Pure forensic sweep (OCR disabled for speed).
* **Features:**
    * **Reasoning Engine:** classifying risk based on Peak-to-Mean noise ratios.
    * **Batch Processing:** Handles merged PDFs and multi-page documents, flagging the entire file if a single page fails.

---

##  Algorithmic Logic & XAI

The core innovation is a **Weighted Peak Scoring Model** calibrated against real-world forgery datasets.

### Integrity Scoring Formula
The system calculates a final integrity score (0.0 - 1.0) using a weighted balance of Global and Local noise:
```python
# 80% weight on Peak Noise ensures surgical edits drive the verdict
final_score = (normalized_mean * 0.2) + (normalized_peak_99_9th * 0.8)
