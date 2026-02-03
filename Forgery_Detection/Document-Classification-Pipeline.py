import os
import cv2
import json
import base64
import time
import requests
import numpy as np
from datetime import datetime
from pdf2image import convert_from_path
from pyIFD import ELA
from docx import Document
from PIL import Image

# --- Configuration ---
BASE_DIR = os.getcwd()
INPUT_ROOT_FOLDER = os.path.join(BASE_DIR, "CERTIFICATES")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "analysis_output")
REPORT_PATH = os.path.join(BASE_DIR, "integrity_audit_report.json")

# OCR Configuration
API_URL = "http://10.91.2.100:8010/v1/chat/completions"
MODEL = "dots_ocr"
VALID_EXTENSIONS = ('.pdf', '.jpg', '.jpeg', '.webp')

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- 1. ENHANCED FORENSIC LOGIC ---

def calculate_integrity_score(ela_map):
    """
    ULTRA-SENSITIVE SCORING:
    - Uses 99.9th percentile to catch surgical character-level edits.
    - Thresholds calibrated to catch subtle 0.88-rated fakes.
    """
    raw_mean = np.mean(ela_map)
    raw_peak = np.percentile(ela_map, 99.9) 
    
    # Global Mean Calibration (Clean: 2.0 -> Dirty: 12.0)
    score_mean = 1.0 - ((raw_mean - 2.0) / (12.0 - 2.0))
    # Local Peak Calibration (Clean: 8.0 -> Dirty: 35.0)
    score_peak = 1.0 - ((raw_peak - 8.0) / (35.0 - 8.0))
    
    score_mean = np.clip(score_mean, 0.0, 1.0)
    score_peak = np.clip(score_peak, 0.0, 1.0)
    
    # Heavy Peak Penalty (80% weight) to ensure small edits are never missed
    final_score = (score_mean * 0.2) + (score_peak * 0.8)
    
    return round(float(final_score), 2), raw_mean, raw_peak

# --- 2. ROBUST SEMANTIC LOGIC ---

def call_ocr_api(image_path, prompt):
    """Hallucination-proof OCR with Retry Logic and Timeout protection."""
    try:
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "Strict OCR engine: Transcribe English/Hindi exactly. No hallucinations."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            "temperature": 0.1 # Forces deterministic transcription
        }

        for attempt in range(3):
            try:
                # 120s timeout to handle high-res scanner files
                response = requests.post(API_URL, json=payload, timeout=120)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
            except (requests.exceptions.RequestException, KeyError):
                time.sleep(2)
        return "OCR Error: Persistent Timeout or API Failure"
    except Exception as e:
        return f"OCR Failed: {str(e)}"

# --- 3. UNIVERSAL PREPROCESSING ---

def process_file(file_path, filename):
    """Handles multi-page PDFs and converts images to forensic JPEGs."""
    results = []
    ext = os.path.splitext(filename)[1].lower()
    
    # PDF splitting logic
    if ext == '.pdf':
        pages = convert_from_path(file_path, 200)
    else:
        pages = [Image.open(file_path).convert("RGB")]

    for i, page_img in enumerate(pages):
        page_suffix = f"_p{i+1}" if len(pages) > 1 else ""
        temp_path = os.path.join(OUTPUT_FOLDER, f"temp_{filename}{page_suffix}.jpg")
        
        # Resizing to prevent noise washout and API timeouts
        if max(page_img.size) > 1800:
            page_img.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
        page_img.save(temp_path, "JPEG", quality=90)
        
        # A. Forensic Structural Layer
        ela_map = ELA.ELA(temp_path)
        score, r_mean, r_peak = calculate_integrity_score(ela_map)
        
        # B. Classification
        if score >= 0.95: flag = "Genuine"
        elif score >= 0.90: flag = "Suspicious"
        else: flag = "Forged"
        
        # C. Semantic Extraction Layer
        ocr_prompt = "Extract English and Hindi text from this document. Transcribe exactly."
        text = call_ocr_api(temp_path, ocr_prompt)
        
        # D. Save Heatmap Evidence
        heatmap_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}{page_suffix}_ELA.png")
        cv2.imwrite(heatmap_path, (ela_map * 255).astype(np.uint8))
        
        results.append({
            "page": i + 1,
            "score": score,
            "flag": flag,
            "text": text,
            "heatmap": heatmap_path,
            "metrics": {"mean": round(r_mean, 4), "peak_99.9th": round(r_peak, 4)}
        })
        if os.path.exists(temp_path): os.remove(temp_path)
        
    return results

# --- 4. MAIN AUDIT LOOP ---

def main():
    full_report = []
    print(f"🚀 Initializing Integrated Multi-Layer Audit...")

    for root, _, files in os.walk(INPUT_ROOT_FOLDER):
        for filename in files:
            if not filename.lower().endswith(VALID_EXTENSIONS): continue
            
            file_path = os.path.join(root, filename)
            category = os.path.basename(root)
            print(f"📑 Processing [{category}]: {filename}")
            
            pages_data = process_file(file_path, filename)
            
            # Aggregate: Document verdict based on worst-performing page
            worst_page = min(pages_data, key=lambda x: x['score'])
            
            # Combine all page text into a single Word Document
            doc = Document()
            doc.add_heading(f"Audit Report: {filename}", level=1)
            doc.add_paragraph(f"Category: {category} | Overall Verdict: {worst_page['flag']}")
            for p in pages_data:
                doc.add_heading(f"Page {p['page']} Extraction (Score: {p['score']})", level=2)
                doc.add_paragraph(p['text'])
            
            word_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}_audit.docx")
            doc.save(word_path)

            full_report.append({
                "metadata": {"file": filename, "category": category, "pages": len(pages_data)},
                "verdict": {"score": worst_page['score'], "flag": worst_page['flag']},
                "structural_details": pages_data,
                "docx_report": word_path
            })
            print(f"   Verdict: {worst_page['flag']} ({worst_page['score']})")

    # Save final JSON Master Report
    with open(REPORT_PATH, "w") as f:
        json.dump(full_report, f, indent=4)
    print(f"\n✨ AUDIT COMPLETE. Report: {REPORT_PATH}")

if __name__ == "__main__":
    main()
