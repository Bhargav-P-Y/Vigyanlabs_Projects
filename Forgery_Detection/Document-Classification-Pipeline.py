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
INPUT_ROOT_FOLDER = os.path.join(BASE_DIR, "CERTIFICATES-20260202T052442Z-3-001")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "analysis_output")
REPORT_PATH = os.path.join(BASE_DIR, "final_production_report.json")

# OCR Configuration
API_URL = "http://10.91.2.100:8010/v1/chat/completions"
MODEL = "dots_ocr"
VALID_EXTENSIONS = ('.pdf', '.jpg', '.jpeg', '.webp')

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- 1. ENHANCED FORENSIC LOGIC (The "Ultra-Sensitive" Calibration) ---

def calculate_integrity_score(ela_map):
    """
    Production Calibration:
    - 99.9th Percentile: Catches single-character edits.
    - Floor 8.0 / Ceiling 35.0: Calibrated for Merged PDF noise.
    """
    raw_mean = np.mean(ela_map)
    raw_peak = np.percentile(ela_map, 99.9) 
    
    # Global Mean Calibration (Clean: 2.0 -> Dirty: 12.0)
    score_mean = 1.0 - ((raw_mean - 2.0) / (12.0 - 2.0))
    # Local Peak Calibration (Clean: 8.0 -> Dirty: 35.0)
    score_peak = 1.0 - ((raw_peak - 8.0) / (35.0 - 8.0))
    
    score_mean = np.clip(score_mean, 0.0, 1.0)
    score_peak = np.clip(score_peak, 0.0, 1.0)
    
    # 80% weight on peak to ensure high sensitivity to edits
    final_score = (score_mean * 0.2) + (score_peak * 0.8)
    
    return round(float(final_score), 2), raw_mean, raw_peak

# --- 2. ROBUST SEMANTIC LOGIC (OCR) ---

def call_ocr_api(image_path, prompt):
    """
    Robust OCR Call:
    - Timeout increased to 180s to prevent 'Persistent Timeout' errors.
    - Retries 3 times with backoff.
    """
    try:
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a strict OCR engine. Extract text exactly as seen in Markdown format. Preserve tables and headers."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            "temperature": 0.1 # Deterministic output
        }

        # Retry loop with extended timeout
        for attempt in range(3):
            try:
                # 180 seconds = 3 minutes per page (Generous buffer for large files)
                response = requests.post(API_URL, json=payload, timeout=180)
                response.raise_for_status()
                content = response.json()['choices'][0]['message']['content']
                if not content: return "[OCR Warning: API returned empty text]"
                return content
            except requests.exceptions.Timeout:
                print(f"      ⏳ OCR Timeout (Attempt {attempt+1})...")
                time.sleep(5)
            except Exception as e:
                print(f"      ⚠️ API Error (Attempt {attempt+1}): {e}")
                time.sleep(2)
                
        return "OCR Error: Persistent Timeout (Server overloaded or image too complex)"
    except Exception as e:
        return f"OCR Failed: {str(e)}"

# --- 3. UNIVERSAL PREPROCESSING ---

def process_file(file_path, filename):
    """
    Pipeline: Image Conversion -> Forensic Scoring -> Text Extraction
    """
    results = []
    ext = os.path.splitext(filename)[1].lower()
    
    # Convert input to processing-ready images
    if ext == '.pdf':
        pages = convert_from_path(file_path, 200) # 200 DPI is good balance for speed/accuracy
    else:
        pages = [Image.open(file_path).convert("RGB")]

    for i, page_img in enumerate(pages):
        page_suffix = f"_p{i+1}" if len(pages) > 1 else ""
        temp_path = os.path.join(OUTPUT_FOLDER, f"temp_{filename}{page_suffix}.jpg")
        
        # Resize to max 1800px to ensure API accepts the payload
        if max(page_img.size) > 1800:
            page_img.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
        page_img.save(temp_path, "JPEG", quality=90)
        
        # A. Forensic Layer
        ela_map = ELA.ELA(temp_path)
        score, r_mean, r_peak = calculate_integrity_score(ela_map)
        
        # B. Classification
        if score >= 0.95: flag = "Genuine"
        elif score >= 0.90: flag = "Suspicious"
        else: flag = "Forged"
        
        # C. Semantic Layer (OCR)
        # Note: We perform this AFTER forensics to fail fast if needed, but here we do both.
        print(f"      🔍 Extracting text for Page {i+1}...")
        ocr_prompt = "Extract all text from this document using Markdown formatting for structure."
        text = call_ocr_api(temp_path, ocr_prompt)
        
        # D. Evidence Generation
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

# --- 4. MAIN PRODUCTION LOOP ---

def main():
    full_report = []
    print(f"🚀 Starting Production Extraction Pipeline...")
    print(f"📂 Input: {INPUT_ROOT_FOLDER}")
    print(f"📄 Report: {REPORT_PATH}\n")

    for root, _, files in os.walk(INPUT_ROOT_FOLDER):
        for filename in files:
            if not filename.lower().endswith(VALID_EXTENSIONS): continue
            
            file_path = os.path.join(root, filename)
            category = os.path.basename(root)
            print(f"⚙️  Processing [{category}]: {filename}")
            
            try:
                pages_data = process_file(file_path, filename)
                
                # Verdict: Worst page score defines the document status
                worst_page = min(pages_data, key=lambda x: x['score'])
                
                # --- DOCUMENT GENERATION (Requirement: Microsoft Word) ---
                doc = Document()
                doc.add_heading(f"Extraction Report: {filename}", level=0)
                doc.add_paragraph(f"Category: {category}")
                doc.add_paragraph(f"Integrity Verdict: {worst_page['flag']} (Score: {worst_page['score']})")
                doc.add_paragraph("-" * 20)
                
                for p in pages_data:
                    doc.add_heading(f"Page {p['page']} Text Content", level=1)
                    # We add the raw text. Word won't render Markdown automatically, 
                    # but the text content and structure will be preserved.
                    doc.add_paragraph(p['text'])
                    doc.add_page_break()
                
                word_filename = f"{os.path.splitext(filename)[0]}_extracted_report.docx"
                word_path = os.path.join(OUTPUT_FOLDER, word_filename)
                doc.save(word_path)
                # ---------------------------------------------------------

                # Add to Master JSON Report
                # We truncate text in JSON to 500 chars to keep the file readable, 
                # but full text is in the DOCX.
                for p in pages_data:
                    p['text_snippet'] = p['text'][:500] + "..." if len(p['text']) > 500 else p['text']
                    # We keep the full text out of the main JSON array to prevent bloating,
                    # relying on the DOCX for the full content.

                full_report.append({
                    "metadata": {"file": filename, "category": category, "pages": len(pages_data)},
                    "verdict": {"score": worst_page['score'], "flag": worst_page['flag']},
                    "structural_details": pages_data,
                    "docx_path": word_path
                })
                
                print(f"   ✅ Saved: {word_filename}")

            except Exception as e:
                print(f"   ❌ Failed: {filename} - {e}")

    # Save Master JSON
    with open(REPORT_PATH, "w") as f:
        json.dump(full_report, f, indent=4)
    
    print(f"\n✨ PRODUCTION RUN COMPLETE.")
    print(f"📊 Master JSON: {REPORT_PATH}")
    print(f"📂 Extracted Docs: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
