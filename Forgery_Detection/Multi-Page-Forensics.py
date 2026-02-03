import os
import cv2
import json
import numpy as np
from datetime import datetime
from pdf2image import convert_from_path
from pyIFD import ELA
from PIL import Image

# --- Configuration ---
BASE_DIR = os.getcwd()
INPUT_FOLDER = os.path.join(BASE_DIR, "multi_page")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "multi_page_analysis")
REPORT_PATH = os.path.join(BASE_DIR, "multi_page_audit_report.json")

VALID_EXTENSIONS = ('.pdf', '.jpg', '.jpeg', '.webp')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- REASONING ENGINE ---

def generate_reasoning(mean, peak, score):
    """
    Translates forensic metrics into human-readable explanations.
    """
    ratio = peak / (mean + 0.001) 
    
    if score >= 0.95:
        return "Pristine: Structural noise is negligible and consistent with an original digital document."
    
    if score >= 0.90:
        return "Suspicious Metadata: Slight noise detected. Typical of high-quality scans or digitally flattened PDFs."
    
    # Logic for scores below 0.90 (catching the 0.60-0.69 range)
    if ratio > 15.0:
        return f"Surgical Forgery: High local noise (Peak: {peak:.1f}) relative to background suggests specific pasted content."
    
    if mean > 8.0:
        return f"Global Manipulation: High uniform noise (Mean: {mean:.1f}) suggests the document was digitally reconstructed or re-saved."
    
    return "Artifact Contamination: Irregular noise patterns detected that deviate from standard compression standards."

# --- CORE FORENSICS ---

def calculate_integrity_score(ela_map):
    """
    Refined scoring using surgical 99.9th percentile peak detection.
    """
    raw_mean = np.mean(ela_map)
    raw_peak = np.percentile(ela_map, 99.9) 
    
    # Calibration tuned for 'Merged' PDFs (Floor: 8.0 / Ceiling: 35.0)
    score_mean = 1.0 - ((raw_mean - 2.0) / (12.0 - 2.0))
    score_peak = 1.0 - ((raw_peak - 8.0) / (35.0 - 8.0))
    
    score_mean = np.clip(score_mean, 0.0, 1.0)
    score_peak = np.clip(score_peak, 0.0, 1.0)
    
    # 80% weight on Peak ensures localized edits define the verdict
    final_score = (score_mean * 0.2) + (score_peak * 0.8)
    
    return round(float(final_score), 2), raw_mean, raw_peak

# --- PIPELINE ---

def main():
    full_report = []
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Folder not found: {INPUT_FOLDER}")
        return

    pdf_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(VALID_EXTENSIONS)]
    if not pdf_files:
        print(f"📂 No valid files found in {INPUT_FOLDER}")
        return

    print(f"\n🚀 Analyzing {len(pdf_files)} Multi-Page Documents (Forensic Reasoning Mode)...")
    print(f"{'DOCUMENT NAME':<40} | {'STATUS':<10} | {'SCORE':<5} | {'REASONING'}")
    print("-" * 120)

    for filename in pdf_files:
        file_path = os.path.join(INPUT_FOLDER, filename)
        
        try:
            # 1. Page Splitting (300 DPI)
            if filename.lower().endswith('.pdf'):
                pages = convert_from_path(file_path, 300)
            else:
                pages = [Image.open(file_path).convert("RGB")]

            page_results = []
            for i, page_img in enumerate(pages):
                page_num = i + 1
                temp_filename = f"temp_{filename}_p{page_num}.jpg"
                temp_path = os.path.join(OUTPUT_FOLDER, temp_filename)
                
                # Standardize size for consistent noise analysis
                if max(page_img.size) > 1800:
                    page_img.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
                page_img.save(temp_path, 'JPEG', quality=95)
                
                # 2. Forensic Analysis
                ela_map = ELA.ELA(temp_path)
                score, r_mean, r_peak = calculate_integrity_score(ela_map)
                reason = generate_reasoning(r_mean, r_peak, score)
                
                # Classification
                if score >= 0.95: status = "Genuine"
                elif score >= 0.90: status = "Suspicious"
                else: status = "Forged"
                
                # Save Heatmap
                heatmap_name = f"{os.path.splitext(filename)[0]}_p{page_num}_ELA.png"
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, heatmap_name), (ela_map * 255).astype(np.uint8))
                
                page_results.append({
                    "page": page_num,
                    "score": score,
                    "status": status,
                    "reasoning": reason,
                    "metrics": {"mean": round(r_mean, 4), "peak": round(r_peak, 4)}
                })
                if os.path.exists(temp_path): os.remove(temp_path)

            # 3. Final Verdict (Worst Page Logic)
            worst_page = min(page_results, key=lambda x: x['score'])
            
            print(f"{filename:<40} | {worst_page['status']:<10} | {worst_page['score']:<5} | {worst_page['reasoning']}")
            
            full_report.append({
                "file": filename,
                "verdict": {"score": worst_page['score'], "status": worst_page['status'], "reason": worst_page['reasoning']},
                "page_details": page_results
            })

        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

    with open(REPORT_PATH, "w") as f:
        json.dump(full_report, f, indent=4)
    print("-" * 120)
    print(f"✨ Audit Complete. JSON Report: {REPORT_PATH}")

if __name__ == "__main__":
    main()
