import cv2
import easyocr
import re
import numpy as np
import io
import os  # This replaces 'fs'
from datetime import datetime
from dateutil.relativedelta import relativedelta
from flask import Flask, request, jsonify

# ADD these 3 lines

from PIL import Image

app = Flask(__name__)

# Keep reader outside the function so it loads only ONCE on startup
reader = easyocr.Reader(['en'])


# Your original function — only change: accept numpy array instead of file path
def scan_medical_document(image_array):

    # REMOVED: reader = easyocr.Reader(['en'])   ← was here before, now moved above
    # REMOVED: cv2.imread(image_path)            ← Node sends the image directly now

    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)

    results = reader.readtext(sharpened, detail=1)
    full_text = " ".join([res[1] for res in results])

    found_name  = "Not found"
    found_visit = "Not found"

    # ---------------- DOCTOR NAME ----------------
    dr_pattern = re.compile(r"\bDr\.?\s+\S+\s+\S+", re.IGNORECASE)
    name_match = dr_pattern.search(full_text)
    if name_match:
        found_name = name_match.group().strip()
        found_name = found_name.replace(".", "").replace(";", "").replace(",", "")

    # ---------------- NEXT VISIT ----------------
    visit_pattern = re.compile(
        r"Next\s*Visit\s*[:\-]?\s*After\s*(\d+)\s*(day|days|month|months|year|years)",
        re.IGNORECASE
    )
    visit_match = visit_pattern.search(full_text)
    if visit_match:
        number = int(visit_match.group(1))
        unit   = visit_match.group(2).lower()
        today  = datetime.today()

        if "day"   in unit: future_date = today + relativedelta(days=number)
        elif "month" in unit: future_date = today + relativedelta(months=number)
        elif "year"  in unit: future_date = today + relativedelta(years=number)

        found_visit = future_date.strftime("%d-%m-%Y")

    return {
        "doctor_name": found_name,
        "next_visit":  found_visit,
        "full_text":   full_text       # ADD: so Node.js can store raw OCR text too
    }


# ── ADD: Flask route — Node.js sends image here ──────────────────────────────
@app.route('/ocr', methods=['POST'])
def ocr():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Read image bytes → convert to OpenCV array (same format as cv2.imread)
        img_bytes = request.files['image'].read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_array = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Call your existing function — nothing inside it changed
        result = scan_medical_document(img_array)

        return jsonify(result)   # sends JSON back to Node.js

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── ADD: health check so Node can verify Python is running ───────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


# ── REPLACE the bottom 2 lines with this ─────────────────────────────────────
# REMOVED: result = scan_medical_document('my_image.jpg')
# REMOVED: print(result)
if __name__ == '__main__':
    print("Loading EasyOCR model...")
    app.run(port=5001, debug=False)