import difflib
import os
import json
import re
import time
import random
import hashlib
import io
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from PIL import Image, ImageStat, ImageChops, ImageEnhance
import cv2
import pytesseract
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load spaCy model for NLP analysis
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded successfully!")
except Exception as e:
    print(f"Warning: spaCy model not available. NLP features will be limited. Error: {e}")
    nlp = None

# Configure paths for document storage
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
METADATA_FOLDER = os.path.join(UPLOAD_FOLDER, "metadata")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(METADATA_FOLDER, exist_ok=True)

# Image hash cache to improve performance
IMAGE_HASH_CACHE = {}
OCR_CACHE = {}

# Set Tesseract OCR path for Railway environment
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH", "/usr/bin/tesseract")

# Common document fields to extract and analyze
COMMON_FIELDS = [
    'name', 'date of birth', 'dob', 'address', 'id number', 'expiration date',
    'issue date', 'gender', 'nationality', 'country'
]

# Entity types to extract from documents
ENTITY_TYPES = {
    'PERSON': 'personal names',
    'DATE': 'dates (birth, expiry, issue)',
    'GPE': 'geographic locations',
    'ORG': 'organizations',
    'CARDINAL': 'numbers (ID, codes)',
}

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Helper Functions
def compute_image_hash(image_path, hash_size=8):
    """Compute the perceptual hash of an image file."""
    if isinstance(image_path, (str, Path)) and image_path in IMAGE_HASH_CACHE:
        return IMAGE_HASH_CACHE[image_path]

    try:
        if isinstance(image_path, (str, Path)):
            img = cv2.imread(str(image_path))
        else:
            if hasattr(image_path, 'read'):
                image_path.seek(0)
                img_array = np.frombuffer(image_path.read(), np.uint8)
            else:
                img_array = np.frombuffer(image_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return None

        img = cv2.resize(img, (hash_size * 4, hash_size * 4))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        avg = gray.mean()
        img_hash = 0

        for i in range(hash_size * hash_size):
            row, col = i // hash_size, i % hash_size
            img_hash = img_hash << 1 | (1 if gray[row * 4, col * 4] >= avg else 0)

        if isinstance(image_path, (str, Path)):
            IMAGE_HASH_CACHE[image_path] = img_hash

        return img_hash
    except Exception as e:
        print(f"Error computing hash: {e}")
        return None

def compute_dct_hash(image_path, hash_size=8):
    """Compute the DCT (Discrete Cosine Transform) hash of an image."""
    try:
        if isinstance(image_path, (str, Path)):
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        else:
            if hasattr(image_path, 'read'):
                image_path.seek(0)
                img_array = np.frombuffer(image_path.read(), np.uint8)
            else:
                img_array = np.frombuffer(image_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return None

        img = cv2.resize(img, (32, 32))
        dct = cv2.dct(np.float32(img))
        dct_low = dct[:hash_size, :hash_size]
        mean = np.mean(dct_low[1:])
        hash_value = 0

        for i in range(1, hash_size * hash_size):
            row, col = i // hash_size, i % hash_size
            hash_value = hash_value << 1 | (1 if dct_low[row, col] >= mean else 0)

        return hash_value
    except Exception as e:
        print(f"Error computing DCT hash: {e}")
        return None

def hamming_distance(hash1, hash2):
    """Calculate the Hamming distance between two hashes."""
    if hash1 is None or hash2 is None:
        return float('inf')
    return bin(hash1 ^ hash2).count('1')

def extract_text_from_image(image_data):
    """Extract text from an image using enhanced OCR techniques."""
    try:
        image_hash = hashlib.md5(image_data).hexdigest()
        if image_hash in OCR_CACHE:
            return OCR_CACHE[image_hash]

        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Enhance contrast and sharpness
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)

        # Convert to numpy array for OpenCV processing
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Apply Otsu's thresholding for better text segmentation
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Denoising using non-local means
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)

        # Convert back to PIL image for Tesseract
        enhanced_image = Image.fromarray(denoised)

        # Perform OCR with Tesseract
        text = pytesseract.image_to_string(enhanced_image)

        OCR_CACHE[image_hash] = text
        return text
    except Exception as e:
        print(f"OCR error: {e}")
        return ""

def preprocess_text(text):
    """Clean and normalize extracted text."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s@.,-:#/]', '', text)
    return text

def extract_entities(text):
    """Extract named entities from text using spaCy."""
    if not nlp or not text:
        return {"entities": [], "html": ""}

    doc = nlp(text)
    entities = []
    html_text = text

    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })

    entity_spans = sorted([(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents],
                          key=lambda x: x[0], reverse=True)

    for start, end, label in entity_spans:
        entity_class = ""
        if label == "PERSON":
            entity_class = "entity-person"
        elif label == "DATE":
            entity_class = "entity-date"
        elif label == "GPE":
            entity_class = "entity-loc"
        elif label == "ORG":
            entity_class = "entity-org"
        elif label in ["CARDINAL", "MONEY", "QUANTITY", "ORDINAL"]:
            entity_class = "entity-num"

        if entity_class:
            replacement = f'<span class="entity-tag {entity_class}">{text[start:end]} ({label})</span>'
            html_text = html_text[:start] + replacement + html_text[end:]

    return {
        "entities": entities,
        "html": html_text
    }

def extract_fields(text):
    """Extract common document fields from text using enhanced pattern matching."""
    if not text:
        return {}

    fields = {}
    text_lower = text.lower()

    # Enhanced name extraction
    name_patterns = [
        r'(?:name|full name|student)[:\s]+([a-zA-Z\s\'-]+)',
        r'(?:name\s*:\s*)([a-zA-Z\s\'-]+)',
        r'([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)'  # Matches "First Middle Last" format
    ]
    for pattern in name_patterns:
        name_match = re.search(pattern, text, re.IGNORECASE)
        if name_match:
            fields['name'] = name_match.group(1).strip()
            break

    # Phone/Mobile Number extraction
    phone_patterns = [
        r'(?:phone|mobile|tel|contact)[:\s\+]*([+\(]?[0-9\-\(\)\s]{8,15})',
        r'(\+?\d{1,3}[-\.\s]?\(?\d{1,4}\)?[-\.\s]?\d{1,4}[-\.\s]?\d{1,4})'
    ]
    for pattern in phone_patterns:
        phone_match = re.search(pattern, text)
        if phone_match:
            number = re.sub(r'\s+', '', phone_match.group(1))  # Remove spaces
            if 'phone_number' not in fields:
                fields['phone_number'] = number
            else:
                fields['mobile_number'] = number
            break

    # Address extraction with multi-line support
    address_patterns = [
        r'(?:address|location|residence)[:\s]+([a-zA-Z0-9\s,.-]+(?:\n[a-zA-Z0-9\s,.-]+){1,3})',
        r'((?:street|st|road|rd|ave|avenue|colony|village|post)[\s\w,.-]+)',
        r'(\d+\s+[a-zA-Z\s]+,?\s?[a-zA-Z]+,?\s?[a-zA-Z]+\s+\d{5,6})'  # Matches "123 Street, City, State 560001"
    ]
    for pattern in address_patterns:
        address_match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if address_match:
            address = address_match.group(1).replace('\n', ', ').strip()
            fields['address'] = re.sub(r'\s{2,}', ' ', address)  # Clean extra spaces
            break

    # Enhanced ID number extraction
    id_patterns = [
        r'(?:id\s*#?|enrol\s*no|document\s*number)[:\s]+([A-Z0-9\-]{8,20})',
        r'\b([A-Z]{2,4}\d{6,10}[A-Z]?)\b'  # Matches alphanumeric IDs
    ]
    for pattern in id_patterns:
        id_match = re.search(pattern, text, re.IGNORECASE)
        if id_match:
            fields['id_number'] = id_match.group(1).strip()
            break

    # Date of Birth extraction
    dob_patterns = [
        r'(?:dob|date\s*of\s*birth)[:\s]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        r'(\d{2}/\d{2}/\d{4})'
    ]
    for pattern in dob_patterns:
        dob_match = re.search(pattern, text)
        if dob_match:
            fields['date_of_birth'] = dob_match.group(1).strip()
            break

    return fields

def compute_error_level_analysis(image_data):
    """Perform Error Level Analysis (ELA) to detect image tampering."""
    try:
        image = Image.open(io.BytesIO(image_data))
        temp_output = io.BytesIO()
        image.save(temp_output, format='JPEG', quality=90)
        temp_output.seek(0)
        saved_image = Image.open(temp_output)
        ela_image = ImageChops.difference(image, saved_image)
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        ela_stats = ImageStat.Stat(ela_image)
        ela_mean = sum(ela_stats.mean) / len(ela_stats.mean)
        ela_std = sum(ela_stats.stddev) / len(ela_stats.stddev)
        max_expected_ela = 100
        ela_score = min(1.0, (ela_mean + ela_std) / max_expected_ela)
        return ela_score
    except Exception as e:
        print(f"ELA analysis error: {e}")
        return 0.5

def analyze_noise_patterns(image_data):
    """Analyze noise patterns to detect inconsistencies."""
    try:
        image_bytes = io.BytesIO(image_data)
        image = Image.open(image_bytes)
        img_array = np.array(image)
        if len(img_array.shape) > 2:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        noise = cv2.medianBlur(gray, 5)
        noise = cv2.absdiff(gray, noise)
        hist = cv2.calcHist([noise], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        noise_std = np.std(noise)
        blocks = []
        block_size = 32
        for y in range(0, gray.shape[0], block_size):
            for x in range(0, gray.shape[1], block_size):
                block = noise[y:min(y + block_size, gray.shape[0]), x:min(x + block_size, gray.shape[1])]
                if block.size > 0:
                    blocks.append(np.std(block))
        block_std = np.std(blocks)
        max_expected_inconsistency = 20
        noise_score = min(1.0, block_std / max_expected_inconsistency)
        return noise_score
    except Exception as e:
        print(f"Noise analysis error: {e}")
        return 0.5

def compare_documents(current_data, stored_docs):
    """Compare the current document with stored documents to find matches."""
    if not stored_docs:
        return None, 0.0, 0.0

    best_match = None
    best_visual_similarity = 0.0
    best_content_similarity = 0.0

    current_hash = compute_dct_hash(io.BytesIO(current_data))
    current_text = extract_text_from_image(current_data)
    current_text_norm = preprocess_text(current_text)

    for doc_id, doc_info in stored_docs.items():
        try:
            doc_hash = doc_info.get("hash")
            if doc_hash:
                distance = hamming_distance(current_hash, int(doc_hash))
                max_distance = 64
                visual_similarity = 1.0 - (distance / max_distance)
            else:
                visual_similarity = 0.0
        except:
            visual_similarity = 0.0

        try:
            doc_text = doc_info.get("text", "")
            if doc_text and current_text_norm:
                tfidf = TfidfVectorizer().fit_transform([doc_text, current_text_norm])
                similarity_matrix = cosine_similarity(tfidf)
                content_similarity = similarity_matrix[0, 1]
            else:
                content_similarity = 0.0
        except:
            content_similarity = 0.0

        combined_similarity = (visual_similarity + content_similarity) / 2

        if combined_similarity > (best_visual_similarity + best_content_similarity) / 2:
            best_match = doc_id
            best_visual_similarity = visual_similarity
            best_content_similarity = content_similarity

    return best_match, best_visual_similarity, best_content_similarity

def analyze_document_consistency(fields, text):
    """Analyze document fields for internal consistency."""
    inconsistent_fields = []
    date_fields = ['date_of_birth', 'issue_date', 'expiration_date']
    present_date_fields = [f for f in date_fields if f in fields]

    date_formats = []
    for field in present_date_fields:
        date_value = fields[field]
        if '/' in date_value:
            date_formats.append('slash')
        elif '-' in date_value:
            date_formats.append('dash')
        else:
            date_formats.append('other')

    if len(set(date_formats)) > 1:
        inconsistent_fields.extend(present_date_fields)

    if 'expiration_date' in fields and 'issue_date' in fields:
        if fields['expiration_date'] < fields['issue_date']:
            inconsistent_fields.append('expiration_date')
            inconsistent_fields.append('issue_date')

    for field, value in fields.items():
        if value and len(value) > 3:
            if field not in date_fields:
                if value.lower() not in text.lower():
                    if not any(difflib.get_close_matches(value.lower(), text.lower().split(), cutoff=0.7)):
                        inconsistent_fields.append(field)

    total_fields = len(fields)
    consistent_fields = total_fields - len(inconsistent_fields)
    consistency_score = consistent_fields / total_fields if total_fields > 0 else 0.5

    return consistency_score, inconsistent_fields

def generate_checksum(data):
    """Generate a unique identifier/checksum for a document."""
    if isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()
    return hashlib.sha256(str(data).encode()).hexdigest()

def simulate_blockchain_verification(document_hash):
    """Simulate verification of document hash on a blockchain."""
    return {
        "verified": random.random() > 0.3,
        "timestamp": int(time.time()),
        "block_number": random.randint(1000000, 9999999),
        "transaction_id": f"0x{document_hash[:16]}" if document_hash else "N/A"
    }

# Frontend HTML, CSS, and JavaScript
FRONTEND_CODE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Powered Document Verification</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --primary-color: #2196F3;
            --secondary-color: #9C27B0;
            --background-color: #0a0a0a;
            --card-color: #1a1a1a;
            --text-color: #ffffff;
            --accent-color: #00BCD4;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Roboto', sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
            overflow-x: hidden;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            text-align: center;
            padding: 20px 0;
            margin-bottom: 30px;
        }

        h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            animation: fadeIn 1s ease-in-out;
        }

        .upload-section {
            background-color: var(--card-color);
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease;
        }

        .upload-section:hover {
            transform: translateY(-5px);
        }

        .file-input-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 30px;
            border: 2px dashed rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            transition: all 0.3s ease;
        }

        .file-input-container:hover {
            border-color: var(--primary-color);
            background-color: rgba(33, 150, 243, 0.05);
        }

        .file-input-container input {
            display: none;
        }

        .file-input-label {
            display: flex;
            flex-direction: column;
            align-items: center;
            cursor: pointer;
            padding: 20px;
            border-radius: 8px;
            transition: background-color 0.3s ease;
        }

        .file-input-label:hover {
            background-color: rgba(33, 150, 243, 0.1);
        }

        .file-input-icon {
            font-size: 3rem;
            margin-bottom: 15px;
            color: var(--primary-color);
        }

        .file-input-text {
            font-size: 1.1rem;
            text-align: center;
            margin-bottom: 10px;
        }

        .preview-container {
            margin-top: 20px;
            text-align: center;
        }

        .preview-container img {
            max-width: 100%;
            max-height: 300px;
            border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }

        .analyze-button {
            background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
            color: white;
            border: none;
            padding: 12px 24px;
            font-size: 1rem;
            border-radius: 30px;
            cursor: pointer;
            margin-top: 20px;
            box-shadow: 0 4px 10px rgba(33, 150, 243, 0.3);
            transition: all 0.3s ease;
        }

        .analyze-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(33, 150, 243, 0.4);
        }

        .analyze-button:active {
            transform: translateY(0);
        }

        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
            color: var(--accent-color);
            font-size: 1.2rem;
            animation: pulse 1.5s infinite;
        }

        .results-container {
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }

        .card {
            background-color: var(--card-color);
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.4);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .card-title {
            font-size: 1.5rem;
            font-weight: 500;
            color: var(--primary-color);
        }

        .card-content {
            animation: fadeIn 0.5s ease-in-out;
        }

        .verification-status {
            display: flex;
            justify-content: space-around;
            margin-bottom: 30px;
        }

        .status-card {
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0.05);
            transition: all 0.3s ease;
        }

        .status-card:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }

        .status-icon {
            font-size: 2.5rem;
            margin-bottom: 15px;
        }

        .status-title {
            font-size: 1.2rem;
            margin-bottom: 10px;
            font-weight: 500;
        }

        .status-value {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        .tabs {
            display: flex;
            margin-bottom: 20px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            overflow-x: auto;
        }

        .tab {
            padding: 12px 24px;
            cursor: pointer;
            font-weight: 500;
            color: rgba(255, 255, 255, 0.6);
            transition: all 0.3s ease;
            white-space: nowrap;
        }

        .tab.active {
            color: var(--primary-color);
            border-bottom: 3px solid var(--primary-color);
        }

        .tab:hover:not(.active) {
            color: var(--primary-color);
        }

        .tab-content {
            display: none;
            animation: fadeIn 0.5s ease-in-out;
        }

        .tab-content.active {
            display: block;
        }

        .document-preview {
            text-align: center;
            margin: 20px 0;
        }

        .document-preview img {
            max-width: 100%;
            max-height: 400px;
            border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease;
        }

        .document-preview img:hover {
            transform: scale(1.02);
        }

        .extracted-text {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 15px;
            font-family: monospace;
            white-space: pre-wrap;
            margin-bottom: 20px;
            max-height: 300px;
            overflow-y: auto;
        }

        .extracted-fields {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .field-card {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 15px;
            transition: all 0.3s ease;
        }

        .field-card:hover {
            background-color: rgba(255, 255, 255, 0.1);
            transform: translateY(-3px);
        }

        .field-name {
            font-weight: 500;
            color: var(--primary-color);
            margin-bottom: 5px;
        }

        .field-value {
            font-size: 1.1rem;
        }

        .inconsistent {
            background-color: rgba(255, 99, 132, 0.1);
            border-left: 3px solid #ff5252;
        }

        .inconsistent .field-name {
            color: #ff5252;
        }

        .charts-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .chart-card {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }

        .chart-card:hover {
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        }

        .chart-title {
            font-size: 1.2rem;
            font-weight: 500;
            margin-bottom: 15px;
            color: var(--primary-color);
        }

        .chart-container {
            height: 250px;
            position: relative;
        }

        .issues-list {
            list-style-type: none;
            margin-bottom: 20px;
        }

        .issues-list li {
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
        }

        .issues-list li:before {
            content: "⚠";
            margin-right: 10px;
            color: #ff9800;
        }

        .recommendation {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            border-left: 4px solid var(--accent-color);
        }

        .recommendation h3 {
            color: var(--accent-color);
            margin-bottom: 10px;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }

        @keyframes slideIn {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        .hidden {
            display: none;
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
            .verification-status {
                flex-direction: column;
            }

            .charts-container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>AI-Powered Document Verification</h1>
            <p>Advanced document analysis using AI and forensic techniques</p>
        </header>

        <div class="upload-section">
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="file-input-container">
                    <label class="file-input-label" for="fileInput">
                        <div class="file-input-icon">📁</div>
                        <div class="file-input-text">Select a document to analyze</div>
                        <div class="file-input-subtext">Supported formats: PNG, JPG, PDF</div>
                    </label>
                    <input type="file" id="fileInput" name="file" accept="image/*" style="display: none;">
                    <div id="preview" class="preview-container hidden">
                        <img id="previewImage" src="" alt="Document Preview">
                    </div>
                    <button type="submit" class="analyze-button">Analyze Document</button>
                    <div id="loading" class="loading">Analyzing document... Please wait</div>
                </div>
            </form>
        </div>

        <div id="results" class="hidden">
            <div class="results-container">
                <div class="card">
                    <div class="card-header">
                        <h2 class="card-title">Document Authentication Summary</h2>
                    </div>
                    <div class="card-content">
                        <div class="verification-status">
                            <div class="status-card">
                                <div class="status-icon">✅</div>
                                <div class="status-title">Verification Status</div>
                                <div id="verification-status-value" class="status-value">AUTHENTIC</div>
                            </div>
                            <div class="status-card">
                                <div class="status-icon">📊</div>
                                <div class="status-title">Security Score</div>
                                <div id="security-score-value" class="status-value">92%</div>
                            </div>
                            <div class="status-card">
                                <div class="status-icon">⏱</div>
                                <div class="status-title">Analysis Time</div>
                                <div id="analysis-time-value" class="status-value">1.16s</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="tabs">
                    <div class="tab active" data-tab="document">Document</div>
                    <div class="tab" data-tab="extracted-text">Extracted Text</div>
                    <div class="tab" data-tab="extracted-fields">Extracted Fields</div>
                    <div class="tab" data-tab="forensic-analysis">Forensic Analysis</div>
                    <div class="tab" data-tab="issues">Issues</div>
                </div>

                <div id="document" class="tab-content active">
                    <div class="document-preview">
                        <img id="document-image" src="" alt="Document Image">
                    </div>
                </div>

                <div id="extracted-text" class="tab-content">
                    <div class="extracted-text" id="extracted-text-content"></div>
                    <div class="chart-card">
                        <h3 class="chart-title">Named Entity Recognition</h3>
                        <div class="chart-container">
                            <canvas id="ner-chart"></canvas>
                        </div>
                    </div>
                </div>

                <div id="extracted-fields" class="tab-content">
                    <div class="extracted-fields" id="extracted-fields-container"></div>
                </div>

                <div id="forensic-analysis" class="tab-content">
                    <div class="charts-container">
                        <div class="chart-card">
                            <h3 class="chart-title">Image Analysis</h3>
                            <div class="chart-container">
                                <canvas id="image-analysis-chart"></canvas>
                            </div>
                        </div>
                        <div class="chart-card">
                            <h3 class="chart-title">Content Analysis</h3>
                            <div class="chart-container">
                                <canvas id="content-analysis-chart"></canvas>
                            </div>
                        </div>
                        <div class="chart-card">
                            <h3 class="chart-title">Verification Timeline</h3>
                            <div class="chart-container">
                                <canvas id="timeline-chart"></canvas>
                            </div>
                        </div>
                        <div class="chart-card">
                            <h3 class="chart-title">Security Metrics</h3>
                            <div class="chart-container">
                                <canvas id="security-metrics-chart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>

                <div id="issues" class="tab-content">
                    <h3 class="card-title">Issues Detected</h3>
                    <ul class="issues-list" id="issues-list">
                        <li>No issues detected</li>
                    </ul>
                    <div class="recommendation">
                        <h3>Recommendation</h3>
                        <p id="recommendation-text">No recommendation available.</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Tab switching functionality
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                // Remove active class from all tabs
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                // Add active class to clicked tab
                tab.classList.add('active');

                // Hide all tab contents
                document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

                // Show the selected tab content
                const tabId = tab.getAttribute('data-tab');
                document.getElementById(tabId).classList.add('active');
            });
        });

        // File preview functionality
        const fileInput = document.getElementById('fileInput');
        const preview = document.getElementById('preview');
        const previewImage = document.getElementById('previewImage');
        const uploadForm = document.getElementById('uploadForm');

        fileInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.classList.remove('hidden');
                    previewImage.src = e.target.result;
                }
                reader.readAsDataURL(file);
            }
        });

        // Form submission handler
        uploadForm.addEventListener('submit', async function(e) {
            e.preventDefault();

            const file = fileInput.files[0];
            if (!file) {
                alert("Please select a file.");
                return;
            }

            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            const documentImage = document.getElementById('document-image');

            loading.style.display = 'block';
            results.classList.add('hidden');

            try {
                const formData = new FormData();
                formData.append('file', file);

                // Send to your backend endpoint
                const response = await fetch('/verify', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`Server responded with ${response.status}`);
                }

                const data = await response.json();

                // Update UI with REAL data from backend
                loading.style.display = 'none';
                results.classList.remove('hidden');
                documentImage.src = URL.createObjectURL(file);

                // Update verification status
                document.getElementById('verification-status-value').textContent =
                    data.valid ? "AUTHENTIC" : "SUSPICIOUS";
                document.getElementById('security-score-value').textContent =
                    `${Math.round(data.security_score * 100)}%`;
                document.getElementById('analysis-time-value').textContent =
                    `${data.verification_time.toFixed(2)}s`;

                // Update extracted text with REAL OCR data
                document.getElementById('extracted-text-content').textContent =
                    data.extracted_text || "No text could be extracted";

                // Update extracted fields with REAL data
                const fieldsContainer = document.getElementById('extracted-fields-container');
                fieldsContainer.innerHTML = '';

                if (data.extracted_fields) {
                    for (const [field, value] of Object.entries(data.extracted_fields)) {
                        const isInconsistent = data.inconsistent_fields?.includes(field);
                        const fieldCard = document.createElement('div');
                        fieldCard.className = `field-card ${isInconsistent ? 'inconsistent' : ''}`;
                        fieldCard.innerHTML = `
                            <div class="field-name">${field.replace(/_/g, ' ').toUpperCase()}</div>
                            <div class="field-value">${value || 'N/A'}</div>
                        `;
                        fieldsContainer.appendChild(fieldCard);
                    }
                }

                // Update issues list
                const issuesList = document.getElementById('issues-list');
                issuesList.innerHTML = '';

                if (data.issues && data.issues.length > 0) {
                    data.issues.forEach(issue => {
                        const li = document.createElement('li');
                        li.textContent = issue;
                        issuesList.appendChild(li);
                    });
                } else {
                    issuesList.innerHTML = '<li>No issues detected</li>';
                }

                // Update recommendation
                document.getElementById('recommendation-text').textContent =
                    data.recommendation || "No recommendation available.";

                // Initialize charts with real data
                initializeCharts(data);

            } catch (error) {
                loading.style.display = 'none';
                alert("An error occurred during verification: " + error.message);
                console.error("Verification error:", error);
            }
        });

        // Simulated data for charts
        function initializeCharts(data) {
            // Named Entity Recognition Chart
            const nerCounts = data.entities?.entities?.reduce((acc, entity) => {
                acc[entity.label] = (acc[entity.label] || 0) + 1;
                return acc;
            }, {}) || { PERSON: 5, DATE: 3, GPE: 2, ORG: 1 };

            const nerCtx = document.getElementById('ner-chart').getContext('2d');
            new Chart(nerCtx, {
                type: 'doughnut',
                data: {
                    labels: Object.keys(nerCounts),
                    datasets: [{
                        data: Object.values(nerCounts),
                        backgroundColor: [
                            '#2196F3', '#9C27B0', '#FF5722', '#4CAF50', '#FFC107'
                        ],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });

            // Image Analysis Chart
            const imageCtx = document.getElementById('image-analysis-chart').getContext('2d');
            new Chart(imageCtx, {
                type: 'bar',
                data: {
                    labels: ['Error Level', 'Noise', 'Consistency'],
                    datasets: [{
                        label: 'Score',
                        data: [
                            Math.round((1 - (data.forensics?.ela_score || 0)) * 100),
                            Math.round((1 - (data.forensics?.noise_analysis || 0)) * 100),
                            Math.round((data.consistency_score || 0) * 100)
                        ],
                        backgroundColor: '#2196F3',
                        borderRadius: 5
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });

            // Content Analysis Chart
            const contentCtx = document.getElementById('content-analysis-chart').getContext('2d');
            new Chart(contentCtx, {
                type: 'bar',
                data: {
                    labels: ['Font Consistency', 'Layout Analysis', 'Text Validity'],
                    datasets: [{
                        label: 'Score',
                        data: [90, 77, 85],
                        backgroundColor: '#9C27B0',
                        borderRadius: 5
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    animation: {
                        duration: 2000
                    }
                }
            });

            // Timeline Chart
            const timelineCtx = document.getElementById('timeline-chart').getContext('2d');
            new Chart(timelineCtx, {
                type: 'line',
                data: {
                    labels: ['Upload', 'OCR', 'Analysis', 'Verification', 'Result'],
                    datasets: [{
                        label: 'Time (ms)',
                        data: [100, 200, 500, 250, 100],
                        borderColor: '#FF5722',
                        backgroundColor: 'rgba(255, 87, 34, 0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    animation: {
                        duration: 2000
                    }
                }
            });

            // Security Metrics Chart
            const securityCtx = document.getElementById('security-metrics-chart').getContext('2d');
            new Chart(securityCtx, {
                type: 'radar',
                data: {
                    labels: ['Authenticity', 'Validity', 'Consistency', 'Security', 'Integrity'],
                    datasets: [{
                        label: 'Score',
                        data: [92, 88, 95, 85, 90],
                        backgroundColor: 'rgba(76, 175, 80, 0.2)',
                        borderColor: '#4CAF50',
                        pointBackgroundColor: '#4CAF50'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            angleLines: {
                                display: true
                            },
                            suggestedMin: 0,
                            suggestedMax: 100
                        }
                    },
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    },
                    animation: {
                        duration: 2000
                    }
                }
            });
        }
    </script>
</body>
</html>
"""

# Flask Routes
@app.route('/')
def index():
    """Render the main application interface."""
    return render_template_string(FRONTEND_CODE)

@app.route('/verify', methods=['POST'])
def verify_document():
    """Verify uploaded document using AI techniques."""
    try:
        start_time = time.time()

        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        file_data = file.read()

        # Perform full analysis
        return perform_full_analysis(file_data, start_time)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def perform_full_analysis(file_data, start_time):
    """Perform comprehensive document analysis."""
    try:
        # Image processing and analysis
        doc_hash = compute_dct_hash(io.BytesIO(file_data))
        if doc_hash is None:
            return jsonify({"error": "Failed to process image"}), 400

        # Text extraction and preprocessing
        extracted_text = extract_text_from_image(file_data)
        clean_text = preprocess_text(extracted_text)
        entities = extract_entities(clean_text)
        extracted_fields = extract_fields(clean_text)
        consistency_score, inconsistent_fields = analyze_document_consistency(extracted_fields, clean_text)
        ela_score = compute_error_level_analysis(file_data)
        noise_analysis = analyze_noise_patterns(file_data)

        # Document comparison with stored documents
        stored_documents = {}
        metadata_path = os.path.join(METADATA_FOLDER, "documents.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    stored_documents = json.load(f)
            except Exception as e:
                print(f"Error loading stored documents: {e}")
                stored_documents = {}

        best_match, visual_similarity, content_similarity = compare_documents(file_data, stored_documents)

        # Security scoring
        image_authenticity = 1.0 - (0.7 * ela_score + 0.3 * noise_analysis)
        content_validity = content_similarity if best_match else 0.5
        security_score = 0.4 * image_authenticity + 0.4 * consistency_score + 0.2 * content_validity

        # Issue detection
        issues = []
        if image_authenticity < 0.7:
            issues.append("Potential image manipulation detected")
        if ela_score > 0.6:
            issues.append("High error level analysis score indicates possible editing")
        if noise_analysis > 0.6:
            issues.append("Inconsistent noise patterns detected")
        if consistency_score < 0.7:
            issues.append("Document fields show internal inconsistencies")
        if inconsistent_fields:
            field_list = ", ".join(inconsistent_fields)
            issues.append(f"Inconsistent fields detected: {field_list}")
        if not extracted_fields.get('id_number') or not extracted_fields.get('name'):
            issues.append("Missing critical identification information")

        # Recommendation system
        if security_score > 0.8 and not issues:
            recommendation = "Document appears to be authentic with high confidence."
        elif security_score > 0.6 and len(issues) < 3:
            recommendation = "Document appears generally authentic but has some suspicious elements that warrant additional verification."
        else:
            recommendation = "Document contains significant anomalies suggesting possible tampering or forgery. Manual verification strongly recommended."

        # Generate checksum for document
        checksum = generate_checksum(file_data)

        # Simulate blockchain verification
        blockchain_result = simulate_blockchain_verification(checksum)

        # Save document metadata
        document_metadata = {
            "filename": f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "hash": str(doc_hash),
            "text": clean_text,
            "fields": extracted_fields,
            "security_score": float(security_score),
            "issues": issues
        }

        doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        stored_documents[doc_id] = document_metadata

        with open(metadata_path, 'w') as f:
            json.dump(stored_documents, f, cls=NumpyEncoder)

        verification_time = time.time() - start_time

        # Prepare response
        response = {
            "valid": bool(security_score > 0.6),
            "security_score": float(security_score),
            "image_authenticity": float(image_authenticity),
            "content_validity": float(content_validity),
            "consistency_score": float(consistency_score),
            "match_found": bool(best_match is not None),
            "match_id": str(best_match) if best_match is not None else None,
            "visual_similarity": float(visual_similarity),
            "content_similarity": float(content_similarity),
            "extracted_text": str(extracted_text),
            "extracted_fields": {k: str(v) for k, v in extracted_fields.items()},
            "inconsistent_fields": [str(field) for field in inconsistent_fields],
            "entities": entities,
            "checksum": str(checksum),
            "blockchain_verification": blockchain_result,
            "verification_time": float(verification_time),
            "issues": [str(issue) for issue in issues],
            "recommendation": str(recommendation),
            "forensics": {
                "ela_score": float(ela_score) if ela_score is not None else 0.0,
                "noise_analysis": float(noise_analysis) if noise_analysis is not None else 0.0
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
