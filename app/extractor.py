"""
Core extraction pipeline — FULLY LOCAL, no external API calls.
  PDF → PyMuPDF text extraction → PaddleOCR → Rule-based structuring → JSON

All processing runs inside the Docker container. Data never leaves the server.
"""

import io
import math
import re
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Stage 1: PDF → high-DPI images
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[tuple[bytes, int]]:
    """Convert each PDF page to a PNG image at the given DPI."""
    doc = fitz.open(pdf_path)
    images = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        png_bytes = pix.tobytes("png")
        images.append((png_bytes, page_num + 1))
    doc.close()
    return images


# ---------------------------------------------------------------------------
# Stage 2: PyMuPDF programmatic text extraction (for CAD-generated PDFs)
# ---------------------------------------------------------------------------

def extract_pdf_text_blocks(pdf_path: str) -> list[dict]:
    """Extract text with position data from PDF. Works if PDF has embedded text from CAD."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        blocks = []
        for b in page.get_text("dict")["blocks"]:
            if b.get("lines"):
                for line in b["lines"]:
                    for span in line["spans"]:
                        blocks.append({
                            "text": span["text"].strip(),
                            "bbox": list(span["bbox"]),
                            "size": span["size"],
                            "font": span["font"],
                        })
        pages.append({
            "page": page_num + 1,
            "raw_text": text,
            "blocks": blocks,
            "has_embedded_text": len(text.strip()) > 20,
        })
    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Stage 3: PaddleOCR extraction (for scanned/rasterized PDFs)
# ---------------------------------------------------------------------------

_ocr_engine = None


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(
            lang="en",
            text_detection_model_name="PP-OCRv4_mobile_det",
            text_recognition_model_name="en_PP-OCRv4_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return _ocr_engine


def ocr_extract(image_bytes: bytes) -> list[dict]:
    """Run PaddleOCR on an image and return detected text with bounding boxes."""
    ocr = get_ocr_engine()
    img = Image.open(io.BytesIO(image_bytes))
    img_array = np.array(img)

    detections = []
    try:
        # PaddleOCR v3.4+ API: predict() returns a result object
        results = ocr.predict(img_array)
        for result in results:
            if hasattr(result, 'rec_texts'):
                # Batch result object
                for i, text in enumerate(result.rec_texts):
                    conf = result.rec_scores[i] if i < len(result.rec_scores) else 0.9
                    bbox = result.dt_polys[i].tolist() if i < len(result.dt_polys) else [[0,0],[0,0],[0,0],[0,0]]
                    detections.append({
                        "text": text,
                        "confidence": round(float(conf), 3),
                        "bbox": bbox,
                    })
            elif isinstance(result, dict):
                # Dict-style result
                texts = result.get('rec_texts', result.get('rec_text', []))
                scores = result.get('rec_scores', result.get('rec_score', []))
                polys = result.get('dt_polys', result.get('dt_poly', []))
                if isinstance(texts, str):
                    texts, scores, polys = [texts], [scores], [polys]
                for i, text in enumerate(texts):
                    conf = scores[i] if i < len(scores) else 0.9
                    bbox = polys[i] if i < len(polys) else [[0,0],[0,0],[0,0],[0,0]]
                    if hasattr(bbox, 'tolist'):
                        bbox = bbox.tolist()
                    detections.append({
                        "text": str(text),
                        "confidence": round(float(conf), 3),
                        "bbox": bbox,
                    })
    except (TypeError, AttributeError):
        # Fallback: try legacy .ocr() API (PaddleOCR v2.x)
        try:
            results = ocr.ocr(img_array, cls=True)
            if results and results[0]:
                for line in results[0]:
                    bbox, (text, confidence) = line
                    detections.append({
                        "text": text,
                        "confidence": round(confidence, 3),
                        "bbox": bbox,
                    })
        except Exception:
            pass

    return detections


# ---------------------------------------------------------------------------
# Stage 4: Rule-based structuring (replaces Claude Vision)
#
# Parses OCR text detections into structured dimensions, tolerances, GD&T,
# title block data, and notes using regex pattern matching.
# ---------------------------------------------------------------------------

# --- Dimension patterns ---

# Diameter: Ø16, ⌀16, ∅16, Dia16, 016 (OCR misread), φ16
RE_DIAMETER = re.compile(
    r'(?P<mult>\d+\s*[Xx×]\s*)?'
    r'[Øø⌀∅ΦφDdia]\.?\s*'
    r'(?P<value>\d+(?:\.\d+)?)',
    re.IGNORECASE,
)

# Radius: R14, 2 X R14
RE_RADIUS = re.compile(
    r'(?P<mult>\d+\s*[Xx×]\s*)?'
    r'R\s*(?P<value>\d+(?:\.\d+)?)',
    re.IGNORECASE,
)

# Countersink: ⌵Ø22.5 or V/Ø22.5 X 90°
RE_COUNTERSINK = re.compile(
    r'(?P<mult>\d+\s*[Xx×]\s*)?'
    r'[⌵Vv]\s*[Øø⌀∅ΦφDdia]?\.?\s*'
    r'(?P<value>\d+(?:\.\d+)?)'
    r'(?:\s*[Xx×]\s*(?P<angle>\d+(?:\.\d+)?)\s*°)?',
    re.IGNORECASE,
)

# Tolerance: ±0.25, +0.13/-0.38, +0.25/0, +0.13\n-0.38
RE_TOLERANCE_BILATERAL = re.compile(r'[±]\s*(?P<tol>\d+(?:\.\d+)?)')
RE_TOLERANCE_UNILATERAL = re.compile(
    r'(?P<upper>[+]\s*\d+(?:\.\d+)?)\s*[/\n]\s*(?P<lower>[-]\s*\d+(?:\.\d+)?)'
)
RE_TOLERANCE_PLUS_ZERO = re.compile(
    r'(?P<upper>[+]\s*\d+(?:\.\d+)?)\s*[/\n]\s*(?P<lower>0(?:\.0+)?)'
)

# Linear dimension: plain number like 55, 130.5, 44.9
RE_LINEAR = re.compile(r'^(?P<value>\d+(?:\.\d+)?)$')

# Angle: 90°±1°, 45°
RE_ANGLE = re.compile(r'(?P<value>\d+(?:\.\d+)?)\s*°')

# Multiplier prefix: "3 X", "4X", "6 x"
RE_MULTIPLIER = re.compile(r'^(\d+)\s*[Xx×]\s*')

# --- GD&T patterns ---
# GD&T symbols that OCR might detect as text or Unicode
GDT_SYMBOLS = {
    '⌖': 'Position', '⊕': 'Position',
    '⏥': 'Flatness', '⌓': 'Flatness',
    '○': 'Circularity', '⌒': 'Circularity',
    '⌭': 'Cylindricity',
    '⊥': 'Perpendicularity',
    '∥': 'Parallelism', '//': 'Parallelism',
    '⌗': 'Profile of a line',
    '⌓': 'Profile of a surface',
    '↗': 'Angularity',
    '◎': 'Concentricity',
    '⌰': 'Runout',
}

GDT_TEXT_MAP = {
    'position': 'Position',
    'flatness': 'Flatness',
    'circularity': 'Circularity',
    'cylindricity': 'Cylindricity',
    'perpendicularity': 'Perpendicularity',
    'parallelism': 'Parallelism',
    'concentricity': 'Concentricity',
    'runout': 'Runout',
    'total runout': 'Total Runout',
    'straightness': 'Straightness',
    'symmetry': 'Symmetry',
}

RE_GDT_FRAME = re.compile(
    r'(?P<symbol>[⌖⊕⏥⌓○⌒⌭⊥∥◎⌰↗//])\s*'
    r'[Øø⌀∅]?\s*(?P<value>\d+(?:\.\d+)?)'
    r'(?:\s*[ⓂM])?\s*'
    r'(?P<datums>[A-Z](?:\s*[ⓂM])?(?:\s+[A-Z](?:\s*[ⓂM])?)*)?',
    re.IGNORECASE,
)

# --- Title block keywords ---
TITLE_KEYWORDS = {
    'part_name': ['title', 'part name', 'part description', 'description'],
    'drawing_number': ['drg no', 'drawing no', 'dwg no', 'part no', 'part number'],
    'material': ['material'],
    'revision': ['rev', 'revision'],
    'drawn_by': ['drn', 'drawn', 'drawn by', 'designer'],
    'checked_by': ['chkd', 'checked', 'checked by', 'checker'],
    'approved_by': ['apvd', 'approved', 'approved by'],
    'date': ['date'],
    'customer': ['customer', 'client'],
    'scale': ['scale'],
    'sheet': ['sheet', 'sheet no'],
    'general_tolerance': ['general tol', 'tolerance', 'unless specified', 'general tolerance'],
    'weight': ['weight', 'mass'],
}

# --- Note patterns ---
RE_NOTE_PREFIX = re.compile(r'^\s*(?:note|notes?)?\s*:?\s*\d+[\.\)]\s*', re.IGNORECASE)


def _bbox_center(bbox) -> tuple[float, float]:
    """Get center point of a bounding box."""
    if len(bbox) == 4 and isinstance(bbox[0], (int, float)):
        # [x1, y1, x2, y2]
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    # PaddleOCR format: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _bbox_area(bbox) -> float:
    if len(bbox) == 4 and isinstance(bbox[0], (int, float)):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _is_in_lower_region(bbox, img_height: float, threshold: float = 0.7) -> bool:
    """Check if detection is in the lower portion (likely title block)."""
    _, cy = _bbox_center(bbox)
    return cy > img_height * threshold


def _parse_tolerance(text: str) -> tuple[str | None, str | None]:
    """Extract tolerance values from text."""
    # ±0.25
    m = RE_TOLERANCE_BILATERAL.search(text)
    if m:
        tol = f"±{m.group('tol')}"
        return tol, tol

    # +0.13/-0.38
    m = RE_TOLERANCE_UNILATERAL.search(text)
    if m:
        return m.group('upper').replace(' ', ''), m.group('lower').replace(' ', '')

    # +0.25/0
    m = RE_TOLERANCE_PLUS_ZERO.search(text)
    if m:
        return m.group('upper').replace(' ', ''), '0'

    return None, None


def _suggest_instrument(dim_type: str, tolerance: str | None) -> str:
    """Suggest measurement instrument based on dimension type and tolerance."""
    if dim_type == 'radius':
        return 'Radius gauge'
    if dim_type == 'countersink':
        return 'Countersink gauge'
    if dim_type == 'angle':
        return 'Protractor'
    if dim_type == 'thread':
        return 'Thread gauge'

    # Based on tolerance precision
    if tolerance:
        try:
            tol_val = float(tolerance.replace('±', '').replace('+', '').replace('-', '').strip())
            if tol_val <= 0.05:
                return 'Micrometer'
            if tol_val <= 0.5:
                return 'Vernier caliper'
        except (ValueError, AttributeError):
            pass

    return 'Vernier caliper'


def _classify_and_parse_detection(text: str, is_title_block: bool = False) -> list[dict]:
    """Classify a single OCR detection into dimension(s)."""
    results = []
    text = text.strip()

    if not text or len(text) < 1:
        return results

    # Never parse title block text as dimensions
    if is_title_block:
        return results

    # Skip common non-dimension text
    skip_words = {'mm', 'note', 'notes', 'note:-', 'material', 'title', 'title:-',
                  'scale', 'proj', 'proj.', 'rev', 'rev by',
                  'date', 'drn', 'drn.', 'chkd', 'chkd.', 'apvd', 'apvd.',
                  'sign', 'sheet', 'sheet no', 'sheet no.', 'size', 'weight',
                  'finish', 'hardness', 'units', 'unless', 'specified', 'general',
                  'a', 'b', 'c', 'ok', 'of', 'the', 'and', 'for', 'all', 'are', 'in',
                  'customer', 'ref', 'ref.drg.no.', 'modification', 'released',
                  'new', 'drawing', 'new drawing released',
                  'remove', 'burrs', 'sharp', 'edges', 'steel', 'grade', 'per',
                  'isometric', 'isometric view', 'view', 'left', 'right', 'hand',
                  'hood', 'bracket', 'left hand hood bracket',
                  'support', 'hose', 'pump', 'detail', 'section',
                  'gaurav', 'patil', 'melric', 'karthik', 'ecr no.', 'ecr no',
                  'cl.', 'cl', 'nts', 'surf.', 'surf', 'india', 'united',
                  'rubber industries', 'united rubber industries',
                  'directors.', 'directors', '-', 'i', 'uri',
                  'tol.', 'tol', 'unless specified',
                  'bhayandar', '01 of 01', 'drg no:', 'drg no',
                  'rev by', 'ecr no.'}
    text_lower = text.lower().strip().rstrip('.')
    if text_lower in skip_words:
        return results
    # Skip long sentences (likely notes or disclaimers) — but allow engineering annotations
    has_eng_symbol = any(s in text for s in ('⌵', 'Ø', '⌀', '°', '±'))
    if len(text.split()) > 5 and not has_eng_symbol:
        return results
    if len(text.split()) > 10:
        return results  # hard limit
    # Skip pure text (no digits at all) — not a dimension
    if not any(c.isdigit() for c in text):
        return results
    # Skip date patterns (14.01.26)
    if re.match(r'^\d{1,2}\.\d{1,2}\.\d{2,4}$', text):
        return results
    # Skip part number patterns (4643710c1_b, URI-4643708C1)
    if re.match(r'^[A-Za-z]*[-_]?\d{4,}', text) and any(c.isalpha() for c in text):
        return results

    # --- Tolerance-only fragments: skip them ---
    # These are standalone tolerance values (+0.13, -0.38, +0.25, 0) near parent dims.
    # They'll be merged with their parent dimension by the proximity merger.
    if re.match(r'^[+]\d+(\.\d+)?$', text):
        return results  # standalone "+0.13"
    if re.match(r'^[-]\d+(\.\d+)?$', text):
        return results  # standalone "-0.38"
    if text == '0':
        return results  # standalone "0" (lower tolerance bound)

    # --- GD&T fragments: skip ---
    # Symbols, modifiers, datum letters handled by GD&T extractor
    if text in ('⌖', '⌖⌀', '⏥', '⌓', '⊕', '⊥', '∥', 'Ⓜ', 'Ⓛ', '⌵⌀', '⌀'):
        return results
    if re.match(r'^[AB]\s*[BⓂM]?$', text):
        return results  # datum references "A B", "A BM"
    if re.match(r'^[⌖⏥⌓⊕⊥∥]', text):
        return results  # GD&T symbol prefix

    # --- Countersink: "4 X ⌵Ø22.5 ±0.25 X 90°±1°" or "4 X V 22.5±0.25 X 90°±1°" ---
    m = re.match(r'(\d+\s*[Xx×]\s*)?[V⌵]\s*[Øø⌀]?(\d+(?:\.\d+)?)\s*([±]\s*\d+(?:\.\d+)?)?\s*[Xx×]\s*(\d+(?:\.\d+)?)\s*°\s*([±]\s*\d+(?:\.\d+)?\s*°)?', text)
    if m:
        mult = (m.group(1) or '').strip()
        dia = m.group(2)
        tol_dia = m.group(3)
        angle = m.group(4)
        tol_angle = m.group(5)
        nominal = f"{mult + ' ' if mult else ''}⌵Ø{dia} × {angle}°"
        tol_u = tol_dia.replace(' ', '') if tol_dia else None
        tol_l = tol_u
        results.append({
            'type': 'countersink',
            'description': f"Countersink {nominal}",
            'nominal_value': nominal,
            'tolerance_upper': tol_u,
            'tolerance_lower': tol_l,
            'unit': 'mm',
        })
        return results

    # --- Countersink (simpler): ⌵⌀22.5 ---
    m = RE_COUNTERSINK.search(text)
    if m and ('⌵' in text or 'V ' in text):
        mult = m.group('mult') or ''
        val = m.group('value')
        angle = m.group('angle')
        tol_u, tol_l = _parse_tolerance(text)
        nominal = f"{mult}⌵Ø{val}"
        if angle:
            nominal += f" × {angle}°"
        results.append({
            'type': 'countersink',
            'description': f"Countersink {nominal}",
            'nominal_value': nominal,
            'tolerance_upper': tol_u,
            'tolerance_lower': tol_l,
            'unit': 'mm',
        })
        return results

    # --- Diameter: Ø16, 3 X Ø11.5, Ø16 +0.13 -0.38 ---
    m = RE_DIAMETER.search(text)
    if m:
        mult = m.group('mult') or ''
        val = m.group('value')
        tol_u, tol_l = _parse_tolerance(text)
        # Also check for space-separated tolerances from merger: "Ø16 +0.13 -0.38"
        if not tol_u:
            m2 = re.search(r'([+]\d+(?:\.\d+)?)\s+([-]\d+(?:\.\d+)?)', text)
            if m2:
                tol_u = m2.group(1)
                tol_l = m2.group(2)
        if not tol_u:
            m2 = re.search(r'([+]\d+(?:\.\d+)?)', text)
            if m2 and '+' not in val:
                tol_u = m2.group(1)
        if not tol_l:
            m2 = re.search(r'([-]\d+(?:\.\d+)?)', text)
            if m2 and '-' not in val:
                tol_l = m2.group(1)
        nominal = f"{mult}Ø{val}".strip()
        results.append({
            'type': 'diameter',
            'description': f"Diameter {nominal}",
            'nominal_value': nominal,
            'tolerance_upper': tol_u,
            'tolerance_lower': tol_l,
            'unit': 'mm',
        })
        return results

    # --- Radius: R14, 2 X R14, 6 X R10 ---
    m = RE_RADIUS.search(text)
    if m:
        mult = m.group('mult') or ''
        val = m.group('value')
        nominal = f"{mult}R{val}".strip()
        results.append({
            'type': 'radius',
            'description': f"Radius {nominal}",
            'nominal_value': nominal,
            'tolerance_upper': None,
            'tolerance_lower': None,
            'unit': 'mm',
        })
        return results

    # --- Number with tolerance: "9±0.25", "8±0.3" ---
    m = re.match(r'^(\d+(?:\.\d+)?)\s*[±]\s*(\d+(?:\.\d+)?)$', text)
    if m:
        val = m.group(1)
        tol = m.group(2)
        results.append({
            'type': 'linear',
            'description': f"Linear dimension {val}",
            'nominal_value': val,
            'tolerance_upper': f"±{tol}",
            'tolerance_lower': f"±{tol}",
            'unit': 'mm',
        })
        return results

    # --- "3 X 11.5" (multiplied dimension without Ø/R prefix) ---
    m = re.match(r'^(\d+)\s*[Xx×]\s*(\d+(?:\.\d+)?)$', text)
    if m:
        mult = m.group(1)
        val = m.group(2)
        nominal = f"{mult} X {val}"
        results.append({
            'type': 'linear',
            'description': f"Dimension {nominal}",
            'nominal_value': nominal,
            'tolerance_upper': None,
            'tolerance_lower': None,
            'unit': 'mm',
        })
        return results

    # --- Linear dimension: plain number like 55, 130.5, 44.9 ---
    m = RE_LINEAR.match(text)
    if m:
        val = m.group('value')
        try:
            num = float(val)
            # Filter: must be a plausible dimension (not a tolerance fragment)
            # Tolerance values are typically < 1.0 (0.25, 0.5, 0.2, 0.3, 0.13, 0.38)
            # Real dimensions are typically >= 1.0
            if num < 1.0:
                return results  # Almost certainly a tolerance fragment, not a dimension
            if num > 9999:
                return results  # Too large
            results.append({
                'type': 'linear',
                'description': f"Linear dimension {val}",
                'nominal_value': val,
                'tolerance_upper': None,
                'tolerance_lower': None,
                'unit': 'mm',
            })
        except ValueError:
            pass
        return results

    return results


def _find_nearest_right(det: dict, candidates: list[dict], max_y_gap: float = 30) -> str | None:
    """Find the nearest detection to the right of `det` on roughly the same Y line."""
    cx, cy = _bbox_center(det['bbox'])
    best = None
    best_dist = float('inf')
    for c in candidates:
        ccx, ccy = _bbox_center(c['bbox'])
        if abs(ccy - cy) > max_y_gap:
            continue  # not on same line
        if ccx <= cx:
            continue  # not to the right
        dist = ccx - cx
        if dist < best_dist:
            best_dist = dist
            best = c['text'].strip()
    return best


def _extract_title_block(detections: list[dict], img_height: float) -> dict:
    """Extract title block information from detections in the lower region of the drawing."""
    title_block = {
        'part_name': '',
        'drawing_number': '',
        'material': '',
        'revision': '',
        'drawn_by': '',
        'checked_by': '',
        'approved_by': '',
        'date': '',
        'customer': '',
        'scale': '',
        'sheet': '',
        'general_tolerance': '',
        'weight': '',
    }

    # Only consider detections in the title block region (bottom 35%)
    tb_dets = [d for d in detections if _is_in_lower_region(d['bbox'], img_height, 0.65)]

    # Strategy: look for known label text, then find the value to its right
    for det in tb_dets:
        text = det['text'].strip()
        text_lower = text.lower().rstrip(':.-')

        # TITLE:- → part name is to the right
        if text_lower in ('title', 'title:-'):
            val = _find_nearest_right(det, tb_dets)
            if val and len(val) > 2:
                title_block['part_name'] = val

        # DRG NO: → drawing number to the right
        elif text_lower in ('drg no', 'drg no:', 'drg no.'):
            val = _find_nearest_right(det, tb_dets)
            if val:
                title_block['drawing_number'] = val

        # CUSTOMER → value to the right
        elif text_lower == 'customer':
            val = _find_nearest_right(det, tb_dets)
            if val:
                title_block['customer'] = val

        # SCALE → value to the right
        elif text_lower == 'scale':
            val = _find_nearest_right(det, tb_dets)
            if val:
                title_block['scale'] = val

        # SHEET NO. → value to the right
        elif text_lower in ('sheet no', 'sheet no.'):
            val = _find_nearest_right(det, tb_dets)
            if val:
                title_block['sheet'] = val

        # DRN. → drawn_by name to the right
        elif text_lower in ('drn', 'drn.'):
            val = _find_nearest_right(det, tb_dets)
            if val:
                title_block['drawn_by'] = val

        # CHKD. → checked_by
        elif text_lower.startswith('chkd'):
            # Often "CHKD.MELRIC" or "CHKD. MELRIC 14.01.26"
            after = text[4:].strip(' .:')
            if after:
                # Split off date if appended
                parts = re.split(r'\s+\d{1,2}\.\d{1,2}\.\d{2,4}', after)
                title_block['checked_by'] = parts[0].strip()
            else:
                val = _find_nearest_right(det, tb_dets)
                if val:
                    parts = re.split(r'\s+\d{1,2}\.\d{1,2}\.\d{2,4}', val)
                    title_block['checked_by'] = parts[0].strip()

        # APVD. → approved_by
        elif text_lower.startswith('apvd'):
            after = text[4:].strip(' .:')
            if after:
                parts = re.split(r'\s*\d{1,2}\.\d{1,2}\.\d{2,4}', after)
                title_block['approved_by'] = parts[0].strip()
            else:
                val = _find_nearest_right(det, tb_dets)
                if val:
                    parts = re.split(r'\s*\d{1,2}\.\d{1,2}\.\d{2,4}', val)
                    title_block['approved_by'] = parts[0].strip()

    # Drawing number: search for URI-XXXX pattern globally
    for det in detections:
        text = det['text'].strip()
        if re.match(r'URI[-_ ]\d{4,}', text, re.IGNORECASE):
            title_block['drawing_number'] = text

    # Date: first DD.MM.YY in title block area
    for det in tb_dets:
        text = det['text'].strip()
        if re.match(r'^\d{1,2}\.\d{1,2}\.\d{2,4}$', text):
            if not title_block['date']:
                title_block['date'] = text

    # General tolerance: look for ±X.XX near "TOL." or "UNLESS SPECIFIED"
    for det in tb_dets:
        text = det['text'].strip()
        m = re.match(r'^[±]\s*(\d+(?:\.\d+)?)$', text)
        if m:
            title_block['general_tolerance'] = f"±{m.group(1)}"
        m = re.match(r'^±(\d+(?:\.\d+)?)$', text)
        if m:
            title_block['general_tolerance'] = text

    # Revision: look for single letter "A" near "CL." in title block
    for det in tb_dets:
        text = det['text'].strip()
        if text == 'A' and not title_block['revision']:
            cx, cy = _bbox_center(det['bbox'])
            # Check if near "CL." label
            for other in tb_dets:
                if other['text'].strip().upper() in ('CL.', 'CL'):
                    ox, oy = _bbox_center(other['bbox'])
                    if abs(cy - oy) < 20:
                        title_block['revision'] = 'A'

    # Notes section: extract material info
    for det in detections:
        text = det['text'].strip()
        m = re.match(r'^\d+\.\s*MATERIAL[:\-\s]+(.+)', text, re.IGNORECASE)
        if m:
            title_block['material'] = m.group(1).strip()

    return title_block


def _extract_notes(detections: list[dict], img_height: float) -> list[str]:
    """Extract notes from the drawing."""
    notes = []
    in_notes = False
    for det in detections:
        text = det['text'].strip()
        if re.match(r'^notes?\s*:?\s*$', text, re.IGNORECASE):
            in_notes = True
            continue
        if in_notes:
            if re.match(r'^\d+[\.\)]\s*', text):
                note_text = re.sub(r'^\d+[\.\)]\s*', '', text)
                if len(note_text) > 5:
                    notes.append(note_text)
            elif text.upper().startswith(('MATERIAL', 'REMOVE', 'ALL DIM', 'HOSE', 'MUST', 'WHITE')):
                notes.append(text)
            else:
                # Check if we've left the notes section
                if any(kw in text.lower() for kw in ('material', 'weight', 'title', 'customer')):
                    in_notes = False

    # Also look for standalone note-like text (numbered sentences with keywords)
    for det in detections:
        text = det['text'].strip()
        m = re.match(r'^(\d+)\.\s*(.+)', text)
        if m and len(m.group(2)) > 10:
            note_text = m.group(2)
            if any(kw in note_text.lower() for kw in ('material', 'remove', 'conform', 'free from', 'marked', 'thickness', 'hose')):
                notes.append(note_text)

    # Deduplicate notes (OCR may detect same note text in multiple views)
    seen = set()
    unique_notes = []
    for note in notes:
        # Normalize aggressively: lowercase, strip all spaces/punctuation/symbols
        key = re.sub(r'[^a-z0-9]', '', note.lower())
        if key not in seen:
            seen.add(key)
            unique_notes.append(note)

    return unique_notes


def _extract_gdt(detections: list[dict], img_height: float) -> list[dict]:
    """
    Extract GD&T feature control frames from detections.

    OCR splits GD&T frames into fragments like:
      "⌖⌀" "0.5" "Ⓜ" "A B" "Ⓜ"   (all at similar Y position)
      "⏥"  "1"                        (flatness)
      "0.5M A BM"                      (merged by OCR)
      "0.2M" "A BM"                    (partially merged)

    Strategy: find GD&T symbol detections, then gather nearby fragments.
    """
    gdt_items = []
    used_indices = set()

    for i, det in enumerate(detections):
        if i in used_indices:
            continue
        text = det['text'].strip()

        # --- Look for GD&T anchor symbols ---
        is_gdt_anchor = False
        symbol_name = None

        # Unicode symbols
        if text.startswith('⌖') or text.startswith('⊕'):
            is_gdt_anchor = True
            symbol_name = 'Position'
        elif text.startswith('⏥'):
            is_gdt_anchor = True
            symbol_name = 'Flatness'
        elif text.startswith('⌓'):
            is_gdt_anchor = True
            symbol_name = 'Circularity'
        elif text.startswith('⊥'):
            is_gdt_anchor = True
            symbol_name = 'Perpendicularity'
        elif text.startswith('∥') or text == '//':
            is_gdt_anchor = True
            symbol_name = 'Parallelism'

        # Text-based: "0.5M A BM" or "0.2M"
        m_merged = re.match(r'^(\d+(?:\.\d+)?)M?\s*((?:[A-Z]\s*(?:M\s*)?)+)?$', text)
        if not is_gdt_anchor and m_merged:
            # Check if nearby detections include GD&T symbols
            cx, cy = _bbox_center(det['bbox'])
            for j, other in enumerate(detections):
                if j == i or j in used_indices:
                    continue
                ox, oy = _bbox_center(other['bbox'])
                if abs(oy - cy) < 30 and abs(ox - cx) < 200:
                    if any(s in other['text'] for s in ('⌖', '⊕', '⏥', '⌓', '⊥', '∥', '①', '④')):
                        is_gdt_anchor = True
                        symbol_name = 'Position'  # default, refine below
                        for s_char, s_name in GDT_SYMBOLS.items():
                            if s_char in other['text']:
                                symbol_name = s_name
                                break
                        used_indices.add(j)
                        break

        if not is_gdt_anchor:
            continue

        used_indices.add(i)
        cx, cy = _bbox_center(det['bbox'])

        # Gather nearby fragments on the same line (within 30px Y, 300px X)
        fragments = [text]
        for j, other in enumerate(detections):
            if j == i or j in used_indices:
                continue
            ox, oy = _bbox_center(other['bbox'])
            if abs(oy - cy) < 30 and abs(ox - cx) < 300:
                other_text = other['text'].strip()
                # Only gather GD&T-related fragments (values, modifiers, datums)
                if re.match(r'^[\d.]+$', other_text) or other_text in ('Ⓜ', 'Ⓛ', 'M') \
                   or re.match(r'^[A-Z](\s+[A-Z])?$', other_text) \
                   or re.match(r'^[A-Z]\s*[BⓂM]', other_text) \
                   or '⌀' in other_text or 'Ø' in other_text:
                    fragments.append(other_text)
                    used_indices.add(j)

        merged = ' '.join(fragments)

        # Parse the merged GD&T frame
        tolerance_value = None
        modifier = None
        datums = []

        # Extract numeric tolerance value
        val_match = re.search(r'(\d+(?:\.\d+)?)', merged)
        if val_match:
            tolerance_value = val_match.group(1)

        # Check if diameter tolerance (⌀ or Ø present) — only for Position
        if symbol_name == 'Position' and ('⌀' in merged or 'Ø' in merged):
            if tolerance_value:
                tolerance_value = f"Ø{tolerance_value}"

        # Modifier
        if 'Ⓜ' in merged or 'M' in merged.replace('MM', '').replace('MATERIAL', ''):
            modifier = 'MMC'
        elif 'Ⓛ' in merged:
            modifier = 'LMC'

        # Datum references
        # Find standalone capital letters (A, B, C) that are datums
        for d_match in re.finditer(r'\b([A-Z])\b', merged):
            letter = d_match.group(1)
            if letter in ('A', 'B', 'C', 'D', 'E', 'F') and letter not in ('M', 'X'):
                datum = letter
                if modifier == 'MMC':
                    datum += '(M)'
                datums.append(datum)

        if tolerance_value:
            gdt_items.append({
                'symbol': symbol_name,
                'tolerance_value': tolerance_value,
                'modifier': modifier,
                'datum_references': datums,
                'applied_to': '',
            })

    # Deduplicate by symbol+value
    seen = set()
    unique = []
    for g in gdt_items:
        key = f"{g['symbol']}_{g['tolerance_value']}"
        if key not in seen:
            seen.add(key)
            unique.append(g)

    return unique


def _is_symbol_fragment(text: str) -> bool:
    """Check if text is an engineering symbol fragment that should be merged."""
    return text in (
        '⌀', 'Ø', 'ø', '∅', 'Φ', 'φ',  # diameter
        '⌵⌀', '⌵', 'V',                  # countersink
        '⌖⌀', '⌖', '⊕',                  # position
        '⏥', '⌓', '⊥', '∥', '//',        # flatness, circularity, perp, parallel
        '±', '°', '°±',                    # tolerance/angle fragments
        'Ⓜ', 'Ⓛ', 'M',                   # modifiers (standalone M)
    ) or re.match(r'^[+\-]\d+(\.\d+)?$', text)  # +0.13, -0.38


def _is_value_fragment(text: str) -> bool:
    """Check if text is a numeric value that could be part of an annotation."""
    return bool(re.match(r'^\d+(\.\d+)?$', text))


def _is_multiplier(text: str) -> bool:
    """Check if text is a multiplier prefix like '3 X', '4X'."""
    return bool(re.match(r'^\d+\s*[Xx×]$', text.strip()))


def _merge_annotation_fragments(detections: list[dict], y_threshold: float = 15) -> list[dict]:
    """
    Pre-processing pass: merge fragmented OCR detections that belong to the
    same engineering annotation.

    OCR often splits symbols from their values:
      "⌀" + "16" → should be "Ø16"
      "3 X" + "⌀" + "11.5" → "3 X Ø11.5"
      "⏥" + "1" → "⏥ 1"
      "⌀" + "9" + "±" + "0.25" → "Ø9±0.25"
      "4 X" + "⌵⌀" + "22.5" + "±" + "0.25 X 90" + "°±" + "1" + "°"

    Strategy: group detections on same Y line, find clusters containing
    engineering symbols, merge them left-to-right.
    """
    if not detections:
        return detections

    # Step 1: Group by Y proximity
    groups = []
    used = set()

    for i, det in enumerate(detections):
        if i in used:
            continue
        cy = _bbox_center(det['bbox'])[1]
        group = [i]
        used.add(i)
        for j, other in enumerate(detections):
            if j in used:
                continue
            oy = _bbox_center(other['bbox'])[1]
            if abs(oy - cy) <= y_threshold:
                group.append(j)
                used.add(j)
        groups.append(group)

    # Step 2: For each group, check if it contains symbol fragments
    merged_detections = []
    consumed = set()

    for group_indices in groups:
        group_dets = [(idx, detections[idx]) for idx in group_indices]
        # Sort by X within the line
        group_dets.sort(key=lambda x: _bbox_center(x[1]['bbox'])[0])

        has_symbol = any(
            _is_symbol_fragment(d['text'].strip()) or _is_multiplier(d['text'].strip())
            for _, d in group_dets
        )

        if not has_symbol or len(group_dets) < 2:
            # No symbols — keep detections as-is
            continue

        # Step 3: Find annotation clusters within this line
        # An annotation cluster starts at a multiplier or symbol and grabs
        # consecutive nearby fragments
        i = 0
        while i < len(group_dets):
            idx, det = group_dets[i]
            text = det['text'].strip()

            # Check if this starts an annotation cluster
            starts_cluster = (
                _is_multiplier(text) or
                _is_symbol_fragment(text) or
                # A number right before a symbol
                (_is_value_fragment(text) and i + 1 < len(group_dets) and
                 _is_symbol_fragment(group_dets[i + 1][1]['text'].strip()))
            )

            if not starts_cluster:
                i += 1
                continue

            # Gather the cluster: grab consecutive fragments that are part of
            # the same annotation (symbols, values, tolerances, modifiers)
            cluster_indices = [idx]
            cluster_texts = [text]
            cx = _bbox_center(det['bbox'])[0]
            j = i + 1

            while j < len(group_dets):
                jdx, jdet = group_dets[j]
                jtext = jdet['text'].strip()
                jcx = _bbox_center(jdet['bbox'])[0]

                # Stop if too far away horizontally (>250px gap)
                if jcx - cx > 250:
                    break

                # Include if it's a symbol, value, modifier, or tolerance-like
                should_include = (
                    _is_symbol_fragment(jtext) or
                    _is_value_fragment(jtext) or
                    _is_multiplier(jtext) or
                    bool(re.match(r'^[\d.]+\s*[Xx×]\s*\d+', jtext)) or  # "0.25 X 90"
                    jtext in ('A', 'B', 'C', 'A B', 'A BM')  # datum refs
                )

                if should_include:
                    cluster_indices.append(jdx)
                    cluster_texts.append(jtext)
                    cx = jcx
                    j += 1
                else:
                    break

            # Only merge if cluster has 2+ items and contains a symbol
            if len(cluster_indices) >= 2 and any(_is_symbol_fragment(t) for t in cluster_texts):
                # Build merged text
                merged_text = ' '.join(cluster_texts)
                # Normalize: replace ⌀ with Ø for consistency
                merged_text = merged_text.replace('⌀', 'Ø').replace('⌵Ø', '⌵Ø')
                # Clean up spacing: "3 X Ø 11.5" → "3 X Ø11.5", "Ø 16" → "Ø16"
                merged_text = re.sub(r'Ø\s+', 'Ø', merged_text)
                merged_text = re.sub(r'⌵\s*Ø', '⌵Ø', merged_text)
                # Clean tolerance: "± 0.25" → "±0.25"
                merged_text = re.sub(r'±\s+', '±', merged_text)
                # Clean: "22.5 ± 0.25 X 90 ° ± 1 °" → "22.5±0.25 X 90°±1°"
                merged_text = re.sub(r'\s*°\s*', '°', merged_text)

                avg_conf = sum(detections[ci]['confidence'] for ci in cluster_indices) / len(cluster_indices)
                merged_detections.append({
                    'text': merged_text,
                    'confidence': round(avg_conf, 3),
                    'bbox': detections[cluster_indices[0]]['bbox'],
                    '_merged': True,
                })
                consumed.update(cluster_indices)

            i = j if j > i + 1 else i + 1

    # Build final list: merged annotations + untouched originals
    result = list(merged_detections)
    for i, det in enumerate(detections):
        if i not in consumed:
            result.append(det)

    return result


def structure_detections(detections: list[dict], img_height: float = 3000) -> dict:
    """
    Main structuring function: takes raw OCR detections and produces
    structured extraction result with dimensions, GD&T, title block, notes.

    This is the LOCAL replacement for Claude Vision.
    """
    # Pre-processing: merge fragmented engineering annotations
    detections = _merge_annotation_fragments(detections)

    # Sort detections top-to-bottom, left-to-right for reading order
    detections_sorted = sorted(detections, key=lambda d: (
        _bbox_center(d['bbox'])[1],  # y first
        _bbox_center(d['bbox'])[0],  # then x
    ))

    # Extract title block
    title_block = _extract_title_block(detections_sorted, img_height)

    # Extract notes
    notes = _extract_notes(detections_sorted, img_height)

    # Extract GD&T
    gdt_items = _extract_gdt(detections_sorted, img_height)

    # Extract dimensions — skip title block region and tolerance-only fragments
    dimensions = []
    seen_values = set()

    for det in detections_sorted:
        # Skip low-confidence detections
        if det['confidence'] < 0.5:
            continue

        is_tb = _is_in_lower_region(det['bbox'], img_height, 0.7)
        text = det['text'].strip()
        parsed = _classify_and_parse_detection(text, is_title_block=is_tb)

        for dim in parsed:
            # Dedup by nominal value
            dedup_key = f"{dim['type']}_{dim['nominal_value']}"
            if dedup_key in seen_values:
                continue
            seen_values.add(dedup_key)

            dim['suggested_instrument'] = _suggest_instrument(
                dim['type'],
                dim.get('tolerance_upper'),
            )
            dim['confidence'] = round(det['confidence'] * 100, 1)
            dim['_bbox'] = det['bbox']  # keep bbox for tolerance proximity merge
            dimensions.append(dim)

    # --- Post-processing: merge nearby tolerance fragments with parent dimensions ---
    # Find orphan tolerance fragments: "+0.13", "-0.38", "+0.25", "0"
    tol_fragments = []
    for det in detections_sorted:
        text = det['text'].strip()
        if re.match(r'^[+]\d+(\.\d+)?$', text):
            tol_fragments.append({'text': text, 'bbox': det['bbox'], 'type': 'upper'})
        elif re.match(r'^[-]\d+(\.\d+)?$', text):
            tol_fragments.append({'text': text, 'bbox': det['bbox'], 'type': 'lower'})

    # For each tolerance fragment, find the nearest dimension and attach it
    for frag in tol_fragments:
        fcx, fcy = _bbox_center(frag['bbox'])
        best_dim = None
        best_dist = float('inf')
        for dim in dimensions:
            if dim.get('_bbox'):
                dcx, dcy = _bbox_center(dim['_bbox'])
                dist = math.sqrt((fcx - dcx) ** 2 + (fcy - dcy) ** 2)
                if dist < best_dist and dist < 200:  # within 200px
                    best_dist = dist
                    best_dim = dim
        if best_dim:
            if frag['type'] == 'upper' and not best_dim.get('tolerance_upper'):
                best_dim['tolerance_upper'] = frag['text']
            elif frag['type'] == 'lower' and not best_dim.get('tolerance_lower'):
                best_dim['tolerance_lower'] = frag['text']

    # Clean up temp fields and update instruments now that tolerances are merged
    for dim in dimensions:
        dim.pop('_bbox', None)
        dim['suggested_instrument'] = _suggest_instrument(
            dim['type'],
            dim.get('tolerance_upper'),
        )

    # Apply general tolerance to dimensions that have no explicit tolerance
    gen_tol = title_block.get('general_tolerance', '')
    if gen_tol:
        for dim in dimensions:
            if not dim.get('tolerance_upper') and dim['type'] in ('linear', 'diameter'):
                dim['tolerance_upper'] = gen_tol
                dim['tolerance_lower'] = gen_tol

    # Sort dimensions: diameters first, then linear, then radius, then others
    type_order = {'diameter': 0, 'linear': 1, 'radius': 2, 'countersink': 3, 'angle': 4, 'thread': 5}
    dimensions.sort(key=lambda d: type_order.get(d['type'], 99))

    # Assign IDs
    for i, dim in enumerate(dimensions, 1):
        dim['id'] = i
    for i, gdt in enumerate(gdt_items, 1):
        gdt['id'] = i

    return {
        'title_block': title_block,
        'dimensions': dimensions,
        'gdt': gdt_items,
        'notes': notes,
        'balloon_count': None,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def extract_drawing(pdf_path: str) -> dict:
    """
    Run the full LOCAL extraction pipeline on a PDF drawing.
    No external API calls. Everything runs on-premise.
    """
    # Step 1: Try programmatic text extraction first (fastest, most accurate)
    pdf_pages = extract_pdf_text_blocks(pdf_path)
    has_embedded = any(p["has_embedded_text"] for p in pdf_pages)

    # Step 2: Rasterize — 150 DPI is enough since we also have embedded text
    images = pdf_to_images(pdf_path, dpi=150)

    all_pages = []
    for png_bytes, page_num in images:
        # Step 3: OCR extraction
        ocr_results = ocr_extract(png_bytes)

        # If we have embedded text, merge it with OCR for better coverage
        if has_embedded and page_num <= len(pdf_pages):
            pdf_blocks = pdf_pages[page_num - 1]["blocks"]
            for block in pdf_blocks:
                if block["text"] and len(block["text"]) > 0:
                    # Add PDF text blocks as high-confidence detections
                    ocr_results.append({
                        "text": block["text"],
                        "confidence": 0.99,
                        "bbox": block["bbox"],
                    })

        # Step 4: Get image dimensions for title block detection
        img = Image.open(io.BytesIO(png_bytes))
        img_height = img.height

        # Step 5: Structure the detections using rule-based parser
        extraction = structure_detections(ocr_results, img_height)

        all_pages.append({
            "page": page_num,
            "ocr_detection_count": len(ocr_results),
            "extraction": extraction,
        })

    # Merge multi-page results
    if len(all_pages) == 1:
        result = all_pages[0]["extraction"]
    else:
        merged = all_pages[0]["extraction"]
        for page_data in all_pages[1:]:
            ext = page_data["extraction"]
            merged["dimensions"].extend(ext.get("dimensions", []))
            merged["gdt"].extend(ext.get("gdt", []))
            merged["notes"].extend(ext.get("notes", []))
        for i, dim in enumerate(merged["dimensions"], 1):
            dim["id"] = i
        for i, gdt in enumerate(merged["gdt"], 1):
            gdt["id"] = i
        result = merged

    result["_meta"] = {
        "pages": len(all_pages),
        "has_embedded_text": has_embedded,
        "ocr_detections": sum(p["ocr_detection_count"] for p in all_pages),
        "engine": "local-paddleocr-rulebased",
    }

    # Include first page preview image to avoid double rasterization in caller
    if images:
        result["_preview_png"] = images[0][0]

    return result
