"""
Tests for the extraction pipeline using real OCR output fixtures.

These fixtures come from the actual PaddleOCR output on URI-4643708C1-01.pdf.
No OCR/PaddleOCR dependency needed to run these tests — they test the
structuring logic only.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.extractor import (
    _classify_and_parse_detection,
    _extract_gdt,
    _extract_notes,
    _extract_title_block,
    _merge_annotation_fragments,
    _parse_tolerance,
    structure_detections,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures: real OCR fragments from URI-4643708C1-01.pdf
# ═══════════════════════════════════════════════════════════════

# These mirror the actual detection output at 150 DPI (img height 1654)
IMG_HEIGHT = 1654


def _det(text, x, y, conf=0.99):
    """Shorthand to build a detection dict."""
    return {"text": text, "confidence": conf, "bbox": [x, y, x + 50, y + 13]}


# Drawing area dimensions (upper 70% of image)
DIMENSION_DETS = [
    _det("⌀", 478, 447),
    _det("16", 494, 447),
    _det("+0.13", 530, 443),
    _det("-0.38", 530, 452),
    _det("3 X", 446, 131),
    _det("⌀", 462, 131),
    _det("11.5", 474, 131),
    _det("+0.25", 516, 127),
    _det("⌀", 295, 390),
    _det("9", 307, 390),
    _det("±", 322, 390),
    _det("0.25", 334, 390),
    _det("55", 378, 236),
    _det("72", 405, 224),
    _det("130.5", 200, 427),
    _det("44.9", 315, 852, conf=0.89),
    _det("2 X R14", 105, 410, conf=0.93),
    _det("6 X R10", 396, 366),
    _det("9±0.25", 864, 1080, conf=1.0),
]

COUNTERSINK_DETS = [
    _det("4 X", 415, 101),
    _det("⌵⌀", 437, 101),
    _det("22.5", 452, 101),
    _det("±", 482, 101),
    _det("0.25 X 90", 492, 101),
    _det("°±", 557, 101),
    _det("1", 572, 101),
    _det("°", 582, 101),
]

GDT_DETS = [
    _det("⏥", 524, 151),
    _det("1", 538, 152),
    _det("⌖⌀", 190, 463),
    _det("0.5", 212, 463),
    _det("Ⓜ", 237, 463),
    _det("A B", 250, 463),
    _det("Ⓜ", 282, 463),
    _det("⌖⌀", 451, 147),
    _det("0.2", 472, 147),
    _det("Ⓜ", 497, 147),
    _det("A", 510, 147),
]

TITLE_BLOCK_DETS = [
    _det("TITLE:-", 601, 1478),
    _det("SUPPORT,HOOD BRACKET-LH", 724, 1478),
    _det("DRG NO:", 1669, 1525),
    _det("URI-4643708C1 B-01", 1848, 1558),
    _det("CUSTOMER", 1702, 1377),
    _det("URI", 1838, 1376),
    _det("DRN.", 1397, 1434),
    _det("GAURAV", 1484, 1424),
    _det("CHKD.MELRIC", 1444, 1502),
    _det("14.01.26", 1578, 1504),
    _det("APVD.KARTHIK14.01.26", 1492, 1572),
    _det("SCALE", 486, 1519),
    _det("NTS", 610, 1519),
    _det("±0.25", 305, 1551),
    _det("CL.", 2096, 1534),
    _det("A", 2097, 1582),
    _det("SHEET NO.", 2193, 1535),
    _det("01 of 01", 2192, 1582),
]

NOTE_DETS = [
    _det("NOTE:-", 1318, 1232),
    _det("1.MATERIAL:- STEEL GRADE 1 OR 2 AS PER MPAPS A-6 PART-1.", 1500, 1260),
    _det("2. MATERIAL THICKNESS 8±0.3 MM.", 1500, 1290),
    _det("3. REMOVE ALL BURRS AND SHARP EDGES.", 1500, 1321),
    # Duplicate from second view
    _det("1.MATERIAL:-STEEL GRADE 1OR 2 AS PER MPAPS A-6 PART-1.", 600, 455, conf=0.97),
    _det("3. REMOVE ALL BURRS AND SHARP EDGES.", 600, 476),
]

ALL_DETS = DIMENSION_DETS + COUNTERSINK_DETS + GDT_DETS + TITLE_BLOCK_DETS + NOTE_DETS


# ═══════════════════════════════════════════════════════════════
# Tests: tolerance parsing
# ═══════════════════════════════════════════════════════════════

class TestToleranceParsing:
    def test_bilateral(self):
        assert _parse_tolerance("±0.25") == ("±0.25", "±0.25")

    def test_unilateral(self):
        assert _parse_tolerance("+0.13/-0.38") == ("+0.13", "-0.38")

    def test_no_tolerance(self):
        assert _parse_tolerance("55") == (None, None)

    def test_bilateral_in_context(self):
        u, l = _parse_tolerance("Ø9±0.25")
        assert u == "±0.25"


# ═══════════════════════════════════════════════════════════════
# Tests: fragment merging
# ═══════════════════════════════════════════════════════════════

class TestFragmentMerger:
    def test_diameter_merged(self):
        dets = [_det("⌀", 478, 447), _det("16", 494, 447)]
        merged = _merge_annotation_fragments(dets)
        merged_texts = [d["text"] for d in merged if d.get("_merged")]
        assert any("Ø16" in t for t in merged_texts)

    def test_multiplied_diameter_merged(self):
        dets = [_det("3 X", 446, 131), _det("⌀", 462, 131), _det("11.5", 474, 131)]
        merged = _merge_annotation_fragments(dets)
        merged_texts = [d["text"] for d in merged if d.get("_merged")]
        assert any("3 X" in t and "11.5" in t for t in merged_texts)

    def test_flatness_merged(self):
        dets = [_det("⏥", 524, 151), _det("1", 538, 152)]
        merged = _merge_annotation_fragments(dets)
        merged_texts = [d["text"] for d in merged if d.get("_merged")]
        assert any("⏥" in t and "1" in t for t in merged_texts)

    def test_plain_numbers_not_merged(self):
        dets = [_det("55", 378, 236), _det("72", 405, 224)]
        merged = _merge_annotation_fragments(dets)
        assert not any(d.get("_merged") for d in merged)

    def test_diameter_with_tolerance_merged(self):
        dets = [
            _det("⌀", 478, 447), _det("16", 494, 447),
            _det("+0.13", 530, 443), _det("-0.38", 530, 452),
        ]
        merged = _merge_annotation_fragments(dets)
        merged_texts = [d["text"] for d in merged if d.get("_merged")]
        assert any("16" in t and "+0.13" in t for t in merged_texts)


# ═══════════════════════════════════════════════════════════════
# Tests: dimension classification
# ═══════════════════════════════════════════════════════════════

class TestDimensionClassification:
    def test_linear(self):
        result = _classify_and_parse_detection("55")
        assert len(result) == 1
        assert result[0]["type"] == "linear"
        assert result[0]["nominal_value"] == "55"

    def test_diameter(self):
        result = _classify_and_parse_detection("Ø16 +0.13 -0.38")
        assert len(result) == 1
        assert result[0]["type"] == "diameter"
        assert result[0]["nominal_value"] == "Ø16"
        assert result[0]["tolerance_upper"] == "+0.13"
        assert result[0]["tolerance_lower"] == "-0.38"

    def test_radius(self):
        result = _classify_and_parse_detection("2 X R14")
        assert len(result) == 1
        assert result[0]["type"] == "radius"
        assert result[0]["nominal_value"] == "2 X R14"

    def test_number_with_tolerance(self):
        result = _classify_and_parse_detection("9±0.25")
        assert len(result) == 1
        assert result[0]["nominal_value"] == "9"
        assert result[0]["tolerance_upper"] == "±0.25"

    def test_sub_1mm_filtered(self):
        """Values < 1.0 are tolerance fragments, not dimensions."""
        assert _classify_and_parse_detection("0.25") == []
        assert _classify_and_parse_detection("0.5") == []
        assert _classify_and_parse_detection("0.13") == []

    def test_title_block_skipped(self):
        assert _classify_and_parse_detection("55", is_title_block=True) == []

    def test_skip_words(self):
        assert _classify_and_parse_detection("GAURAV") == []
        assert _classify_and_parse_detection("ISOMETRIC VIEW") == []
        assert _classify_and_parse_detection("NTS") == []

    def test_date_skipped(self):
        assert _classify_and_parse_detection("14.01.26") == []

    def test_tolerance_fragments_skipped(self):
        assert _classify_and_parse_detection("+0.13") == []
        assert _classify_and_parse_detection("-0.38") == []
        assert _classify_and_parse_detection("0") == []

    def test_countersink(self):
        result = _classify_and_parse_detection("4 X ⌵Ø22.5 ±0.25 X 90°±1°")
        assert len(result) == 1
        assert result[0]["type"] == "countersink"


# ═══════════════════════════════════════════════════════════════
# Tests: GD&T extraction
# ═══════════════════════════════════════════════════════════════

class TestGDT:
    def test_flatness_detected(self):
        dets = _merge_annotation_fragments([_det("⏥", 524, 151), _det("1", 538, 152)])
        gdt = _extract_gdt(dets, IMG_HEIGHT)
        assert len(gdt) >= 1
        flatness = [g for g in gdt if g["symbol"] == "Flatness"]
        assert len(flatness) == 1
        assert flatness[0]["tolerance_value"] == "1"

    def test_position_with_datums(self):
        dets = _merge_annotation_fragments([
            _det("⌖⌀", 190, 463), _det("0.5", 212, 463),
            _det("Ⓜ", 237, 463), _det("A B", 250, 463), _det("Ⓜ", 282, 463),
        ])
        gdt = _extract_gdt(dets, IMG_HEIGHT)
        positions = [g for g in gdt if g["symbol"] == "Position"]
        assert len(positions) >= 1
        pos = positions[0]
        assert "0.5" in pos["tolerance_value"]
        assert len(pos["datum_references"]) >= 1


# ═══════════════════════════════════════════════════════════════
# Tests: title block
# ═══════════════════════════════════════════════════════════════

class TestTitleBlock:
    def test_part_name(self):
        tb = _extract_title_block(TITLE_BLOCK_DETS, IMG_HEIGHT)
        assert tb["part_name"] == "SUPPORT,HOOD BRACKET-LH"

    def test_drawing_number(self):
        tb = _extract_title_block(TITLE_BLOCK_DETS, IMG_HEIGHT)
        assert "URI-4643708C1" in tb["drawing_number"]

    def test_customer(self):
        tb = _extract_title_block(TITLE_BLOCK_DETS, IMG_HEIGHT)
        assert tb["customer"] == "URI"

    def test_drawn_by(self):
        tb = _extract_title_block(TITLE_BLOCK_DETS, IMG_HEIGHT)
        assert tb["drawn_by"] == "GAURAV"

    def test_checked_by(self):
        tb = _extract_title_block(TITLE_BLOCK_DETS, IMG_HEIGHT)
        assert tb["checked_by"] == "MELRIC"

    def test_scale(self):
        tb = _extract_title_block(TITLE_BLOCK_DETS, IMG_HEIGHT)
        assert tb["scale"] == "NTS"

    def test_general_tolerance(self):
        tb = _extract_title_block(TITLE_BLOCK_DETS, IMG_HEIGHT)
        assert tb["general_tolerance"] == "±0.25"


# ═══════════════════════════════════════════════════════════════
# Tests: notes deduplication
# ═══════════════════════════════════════════════════════════════

class TestNotes:
    def test_notes_extracted(self):
        notes = _extract_notes(NOTE_DETS, IMG_HEIGHT)
        assert len(notes) >= 2

    def test_notes_deduplicated(self):
        notes = _extract_notes(NOTE_DETS, IMG_HEIGHT)
        # "REMOVE ALL BURRS" appears twice in fixtures — should be deduped
        burr_notes = [n for n in notes if "BURRS" in n.upper()]
        assert len(burr_notes) == 1


# ═══════════════════════════════════════════════════════════════
# Tests: full pipeline integration (no OCR, no PDF)
# ═══════════════════════════════════════════════════════════════

class TestFullStructuring:
    def test_all_dimension_types_found(self):
        result = structure_detections(ALL_DETS, IMG_HEIGHT)
        types = {d["type"] for d in result["dimensions"]}
        assert "linear" in types
        assert "diameter" in types
        assert "radius" in types

    def test_no_tolerance_fragments_as_dimensions(self):
        result = structure_detections(ALL_DETS, IMG_HEIGHT)
        for d in result["dimensions"]:
            val = d["nominal_value"]
            assert val not in ("0.25", "0.5", "0.2", "0.13", "0.38", "0")

    def test_gdt_present(self):
        result = structure_detections(ALL_DETS, IMG_HEIGHT)
        assert len(result["gdt"]) >= 2

    def test_title_block_populated(self):
        result = structure_detections(ALL_DETS, IMG_HEIGHT)
        tb = result["title_block"]
        assert tb["part_name"] != ""
        assert tb["drawing_number"] != ""

    def test_general_tolerance_applied(self):
        result = structure_detections(ALL_DETS, IMG_HEIGHT)
        bare_linear = [d for d in result["dimensions"]
                       if d["type"] == "linear" and d["nominal_value"] == "55"]
        if bare_linear:
            assert bare_linear[0].get("tolerance_upper") == "±0.25"

    def test_dimension_ids_sequential(self):
        result = structure_detections(ALL_DETS, IMG_HEIGHT)
        ids = [d["id"] for d in result["dimensions"]]
        assert ids == list(range(1, len(ids) + 1))
