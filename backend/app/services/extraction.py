"""
Step 4: Content Extraction from images and documents using real AI models.

Uses:
- PIL/EXIF for image metadata
- YOLO for object detection
- Tesseract OCR for text in images
- pdfplumber for PDF text extraction
- BERT/spaCy NER for entity extraction
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import re

from PIL import Image, ExifTags
import pdfplumber
import pytesseract

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

from app.config import settings
from app.services.model_manager import model_manager


class ExtractionService:
    """
    Content Extraction Service with real AI models.
    """

    _LOCATION_KEYWORDS = {
        "road",
        "street",
        "st.",
        "avenue",
        "lane",
        "drive",
        "highway",
        "bypass",
        "bridge",
        "market",
        "mall",
        "shop",
        "store",
        "hospital",
        "clinic",
        "medical",
        "center",
        "centre",
        "station",
        "police station",
        "ps ",
        "village",
        "town",
        "district",
        "city",
        "colony",
        "nagar",
        "layout",
        "block",
        "building",
        "apartment",
        "residency",
        "camp",
        "chowk",
        "square",
        "park",
        "garden",
    }

    _MONTH_NAMES = {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    }

    _NON_PERSON_TERMS = {
        "witness",
        "statement",
        "section",
        "action",
        "taken",
        "incident",
        "report",
        "police",
        "station",
        "case",
        "evidence",
        "document",
        "summary",
        "complaint",
        "victim",
        "suspect",
        "accused",
        "officer",
        "action taken",
    }

    _KNOWN_LOCATIONS = {
        "mumbai",
        "pune",
        "nagpur",
        "delhi",
        "new delhi",
        "bangalore",
        "bengaluru",
        "hyderabad",
        "kolkata",
        "chennai",
        "ahmedabad",
        "surat",
        "thane",
        "nashik",
        "aurangabad",
        "kolhapur",
        "jaipur",
        "lucknow",
        "indore",
        "bhopal",
        "patna",
        "kochi",
        "goa",
    }

    _LOCATION_SUFFIXES = {
        "pur",
        "pura",
        "garh",
        "nagar",
        "abad",
        "ville",
        "city",
        "town",
        "gram",
        "wadi",
        "wadi.",
    }

    _PERSON_TITLES = {
        "mr",
        "mrs",
        "ms",
        "miss",
        "smt",
        "shri",
        "dr",
        "adv",
        "advocate",
        "inspector",
        "constable",
        "si",
        "asi",
        "pi",
        "sub-inspector",
    }

    def extract(self, artifact_path: Path) -> Dict[str, Any]:
        """
        Extract content from evidence file.
        
        Returns:
            Dict with extracted metadata, entities, objects, OCR text, etc.
        """
        extension = artifact_path.suffix.lower()
        if extension in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
            return self._extract_image_metadata(artifact_path)
        return self._extract_document_metadata(artifact_path)

    # -----------------
    # Image extraction
    # -----------------
    def _extract_image_metadata(self, artifact_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from images:
        - EXIF timestamps and GPS
        - Object detection using YOLO
        - OCR text using Tesseract
        """
        timestamp = None
        gps = None
        gps_coords = None

        # Extract EXIF metadata
        try:
            image = Image.open(artifact_path)
            exif = {
                ExifTags.TAGS.get(k, k): v
                for k, v in (image._getexif() or {}).items()
                if k in ExifTags.TAGS
            }
            raw_dt = exif.get("DateTimeOriginal") or exif.get("DateTime")
            if raw_dt:
                try:
                    timestamp = datetime.strptime(raw_dt, "%Y:%m:%d %H:%M:%S").isoformat()
                except ValueError:
                    # Try alternative formats
                    try:
                        timestamp = datetime.strptime(raw_dt, "%Y-%m-%d %H:%M:%S").isoformat()
                    except ValueError:
                        pass

            # Extract GPS coordinates
            gps_info = exif.get("GPSInfo")
            if gps_info:
                gps = "GPS_AVAILABLE"
                # Convert GPS to decimal degrees
                try:
                    lat_ref = gps_info.get(1, "N")
                    lat_data = gps_info.get(2, (0, 0, 0))
                    lon_ref = gps_info.get(3, "E")
                    lon_data = gps_info.get(4, (0, 0, 0))

                    lat = float(lat_data[0]) + float(lat_data[1]) / 60.0 + float(lat_data[2]) / 3600.0
                    if lat_ref == "S":
                        lat = -lat

                    lon = float(lon_data[0]) + float(lon_data[1]) / 60.0 + float(lon_data[2]) / 3600.0
                    if lon_ref == "W":
                        lon = -lon

                    gps_coords = {"latitude": lat, "longitude": lon}
                except Exception:
                    pass
        except Exception as e:
            print(f"EXIF extraction error: {e}")

        # Object detection using YOLO
        objects_detected = []
        if settings.enable_real_ai:
            try:
                objects_detected = model_manager.detect_objects_yolo(artifact_path)
            except Exception as e:
                print(f"YOLO detection error: {e}")

        # Identify persons and weapons from detected objects
        persons_detected = []
        weapons_detected = []
        try:
            for obj in objects_detected:
                cls_name = obj.get("class") or obj.get("name")
                if not cls_name:
                    continue
                if cls_name.lower() == "person":
                    persons_detected.append(obj.get("class") or "person")
                if cls_name.lower() in ["knife", "gun", "blade", "weapon"]:
                    weapons_detected.append(cls_name)
        except Exception:
            pass

        # OCR text extraction using Tesseract
        ocr_text = ""
        ocr_timestamps = []
        ocr_locations = []
        try:
            image = Image.open(artifact_path)
            ocr_text = pytesseract.image_to_string(image)
            
            # Extract timestamps from OCR text
            time_patterns = [
                r"\b\d{1,2}:\d{2}:\d{2}\b",  # HH:MM:SS
                r"\b\d{1,2}:\d{2}\s?(AM|PM|am|pm)\b",  # HH:MM AM/PM
                r"\b\d{4}[-/]\d{2}[-/]\d{2}\s+\d{1,2}:\d{2}\b",  # Date + time
            ]
            for pattern in time_patterns:
                matches = re.findall(pattern, ocr_text)
                ocr_timestamps.extend(matches)

            # Extract location-like text (capitalized words, street names, etc.)
            location_patterns = [
                r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(Road|Street|Avenue|Lane|Drive)\b",
                r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(Shop|Store|Market|Mall)\b",
            ]
            for pattern in location_patterns:
                matches = re.findall(pattern, ocr_text)
                ocr_locations.extend([m[0] if isinstance(m, tuple) else m for m in matches])
        except Exception as e:
            print(f"OCR extraction error: {e}")

        notes_text = "EXIF and OCR extraction completed."
        if settings.enable_real_ai and getattr(settings, 'enable_object_detection', True):
            notes_text = "EXIF, YOLO object detection, and OCR extraction completed."
        else:
            notes_text = "EXIF and OCR extraction completed. Object detection disabled."

        return {
            "type": "image",
            "timestamp": timestamp,
            "location": gps or "UNKNOWN",
            "gps_coordinates": gps_coords,
            "objects": objects_detected,
            "persons": list(set(persons_detected)),
            "ocr_text": ocr_text[:1000],  # Limit OCR text length
            "ocr_timestamps": list(set(ocr_timestamps))[:10],
            "ocr_locations": list(set(ocr_locations))[:10],
            "timestamps": [timestamp] + list(set(ocr_timestamps))[:10] if timestamp else list(set(ocr_timestamps))[:10],
            "weapons": list(set(weapons_detected)),
            "notes": notes_text,
        }

    # -----------------
    # Document extraction
    # -----------------
    def _extract_document_metadata(self, artifact_path: Path) -> Dict[str, Any]:
        """
        Extract content from documents:
        - Text extraction from PDFs
        - Named Entity Recognition using BERT/spaCy
        - Time expression extraction
        - Event/action extraction
        """
        text = self._read_pdf_text(artifact_path)
        
        # Extract entities using BERT/spaCy NER
        entities = []
        if settings.enable_real_ai and text:
            try:
                entities = model_manager.extract_entities_bert(text)
            except Exception as e:
                print(f"Entity extraction error: {e}")
                entities = self._simple_entity_guess(text)
        else:
            entities = self._simple_entity_guess(text)

        # Add custom legal entities (IPC, FIR numbers, MLC, Hospital, vehicle numbers, injuries)
        legal_entities = self._extract_legal_entities(text)
        # Merge legal entities into entities list
        for key, values in legal_entities.items():
            if values:
                entities.append({"entity_type": key, "values": values})

        # Extract time expressions
        time_mentions = self._extract_time_expressions(text)

        # Extract dates
        dates = self._extract_dates(text)

        # Extract actions/events (simple pattern matching)
        events = self._extract_events(text)

        summary = text[:500] + "..." if len(text) > 500 else text

        # Normalize entities and extract per-file structured lists
        # Persons and locations from NER
        persons = set()
        locations = set()
        for ent in entities:
            if isinstance(ent, dict):
                ent_text = ent.get("entity") or ent.get("text")
                ent_label = ent.get("label")
            else:
                ent_text = ent
                ent_label = None
            if not ent_text or not isinstance(ent_text, str):
                continue
            # Filter out tokenization artifacts and formatting characters
            ent_text_clean = self._clean_entity_text(ent_text.strip())
            if not ent_text_clean or len(ent_text_clean) < 2:
                continue
            # Skip single letters, tokenization artifacts, and formatting characters
            if len(ent_text_clean) == 1 or ent_text_clean.startswith("##"):
                continue
            # Skip ASCII box-drawing characters and formatting artifacts
            if self._is_formatting_artifact(ent_text_clean):
                continue
            # Skip bullet points and list markers
            if ent_text_clean.startswith("•") or ent_text_clean.startswith("-") or ent_text_clean.startswith("*"):
                continue
            classification = self._classify_entity(ent_text_clean, ent_label)
            if classification == "person":
                persons.add(ent_text_clean)
            elif classification == "location":
                locations.add(ent_text_clean)

        # Injuries from legal_entities - filter out negated terms
        injuries = set()
        text_lower = (text or "").lower()
        for injury_term in legal_entities.get("injury_terms", []):
            # Check if injury is negated (e.g., "no fracture")
            if not self._is_negated_injury(text_lower, injury_term.lower()):
                injuries.add(injury_term)

        # Weapons from text heuristic
        weapons = set()
        for kw in ["knife", "gun", "blade", "weapon", "sharp"]:
            if kw in (text or "").lower():
                weapons.add(kw)

        # Normalize timestamps: try to parse time_mentions and dates to ISO if possible
        timestamps = []
        try:
            from dateutil import parser as _parser
        except Exception:
            _parser = None
        for raw in (time_mentions or []):
            parsed = raw
            if _parser:
                try:
                    dt = _parser.parse(raw, fuzzy=True)
                    parsed = dt.isoformat()
                except Exception:
                    parsed = raw
            timestamps.append(parsed)
        for raw in (dates or []):
            parsed = raw
            if _parser:
                try:
                    dt = _parser.parse(raw, fuzzy=True)
                    parsed = dt.isoformat()
                except Exception:
                    parsed = raw
            timestamps.append(parsed)

        return {
            "type": "document",
            "raw_text": text,
            "entities": entities,
            "time_mentions": time_mentions,
            "dates": dates,
            "events": events,
            "legal_entities": legal_entities,
            "persons": list(persons),
            "locations": list(locations),
            "timestamps": timestamps,
            "injuries": list(injuries),
            "weapons": list(weapons),
            "summary": summary,
        }

    def _read_pdf_text(self, artifact_path: Path) -> str:
        """Extract text from PDF or plain text files."""
        if artifact_path.suffix.lower() != ".pdf":
            try:
                return artifact_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return ""

        chunks: List[str] = []
        try:
            with pdfplumber.open(artifact_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    chunks.append(page_text)
        except Exception:
            # Fallback: try OCR on scanned PDFs
            if PDF2IMAGE_AVAILABLE:
                try:
                    images = convert_from_path(str(artifact_path), dpi=200)
                    for img in images:
                        text = pytesseract.image_to_string(img)
                        chunks.append(text)
                except Exception as e:
                    print(f"PDF OCR extraction error: {e}")
            else:
                print("pdf2image not available. Install poppler for scanned PDF OCR.")

        return "\n".join(chunks)

    def _simple_entity_guess(self, text: str) -> List[Dict[str, str]]:
        """Fallback regex-based entity extraction."""
        entities: List[Dict[str, str]] = []
        # Look for capitalized names (potential persons or locations)
        for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text[:5000]):
            candidate = match.group(1)
            if len(candidate.split()) <= 4:
                classification = self._classify_entity(candidate, None)
                if classification == "person":
                    label = "PERSON"
                elif classification == "location":
                    label = "LOCATION"
                else:
                    label = "PERSON_OR_LOCATION"
                entities.append({"entity": candidate, "label": label})
        return entities[:50]

    def _classify_entity(self, text: str, label: Optional[str]) -> str:
        """
        Classify entity text into person/location/other using model labels + heuristics.
        Returns one of {"person", "location", "other"}.
        """
        if not text:
            return "other"

        normalized = text.strip()
        lower_text = normalized.lower()
        tokens = normalized.split()
        token_count = len(tokens)

        if self._looks_like_datetime(normalized):
            return "other"

        # Trust spaCy/BERT labels when confident
        if label == "PERSON":
            if self._looks_like_person_name(tokens):
                return "person"
        if label in {"GPE", "LOC"}:
            return "location"
        if label == "FAC":
            if self._looks_like_person_name(tokens):
                return "person"
            return "location"
        if label == "ORG":
            if self._looks_like_location_name(normalized, lower_text, token_count):
                return "location"
            if self._looks_like_person_name(tokens):
                return "person"

        # Person heuristics
        if self._looks_like_person_name(tokens):
            return "person"
        if token_count <= 4 and token_count >= 2:
            first_token = tokens[0].rstrip(".").lower()
            if first_token in self._PERSON_TITLES:
                return "person"
            if normalized.replace(".", "").replace(",", "").istitle():
                if not self._looks_like_location_name(normalized, lower_text, token_count):
                    return "person"

        # Location heuristics
        if self._looks_like_location_name(normalized, lower_text, token_count):
            return "location"
        if token_count == 1:
            if lower_text in self._KNOWN_LOCATIONS:
                return "location"
            if normalized.isupper() and len(normalized) > 3:
                return "other"

        return "other"

    def _looks_like_person_name(self, tokens: List[str]) -> bool:
        """Check if tokens resemble a human name (2-4 capitalized words, no digits)."""
        cleaned_tokens = [token.strip(".,") for token in tokens if token.strip(".,")]
        if not 2 <= len(cleaned_tokens) <= 4:
            return False
        has_digit = any(any(char.isdigit() for char in token) for token in cleaned_tokens)
        if has_digit:
            return False
        capitalized_tokens = all(token[0].isupper() for token in cleaned_tokens if token)
        if not capitalized_tokens:
            return False
        lower_tokens = [token.lower() for token in cleaned_tokens]
        if any(token in self._LOCATION_KEYWORDS for token in lower_tokens):
            return False
        if any(token in self._MONTH_NAMES for token in lower_tokens):
            return False
        if any(token in self._NON_PERSON_TERMS for token in lower_tokens):
            return False
        combined = " ".join(lower_tokens)
        if combined in self._NON_PERSON_TERMS:
            return False
        return True

    def _looks_like_location_name(self, normalized: str, lower_text: str, token_count: int) -> bool:
        """Determine if entity resembles a location/address."""
        if any(keyword in lower_text for keyword in self._LOCATION_KEYWORDS):
            return True
        if any(lower_text.endswith(suffix) for suffix in self._LOCATION_SUFFIXES):
            return True
        if lower_text in self._KNOWN_LOCATIONS:
            return True
        if any(char.isdigit() for char in normalized) and token_count >= 2:
            return True
        return False

    def _looks_like_datetime(self, text: str) -> bool:
        """Identify date or time expressions that shouldn't be treated as names."""
        lower_text = text.lower()
        if any(month in lower_text for month in self._MONTH_NAMES):
            return True
        if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text):
            return True
        if re.search(r"\b\d{4}-\d{2}-\d{2}\b", text):
            return True
        if re.search(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", text):
            return True
        if re.search(r"\b\d{1,2}\s+(am|pm)\b", lower_text):
            return True
        return False

    def _extract_time_expressions(self, text: str) -> List[str]:
        """Extract time expressions from text."""
        patterns = [
            r"\b\d{1,2}:\d{2}\s?(AM|PM|am|pm)\b",
            r"\b\d{1,2}:\d{2}:\d{2}\b",
            r"\b\d{1,2}:\d{2}\b",
            r"\b(at|around|approximately)\s+\d{1,2}\s?(AM|PM|am|pm)\b",
        ]
        found: List[str] = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            found.extend([m if isinstance(m, str) else " ".join(m) for m in matches])
        return list(set(found))[:20]

    def _extract_dates(self, text: str) -> List[str]:
        """Extract date expressions from text."""
        patterns = [
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            r"\b\d{4}[-/]\d{2}[-/]\d{2}\b",
            r"\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",  # "14 March 2025"
        ]
        found: List[str] = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Reconstruct date string from tuple
                    found.append(" ".join(match))
                else:
                    found.append(match)
        return list(set(found))[:20]
    
    def _clean_entity_text(self, text: str) -> str:
        """Remove tokenization artifacts and clean entity text."""
        if not text:
            return ""
        # Remove BERT tokenization artifacts
        text = re.sub(r'^##+', '', text)
        # Remove ASCII box-drawing characters and formatting artifacts
        text = re.sub(r'[═║╔╗╚╝╠╣╦╩╬─│┼┴┬├┤└┘┌┐]', '', text)
        # Remove single letter entities that are likely artifacts
        if len(text.strip()) == 1 and text.strip().isalpha():
            return ""
        # Remove entities that are just punctuation or numbers
        if text.strip().isdigit() or not any(c.isalpha() for c in text):
            return ""
        # Remove common formatting prefixes
        text = re.sub(r'^[•\-\*]\s*', '', text)
        return text.strip()
    
    def _is_formatting_artifact(self, text: str) -> bool:
        """Check if text is a formatting artifact (box-drawing, separators, etc.)."""
        if not text:
            return True
        # Check for ASCII box-drawing characters
        box_chars = set('═║╔╗╚╝╠╣╦╩╬─│┼┴┬├┤└┘┌┐')
        if any(c in box_chars for c in text):
            return True
        # Check if it's mostly formatting characters
        if len(text) <= 3 and all(c in '═║─│•\-\*' for c in text):
            return True
        # Check for common formatting patterns
        formatting_patterns = [
            r'^[═\-_]{2,}$',  # Lines of dashes/equals
            r'^[│|]{1,3}$',  # Vertical bars
            r'^[•\-\*]\s*$',  # Just a bullet
        ]
        for pattern in formatting_patterns:
            if re.match(pattern, text):
                return True
        return False
    
    def _is_negated_injury(self, text_lower: str, injury_term: str) -> bool:
        """Check if an injury term is negated in the text."""
        negators = ["no", "without", "absent", "ruled out", "negative for", "not"]
        idx = text_lower.find(injury_term)
        while idx != -1:
            window_start = max(0, idx - 50)
            window = text_lower[window_start:idx]
            if any(neg in window for neg in negators):
                return True
            idx = text_lower.find(injury_term, idx + 1)
        return False

    def _extract_events(self, text: str) -> List[Dict[str, str]]:
        """Extract action/event phrases from text."""
        # Simple pattern matching for common crime-related actions
        action_patterns = [
            (r"(entered|went into|arrived at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "location_entry"),
            (r"(heard|witnessed|saw|observed)\s+([a-z]+(?:\s+[a-z]+)*)", "observation"),
            (r"(injured|stabbed|hit|attacked|assaulted)", "violence"),
            (r"(left|exited|departed)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "location_exit"),
        ]
        events = []
        for pattern, event_type in action_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                events.append({
                    "event": match.group(0),
                    "type": event_type,
                    "context": text[max(0, match.start() - 50):match.end() + 50],
                })
        return events[:30]

    def _extract_legal_entities(self, text: str) -> Dict[str, list]:
        """
        Extract legal-specific entities using regex and heuristics.
        Returns a dictionary of entity lists.
        """
        entities = {
            "ipc_sections": [],
            "fir_numbers": [],
            "mlc_numbers": [],
            "hospital_names": [],
            "vehicle_numbers": [],
            "injury_terms": [],
        }

        try:
            # IPC sections
            ipc_matches = re.findall(r"IPC\s*(?:Section)?\s*[:#-]?\s*(\d{1,4})", text, re.IGNORECASE)
            entities["ipc_sections"] = list(set(ipc_matches))

            # FIR numbers (simple heuristic)
            fir_matches = re.findall(r"\bFIR\s*(?:No\.?|#|:)?\s*([\d\-/]+)\b", text, re.IGNORECASE)
            entities["fir_numbers"] = list(set(fir_matches))

            # MLC numbers
            mlc_matches = re.findall(r"\bMLC\s*(?:No\.?|#|:)?\s*([\d\-/]+)\b", text, re.IGNORECASE)
            entities["mlc_numbers"] = list(set(mlc_matches))

            # Hospital names - search for '(.*) Hospital' or '(.*) Clinic'
            hosp_matches = re.findall(r"([A-Z][a-zA-Z\s]+(?:Hospital|Clinic|Medical Center|Medical Centre|Infirmary))", text)
            entities["hospital_names"] = list(set([m.strip() for m in hosp_matches]))

            # Vehicle registration numbers (India-like format heuristic, general fallback)
            vehicle_matches = re.findall(r"\b([A-Z]{2,3}-?\d{1,4}-?[A-Z]{1,3}-?\d{1,4})\b", text)
            entities["vehicle_numbers"] = list(set(vehicle_matches))

            # Injury terms with some context - but exclude negated ones
            injury_patterns = [
                r"stabbed|stabb?ing|stab\s*\w*", 
                r"laceration[s]?\s*\d{1,2}\s*cm", 
                r"laceration[s]?", 
                r"fracture[s]?|compound fracture", 
                r"blunt force|blunt trauma|blunt force injury|deep laceration",
                r"cut\s+wound",
                r"bleeding",
            ]
            injuries_found = []
            text_lower = text.lower()
            for pat in injury_patterns:
                matches = re.findall(pat, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match_str = " ".join(match)
                    else:
                        match_str = match
                    # Only add if not negated
                    if not self._is_negated_injury(text_lower, match_str.lower()):
                        injuries_found.append(match_str)
            entities["injury_terms"] = list(set(injuries_found))
        except Exception as e:
            print(f"Legal entity extraction error: {e}")

        return entities
