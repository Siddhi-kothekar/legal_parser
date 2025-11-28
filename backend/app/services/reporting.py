from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from app.config import settings
from app.models.schemas import ReportSummary
from app.services.case_graph import CaseGraphService
from app.services.timeline import TimelineEvent
from app.services.model_manager import model_manager
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class ReportingService:
    def __init__(self) -> None:
        self.case_graph = CaseGraphService()

    def persist_summary(
        self,
        case_id: str,
        events: List[TimelineEvent],
        inconsistencies: List[dict[str, str]],
        missing: List[str],
        extracted_data: List[dict] = None,
        system_warnings: List[str] | None = None,
    ) -> ReportSummary:
        # Build detailed report with all extracted content
        warnings = list(system_warnings or [])
        detailed_report = {
            "case_id": case_id,
            "generated_at": datetime.utcnow().isoformat(),
            "timeline_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "source": event.source,
                    "description": event.description,
                }
                for event in events
            ],
            "inconsistencies": inconsistencies,
            "missing_evidence": missing,
            "extracted_content": [],
            "warnings": warnings,
        }
        
        # Add all extracted data (text, entities, objects, OCR, etc.)
        if extracted_data:
            for data in extracted_data:
                filename = data.get("filename") or data.get("file") or data.get("source") or "unknown"
                content_item = {
                    "file_type": data.get("type", "unknown"),
                    "classification": data.get("classification", {}),
                    "extracted_text": "",
                    "entities": [],
                    "objects_detected": [],
                    "ocr_text": "",
                    "timestamps": [],
                    "locations": [],
                    "summary": "",
                    "filename": filename,
                }
                
                # Document content
                if data.get("type") == "document":
                    content_item["extracted_text"] = data.get("raw_text", "")  # Full text, no truncation
                    # Filter entities to remove non-entity words
                    raw_entities = data.get("entities", [])
                    filtered_entities = []
                    non_entity_words = {"time visible", "action taken", "case reference", "incident description", 
                                       "sections applied", "doctor's note", "findings", "notes", "the", "date", 
                                       "march", "frame", "pm", "am", "name", "age", "occupation", "address", "phone"}
                    for entity in raw_entities:
                        entity_text = (entity.get("entity") or entity.get("text") or "").strip()
                        if entity_text and len(entity_text) > 1:
                            lower_text = entity_text.lower()
                            # Skip if it's a non-entity word or too short
                            if lower_text not in non_entity_words and len(entity_text) > 2:
                                # Skip single words that are common English words
                                if not (len(entity_text.split()) == 1 and lower_text in {"the", "a", "an", "is", "are", "was", "were"}):
                                    filtered_entities.append(entity)
                    content_item["entities"] = filtered_entities
                    # Preserve original timestamp formats from time_mentions and dates
                    time_mentions = data.get("time_mentions", [])
                    dates = data.get("dates", [])
                    content_item["timestamps"] = time_mentions + dates  # Keep original formats
                    content_item["dates"] = dates
                    content_item["events"] = data.get("events", [])
                    content_item["legal_entities"] = data.get("legal_entities", {})
                    content_item["summary"] = data.get("summary", "")
                    content_item["filename"] = data.get("filename") or data.get("file") or data.get("source") or "unknown"
                    # Use normalized persons/locations from extraction
                    content_item["persons"] = data.get("persons", [])
                    content_item["locations"] = data.get("locations", [])
                    content_item["injuries"] = data.get("injuries", [])
                    content_item["weapons"] = data.get("weapons", [])
                    content_item["camera_id"] = data.get("camera_id")  # Add camera ID if present
                
                # Image content
                elif data.get("type") == "image":
                    content_item["ocr_text"] = data.get("ocr_text", "")  # Full OCR text
                    content_item["objects_detected"] = data.get("objects", [])
                    content_item["timestamps"] = data.get("ocr_timestamps", [])
                    content_item["locations"] = data.get("ocr_locations", [])
                    content_item["gps_coordinates"] = data.get("gps_coordinates")
                    content_item["image_timestamp"] = data.get("timestamp")
                    content_item["filename"] = filename
                    content_item["persons"] = data.get("persons", [])
                    content_item["weapons"] = data.get("weapons", [])
                
                detailed_report["extracted_content"].append(content_item)
        
        # Build evidence map and add to the detailed report
        evidence_map = self._build_evidence_map(extracted_data)
        detailed_report["evidence_map"] = evidence_map

        # Generate case summary - prefer LLM summarizer, fallback to rule-based
        case_summary_text, summary_warning = self._summarize_case(case_id, evidence_map, detailed_report)
        if summary_warning:
            warnings.append(summary_warning)
        detailed_report["case_summary"] = case_summary_text
        detailed_report["relationships"] = self.case_graph.compute_relationships_for_case(
            case_id, detailed_report
        )
        # Build evidence-to-evidence relationships
        detailed_report["evidence_relationships"] = self._build_evidence_relationships(detailed_report.get("extracted_content", []))
        
        summary = ReportSummary(
            case_id=case_id,
            generated_at=datetime.utcnow(),
            timeline_events=len(events),
            inconsistencies=len(inconsistencies),
            missing_evidence=missing,
            report_path=str(self._pdf_path(case_id)),
            preview={
                "timeline": [],
                "inconsistencies": inconsistencies,
                "generated_at": datetime.utcnow().isoformat(),
                "evidence_map": detailed_report.get("evidence_map", {}),
                "case_summary": case_summary_text,
                "relationships": detailed_report.get("relationships", []),
                "evidence_relationships": detailed_report.get("evidence_relationships", []),
                "warnings": warnings,
            },
        )
        
        # Save detailed JSON report
        self._json_path(case_id).write_text(
            json.dumps(detailed_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # Ensure reciprocal case files include the current relationships
        self.case_graph.refresh_relationship_references(case_id, detailed_report.get("relationships", []))
        
        # Save basic summary for API response
        summary_path = settings.reports_dir / f"{case_id}_summary.json"
        summary_path.write_text(summary.model_dump_json(), encoding="utf-8")
        
        # Generate PDF placeholder
        self._pdf_path(case_id).write_text(
            "PDF generation placeholder. Integrate ReportLab/WeasyPrint.", encoding="utf-8"
        )
        return summary

    def _build_evidence_map(self, extracted_data: List[dict]) -> dict:
        """Build a consolidated evidence map aggregating key entities across all artifacts."""
        evidence_map = {
            "case_persons": [],
            "case_locations": [],
            "case_timestamps": [],
            "case_injuries": [],
            "case_weapons": [],
            "ipc_sections": [],
            "fir_numbers": [],
            "mlc_numbers": [],
            "hospital_names": [],
            "vehicle_numbers": [],
            "files": [],
            "consolidated_text": "",
        }

        persons = set()
        locations = set()
        timestamps = set()
        injuries = set()
        weapons = set()
        ipc_sections = set()
        fir_numbers = set()
        mlc_numbers = set()
        hospitals = set()
        vehicles = set()
        consolidated_texts = []

        weapon_keywords = ["Knife", "knife", "gun", "firearm", "pistol", "revolver", "weapon", "blade"]

        # entity -> files mapping
        entity_files = {
            "persons": {},
            "locations": {},
            "timestamps": {},
            "injuries": {},
            "weapons": {},
        }

        for data in extracted_data:
            entry = {
                "file": data.get("filename") or data.get("file") or data.get("source") or "unknown",
                "type": data.get("type", "unknown"),
                "classification": data.get("classification", {}),
                "summary": data.get("summary", "")[:500],
                "persons": data.get("persons", []),
                "locations": data.get("locations", []),
                "timestamps": data.get("timestamps", []) or data.get("time_mentions", []),
                "injuries": data.get("injuries", []),
                "weapons": data.get("weapons", []),
                "legal_entities": data.get("legal_entities", {}),
            }
            evidence_map["files"].append(entry)

            # Consolidate document text
            if data.get("type") == "document":
                raw_text = data.get("raw_text", "")
                if raw_text:
                    consolidated_texts.append(raw_text)
                # Gather normalized lists from extraction output
                doc_persons = data.get("persons", []) or []
                doc_locations = data.get("locations", []) or []
                doc_timestamps = data.get("timestamps", []) or []
                doc_injuries = data.get("injuries", []) or []
                doc_weapons = data.get("weapons", []) or []

                for p in doc_persons:
                    persons.add(p)
                    entity_files["persons"].setdefault(p, []).append(entry["file"])
                for l in doc_locations:
                    locations.add(l)
                    entity_files["locations"].setdefault(l, []).append(entry["file"])
                for t in doc_timestamps:
                    timestamps.add(t)
                    entity_files["timestamps"].setdefault(t, []).append(entry["file"])
                for inj in doc_injuries:
                    injuries.add(inj)
                    entity_files["injuries"].setdefault(inj, []).append(entry["file"])
                for w in doc_weapons:
                    weapons.add(w)
                    entity_files["weapons"].setdefault(w, []).append(entry["file"])

            # Legal entities
            legal = data.get("legal_entities") or {}
            for k, v in (legal or {}).items():
                if not v:
                    continue
                if k == "ipc_sections":
                    for _ in v:
                        ipc_sections.add(_)
                if k == "fir_numbers":
                    for _ in v:
                        fir_numbers.add(_)
                if k == "mlc_numbers":
                    for _ in v:
                        mlc_numbers.add(_)
                if k == "hospital_names":
                    for _ in v:
                        hospitals.add(_)
                if k == "vehicle_numbers":
                    for _ in v:
                        vehicles.add(_)
                if k == "injury_terms":
                    for _ in v:
                        injuries.add(_)

            # From image data
            if data.get("type") == "image":
                # OCR timestamps
                for t in data.get("ocr_timestamps", []) or []:
                    timestamps.add(t)
                # EXIF timestamp
                if data.get("timestamp"):
                    timestamps.add(data.get("timestamp"))
                # GPS locations
                if data.get("gps_coordinates"):
                    loc = data.get("gps_coordinates")
                    locations.add(f"GPS({loc.get('latitude')},{loc.get('longitude')})")
                # Objects detected
                for obj in data.get("objects", []) or []:
                    class_name = obj.get("class")
                    if not class_name:
                        continue
                    if class_name.lower() in [w.lower() for w in weapon_keywords]:
                        weapons.add(class_name)
                    if class_name.lower() == "person":
                        # optional face id mapping
                        persons.add(obj.get("label") or obj.get("class") or "person")
                # Also read normalized fields from image extraction
                for p in data.get("persons", []) or []:
                    persons.add(p)
                    entity_files["persons"].setdefault(p, []).append(entry["file"])
                for w in data.get("weapons", []) or []:
                    weapons.add(w)
                    entity_files["weapons"].setdefault(w, []).append(entry["file"])
                for l in data.get("ocr_locations", []) or []:
                    locations.add(l)
                    entity_files["locations"].setdefault(l, []).append(entry["file"])
                for t in data.get("timestamps", []) or []:
                    timestamps.add(t)
                    entity_files["timestamps"].setdefault(t, []).append(entry["file"])
            # Injuries and weapons mention in doc text
            try:
                raw_text = data.get("raw_text", "") or data.get("ocr_text", "")
                if raw_text:
                    for kw in weapon_keywords:
                        if kw.lower() in raw_text.lower():
                            weapons.add(kw)
                    # look for timestamp patterns
                    for pat in [r"\b\d{1,2}:\d{2}:\d{2}\b", r"\b\d{1,2}:\d{2}\s?(am|pm)\b"]:
                        matches = re.findall(pat, raw_text, flags=re.IGNORECASE)
                        for m in matches:
                            timestamps.add(m if isinstance(m, str) else str(m))
            except Exception:
                pass

        # Build consolidated_text
        evidence_map["consolidated_text"] = "\n---\n".join(consolidated_texts)[:20000]

        # Populate final lists
        evidence_map["case_persons"] = sorted(list(persons))
        evidence_map["case_locations"] = sorted(list(locations))
        evidence_map["case_timestamps"] = sorted(list(timestamps))
        evidence_map["case_injuries"] = sorted(list(injuries))
        evidence_map["case_weapons"] = sorted(list(weapons))
        evidence_map["ipc_sections"] = sorted(list(ipc_sections))
        evidence_map["fir_numbers"] = sorted(list(fir_numbers))
        evidence_map["mlc_numbers"] = sorted(list(mlc_numbers))
        evidence_map["hospital_names"] = sorted(list(hospitals))
        evidence_map["vehicle_numbers"] = sorted(list(vehicles))
        evidence_map["entity_files"] = entity_files

        return evidence_map

    def _summarize_case(self, case_id: str, evidence_map: dict, detailed_report: dict) -> tuple[str, Optional[str]]:
        """Summarize a case using local TF-IDF or LLM when available."""
        # Use OpenAI LLM if available with model_manager.llm_reasoning
        if settings.enable_real_ai and settings.openai_api_key:
            prompt = (
                "Create a concise case summary from the following evidence map and extracted content. "
                "Include top persons, locations, timestamps, injuries, weapons and missing evidence suggestions. \n\n"
                f"Evidence map: {json.dumps(evidence_map)[:5000]}\n\n"
                "Provide the summary in 4-6 bullet points."
            )
            try:
                resp = model_manager.llm_reasoning(prompt, model=settings.openai_model)
                if resp.lower().startswith("llm reasoning error"):
                    warning = self._humanize_llm_error(resp)
                else:
                    return resp, None
            except Exception as e:
                print(f"LLM summarization failed: {e}")
                warning = self._humanize_llm_error(str(e))
        else:
            warning = None

        # Fallback: extractive summary using TFIDF
        try:
            text = evidence_map.get("consolidated_text", "")
            if not text:
                return "No textual content available to summarize.", warning
            # Split to sentences
            import re
            sentences = re.split(r'(?<=[.!?]) +', text)
            if len(sentences) <= 6:
                return "\n".join([s.strip() for s in sentences[:6]]), warning

            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(sentences)
            scores = np.array(X.sum(axis=1)).ravel()
            top_idxs = (-scores).argsort()[:5]
            summary = "\n".join([sentences[i].strip() for i in sorted(top_idxs)])
            return summary, warning
        except Exception as e:
            print(f"Extractive summary failed: {e}")
            return "Summary generation failed.", warning

    def _json_path(self, case_id: str) -> Path:
        return settings.reports_dir / f"{case_id}.json"

    def _pdf_path(self, case_id: str) -> Path:
        return settings.reports_dir / f"{case_id}.pdf"

    def _humanize_llm_error(self, raw_error: str) -> str:
        lower = raw_error.lower()
        if "429" in lower or "insufficient_quota" in lower:
            return "LLM summary skipped: OpenAI quota exceeded."
        if "api key" in lower:
            return "LLM summary skipped: OpenAI API key missing or invalid."
        return "LLM summary skipped due to upstream error."
    
    def _build_evidence_relationships(self, extracted_content: List[dict]) -> List[dict]:
        """Build relationships between evidence files (CCTV → FIR, FIR → Medical Report, etc.)."""
        relationships = []
        if not extracted_content or len(extracted_content) < 2:
            return relationships
        
        # Get file names and classifications
        files = []
        for content in extracted_content:
            filename = content.get("filename") or content.get("file") or "unknown"
            file_type = content.get("file_type", "unknown")
            classification = content.get("classification", {})
            label = classification.get("label", "").lower()
            files.append({
                "filename": filename,
                "type": file_type,
                "label": label,
                "persons": set(content.get("persons", []) or []),
                "locations": set(content.get("locations", []) or []),
                "timestamps": set(content.get("timestamps", []) or []),
                "injuries": set(content.get("injuries", []) or []),
                "weapons": set(content.get("weapons", []) or []),
            })
        
        # Compare each file with every other file
        for i, file1 in enumerate(files):
            for j, file2 in enumerate(files[i+1:], start=i+1):
                matches = []
                
                # Check for matching persons
                common_persons = file1["persons"] & file2["persons"]
                if common_persons:
                    matches.append(f"Matches persons: {', '.join(sorted(common_persons))}")
                
                # Check for matching locations
                common_locations = file1["locations"] & file2["locations"]
                if common_locations:
                    matches.append(f"Matches location: {', '.join(sorted(common_locations))}")
                
                # Check for matching timestamps
                common_timestamps = file1["timestamps"] & file2["timestamps"]
                if common_timestamps:
                    matches.append(f"Matches timeline ({', '.join(sorted(common_timestamps))})")
                
                # Check for matching injuries
                common_injuries = file1["injuries"] & file2["injuries"]
                if common_injuries:
                    matches.append(f"Injury matches: {', '.join(sorted(common_injuries))}")
                
                # Check for matching weapons
                common_weapons = file1["weapons"] & file2["weapons"]
                if common_weapons:
                    matches.append(f"Weapon: {', '.join(sorted(common_weapons))}")
                
                # Determine relationship type based on file types
                source_name = file1["filename"]
                target_name = file2["filename"]
                
                # Map common file patterns to readable names
                if "cctv" in source_name.lower() or "camera" in source_name.lower():
                    source_display = "CCTV"
                elif "fir" in source_name.lower():
                    source_display = "FIR"
                elif "med" in source_name.lower() or "medical" in source_name.lower() or "mlc" in source_name.lower():
                    source_display = "Medical Report"
                elif "witness" in source_name.lower():
                    source_display = "Witness Statement"
                else:
                    source_display = source_name
                
                if "cctv" in target_name.lower() or "camera" in target_name.lower():
                    target_display = "CCTV"
                elif "fir" in target_name.lower():
                    target_display = "FIR"
                elif "med" in target_name.lower() or "medical" in target_name.lower() or "mlc" in target_name.lower():
                    target_display = "Medical Report"
                elif "witness" in target_name.lower():
                    target_display = "Witness Statement"
                else:
                    target_display = target_name
                
                if matches:
                    relationships.append({
                        "source": source_display,
                        "target": target_display,
                        "matches": matches,
                    })
        
        return relationships
    
    def _build_structured_timeline(self, events: List[TimelineEvent], inconsistencies: List[dict], extracted_data: List[dict]) -> List[dict]:
        """Build a structured timeline with source, event, and conflict notes."""
        structured = []
        
        # Map inconsistencies to events
        inconsistency_map = {}
        for inc in inconsistencies:
            inc_type = inc.get("type", "")
            details = inc.get("details", "")
            # Try to extract time from inconsistency details
            time_match = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm))', details, re.IGNORECASE)
            if time_match:
                time_key = time_match.group(1)
                if time_key not in inconsistency_map:
                    inconsistency_map[time_key] = []
                inconsistency_map[time_key].append(inc)
        
        if not events:
            print("[Reporting] Warning: No events to build structured timeline")
            return []
        
        for event in events:
            try:
                # Format time
                if event.timestamp and hasattr(event.timestamp, 'strftime'):
                    time_str = event.timestamp.strftime("%I:%M:%S %p") if event.timestamp.second else event.timestamp.strftime("%I:%M %p")
                else:
                    time_str = "N/A"
                
                # Check for conflicts at this time
                notes = []
                if event.timestamp and hasattr(event.timestamp, 'strftime'):
                    event_time_str = event.timestamp.strftime("%I:%M %p")
                    if event_time_str in inconsistency_map:
                        for inc in inconsistency_map[event_time_str]:
                            notes.append(f"{inc.get('type', 'conflict')}: {inc.get('details', '')[:100]}")
                
                structured.append({
                    "time": time_str,
                    "source": event.source or "Unknown",
                    "event": event.description[:200] if event.description else "N/A",
                    "notes": "; ".join(notes) if notes else None,
                })
            except Exception as e:
                print(f"[Reporting] Error formatting event: {e}")
                structured.append({
                    "time": "N/A",
                    "source": "Unknown",
                    "event": str(event.description)[:200] if hasattr(event, 'description') else "N/A",
                    "notes": None,
                })
        
        return structured
    
    def _generate_case_summary(self, events: List[TimelineEvent], inconsistencies: List[dict], extracted_data: List[dict]) -> str:
        """Generate a concise case summary consolidating all evidence."""
        summary_parts = []
        
        # Extract key information from events
        if events:
            summary_parts.append("TIMELINE OF EVENTS:")
            for event in events[:10]:  # Limit to first 10 events
                time_str = event.timestamp.strftime("%I:%M %p")
                summary_parts.append(f"  • {time_str} ({event.source}): {event.description[:150]}")
        
        # Add inconsistencies
        if inconsistencies:
            summary_parts.append("\nINCONSISTENCIES IDENTIFIED:")
            for inc in inconsistencies:
                inc_type = inc.get("type", "Unknown")
                details = inc.get("details", "")
                severity = inc.get("severity", "moderate")
                summary_parts.append(f"  • [{severity.upper()}] {inc_type}: {details[:200]}")
        
        # Extract key entities and locations
        locations = set()
        persons = set()
        for data in extracted_data:
            if data.get("type") == "document":
                for entity in data.get("entities", []):
                    if entity.get("label") in ["GPE", "LOC", "FAC"]:
                        locations.add(entity.get("entity", ""))
                    elif entity.get("label") == "PERSON":
                        persons.add(entity.get("entity", ""))
        
        if locations or persons:
            summary_parts.append("\nKEY ENTITIES:")
            if persons:
                summary_parts.append(f"  • Persons: {', '.join(list(persons)[:5])}")
            if locations:
                summary_parts.append(f"  • Locations: {', '.join(list(locations)[:5])}")
        
        return "\n".join(summary_parts)

