"""
Step 7 & 8: Inconsistency Detection and Missing Evidence Recommendation.

Uses:
- Rule-based logic for deterministic checks
- OpenAI LLM for advanced reasoning and contradiction detection
- Semantic similarity for grouping related evidence
"""

from typing import List, Dict, Any
import json
import re

from app.config import settings
from app.services.model_manager import model_manager
from app.services.timeline import TimelineEvent


class ReasoningService:
    """
    Reasoning Service for inconsistency detection and missing evidence suggestions.
    """

    def __init__(self) -> None:
        self._warnings: List[str] = []

    def detect_inconsistencies(
        self,
        events: List[TimelineEvent],
        extracted_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Detect inconsistencies using rule-based logic and LLM reasoning.
        
        Args:
            events: Timeline events
            extracted_data: All extracted evidence data
        
        Returns:
            List of inconsistency reports
        """
        self._warnings.clear()
        conflicts: List[Dict[str, Any]] = []

        # First, check if document explicitly mentions inconsistencies
        explicit_inconsistencies = self._extract_explicit_inconsistencies(extracted_data)
        conflicts.extend(explicit_inconsistencies)

        # Rule-based checks (only if we don't already have explicit inconsistencies)
        if not explicit_inconsistencies:
            conflicts.extend(self._check_time_conflicts(events))
            conflicts.extend(self._check_location_conflicts(events, extracted_data))
            conflicts.extend(self._check_injury_severity_conflicts(extracted_data))
            conflicts.extend(self._check_description_conflicts(extracted_data))
        
        # Deduplicate inconsistencies
        conflicts = self._deduplicate_inconsistencies(conflicts)

        # LLM-based reasoning for complex contradictions
        if settings.enable_real_ai and settings.openai_api_key and settings.openai_api_key != "" and settings.openai_api_key != "your_openai_api_key_here":
            print(f"[Reasoning] Using OpenAI LLM for inconsistency detection...")
            llm_conflicts = self._llm_detect_inconsistencies(events, extracted_data)
            print(f"[Reasoning] LLM found {len(llm_conflicts)} inconsistencies")
            conflicts.extend(llm_conflicts)
        else:
            warning = "LLM inconsistency detection skipped: OpenAI key missing or disabled."
            print(f"[Reasoning] ⚠️ {warning}")
            self._record_warning(warning)

        return conflicts
    
    def _extract_explicit_inconsistencies(self, extracted_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract inconsistencies that are explicitly mentioned in the document."""
        inconsistencies = []
        
        for data in extracted_data:
            if data.get("type") != "document":
                continue
                
            text = data.get("raw_text", "").lower()
            
            # Look for sections that explicitly list inconsistencies
            inconsistency_keywords = [
                "inconsistencies identified",
                "inconsistencies found",
                "discrepancy",
                "conflict",
                "contradiction",
                "note:",
                "warning:",
            ]
            
            # Find sections mentioning inconsistencies
            for keyword in inconsistency_keywords:
                if keyword in text:
                    # Extract the section
                    idx = text.find(keyword)
                    section = text[idx:idx+2000]  # Get next 2000 chars
                    
                    # Look for specific inconsistency patterns
                    # Time discrepancy
                    if "time discrepancy" in section or "time conflict" in section:
                        # Extract the details
                        time_pattern = r'(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm))'
                        times = re.findall(time_pattern, section, re.IGNORECASE)
                        if len(times) >= 2:
                            inconsistencies.append({
                                "type": "time_discrepancy",
                                "severity": "moderate",
                                "details": f"Time discrepancy mentioned in document: {times[0]} vs {times[1]}",
                                "source": "document_explicit",
                            })
                    
                    # Clothing/appearance conflict
                    if ("clothing" in section and "change" in section) or ("shirt" in section and ("black" in section and "white" in section)):
                        inconsistencies.append({
                            "type": "description_conflict",
                            "severity": "moderate",
                            "details": "Clothing change mentioned in document (e.g., entered in black shirt, exited in white shirt)",
                            "source": "document_explicit",
                        })
                    
                    # Injury severity conflict
                    if ("minor" in section and ("serious" in section or "severe" in section or "fracture" in section or "deep" in section)):
                        inconsistencies.append({
                            "type": "statement_vs_medical",
                            "severity": "critical",
                            "details": "Injury severity conflict: witness described as minor but medical report shows serious injuries",
                            "source": "document_explicit",
                        })
        
        return inconsistencies
    
    def _deduplicate_inconsistencies(self, inconsistencies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate inconsistencies."""
        seen = set()
        unique = []
        
        for inc in inconsistencies:
            # Create a unique key from type and details
            inc_key = f"{inc.get('type', '')}_{inc.get('details', '')[:100]}"
            if inc_key not in seen:
                seen.add(inc_key)
                unique.append(inc)
        
        return unique

    def _check_time_conflicts(self, events: List[TimelineEvent]) -> List[Dict[str, Any]]:
        """Check for time ordering conflicts."""
        conflicts = []
        if len(events) >= 2:
            for i, (prev, curr) in enumerate(zip(events, events[1:])):
                if curr.timestamp < prev.timestamp:
                    conflicts.append({
                        "type": "time_conflict",
                        "severity": "moderate",
                        "details": f"Event '{curr.description}' appears earlier than previous event '{prev.description}'",
                        "previous_event": prev.description,
                        "previous_time": prev.timestamp.isoformat(),
                        "current_event": curr.description,
                        "current_time": curr.timestamp.isoformat(),
                    })
        return conflicts

    def _check_location_conflicts(
        self,
        events: List[TimelineEvent],
        extracted_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Check for location conflicts (same person at different places at same time)."""
        conflicts = []
        
        # Only check if we have multiple distinct locations mentioned
        all_locations = set()
        for data in extracted_data:
            if data.get("type") == "document":
                text = data.get("raw_text", "").lower()
                # Extract location names (proper nouns, place names)
                location_patterns = [
                    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(Supermarket|Store|Shop|Road|Street|Hospital|Clinic)',
                    r'\b(MG Road|ABC Supermarket|Main Street)',
                ]
                for pattern in location_patterns:
                    matches = re.findall(pattern, data.get("raw_text", ""))
                    for match in matches:
                        if isinstance(match, tuple):
                            all_locations.add(" ".join(match))
                        else:
                            all_locations.add(match)
        
        # Only flag conflicts if we have multiple distinct locations AND events at same time
        if len(all_locations) <= 1:
            return conflicts  # No location conflict if only one location
        
        # Group events by time windows (only check if multiple events at same time)
        time_windows = {}
        for event in events:
            # Round to nearest 5 minutes for grouping
            time_key = event.timestamp.replace(second=0, microsecond=0)
            time_key = time_key.replace(minute=(time_key.minute // 5) * 5)
            if time_key not in time_windows:
                time_windows[time_key] = []
            time_windows[time_key].append(event)

        # Check for conflicting locations in same time window
        for time_key, window_events in time_windows.items():
            if len(window_events) >= 2:
                locations = []
                for event in window_events:
                    # Extract location from event description
                    desc = event.description
                    for loc in all_locations:
                        if loc.lower() in desc.lower():
                            locations.append(loc)
                
                # Only flag if we have multiple distinct locations at the same time
                unique_locations = list(set(locations))
                if len(unique_locations) > 1:
                    conflicts.append({
                        "type": "location_conflict",
                        "severity": "moderate",
                        "details": f"Multiple conflicting locations mentioned around {time_key.strftime('%I:%M %p')}: {', '.join(unique_locations)}",
                        "locations": unique_locations,
                        "time": time_key.isoformat(),
                    })

        return conflicts

    def _check_injury_severity_conflicts(
        self,
        extracted_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Check for conflicts between witness statements and medical reports."""
        conflicts = []

        severity_records = []
        for data in extracted_data:
            if data.get("type") != "document":
                continue
            text = data.get("raw_text", "")
            severity_result = self._assess_injury_severity(text)
            if severity_result["label"] == "unknown":
                continue
            severity_records.append({
                "label": severity_result["label"],
                "phrases": severity_result["phrases"],
                "source": data.get("summary", data.get("filename", "document")),
            })

        labels = {record["label"] for record in severity_records}
        if "major" in labels and "minor" in labels:
            major_sources = [rec for rec in severity_records if rec["label"] == "major"]
            minor_sources = [rec for rec in severity_records if rec["label"] == "minor"]
            conflicts.append({
                "type": "statement_vs_medical",
                "severity": "critical",
                "details": (
                    "Conflicting injury severity descriptions: "
                    f"major indicators ({self._format_phrases(major_sources)}) "
                    f"vs minor indicators ({self._format_phrases(minor_sources)})."
                ),
                "sources": {
                    "major": [rec["source"] for rec in major_sources],
                    "minor": [rec["source"] for rec in minor_sources],
                },
            })

        return conflicts

    def _check_description_conflicts(
        self,
        extracted_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Check for description conflicts (e.g., clothing color, appearance)."""
        conflicts = []
        
        # Look for clothing/appearance descriptions
        descriptions = {}
        for data in extracted_data:
            text = data.get("raw_text", "").lower()
            # Simple pattern matching for clothing colors
            color_patterns = {
                "black": r"\b(black|dark)\s+(shirt|top|clothing|clothes|outfit)\b",
                "white": r"\b(white|light)\s+(shirt|top|clothing|clothes|outfit)\b",
                "red": r"\b(red)\s+(shirt|top|clothing|clothes|outfit)\b",
            }
            
            for color, pattern in color_patterns.items():
                if re.search(pattern, text):
                    if color not in descriptions:
                        descriptions[color] = []
                    descriptions[color].append(data.get("summary", "unknown"))

        # If multiple conflicting colors mentioned
        if len(descriptions) > 1:
            conflicts.append({
                "type": "description_conflict",
                "severity": "moderate",
                "details": "Conflicting clothing/appearance descriptions found",
                "descriptions": descriptions,
            })

        return conflicts

    def _assess_injury_severity(self, text: str) -> Dict[str, Any]:
        """Assess injury severity indicators within a document."""
        text_lower = (text or "").lower()
        if not text_lower.strip():
            return {"label": "unknown", "phrases": []}

        major_terms = [
            "compound fracture",
            "fracture",
            "deep laceration",
            "severe bleeding",
            "severe injury",
            "serious injury",
            "major vessel",
            "critical condition",
        ]
        minor_terms = [
            "minor injury",
            "minor cut",
            "superficial",
            "abrasion",
            "scratch",
            "no fracture",
            "no major vessel injury",
            "stable condition",
        ]

        major_hits = self._find_affirmed_hits(text_lower, major_terms)
        minor_hits = self._find_affirmed_hits(text_lower, minor_terms)
        negated_major = self._find_negated_hits(text_lower, major_terms)
        if negated_major and not minor_hits:
            minor_hits.extend([f"no {term}" for term in negated_major])

        if major_hits:
            return {"label": "major", "phrases": major_hits}
        if minor_hits:
            return {"label": "minor", "phrases": minor_hits}

        return {"label": "unknown", "phrases": []}

    def _find_affirmed_hits(self, text_lower: str, phrases: List[str]) -> List[str]:
        hits = []
        for phrase in phrases:
            normalized = phrase.lower()
            if normalized in text_lower and not self._is_negated_phrase(text_lower, normalized):
                hits.append(phrase)
        return hits

    def _find_negated_hits(self, text_lower: str, phrases: List[str]) -> List[str]:
        hits = []
        for phrase in phrases:
            normalized = phrase.lower()
            if normalized in text_lower and self._is_negated_phrase(text_lower, normalized):
                hits.append(phrase)
        return hits

    def _is_negated_phrase(self, text_lower: str, phrase: str) -> bool:
        negators = ["no", "without", "absent", "ruled out", "negative for"]
        idx = text_lower.find(phrase)
        while idx != -1:
            window_start = max(0, idx - 40)
            window = text_lower[window_start:idx]
            if any(neg in window for neg in negators):
                return True
            idx = text_lower.find(phrase, idx + 1)
        return False

    def _format_phrases(self, records: List[Dict[str, Any]]) -> str:
        phrases = []
        for rec in records:
            for phrase in rec.get("phrases", []):
                phrases.append(phrase)
        return ", ".join(sorted(set(phrases))) or "unspecified indicators"

    def _record_warning(self, message: str) -> None:
        if message not in self._warnings:
            self._warnings.append(message)

    def get_warnings(self) -> List[str]:
        return list(self._warnings)

    def _record_llm_error(self, context: str, raw_error: str) -> None:
        friendly = f"{context} skipped: {self._humanize_llm_error(raw_error)}"
        self._record_warning(friendly)

    def _humanize_llm_error(self, raw_error: str) -> str:
        lower = raw_error.lower()
        if "429" in lower or "insufficient_quota" in lower or "insufficient quota" in lower:
            return "OpenAI quota exceeded. Please update billing or try later."
        if "api key" in lower:
            return "OpenAI API key missing or invalid."
        return "LLM service failed."

    def _llm_detect_inconsistencies(
        self,
        events: List[TimelineEvent],
        extracted_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Use LLM to detect complex inconsistencies."""
        if not events and not extracted_data:
            return []

        # Prepare context for LLM
        events_summary = "\n".join([
            f"- {e.timestamp.isoformat()}: {e.description} (Source: {e.source})"
            for e in events[:20]  # Limit to first 20 events
        ])

        extracted_summary = "\n".join([
            f"- {data.get('type', 'unknown')}: {data.get('summary', '')[:200]}"
            for data in extracted_data[:10]
        ])

        prompt = f"""Analyze the following crime evidence timeline and extracted data for inconsistencies.

Timeline Events:
{events_summary}

Extracted Evidence:
{extracted_summary}

Identify any contradictions, inconsistencies, or conflicts in:
1. Time sequences (events out of order, impossible timelines)
2. Location conflicts (same person at different places simultaneously)
3. Injury severity mismatches (witness says minor, medical says severe)
4. Description conflicts (different clothing, appearance, actions)
5. Statement contradictions (witnesses contradicting each other)

For each inconsistency found, provide:
- Type of inconsistency
- Severity (critical, moderate, info)
- Brief description
- Which evidence sources conflict

Return your analysis as a JSON array of objects with keys: type, severity, details, sources.
If no inconsistencies found, return an empty array [].
"""

        system_prompt = "You are an expert crime analyst reviewing evidence for inconsistencies. Be precise and cite specific evidence."

        try:
            response = model_manager.llm_reasoning(
                prompt,
                system_prompt=system_prompt,
                model=settings.openai_model,
                temperature=0.2,
            )

            # Try to parse JSON response
            if response.lower().startswith("llm reasoning error"):
                self._record_llm_error("LLM inconsistency detection", response)
                return []
            if response.startswith("["):
                llm_conflicts = json.loads(response)
                return llm_conflicts
            else:
                # Fallback: extract JSON from markdown or text
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    llm_conflicts = json.loads(json_match.group(0))
                    return llm_conflicts
        except Exception as e:
            print(f"LLM inconsistency detection error: {e}")
            self._record_llm_error("LLM inconsistency detection", str(e))

        return []

    def suggest_missing_evidence(
        self,
        events: List[TimelineEvent],
        extracted_data: List[Dict[str, Any]],
        case_type: str = "general",
    ) -> List[str]:
        """
        Suggest missing evidence using coverage analysis and LLM reasoning.
        Only suggests crime-related evidence if the document is actually crime-related.
        
        Args:
            events: Timeline events
            extracted_data: All extracted evidence
            case_type: Type of case (robbery, assault, etc.)
        
        Returns:
            List of missing evidence suggestions
        """
        suggestions: List[str] = []

        if not events and not extracted_data:
            suggestions.append("No evidence uploaded. Please upload evidence files to start analysis.")
            return suggestions

        # Check if this is a crime-related document
        is_crime_related = False
        evidence_types = set()
        classifications = []
        evidence_mentioned = {
            "cctv": False,
            "witness": False,
            "medical": False,
            "fir": False,
            "forensic": False,
            "photographs": False,
        }
        
        for data in extracted_data:
            evidence_types.add(data.get("type", "unknown"))
            if "classification" in data:
                classification_label = data["classification"].get("label", "unknown")
                classifications.append(classification_label)
                # Check if it's crime-related
                if classification_label in ["witness_statement", "medical_report", "fir", "police_memo"]:
                    is_crime_related = True
                # Also check text content for crime-related keywords
                text_content = str(data.get("raw_text", "") + " " + data.get("summary", "")).lower()
                crime_keywords = ["crime", "suspect", "victim", "police", "arrest", "incident", "assault", "robbery", "theft", "murder", "investigation"]
                if any(keyword in text_content for keyword in crime_keywords):
                    is_crime_related = True
                
                # Check what evidence is already mentioned in the document
                if "cctv" in text_content or "surveillance" in text_content or "camera" in text_content:
                    evidence_mentioned["cctv"] = True
                if "witness" in text_content or "statement" in text_content:
                    evidence_mentioned["witness"] = True
                if "medical" in text_content or "injury" in text_content or "hospital" in text_content:
                    evidence_mentioned["medical"] = True
                if "fir" in text_content or "first information report" in text_content:
                    evidence_mentioned["fir"] = True
                if "forensic" in text_content or "dna" in text_content or "fingerprint" in text_content:
                    evidence_mentioned["forensic"] = True
                if "photograph" in text_content or "photo" in text_content or "image" in text_content:
                    evidence_mentioned["photographs"] = True

        # Only suggest crime-related evidence if document is crime-related
        if is_crime_related:
            # Check for common missing evidence types (only if not already mentioned)
            if not evidence_mentioned["cctv"] and "image" not in evidence_types:
                suggestions.append("No CCTV footage or surveillance images found. Consider collecting CCTV from nearby locations.")

            if not evidence_mentioned["witness"] and not any(
                "witness" in str(data.get("classification", {})).lower()
                for data in extracted_data
            ):
                suggestions.append("No witness statements found. Consider obtaining statements from witnesses and bystanders.")

            if not evidence_mentioned["medical"] and not any("medical" in str(data.get("classification", {})).lower() for data in extracted_data):
                suggestions.append("No medical reports found. If injuries occurred, obtain medical examination reports.")

            if not evidence_mentioned["fir"] and not any("fir" in str(data.get("classification", {})).lower() for data in extracted_data):
                suggestions.append("No FIR (First Information Report) found. Ensure FIR is included in case files.")
            
            if not evidence_mentioned["forensic"]:
                suggestions.append("Check for forensic reports (fingerprints, DNA analysis, blood spatter analysis).")

            # Check time coverage
            if events:
                time_span = (events[-1].timestamp - events[0].timestamp).total_seconds() / 3600
                if time_span < 1:
                    suggestions.append("Limited time coverage. Consider collecting evidence from extended time periods before and after the incident.")

            # LLM-based suggestions (only for crime cases)
            if settings.enable_real_ai and settings.openai_api_key and settings.openai_api_key != "" and settings.openai_api_key != "your_openai_api_key_here":
                print(f"[Reasoning] Using OpenAI LLM for missing evidence suggestions...")
                llm_suggestions = self._llm_suggest_missing_evidence(events, extracted_data, case_type)
                print(f"[Reasoning] LLM suggested {len(llm_suggestions)} missing evidence items")
                suggestions.extend(llm_suggestions)
            else:
                print(f"[Reasoning] ⚠️ OpenAI LLM not available for missing evidence suggestions")

            # Only add standard recommendations if not already mentioned
            if not evidence_mentioned["forensic"]:
                suggestions.append("Check for forensic reports (fingerprints, DNA analysis, blood spatter analysis).")
            suggestions.append("Collect phone records and call logs if applicable.")
            suggestions.append("Obtain statements from neighbors and nearby residents.")
            suggestions.append("Verify alibis and cross-check with CCTV timestamps.")
        else:
            # For non-crime documents, provide generic suggestions
            suggestions.append("Document processed successfully. All text and entities have been extracted.")
            
            # Check if document has basic information
            has_entities = any(
                data.get("entities") and len(data.get("entities", [])) > 0
                for data in extracted_data
            )
            if not has_entities:
                suggestions.append("No named entities (names, locations, dates) found in the document.")
            
            has_timestamps = any(
                data.get("time_mentions") and len(data.get("time_mentions", [])) > 0
                for data in extracted_data
            )
            if not has_timestamps:
                suggestions.append("No timestamps or time expressions found in the document.")

        return list(set(suggestions))  # Remove duplicates

    def _llm_suggest_missing_evidence(
        self,
        events: List[TimelineEvent],
        extracted_data: List[Dict[str, Any]],
        case_type: str,
    ) -> List[str]:
        """Use LLM to suggest missing evidence based on case analysis."""
        events_summary = "\n".join([
            f"- {e.timestamp.isoformat()}: {e.description}"
            for e in events[:15]
        ])

        evidence_summary = "\n".join([
            f"- {data.get('type', 'unknown')}: {data.get('summary', '')[:150]}"
            for data in extracted_data[:8]
        ])

        prompt = f"""You are an expert crime investigator analyzing a {case_type} case.

Current Evidence Timeline:
{events_summary}

Current Evidence Files:
{evidence_summary}

Based on standard investigative practices for this type of case, what additional evidence should be collected?

Consider:
- Forensic evidence (DNA, fingerprints, ballistics, toxicology)
- Digital evidence (phone records, CCTV, social media, GPS data)
- Witness statements (independent witnesses, neighbors, employees)
- Medical evidence (detailed injury reports, autopsy if applicable)
- Physical evidence (weapons, clothing, vehicles, scene photos)
- Documentation (FIR, charge sheet, police memos, legal documents)

Provide 3-5 specific, actionable recommendations. Be concise and practical.
Return as a JSON array of strings, e.g., ["Recommendation 1", "Recommendation 2"].
"""

        try:
            response = model_manager.llm_reasoning(
                prompt,
                system_prompt="You are an expert crime investigator providing evidence collection recommendations.",
                model=settings.openai_model,
                temperature=0.4,
            )

            # Try to parse JSON
            import re
            if response.lower().startswith("llm reasoning error"):
                self._record_llm_error("LLM missing-evidence suggestions", response)
                return []
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                suggestions = json.loads(json_match.group(0))
                return suggestions if isinstance(suggestions, list) else []
        except Exception as e:
            print(f"LLM missing evidence suggestion error: {e}")
            self._record_llm_error("LLM missing-evidence suggestions", str(e))

        return []
