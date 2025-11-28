import re
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from typing import List, Optional

try:
    from dateutil import parser as date_parser
except Exception:  # pragma: no cover
    date_parser = None

from app.config import settings


@dataclass
class TimelineEvent:
    """
    Canonical event used in Step 6: Timeline Construction.
    """

    timestamp: datetime
    source: str
    description: str


class TimelineService:
    def build(self, case_id: str, normalized_data: list[dict[str, str]]) -> List[TimelineEvent]:
        """
        Build a simple, sorted timeline from normalized records.

        - For images, use their timestamp and type as a photo/CCTV event.
        - For documents, use time mentions (if any) and summary text.
        """
        events: List[TimelineEvent] = []
        case_anchor = self._derive_case_anchor(normalized_data)

        for idx, record in enumerate(normalized_data):
            base_source = record.get("type", "unknown")
            source_label = self._resolve_source_label(record, base_source, idx)
            record_anchor = self._derive_record_anchor(record, case_anchor)

            # Image-based event
            if record.get("type") == "image":
                ts_raw = record.get("timestamp") or datetime.utcnow().isoformat()
                dt = self._safe_parse_datetime(ts_raw) or datetime.utcnow()
                desc = f"Image evidence ({record.get('location', 'UNKNOWN')})"
                # If object detection is disabled or image classification is disabled,
                # mark the event as supporting evidence so the UI can prioritize document events
                if not getattr(settings, 'enable_object_detection', True) or not getattr(settings, 'enable_image_classification', True):
                    source_label = f"supporting_{source_label}"
                events.append(
                    TimelineEvent(
                        timestamp=dt,
                        source=source_label,
                        description=desc,
                    )
                )

            # Document-based events from text time mentions
            elif record.get("type") == "document":
                raw_text = record.get("raw_text", "")
                time_mentions = record.get("time_mentions") or []
                
                # Extract actual events from the document text
                extracted_events = self._extract_events_from_text(
                    raw_text,
                    time_mentions,
                    record_anchor,
                    source_label,
                )
                
                if extracted_events:
                    print(f"[Timeline] Extracted {len(extracted_events)} events from document")
                    events.extend(extracted_events)
                elif time_mentions:
                    print(f"[Timeline] Using fallback extraction for {len(time_mentions)} time mentions")
                    # Fallback: create events from time mentions
                    for t in time_mentions:
                        try:
                            time_obj = self._parse_time_only(t)
                            dt = self._combine_date_time(record_anchor, time_obj)
                            # Find context around this time in the text
                            context = self._find_time_context(raw_text, t)
                            events.append(
                                TimelineEvent(
                                    timestamp=dt,
                                    source=source_label,
                                    description=context or f"Event at {t}",
                                )
                            )
                        except Exception:
                            pass
                else:
                    # No time mentions - use document date or current time
                    dt = record_anchor or datetime.utcnow()
                    events.append(
                        TimelineEvent(
                            timestamp=dt,
                            source=source_label,
                            description=record.get("summary", "Document evidence")[:200],
                        )
                    )

        events.sort(key=lambda e: e.timestamp)
        return events
    
    def _extract_events_from_text(self, text: str, time_mentions: List[str], anchor_date: Optional[datetime], source_label: str) -> List[TimelineEvent]:
        """Extract structured events from document text with source identification."""
        events = []
        seen_events = set()  # Track to avoid duplicates
        
        # Pattern to find events with times: "at 8:12:34 PM", "around 8:15 PM", etc.
        time_pattern = r'(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm))'
        
        # Identify source types in text
        source_keywords = {
            "CCTV": ["cctv", "camera", "surveillance", "footage"],
            "Witness": ["witness", "statement", "saw", "heard", "observed"],
            "Medical": ["medical", "hospital", "injury", "laceration", "fracture", "examination"],
            "Police": ["police", "officer", "investigation", "memo", "fir"],
        }
        
        # Find all time mentions with context
        for match in re.finditer(time_pattern, text, re.IGNORECASE):
            time_str = match.group(1)
            start_pos = max(0, match.start() - 200)
            end_pos = min(len(text), match.end() + 200)
            context = text[start_pos:end_pos].strip()
            
            # Clean up context - remove extra whitespace
            context = re.sub(r'\s+', ' ', context)
            
            # Identify source from context
            source = "Document"
            context_lower = context.lower()
            for src_name, keywords in source_keywords.items():
                if any(kw in context_lower for kw in keywords):
                    source = src_name
                    break
            
            # Extract meaningful event description
            # Look for action verbs and key phrases
            event_desc = self._extract_event_description(context, time_str)
            
            # Create unique key to avoid duplicates
            event_key = f"{time_str}_{event_desc[:50]}"
            if event_key in seen_events:
                continue
            seen_events.add(event_key)
            
            # Parse time
            try:
                time_obj = self._parse_time_only(time_str)
                dt = self._combine_date_time(anchor_date, time_obj)
                
                events.append(
                    TimelineEvent(
                        timestamp=dt,
                        source=source if source != "Document" else source_label,
                        description=event_desc,
                    )
                )
            except Exception as e:
                print(f"Error parsing time {time_str}: {e}")
                pass
        
        return events
    
    def _extract_event_description(self, context: str, time_str: str) -> str:
        """Extract a concise event description from context."""
        # Look for key action phrases
        action_patterns = [
            r'(entered|enters|went into|arrived at)\s+([^.]{0,100})',
            r'(exited|exits|left|departed)\s+([^.]{0,100})',
            r'(altercation|argument|conflict|incident)\s+([^.]{0,100})',
            r'(sustained|received|suffered)\s+(injury|injuries|damage)\s+([^.]{0,100})',
            r'(wearing|wore|wearing)\s+([^.]{0,100})',
            r'(heard|saw|observed|witnessed)\s+([^.]{0,100})',
        ]
        
        context_lower = context.lower()
        for pattern in action_patterns:
            match = re.search(pattern, context_lower, re.IGNORECASE)
            if match:
                # Extract the relevant sentence
                sentences = re.split(r'[.!?]\s+', context)
                for sentence in sentences:
                    if time_str in sentence or any(match.group(1).lower() in sentence.lower() for match in [re.search(pattern, sentence, re.IGNORECASE)] if match):
                        # Clean up the sentence
                        desc = sentence.strip()
                        # Remove excessive whitespace
                        desc = re.sub(r'\s+', ' ', desc)
                        # Limit length
                        return desc[:200]
        
        # Fallback: return first sentence containing the time
        sentences = re.split(r'[.!?]\s+', context)
        for sentence in sentences:
            if time_str in sentence:
                return sentence.strip()[:200]
        
        # Last resort: return context snippet
        return context[:200]
    
    def _find_time_context(self, text: str, time_str: str) -> str:
        """Find context around a time mention in the text."""
        # Find the time in the text
        pattern = re.escape(time_str)
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end].strip()
            # Extract first sentence
            sentences = re.split(r'[.!?]\s+', context)
            if sentences:
                return sentences[0][:150]
            return context[:150]
        return f"Event at {time_str}"

    def _resolve_source_label(self, record: dict, fallback: str, idx: int) -> str:
        """Create a stable source identifier for timeline rows."""
        label = record.get("classification", {}).get("label") or fallback
        filename = record.get("filename") or record.get("ingestion", {}).get("filename")
        parts = [label or fallback, str(idx)]
        source = "_".join(filter(None, parts))
        if filename:
            source = f"{source}:{filename}"
        return source

    def _derive_case_anchor(self, records: list[dict]) -> Optional[datetime]:
        """Determine the incident date anchor for the entire case."""
        candidates: List[datetime] = []
        for record in records:
            # Check dates field first (more reliable for incident dates)
            for raw in record.get("dates", []) or []:
                parsed = self._safe_parse_datetime(raw)
                if parsed and parsed.year >= 2020:  # Only accept reasonable dates
                    candidates.append(parsed)
            # Check timestamps
            for raw in record.get("timestamps", []) or []:
                parsed = self._safe_parse_datetime(raw)
                if parsed and parsed.year >= 2020:
                    candidates.append(parsed)
            # Also check raw_text for date mentions
            raw_text = record.get("raw_text", "")
            if raw_text:
                # Look for "16 March 2025" or "March 16, 2025" patterns
                date_patterns = [
                    r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
                    r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b',
                ]
                for pattern in date_patterns:
                    matches = re.findall(pattern, raw_text, re.IGNORECASE)
                    for match in matches:
                        try:
                            if len(match) == 3:
                                if match[0].isdigit():
                                    # "16 March 2025"
                                    day, month_name, year = match
                                else:
                                    # "March 16, 2025"
                                    month_name, day, year = match
                                month_map = {
                                    'january': 1, 'february': 2, 'march': 3, 'april': 4,
                                    'may': 5, 'june': 6, 'july': 7, 'august': 8,
                                    'september': 9, 'october': 10, 'november': 11, 'december': 12
                                }
                                month = month_map.get(month_name.lower())
                                if month:
                                    dt = datetime(int(year), month, int(day))
                                    if dt.year >= 2020:
                                        candidates.append(dt)
                        except (ValueError, IndexError):
                            continue
        if not candidates:
            return None
        # Return the earliest date (most likely to be the incident date)
        return min(candidates)

    def _derive_record_anchor(self, record: dict, case_anchor: Optional[datetime]) -> Optional[datetime]:
        """Pick the best anchor date for a single artifact."""
        for field in ("dates", "timestamps"):
            for raw in record.get(field, []) or []:
                parsed = self._safe_parse_datetime(raw)
                if parsed:
                    return parsed
        return case_anchor

    def _safe_parse_datetime(self, value: str) -> Optional[datetime]:
        if not value or not isinstance(value, str):
            return None
        value = value.strip()
        formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y:%m:%d %H:%M:%S",
            "%Y-%m-%d",
            "%d %B %Y",  # "14 March 2025"
            "%d %b %Y",  # "14 Mar 2025"
            "%B %d, %Y",  # "March 14, 2025"
            "%b %d, %Y",  # "Mar 14, 2025"
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(value, fmt)
                if fmt in ["%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y"]:
                    return datetime.combine(dt.date(), dt_time(0, 0))
                return dt
            except ValueError:
                continue
        if date_parser:
            try:
                parsed = date_parser.parse(value, fuzzy=True, default=datetime.utcnow())
                return parsed
            except Exception:
                return None
        return None

    def _parse_time_only(self, value: str) -> dt_time:
        """Parse a time expression without date info."""
        value = value.strip()
        formats = ["%I:%M %p", "%H:%M", "%I:%M:%S %p", "%H:%M:%S"]
        for fmt in formats:
            try:
                return datetime.strptime(value, fmt).time()
            except ValueError:
                continue
        if date_parser:
            parsed = date_parser.parse(value)
            return parsed.time()
        raise ValueError(f"Could not parse time: {value}")

    def _combine_date_time(self, anchor: Optional[datetime], time_obj: dt_time) -> datetime:
        """Combine an anchor date with a parsed time; fallback to current date."""
        base = anchor or datetime.utcnow()
        return datetime.combine(base.date(), time_obj)

